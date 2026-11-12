"""Ingesta complementaria de lo reciente de SECOP II a bronze (paso 0 de la noche).

Problema que resuelve: la Lambda proyecto-integrador-ingesta (de ingeniería de datos) pide cada dataset con
``:updated_at >= watermark``. SECOP reescribe ``:updated_at`` en millones de filas al día y la Lambda se corta en
~200 mil, así que casi no le llegan los registros nuevos. Proveedores ni siquiera tiene ingesta diaria.

Este script pide solo lo de los últimos N días según una fecha de negocio de cada tabla, con la MISMA paginación
keyset (``$order=:updated_at,:id``) y el MISMO formato que la Lambda (una fila JSON por línea, como la devuelve la
API con ``$select=:*, *``, en gzip), en la carpeta que lee el job de silver de esa tabla:

    s3://proyecto-integrador-bucket-01/bronze/<carpeta>/ingesta/fecha=AAAA-MM-DD/part-recientes-HHMMSS-NNNNN.json.gz

Cada job de silver lee todo ``ingesta/fecha=*/`` y deduplica con "gana la ingesta y luego la versión más
reciente" (procesos: una fila por id_proceso; proponentes: por proceso y proveedor; contratos: por id_contrato;
proveedores: por código y luego por proveedor_key). Lo que traigan este script y la Lambda a la vez no se duplica.

    python -m jobs.ingesta_recientes --tabla contratos --destino /tmp/prueba   # prueba local (no toca S3)
    python -m jobs.ingesta_recientes --tabla procesos                            # escribe en bronze
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

BUCKET = "proyecto-integrador-bucket-01"
BASE_URL = "https://www.datos.gov.co"
PAGE_SIZE = 10_000  # igual que la Lambda (contratos trae muchas columnas de texto largo)


@dataclass(frozen=True)
class Tabla:
    dataset: str  # id en datos.gov.co (Socrata)
    columna_fecha: str  # fecha de negocio para la ventana (no :updated_at, que SECOP reescribe)
    carpeta: str  # carpeta de bronze que lee el job de silver
    dias: int = 45


TABLAS = {
    "procesos": Tabla("p6dx-8zbt", "fecha_de_publicacion_del", "procesos-contratacion"),
    "proponentes": Tabla("hgi6-6wh3", "fecha_publicaci_n", "proponentes-procesos"),  # publicación del proceso
    "contratos": Tabla("jbjy-vk9h", "fecha_de_firma", "contratos-electronicos"),
    "proveedores": Tabla("qmzu-gj57", "fecha_creacion", "proveedores-registrados"),  # la Lambda no la ingesta
}


def http_get_json(url: str, token: str | None, reintentos: int = 5):
    headers = {"Accept": "application/json"}
    if token:
        headers["X-App-Token"] = token
    for i in range(reintentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:  # timeouts o 5xx de Socrata: reintento con espera exponencial (como la Lambda)
            if i == reintentos - 1:
                raise
            print(f"  error ({e}); reintento en {2**i * 5} s", flush=True)
            time.sleep(2**i * 5)


def url_pagina(t: Tabla, desde: str, ultimo: tuple[str, str] | None) -> str:
    cond = [f"{t.columna_fecha} >= '{desde}T00:00:00'"]
    if ultimo:
        u, i = ultimo
        cond.append(f"(:updated_at > '{u}' OR (:updated_at = '{u}' AND :id > '{i}'))")
    params = {"$select": ":*, *", "$order": ":updated_at, :id", "$limit": PAGE_SIZE, "$where": " AND ".join(cond)}
    return f"{BASE_URL}/resource/{t.dataset}.json?" + urllib.parse.urlencode(params)


def comprimir(filas: list[dict]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in filas:
            gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tabla", choices=sorted(TABLAS), default="procesos")
    ap.add_argument("--dias", type=int, help="ventana en días (por defecto, la de la tabla)")
    ap.add_argument("--destino", help="carpeta local para probar; si no se da, escribe en bronze (S3)")
    ap.add_argument("--max-paginas", type=int, default=0, help="corta tras N páginas (pruebas)")
    a = ap.parse_args(argv)
    t = TABLAS[a.tabla]

    import boto3

    ahora = datetime.now(UTC)
    fecha_ingesta, run_id = ahora.strftime("%Y-%m-%d"), ahora.strftime("%H%M%S")  # misma convención que la Lambda
    desde = (ahora - timedelta(days=a.dias or t.dias)).strftime("%Y-%m-%d")
    token = boto3.client("ssm").get_parameter(Name="/secop/app-token", WithDecryption=True)["Parameter"]["Value"]
    s3 = None if a.destino else boto3.client("s3")
    prefijo = f"bronze/{t.carpeta}/ingesta/fecha={fecha_ingesta}/part-recientes-{run_id}"
    print(
        f"{a.tabla} ({t.dataset}): {t.columna_fecha} >= {desde} → {prefijo}-* "
        f"({'local: ' + a.destino if a.destino else 's3'})"
    )

    t0, total, pagina, ultimo, fechas = time.time(), 0, 0, None, []
    while True:
        filas = http_get_json(url_pagina(t, desde, ultimo), token)
        if not filas:
            break
        clave = f"{prefijo}-{pagina:05d}.json.gz"
        cuerpo = comprimir(filas)
        if s3:
            s3.put_object(Bucket=BUCKET, Key=clave, Body=cuerpo)
        else:
            ruta = Path(a.destino) / clave
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(cuerpo)
        total += len(filas)
        fechas += [r.get(t.columna_fecha) for r in filas if r.get(t.columna_fecha)]
        ultimo = (filas[-1].get(":updated_at"), filas[-1].get(":id"))
        print(f"  página {pagina}: {len(filas):,} filas (acumulado {total:,}) → {clave}", flush=True)
        if len(filas) < PAGE_SIZE or (a.max_paginas and pagina + 1 >= a.max_paginas):
            break
        pagina += 1
    rango = f"{min(fechas)[:10]} → {max(fechas)[:10]}" if fechas else "-"
    print(
        f"RESUMEN tabla={a.tabla} filas={total:,} paginas={pagina + 1 if total else 0} fechas={rango} "
        f"duracion={time.time() - t0:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
