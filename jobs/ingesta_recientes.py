"""Ingesta complementaria de procesos recientes de SECOP II a bronze (PREPARADA, todavía no se despliega).

Problema que resuelve: la Lambda proyecto-integrador-ingesta (de ingeniería de datos) pide procesos-contratacion
con ``:updated_at >= watermark``. SECOP reescribe ``:updated_at`` en millones de filas al día y la Lambda se corta
en ~200 mil, así que casi no le llegan los procesos nuevos.

Este script pide solo lo publicado en los últimos N días (``fecha_de_publicacion_del``), con la MISMA paginación
keyset (``$order=:updated_at,:id``) y el MISMO formato que la Lambda (una fila JSON por línea, como la devuelve la
API con ``$select=:*, *``, en gzip), en la misma carpeta que lee secop-silver-procesos:

    s3://proyecto-integrador-bucket-01/bronze/procesos-contratacion/ingesta/fecha=AAAA-MM-DD/part-recientes-HHMMSS-NNNNN.json.gz

secop-silver-procesos lee todo ``ingesta/fecha=*/`` y su regla R4 deja una fila por id_proceso (gana la versión
más reciente): lo que traiga este script y la Lambda a la vez no se duplica.

    python -m jobs.ingesta_recientes --destino /tmp/prueba          # prueba local (no toca S3)
    python -m jobs.ingesta_recientes                                  # escribe en bronze (solo cuando se despliegue)
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
from datetime import UTC, datetime, timedelta
from pathlib import Path

BUCKET = "proyecto-integrador-bucket-01"
BASE_URL = "https://www.datos.gov.co"
DATASET = "p6dx-8zbt"  # procesos-contratacion
CARPETA = "bronze/procesos-contratacion/ingesta"
PAGE_SIZE = 10_000  # igual que la Lambda
COLUMNA_FECHA = "fecha_de_publicacion_del"


def http_get_json(url: str, token: str | None, reintentos: int = 4):
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


def url_pagina(desde: str, ultimo: tuple[str, str] | None) -> str:
    cond = [f"{COLUMNA_FECHA} >= '{desde}T00:00:00'"]
    if ultimo:
        u, i = ultimo
        cond.append(f"(:updated_at > '{u}' OR (:updated_at = '{u}' AND :id > '{i}'))")
    params = {"$select": ":*, *", "$order": ":updated_at, :id", "$limit": PAGE_SIZE, "$where": " AND ".join(cond)}
    return f"{BASE_URL}/resource/{DATASET}.json?" + urllib.parse.urlencode(params)


def comprimir(filas: list[dict]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in filas:
            gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dias", type=int, default=45, help="procesos publicados en los últimos N días")
    ap.add_argument("--destino", help="carpeta local para probar; si no se da, escribe en bronze (S3)")
    ap.add_argument("--max-paginas", type=int, default=0, help="corta tras N páginas (pruebas)")
    a = ap.parse_args(argv)

    import boto3

    ahora = datetime.now(UTC)
    fecha_ingesta, run_id = ahora.strftime("%Y-%m-%d"), ahora.strftime("%H%M%S")  # misma convención que la Lambda
    desde = (ahora - timedelta(days=a.dias)).strftime("%Y-%m-%d")
    token = boto3.client("ssm").get_parameter(Name="/secop/app-token", WithDecryption=True)["Parameter"]["Value"]
    s3 = None if a.destino else boto3.client("s3")
    print(f"procesos publicados desde {desde} → fecha={fecha_ingesta} ({'local: ' + a.destino if a.destino else 's3'})")

    t0, total, pagina, ultimo, fechas = time.time(), 0, 0, None, []
    while True:
        filas = http_get_json(url_pagina(desde, ultimo), token)
        if not filas:
            break
        clave = f"{CARPETA}/fecha={fecha_ingesta}/part-recientes-{run_id}-{pagina:05d}.json.gz"
        cuerpo = comprimir(filas)
        if s3:
            s3.put_object(Bucket=BUCKET, Key=clave, Body=cuerpo)
        else:
            ruta = Path(a.destino) / clave
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(cuerpo)
        total += len(filas)
        fechas += [r.get(COLUMNA_FECHA) for r in filas if r.get(COLUMNA_FECHA)]
        ultimo = (filas[-1].get(":updated_at"), filas[-1].get(":id"))
        print(f"  página {pagina}: {len(filas):,} filas (acumulado {total:,}) → {clave}", flush=True)
        if len(filas) < PAGE_SIZE or (a.max_paginas and pagina + 1 >= a.max_paginas):
            break
        pagina += 1
    rango = f"{min(fechas)[:10]} → {max(fechas)[:10]}" if fechas else "-"
    print(f"RESUMEN filas={total:,} paginas={pagina + 1 if total else 0} publicados={rango} "
          f"duracion={time.time() - t0:.0f} s")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
