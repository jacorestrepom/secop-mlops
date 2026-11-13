"""Publicación atómica de las recomendaciones en el esquema serving (guía de traspaso, Paso 4).

1. Crea una publicación en_progreso (se confirma aparte: si algo falla, queda el rastro como 'fallida').
2. Carga empresas, procesos, recomendaciones y estadísticas con COPY, en una transacción.
3. En UNA transacción: la anterior deja de ser actual, la nueva pasa a publicada y es_actual, se borran otras
   versiones de la misma fecha_corte (republicar el mismo día reemplaza, no duplica) y las publicadas más
   viejas que las últimas RETENER.
Si algo falla: la nueva queda 'fallida', la versión actual NO cambia y el proceso sale con código 1.

Uso:
    python -m jobs.publicar --simulados api/datos_simulados.json --secreto secop/db-job   # publicación 0
    DATABASE_URL=postgresql://... python -m jobs.publicar --simulados api/datos_simulados.json
"""

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date

import psycopg2

RETENER = 7
CANDADO = 731_906_405  # pg_advisory_xact_lock: una sola publicación a la vez

COLUMNAS = {
    "empresas": ["proveedor_key", "nombre", "departamento", "es_pyme", "tiene_historial"],
    "procesos": [
        "id_proceso", "entidad", "objeto", "categoria", "unspsc_familia", "departamento", "municipio",
        "modalidad", "precio_base", "fecha_publicacion", "fecha_limite_ofertas", "oferentes_esperados",
        "url_proceso",
    ],
    "recomendaciones": [
        "proveedor_key", "lista", "posicion", "id_proceso", "probabilidad", "score_ranking", "score_retrieval",
        "oferentes_esperados", "motivos", "unspsc_inferido",
    ],
    "estadisticas": ["indicador", "descripcion", "unidad", "datos"],
}  # fmt: skip


class PublicacionFallida(RuntimeError):
    pass


@dataclass
class Publicacion:
    fecha_corte: date | str
    versiones: dict[str, str]  # {"retrieval": ..., "pgana": ..., "ranking": ...}
    empresas: list[dict]
    procesos: list[dict]
    recomendaciones: list[dict]
    estadisticas: list[dict] = field(default_factory=list)
    silver_snapshots: dict = field(default_factory=dict)
    metricas: dict = field(default_factory=dict)  # lo que sirve GET /metadata
    datos_simulados: bool = False


def _antes_de_activar(cur, version_id: int) -> None:
    """Punto de enganche para las pruebas (simular una falla justo antes del cambio de versión)."""


def _csv(filas: list[dict], columnas: list[str]) -> io.StringIO:
    buf = io.StringIO()
    w = csv.writer(buf)
    for f in filas:
        fila = []
        for c in columnas:
            v = f.get(c)
            if isinstance(v, dict | list):
                v = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, bool):
                v = "t" if v else "f"
            fila.append("" if v is None else v)  # vacío sin comillas = NULL en COPY csv
        w.writerow(fila)
    buf.seek(0)
    return buf


# Valores por defecto de columnas agregadas después (las publicaciones de demo no las traen)
DEFECTOS = {"recomendaciones": {"unspsc_inferido": False}}


def _copiar(cur, tabla: str, version_id: int, filas: list[dict]) -> None:
    cols = ["version_id"] + COLUMNAS[tabla]
    datos = [{"version_id": version_id, **DEFECTOS.get(tabla, {}), **f} for f in filas]
    cur.copy_expert(f"COPY serving.{tabla} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv)", _csv(datos, cols))


def publicar(dsn: str, pub: Publicacion, retener: int = RETENER) -> int:
    """Publica `pub` de forma atómica y devuelve su version_id. Lanza PublicacionFallida si algo sale mal."""
    con = psycopg2.connect(dsn)
    version_id = None
    try:
        with con, con.cursor() as cur:  # 1 · registro en_progreso (se confirma ya)
            cur.execute(
                """INSERT INTO serving.publicaciones (fecha_corte, version_retrieval, version_pgana, version_ranking,
                       silver_snapshots, n_empresas, metricas, datos_simulados)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING version_id""",
                (pub.fecha_corte, pub.versiones["retrieval"], pub.versiones["pgana"], pub.versiones["ranking"],
                 json.dumps(pub.silver_snapshots), len(pub.empresas), json.dumps(pub.metricas, ensure_ascii=False),
                 pub.datos_simulados),
            )  # fmt: skip
            version_id = cur.fetchone()[0]

        with con, con.cursor() as cur:  # 2 · carga con COPY
            _copiar(cur, "empresas", version_id, pub.empresas)
            _copiar(cur, "procesos", version_id, pub.procesos)
            _copiar(cur, "recomendaciones", version_id, pub.recomendaciones)
            _copiar(cur, "estadisticas", version_id, pub.estadisticas)

        with con, con.cursor() as cur:  # 3 · cambio de versión, en una sola transacción
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (CANDADO,))
            _antes_de_activar(cur, version_id)
            cur.execute("UPDATE serving.publicaciones SET es_actual = false WHERE es_actual")
            cur.execute(
                """UPDATE serving.publicaciones SET es_actual = true, estado = 'publicada', publicada_en = now()
                   WHERE version_id = %s AND estado = 'en_progreso'""",
                (version_id,),
            )
            if cur.rowcount != 1:
                raise PublicacionFallida(f"la versión {version_id} ya no estaba en_progreso")
            # Republicar el mismo día reemplaza: se borran las otras versiones de esa fecha_corte
            cur.execute(
                """DELETE FROM serving.publicaciones
                   WHERE fecha_corte = %s AND version_id <> %s AND estado <> 'en_progreso'""",
                (pub.fecha_corte, version_id),
            )
            # Retención: solo las últimas `retener` publicadas (las fallidas viejas también se limpian)
            cur.execute(
                """DELETE FROM serving.publicaciones
                   WHERE estado <> 'en_progreso' AND version_id <> %s AND version_id NOT IN (
                       SELECT version_id FROM serving.publicaciones WHERE estado = 'publicada'
                       ORDER BY version_id DESC LIMIT %s)""",
                (version_id, retener),
            )
        return version_id
    except Exception as e:
        if version_id is not None:
            with psycopg2.connect(dsn) as aparte, aparte.cursor() as cur:
                cur.execute(
                    """UPDATE serving.publicaciones SET estado = 'fallida', error = %s
                       WHERE version_id = %s AND NOT es_actual""",
                    (f"{type(e).__name__}: {e}"[:2000], version_id),
                )
            aparte.close()
        raise PublicacionFallida(f"publicación {version_id} fallida: {type(e).__name__}: {e}") from e
    finally:
        con.close()


def desde_simulados(ruta: str, fecha_corte: str | None = None) -> Publicacion:
    """Convierte api/datos_simulados.json en una Publicación (publicación 0, datos_simulados=true)."""
    d = json.load(open(ruta, encoding="utf-8"))
    procesos = {p["id_proceso"]: p for p in d["procesos"]}
    recs = []
    for key, listas in d["recomendaciones"].items():
        for lista, filas in listas.items():
            for r in filas:
                recs.append({"proveedor_key": key, "lista": lista, "posicion": r["posicion"],
                             "id_proceso": r["id_proceso"], "probabilidad": r["probabilidad"],
                             "oferentes_esperados": procesos[r["id_proceso"]].get("oferentes_esperados"),
                             "motivos": r["motivos"]})  # fmt: skip
    v = d["version_modelos"]
    return Publicacion(
        fecha_corte=fecha_corte or d["fecha_corte"],
        versiones={"retrieval": v["retrieval"], "pgana": v["pgana"], "ranking": v["ranking"]},
        empresas=d["empresas"],
        procesos=d["procesos"],
        recomendaciones=recs,
        estadisticas=list(d["estadisticas"].values()),
        metricas=d["metricas"],
        datos_simulados=True,
    )


def dsn_desde_secreto(nombre: str, host: str | None = None, port: int | None = None) -> str:
    import boto3

    s = json.loads(boto3.client("secretsmanager").get_secret_value(SecretId=nombre)["SecretString"])
    return psycopg2.extensions.make_dsn(
        host=host or s["host"], port=port or s["port"], dbname=s["dbname"], user=s["username"],
        password=s["password"], sslmode="require",
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--simulados", metavar="JSON", help="publica datos simulados (publicación 0)")
    ap.add_argument("--fecha-corte", help="fecha_corte a publicar (por defecto, la del archivo)")
    ap.add_argument("--secreto", help="toma la conexión de este secreto (si no, DATABASE_URL)")
    ap.add_argument("--host", help="sobrescribe el host del secreto (p. ej. localhost por un túnel)")
    ap.add_argument("--port", type=int, help="sobrescribe el puerto del secreto")
    a = ap.parse_args(argv)
    if not a.simulados:
        ap.error("por ahora solo --simulados; las listas reales las publicará score.py llamando a publicar()")
    dsn = dsn_desde_secreto(a.secreto, a.host, a.port) if a.secreto else os.environ["DATABASE_URL"]
    try:
        v = publicar(dsn, desde_simulados(a.simulados, a.fecha_corte))
    except PublicacionFallida as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Publicada la versión {v} (es_actual)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
