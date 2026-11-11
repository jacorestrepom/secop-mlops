"""Fixtures compartidas de las pruebas.

1. Base PostgreSQL para las pruebas de publicación y de la fuente RDS. Se activa con TEST_DATABASE_URL (en CI lo
   pone el servicio postgres del workflow). En local:
    docker run -d --name secop-pg-prueba -e POSTGRES_PASSWORD=prueba -e POSTGRES_DB=secop -p 55432:5432 postgres:16
    export TEST_DATABASE_URL=postgresql://postgres:prueba@localhost:55432/secop

2. Datos reales del modelo (paridad, equivalencia y limpieza). Si falta SECOP_DATA_DIR, esas pruebas se saltan:
    SECOP_DATA_DIR      carpeta data/ con clean/, retrieval_v2/ y final/
    SECOP_MODELOS_DIR   (opcional) carpeta con g1_pgana.txt, ranking_v2_final.txt y M1_final_unico.txt;
                        por defecto data/final/modelos/ y data/retrieval_v2/
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pandas as pd
import pytest

# --------------------------------------------------------------------------
# 1. PostgreSQL
# --------------------------------------------------------------------------


RAIZ = Path(__file__).resolve().parent.parent
SQL_SERVING = RAIZ / "infra" / "sql" / "serving.sql"
SIMULADOS = RAIZ / "api" / "datos_simulados.json"


@pytest.fixture
def pg_dsn():
    """DSN de una base con el esquema serving recién creado (vacío)."""
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL no está definido")
    psycopg2 = pytest.importorskip("psycopg2")
    con = psycopg2.connect(dsn)
    con.autocommit = True
    with con.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS serving CASCADE")
        cur.execute(SQL_SERVING.read_text())
    con.close()
    return dsn


# --------------------------------------------------------------------------
# 2. Datos reales del modelo
# --------------------------------------------------------------------------

MES = "202603"
FECHA_CORTE = "2026-03-01"
DATA = Path(os.environ["SECOP_DATA_DIR"]) if os.environ.get("SECOP_DATA_DIR") else None
HAY_DATOS = DATA is not None and (DATA / "clean").is_dir()
sin_datos = pytest.mark.skipif(not HAY_DATOS, reason="SECOP_DATA_DIR no está definido o no tiene clean/")

NOMBRES_LISTAS = {"1 · tu historial (ranking v2)": "historial", "2 · oportunidades (G1 + filtros)": "oportunidad"}


def _ruta_modelo(nombre: str, por_defecto: Path) -> Path:
    d = os.environ.get("SECOP_MODELOS_DIR")
    return Path(d) / nombre if d else por_defecto / nombre


@pytest.fixture(scope="session")
def modelos():
    return {
        "m1": _ruta_modelo("M1_final_unico.txt", DATA / "retrieval_v2"),
        "g1": _ruta_modelo("g1_pgana.txt", DATA / "final" / "modelos"),
        "v2": _ruta_modelo("ranking_v2_final.txt", DATA / "final" / "modelos"),
    }


@pytest.fixture(scope="session")
def retrieval_ml(modelos):
    """recuperar() sobre las tablas limpias, con las empresas y procesos que el notebook 03 usó en el mes.
    Devuelve (candidatos, positivos del mes) para poder medir el Recall@150."""
    from ml.retrieval import HistoriaRetrieval, recuperar

    h = HistoriaRetrieval.desde_carpeta(DATA / "clean")
    empresas, procesos = h.entradas_del_mes(FECHA_CORTE)
    top = recuperar(empresas, procesos, FECHA_CORTE, historia=h, modelo=modelos["m1"])
    positivos = h.positivos_del_mes(FECHA_CORTE)
    del h
    gc.collect()
    return top, positivos


@pytest.fixture(scope="session")
def candidatos(retrieval_ml):
    return retrieval_ml[0]


@pytest.fixture(scope="session")
def historia_features():
    from ml.features import Historia

    return Historia.desde_carpeta(DATA / "clean")


@pytest.fixture(scope="session")
def puntuados_ml(candidatos, historia_features, modelos):
    """Punta a punta: recuperar → construir → rankear."""
    from ml.features import construir
    from ml.ranking import rankear

    return rankear(construir(candidatos, historia=historia_features), g1=modelos["g1"], v2=modelos["v2"])


@pytest.fixture(scope="session")
def listas_ml(puntuados_ml):
    from ml.listas import armar_listas

    return armar_listas(puntuados_ml)


@pytest.fixture(scope="session")
def top_notebook():
    """Candidatos del notebook 03 (puestos 1–150) traducidos a llaves, como la sección 6 del notebook final."""
    top = pd.read_parquet(
        DATA / "retrieval_v2" / "top_retrieval" / f"top_{MES}.parquet",
        columns=["pk", "pid", "score_retrieval", "rank_retrieval", "es_frio"],
    )
    return (
        top[top.rank_retrieval <= 150]
        .merge(pd.read_parquet(DATA / "clean" / "map_pk.parquet"), on="pk", how="left")
        .merge(pd.read_parquet(DATA / "clean" / "map_pid.parquet"), on="pid", how="left")
        .reset_index(drop=True)
    )


@pytest.fixture(scope="session")
def listas_notebook():
    ref = pd.read_parquet(DATA / "final" / "listas" / f"listas_{MES}.parquet")
    return ref[ref.lista.isin(NOMBRES_LISTAS)].assign(lista=lambda x: x.lista.map(NOMBRES_LISTAS))
