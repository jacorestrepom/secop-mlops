"""Fixtures compartidas por las pruebas que usan datos reales (paridad y equivalencia).

Variables de entorno (si falta SECOP_DATA_DIR, esas pruebas se saltan):
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
    return {"m1": _ruta_modelo("M1_final_unico.txt", DATA / "retrieval_v2"),
            "g1": _ruta_modelo("g1_pgana.txt", DATA / "final" / "modelos"),
            "v2": _ruta_modelo("ranking_v2_final.txt", DATA / "final" / "modelos")}


@pytest.fixture(scope="session")
def candidatos(modelos):
    """recuperar() sobre las tablas limpias, con las empresas y procesos que el notebook 03 usó en el mes."""
    from ml.retrieval import HistoriaRetrieval, recuperar

    h = HistoriaRetrieval.desde_carpeta(DATA / "clean")
    empresas, procesos = h.entradas_del_mes(FECHA_CORTE)
    top = recuperar(empresas, procesos, FECHA_CORTE, historia=h, modelo=modelos["m1"])
    del h
    gc.collect()
    return top


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
def listas_notebook():
    ref = pd.read_parquet(DATA / "final" / "listas" / f"listas_{MES}.parquet")
    return ref[ref.lista.isin(NOMBRES_LISTAS)].assign(lista=lambda x: x.lista.map(NOMBRES_LISTAS))
