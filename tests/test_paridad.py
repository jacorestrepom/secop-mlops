"""Prueba de paridad exacta de ml/ (marzo de 2026).

Dos partes:

1. Contra la referencia generada con ml/ dentro del contenedor de pruebas (tests/referencia/): cada etapa de
   punta a punta (recuperar, construir, rankear, armar_listas) debe dar exactamente lo mismo, con p_gana a menos
   de 1e-9. Solo vale DENTRO de esa imagen linux/amd64: los empates y la numérica del SVD cambian entre
   plataformas. Necesita SECOP_REFERENCIA_DIR (carpeta con 202603/candidatos.parquet, ...); si falta, se salta.

2. Contra el notebook, para los módulos que no dependen de la plataforma: config.json, y rankear + armar_listas
   sobre las features del notebook (cand_202603) deben reproducir exactamente sus puntajes y sus listas.

La equivalencia con las listas del notebook de punta a punta está en tests/test_equivalencia.py.
Variables de entorno: SECOP_DATA_DIR (obligatoria), SECOP_REFERENCIA_DIR, SECOP_MODELOS_DIR (ver conftest.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import DATA, MES, sin_datos

pytestmark = sin_datos

REF = Path(os.environ["SECOP_REFERENCIA_DIR"]) / MES if os.environ.get("SECOP_REFERENCIA_DIR") else None
con_referencia = pytest.mark.skipif(REF is None or not REF.is_dir(),
                                    reason="SECOP_REFERENCIA_DIR no está definido o no tiene la carpeta del mes")


def _iguales(a: pd.Series, b: pd.Series) -> bool:
    if isinstance(a.dtype, pd.CategoricalDtype) or a.dtype == object or pd.api.types.is_string_dtype(a.dtype):
        return bool((a.astype(str).to_numpy() == b.astype(str).to_numpy()).all())
    return bool(np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True))


def _comparar(nuestro: pd.DataFrame, ref: pd.DataFrame, etapa: str, tol: dict[str, float] | None = None) -> None:
    tol = tol or {}
    assert len(nuestro) == len(ref), f"{etapa}: {len(nuestro)} filas frente a {len(ref)} de la referencia"
    nuestro = nuestro.reset_index(drop=True)
    distintas = []
    for c in ref.columns:
        if c in tol:
            if not np.nanmax(np.abs(nuestro[c].to_numpy(float) - ref[c].to_numpy(float))) < tol[c]:
                distintas.append(c)
        elif not _iguales(nuestro[c], ref[c]):
            distintas.append(c)
    assert not distintas, f"{etapa}: columnas distintas de la referencia: {distintas}"


# ---------------------------------------------------------------------------
# 1. Paridad exacta contra la referencia del contenedor
# ---------------------------------------------------------------------------
@con_referencia
def test_paridad_retrieval(candidatos):
    _comparar(candidatos, pd.read_parquet(REF / "candidatos.parquet"), "recuperar")


@con_referencia
def test_paridad_features(puntuados_ml):
    ref = pd.read_parquet(REF / "features.parquet")
    _comparar(puntuados_ml[ref.columns], ref, "construir")


@con_referencia
def test_paridad_puntajes(puntuados_ml):
    ref = pd.read_parquet(REF / "puntuados.parquet")
    _comparar(puntuados_ml[ref.columns], ref, "rankear", tol={"s_v2": 1e-9, "p_gana": 1e-9})


@con_referencia
def test_paridad_listas(listas_ml):
    ref = pd.read_parquet(REF / "listas.parquet")
    _comparar(listas_ml[ref.columns], ref, "armar_listas", tol={"s_v2": 1e-9, "p_gana": 1e-9})


# ---------------------------------------------------------------------------
# 2. Paridad exacta contra el notebook (módulos independientes de la plataforma)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def puntuados_notebook(modelos):
    """rankear() sobre las features del notebook: aísla ranking y listas de las diferencias de construir()."""
    from ml.ranking import rankear

    cand = pd.read_parquet(DATA / "final" / "cand" / f"cand_{MES}.parquet")
    return rankear(cand, g1=modelos["g1"], v2=modelos["v2"])


def test_config_coincide():
    from ml.config import FEATS_G1, FEATS_V2, verificar_config
    from ml.ranking import cargar_modelo

    verificar_config(DATA / "final" / "config.json")
    assert cargar_modelo(DATA / "final" / "modelos" / "g1_pgana.txt").feature_name() == FEATS_G1
    assert cargar_modelo(DATA / "final" / "modelos" / "ranking_v2_final.txt").feature_name() == FEATS_V2


def test_puntajes_igual_al_notebook(puntuados_notebook):
    v2 = pd.read_parquet(DATA / "final" / "pred" / f"v2_cand_{MES}.parquet")
    g1 = pd.read_parquet(DATA / "final" / "pred" / f"pgana_cand_{MES}.parquet")
    m = puntuados_notebook[["pk", "pid", "s_v2", "p_gana"]].merge(v2, on=["pk", "pid"]).merge(g1, on=["pk", "pid"])
    assert len(m) == len(puntuados_notebook) == len(v2)
    assert np.abs(m.s_v2_x.astype(float) - m.s_v2_y.astype(float)).max() < 1e-9
    assert np.abs(m.p_gana.astype(float) - m.p_G1.astype(float)).max() < 1e-9


def test_listas_igual_al_notebook(puntuados_notebook, listas_notebook):
    from ml.listas import armar_listas

    nuestras = armar_listas(puntuados_notebook)
    clave = ["proveedor_key", "lista", "pos"]
    a = nuestras.assign(proveedor_key=nuestras.proveedor_key.astype(str)).sort_values(clave).reset_index(drop=True)
    b = listas_notebook.assign(proveedor_key=listas_notebook.proveedor_key.astype(str)).sort_values(clave)
    b = b.reset_index(drop=True)
    assert len(a) == len(b), f"filas: {len(a)} frente a {len(b)} del notebook"
    cols = clave + ["id_proceso"]
    distinto = (a[cols].astype(str).to_numpy() != b[cols].astype(str).to_numpy()).any(1)
    if distinto.any():
        i = int(np.flatnonzero(distinto)[0])
        pytest.fail(f"Primera diferencia (fila {i}): {a.loc[i, cols].to_dict()} frente a {b.loc[i, cols].to_dict()}")
    assert np.abs(a.p_gana.astype(float) - b.p_G1.astype(float)).max() < 1e-9
    assert (a.pos_v2.to_numpy() == b.pos_v2.to_numpy()).all()
    assert (a.pasa_filtros.to_numpy() == b.pasa_filtros.to_numpy()).all()
