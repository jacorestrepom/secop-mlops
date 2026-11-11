"""Puntuación de los candidatos (notebook final, secciones 7 y 8).

- G1 (LightGBM binario) → ``p_gana``: P(gana) si la empresa se presenta.
- Ranking v2 (LightGBM LambdaRank) → ``s_v2``: puntaje que solo sirve para ordenar.

Las categorías de las columnas categóricas son las del entrenamiento: vienen guardadas dentro de cada
modelo (``pandas_categorical``) y son las mismas que el notebook toma de ``feat_base.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml import config as C
from ml.pit import a_categoria


def cargar_modelo(modelo: lgb.Booster | str | Path) -> lgb.Booster:
    """Acepta un ``lgb.Booster`` o la ruta a su archivo .txt."""
    return modelo if isinstance(modelo, lgb.Booster) else lgb.Booster(model_file=str(modelo))


def _categorias(modelo: lgb.Booster, feats: list[str]) -> dict[str, list]:
    cats = [c for c in feats if c in C.CAT_G]
    pc = modelo.pandas_categorical
    if pc is None or len(pc) != len(cats):
        raise ValueError("El modelo no trae las categorías del entrenamiento (pandas_categorical)")
    return dict(zip(cats, pc, strict=True))


def _validar_columnas(modelo: lgb.Booster, feats: list[str], nombre: str) -> None:
    if modelo.feature_name() != feats:
        raise ValueError(f"Las features del modelo {nombre} no están en el orden de ml.config")


def predecir_g1(features: pd.DataFrame, g1: lgb.Booster | str | Path) -> np.ndarray:
    """Sección 7 (b): P(gana) de cada candidato."""
    g1 = cargar_modelo(g1)
    _validar_columnas(g1, C.FEATS_G1, "G1")
    categ = _categorias(g1, C.FEATS_G1)
    d = features[C.FEATS_G1].copy()
    for c in C.CAT_G:
        d[c] = a_categoria(d[c], categ[c])
    return g1.predict(d[C.FEATS_G1]).astype(np.float32)


def predecir_v2(features: pd.DataFrame, v2: lgb.Booster | str | Path) -> np.ndarray:
    """Sección 8: puntaje del ranking v2 de cada candidato (``cargar_cand`` sin el orden por pk)."""
    v2 = cargar_modelo(v2)
    _validar_columnas(v2, C.FEATS_V2, "ranking v2")
    categ = _categorias(v2, C.FEATS_V2)
    d = features[C.FEATS_V2].copy()
    for c in C.CAT_G:
        d[c] = a_categoria(d[c], categ[c])
    for c in C.FEATS_V2:
        if c not in C.CAT_G:
            d[c] = d[c].astype(np.float32)
    return v2.predict(d[C.FEATS_V2]).astype(np.float32)


def rankear(features: pd.DataFrame, g1: lgb.Booster | str | Path, v2: lgb.Booster | str | Path) -> pd.DataFrame:
    """Interfaz fija del Paso 1: features → ``s_v2`` (ranking v2) y ``p_gana`` (G1).

    ``features`` debe traer las columnas de ``FEATS_V2`` (score_retrieval, rank_retrieval, es_frio y las 102
    del G1). Devuelve una copia de ``features`` con las columnas ``s_v2`` y ``p_gana`` (float32) al final.
    """
    out = features.copy()
    out["s_v2"] = predecir_v2(features, v2)
    out["p_gana"] = predecir_g1(features, g1)
    return out
