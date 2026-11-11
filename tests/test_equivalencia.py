"""Prueba de equivalencia de ml/ frente al notebook (marzo de 2026, de punta a punta).

ml/ no reproduce bit a bit las listas del notebook: los empates (texto del retrieval, vecinos del knn) y la
numérica del SVD dependen de la plataforma. Esta prueba exige que el resultado sea equivalente:

- las métricas del notebook (Precision@10, Recall@10 y NDCG@10 de la lista 1; % de oportunidades plausibles de
  la lista 2, como fracción 0–1) quedan a menos de 0,005 de las del notebook en el mismo mes;
- al menos el 98 % de las empresas recibe la misma lista (lista 1 y lista 2 por separado).

Las métricas del notebook se recalculan para marzo con las mismas fórmulas (ml/evaluacion.py) sobre
listas_202603.parquet: data/final/metricas promedia los 13 meses de prueba y no sirve para comparar un mes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import DATA, MES, sin_datos

pytestmark = sin_datos

TOL_METRICA = 0.005
MIN_COINCIDENCIA = 0.98


@pytest.fixture(scope="module")
def prop_m():
    from ml.evaluacion import participaciones

    return participaciones(pd.read_parquet(DATA / "clean" / "proponentes.parquet"))


def _metricas(listas, prop_m):
    from ml.evaluacion import metricas_lista1, pct_plausibles

    m = metricas_lista1(listas[listas.lista == "historial"], prop_m, MES)
    m["% plausibles"] = pct_plausibles(listas[listas.lista == "oportunidad"], prop_m, MES) / 100
    return m


@pytest.mark.parametrize("metrica", ["Precision@10", "Recall@10", "NDCG@10", "% plausibles"])
def test_metricas_equivalentes(metrica, listas_ml, listas_notebook, prop_m):
    ml_, nb = _metricas(listas_ml, prop_m)[metrica], _metricas(listas_notebook, prop_m)[metrica]
    assert abs(ml_ - nb) < TOL_METRICA, f"{metrica}: ml/ {ml_:.4f} frente a notebook {nb:.4f}"


@pytest.mark.parametrize("lista", ["historial", "oportunidad"])
def test_listas_coinciden_por_empresa(lista, listas_ml, listas_notebook):
    from ml.evaluacion import coincidencia_por_empresa

    c = coincidencia_por_empresa(listas_ml[listas_ml.lista == lista], listas_notebook[listas_notebook.lista == lista])
    assert c >= MIN_COINCIDENCIA, f"lista {lista}: {c:.2%} de empresas con la misma lista (mín. {MIN_COINCIDENCIA:.0%})"
