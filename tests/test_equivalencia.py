"""Prueba de equivalencia de ml/ frente al notebook (marzo de 2026), en dos niveles.

(a) Componentes, con las mismas entradas: sobre los candidatos del notebook (top_202603),
    construir → rankear → armar_listas de ml/ da la misma lista en ≥ 98 % de las empresas (lista 1 y lista 2
    por separado). Aquí solo pesan los empates de los vecinos (knn) y la numérica del SVD.

(b) Punta a punta (recuperar → construir → rankear → armar_listas de ml/): el modelo rinde igual. Las métricas de
    marzo quedan a menos de 0,005 de las del notebook: Recall@150 del retrieval, Precision@10, Recall@10 y NDCG@10
    de la lista 1, y % de plausibles de la lista 2 (como fracción 0–1). La identidad lista por lista no se exige
    de punta a punta: el desempate fijo del retrieval cambia candidatos de forma legítima. El solapamiento de
    pares se imprime como información.

Las métricas del notebook se recalculan para marzo con las mismas fórmulas (ml/evaluacion.py): data/final/metricas
promedia los 13 meses de prueba.
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


@pytest.fixture(scope="module")
def listas_componentes(top_notebook, historia_features, modelos):
    from ml.features import construir
    from ml.listas import armar_listas
    from ml.ranking import rankear

    cols = ["pk", "pid", "proveedor_key", "id_proceso", "score_retrieval", "rank_retrieval", "es_frio"]
    feats = construir(top_notebook[cols], historia=historia_features)
    return armar_listas(rankear(feats, g1=modelos["g1"], v2=modelos["v2"]))


def metricas(listas, candidatos, positivos, prop_m) -> dict[str, float]:
    from ml.evaluacion import metricas_lista1, pct_plausibles, recall_retrieval

    m = {"Recall@150 retrieval": recall_retrieval(candidatos, positivos)}
    m.update({k: v for k, v in metricas_lista1(listas[listas.lista == "historial"], prop_m, MES).items()
              if k != "empresas"})
    m["% plausibles lista 2"] = pct_plausibles(listas[listas.lista == "oportunidad"], prop_m, MES) / 100
    return m


@pytest.mark.parametrize("lista", ["historial", "oportunidad"])
def test_componentes_listas_coinciden(lista, listas_componentes, listas_notebook):
    from ml.evaluacion import coincidencia_por_empresa

    c = coincidencia_por_empresa(listas_componentes[listas_componentes.lista == lista],
                                 listas_notebook[listas_notebook.lista == lista])
    print(f"\n[componentes] lista {lista}: {c:.2%} de empresas con la misma lista")
    assert c >= MIN_COINCIDENCIA, f"lista {lista}: {c:.2%} de empresas con la misma lista (mín. {MIN_COINCIDENCIA:.0%})"


def test_punta_a_punta_metricas(retrieval_ml, listas_ml, listas_notebook, top_notebook, prop_m):
    candidatos, positivos = retrieval_ml
    nb = metricas(listas_notebook, top_notebook, positivos, prop_m)
    ml_ = metricas(listas_ml, candidatos, positivos, prop_m)
    solape = len(candidatos[["pk", "pid"]].merge(top_notebook[["pk", "pid"]])) / len(top_notebook)
    print(f"\n[punta a punta] solapamiento de pares del retrieval: {solape:.2%} (informativo)")
    print("métrica | notebook | ml/ | diferencia")
    for k in nb:
        print(f"{k} | {nb[k]:.5f} | {ml_[k]:.5f} | {ml_[k] - nb[k]:+.5f}")
    fuera = {k: round(ml_[k] - nb[k], 5) for k in nb if abs(ml_[k] - nb[k]) >= TOL_METRICA}
    assert not fuera, f"métricas a más de {TOL_METRICA} del notebook: {fuera}"
