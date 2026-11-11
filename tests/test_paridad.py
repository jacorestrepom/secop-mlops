"""Prueba de paridad del paquete ``ml/`` contra el notebook (marzo de 2026).

Compara cada módulo con lo que dejaron los notebooks en ``data/``. Cada etapa se prueba aislada (con la entrada
que usó el notebook) y además hay una prueba de punta a punta ``recuperar → construir → rankear → armar_listas``:

    retrieval   → data/retrieval_v2/top_retrieval/top_202603.parquet (puestos 1–150)
    features    → data/final/cand/cand_202603.parquet
    puntajes    → data/final/pred/v2_cand_202603.parquet y pgana_cand_202603.parquet
    listas      → data/final/listas/listas_202603.parquet (el árbitro: mismos procesos, mismo orden)

Las features y los puntajes de referencia los generó el notebook 05 (el final los lee de disco con FORZAR = False);
la lógica del constructor es la misma en los dos.

Variables de entorno (si faltan, la prueba se salta):
    SECOP_DATA_DIR      carpeta data/ con clean/, retrieval_v2/ y final/
    SECOP_MODELOS_DIR   (opcional) carpeta con g1_pgana.txt, ranking_v2_final.txt y M1_final_unico.txt;
                        por defecto se toman de data/final/modelos/ y data/retrieval_v2/

Tarda ~10 minutos y usa varios GB de RAM. Exportar OMP_NUM_THREADS=4 antes de correrla.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MES = "202603"
FECHA_CORTE = "2026-03-01"

DATA = Path(os.environ["SECOP_DATA_DIR"]) if os.environ.get("SECOP_DATA_DIR") else None
pytestmark = pytest.mark.skipif(DATA is None or not (DATA / "clean").is_dir(),
                                reason="SECOP_DATA_DIR no está definido o no tiene clean/")


def _ruta_modelo(nombre: str, por_defecto: Path) -> Path:
    d = os.environ.get("SECOP_MODELOS_DIR")
    return Path(d) / nombre if d else por_defecto / nombre


@pytest.fixture(scope="module")
def modelos():
    return {"m1": _ruta_modelo("M1_final_unico.txt", DATA / "retrieval_v2"),
            "g1": _ruta_modelo("g1_pgana.txt", DATA / "final" / "modelos"),
            "v2": _ruta_modelo("ranking_v2_final.txt", DATA / "final" / "modelos")}


@pytest.fixture(scope="module")
def candidatos(modelos):
    import gc

    from ml.retrieval import HistoriaRetrieval, recuperar

    h = HistoriaRetrieval.desde_carpeta(DATA / "clean")
    empresas, procesos = h.entradas_del_mes(FECHA_CORTE)
    top = recuperar(empresas, procesos, FECHA_CORTE, historia=h, modelo=modelos["m1"])
    del h
    gc.collect()
    return top


@pytest.fixture(scope="module")
def historia_features():
    from ml.features import Historia

    return Historia.desde_carpeta(DATA / "clean")


@pytest.fixture(scope="module")
def top_notebook():
    """Candidatos del notebook 03 (puestos 1–150) traducidos a llaves, como la sección 6 del notebook final."""
    top = pd.read_parquet(DATA / "retrieval_v2" / "top_retrieval" / f"top_{MES}.parquet",
                          columns=["pk", "pid", "score_retrieval", "rank_retrieval", "es_frio"])
    return (top[top.rank_retrieval <= 150]
            .merge(pd.read_parquet(DATA / "clean" / "map_pk.parquet"), on="pk", how="left")
            .merge(pd.read_parquet(DATA / "clean" / "map_pid.parquet"), on="pid", how="left").reset_index(drop=True))


@pytest.fixture(scope="module")
def features(top_notebook, historia_features):
    from ml.features import construir

    cols = ["pk", "pid", "proveedor_key", "id_proceso", "score_retrieval", "rank_retrieval", "es_frio"]
    return construir(top_notebook[cols], historia=historia_features)


@pytest.fixture(scope="module")
def cand_notebook():
    return pd.read_parquet(DATA / "final" / "cand" / f"cand_{MES}.parquet")


@pytest.fixture(scope="module")
def puntuados(cand_notebook, modelos):
    """rankear() sobre las features del notebook: aísla ranking y listas de las diferencias de construir()."""
    from ml.ranking import rankear

    return rankear(cand_notebook, g1=modelos["g1"], v2=modelos["v2"])


def _iguales(a: pd.Series, b: pd.Series) -> bool:
    if isinstance(a.dtype, pd.CategoricalDtype) or a.dtype == object or pd.api.types.is_string_dtype(a.dtype):
        return bool((a.astype(str).to_numpy() == b.astype(str).to_numpy()).all())
    return bool(np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True))


def test_config_coincide():
    from ml.config import FEATS_G1, FEATS_V2, verificar_config
    from ml.ranking import cargar_modelo

    verificar_config(DATA / "final" / "config.json")
    assert cargar_modelo(DATA / "final" / "modelos" / "g1_pgana.txt").feature_name() == FEATS_G1
    assert cargar_modelo(DATA / "final" / "modelos" / "ranking_v2_final.txt").feature_name() == FEATS_V2


def test_paridad_retrieval(candidatos):
    ref = pd.read_parquet(DATA / "retrieval_v2" / "top_retrieval" / f"top_{MES}.parquet",
                          columns=["pk", "pid", "score_retrieval", "rank_retrieval", "es_frio"])
    ref = ref[ref.rank_retrieval <= 150].reset_index(drop=True)
    assert len(candidatos) == len(ref)
    for c in ["pk", "pid", "rank_retrieval", "es_frio", "score_retrieval"]:
        assert _iguales(candidatos[c], ref[c]), f"retrieval: la columna {c} no coincide"
    mapas = (pd.read_parquet(DATA / "clean" / "map_pk.parquet").set_index("pk").proveedor_key,
             pd.read_parquet(DATA / "clean" / "map_pid.parquet").set_index("pid").id_proceso)
    assert (candidatos.proveedor_key.to_numpy() == mapas[0].reindex(ref.pk).to_numpy()).all()
    assert (candidatos.id_proceso.to_numpy() == mapas[1].reindex(ref.pid).to_numpy()).all()


def test_paridad_features(features, cand_notebook):
    from ml.config import OK3, ORDEN_D, ORDEN_D3

    ref = cand_notebook
    assert len(features) == len(ref)
    distintas = [c for c in ["pk", "pid"] + ORDEN_D + ORDEN_D3 + OK3 if not _iguales(features[c], ref[c])]
    assert not distintas, f"features distintas: {distintas}"


def test_paridad_puntajes(puntuados):
    v2 = pd.read_parquet(DATA / "final" / "pred" / f"v2_cand_{MES}.parquet")
    g1 = pd.read_parquet(DATA / "final" / "pred" / f"pgana_cand_{MES}.parquet")
    m = puntuados[["pk", "pid", "s_v2", "p_gana"]].merge(v2, on=["pk", "pid"]).merge(g1, on=["pk", "pid"])
    assert len(m) == len(puntuados) == len(v2)
    assert np.abs(m.s_v2_x.astype(float) - m.s_v2_y.astype(float)).max() < 1e-9
    assert np.abs(m.p_gana.astype(float) - m.p_G1.astype(float)).max() < 1e-9


def _comparar_listas(puntuados):
    from ml.listas import LISTA_HISTORIAL, LISTA_OPORTUNIDAD, armar_listas

    nombres = {"1 · tu historial (ranking v2)": LISTA_HISTORIAL, "2 · oportunidades (G1 + filtros)": LISTA_OPORTUNIDAD}
    ref = pd.read_parquet(DATA / "final" / "listas" / f"listas_{MES}.parquet")
    ref = ref[ref.lista.isin(nombres)].assign(lista=lambda x: x.lista.map(nombres))
    nuestras = armar_listas(puntuados)

    clave = ["proveedor_key", "lista", "pos"]
    a = nuestras.assign(proveedor_key=nuestras.proveedor_key.astype(str)).sort_values(clave).reset_index(drop=True)
    b = ref.assign(proveedor_key=ref.proveedor_key.astype(str)).sort_values(clave).reset_index(drop=True)
    assert len(a) == len(b), f"filas: {len(a)} frente a {len(b)} del notebook"
    for lista in [LISTA_HISTORIAL, LISTA_OPORTUNIDAD]:
        assert (a.lista == lista).sum() == (b.lista == lista).sum(), f"tamaño de la lista {lista}"
    cols = clave + ["id_proceso"]
    distinto = (a[cols].astype(str).to_numpy() != b[cols].astype(str).to_numpy()).any(1)
    if distinto.any():
        i = int(np.flatnonzero(distinto)[0])
        pytest.fail(f"Primera diferencia (fila {i}): ml/ {a.loc[i, clave + ['id_proceso']].to_dict()} "
                    f"frente al notebook {b.loc[i, clave + ['id_proceso']].to_dict()}")
    assert np.abs(a.p_gana.astype(float) - b.p_G1.astype(float)).max() < 1e-9
    assert np.abs(a.s_v2.astype(float) - b.s_v2.astype(float)).max() < 1e-9
    assert (a.pos_v2.to_numpy() == b.pos_v2.to_numpy()).all()
    assert (a.pasa_filtros.to_numpy() == b.pasa_filtros.to_numpy()).all()


def test_paridad_listas(puntuados):
    """Listas armadas con las features del notebook."""
    _comparar_listas(puntuados)


def test_paridad_punta_a_punta(candidatos, historia_features, modelos):
    """El árbitro: tablas limpias → recuperar → construir → rankear → armar_listas."""
    from ml.features import construir
    from ml.ranking import rankear

    feats = construir(candidatos, historia=historia_features)
    _comparar_listas(rankear(feats, g1=modelos["g1"], v2=modelos["v2"]))
