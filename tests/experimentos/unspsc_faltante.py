"""Experimento: ¿cuánto pierde ml/ si los procesos abiertos llegan sin categoría UNSPSC?

Desde oct-2026 SECOP publica codigo_principal_de_categoria = 'UNSPECIFIED' en ~99,6 % de los procesos nuevos.
En silver y en data/clean eso queda como NULO en unspsc_segmento, unspsc_familia y unspsc_clase (el notebook 02
trata "unspecified" como relleno; limpiar() no toca esas columnas). Se simula poniendo NaN en esas columnas SOLO
para los procesos publicados en el mes evaluado (los candidatos); el historial queda intacto.

    python -m tests.experimentos.unspsc_faltante --data <data/> --salida <dir> --variante normal
    python -m tests.experimentos.unspsc_faltante --data <data/> --salida <dir> --variante sin_unspsc
    python -m tests.experimentos.unspsc_faltante --data <data/> --salida <dir> --comparar
    (--mes 2025-10-01 para evaluar otro mes; por defecto marzo de 2026)

Cada variante corre en su propio proceso (la memoria pico del mes completo es ~18 GB).
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

MES, FECHA = "202603", pd.Timestamp("2026-03-01")  # se cambia con --mes (fijar_mes)
COLS_UNSPSC = ["unspsc_segmento", "unspsc_familia", "unspsc_clase"]


def fijar_mes(mes: str) -> None:
    """mes = primer día del mes evaluado, p. ej. 2025-10-01."""
    global MES, FECHA
    FECHA = pd.Timestamp(mes)
    MES = FECHA.strftime("%Y%m")


def _procesos_del_mes(fechas: pd.Series) -> np.ndarray:
    f = pd.to_datetime(fechas)
    return ((f >= FECHA) & (f < FECHA + pd.offsets.MonthBegin(1))).values


def _imputar(tabla: pd.DataFrame, m: np.ndarray, imputacion: pd.DataFrame | None, indice: bool = False) -> None:
    """Rellena el UNSPSC de los procesos del mes con la imputación (lo que no venga queda nulo)."""
    if imputacion is None:
        return
    ids = pd.Index(tabla.index if indice else tabla["id_proceso"])[m].astype(str)
    imp = imputacion.set_index(imputacion.id_proceso.astype(str))
    for c in ["unspsc_segmento", "unspsc_familia"]:
        tabla.loc[m, c] = imp[c].reindex(ids).values


def correr(data: Path, salida: Path, variante: str, imputacion: pd.DataFrame | None = None) -> None:
    """imputacion (solo variante "imputado"): id_proceso, unspsc_segmento, unspsc_familia para los procesos del mes."""
    from ml.features import Historia, construir
    from ml.listas import armar_listas
    from ml.ranking import rankear
    from ml.retrieval import HistoriaRetrieval, recuperar

    t0 = time.time()
    hr = HistoriaRetrieval.desde_carpeta(data / "clean")
    if variante != "normal":  # solo los procesos consultados; part_all (historial) ya quedó armado
        m = _procesos_del_mes(hr.proc.fecha_publicacion)
        hr.proc.loc[m, [c for c in COLS_UNSPSC if c in hr.proc]] = np.nan
        _imputar(hr.proc, m, imputacion)
        print(f"retrieval: {m.sum():,} procesos del mes sin UNSPSC", flush=True)
    empresas, procesos = hr.entradas_del_mes(FECHA)
    cand = recuperar(empresas, procesos, FECHA, historia=hr, modelo=data / "retrieval_v2" / "M1_final_unico.txt")
    hr.positivos_del_mes(FECHA).to_parquet(salida / "positivos.parquet", index=False)
    del hr
    gc.collect()
    print(f"recuperar: {len(cand):,} pares | {time.time() - t0:.0f} s", flush=True)

    h = Historia.desde_carpeta(data / "clean")
    if variante != "normal":  # PROC = datos de los procesos candidatos; los eventos (EV) no se tocan
        m = _procesos_del_mes(h.PROC.fecha_publicacion)
        h.PROC.loc[m, [c for c in COLS_UNSPSC if c in h.PROC]] = np.nan
        _imputar(h.PROC, m, imputacion, indice=True)
    feats = construir(cand, historia=h)
    del h
    gc.collect()
    mod = data / "final" / "modelos"
    punt = rankear(feats, g1=mod / "g1_pgana.txt", v2=mod / "ranking_v2_final.txt")
    listas = armar_listas(punt)
    punt.to_parquet(salida / f"puntuados_{variante}.parquet", index=False)
    listas.drop(columns=[c for c in listas.columns if listas[c].dtype.name == "category"]).to_parquet(
        salida / f"listas_{variante}.parquet", index=False
    )
    print(f"{variante}: listo | {time.time() - t0:.0f} s", flush=True)


def comparar(data: Path, salida: Path) -> None:
    import lightgbm as lgb

    from ml.evaluacion import (
        coincidencia_por_empresa,
        metricas_lista1,
        participaciones,
        pct_plausibles,
        recall_retrieval,
    )

    pm = participaciones(pd.read_parquet(data / "clean" / "proponentes.parquet"))
    pos = pd.read_parquet(salida / "positivos.parquet")
    res = {}
    for v in ["normal", "sin_unspsc"]:
        p = pd.read_parquet(salida / f"puntuados_{v}.parquet", columns=["pk", "pid", "rank_retrieval"])
        L = pd.read_parquet(salida / f"listas_{v}.parquet")
        m = {"Recall@150 retrieval": recall_retrieval(p, pos)}
        m.update({k: x for k, x in metricas_lista1(L[L.lista == "historial"], pm, MES).items() if k != "empresas"})
        m["% plausibles lista 2"] = pct_plausibles(L[L.lista == "oportunidad"], pm, MES) / 100
        res[v] = (m, L, p)
    print(f"métrica | {MES} normal | {MES} sin UNSPSC | diferencia")
    for k in res["normal"][0]:
        a, b = res["normal"][0][k], res["sin_unspsc"][0][k]
        print(f"{k} | {a:.5f} | {b:.5f} | {b - a:+.5f} ({100 * (b - a) / a:+.1f} %)")
    (_, La, pa), (_, Lb, pb) = res["normal"], res["sin_unspsc"]
    sol = len(pa[["pk", "pid"]].merge(pb[["pk", "pid"]])) / len(pa)
    print(f"solapamiento de candidatos del retrieval: {sol:.2%}")
    for lista in ["historial", "oportunidad"]:
        a, b = La[La.lista == lista], Lb[Lb.lista == lista]
        ident = coincidencia_por_empresa(b, a)
        sa = a.groupby("proveedor_key").id_proceso.apply(set)
        sb = b.groupby("proveedor_key").id_proceso.apply(set).reindex(sa.index)
        jac = np.mean([len(x & y) / max(len(x), 1) if isinstance(y, set) else 0.0 for x, y in zip(sa, sb, strict=True)])
        print(
            f"lista {lista}: {ident:.2%} de empresas con la misma lista; "
            f"en promedio {jac:.2%} de los procesos de la lista normal siguen en la lista sin UNSPSC"
        )

    print("\nImportancia (% de la ganancia) de las features que dependen del UNSPSC:")
    dep = lambda f: "unspsc" in f or "_fam" in f or f in {"f_seg"}  # noqa: E731
    for nombre, ruta in [
        ("M1 retrieval", data / "retrieval_v2" / "M1_final_unico.txt"),
        ("G1", data / "final" / "modelos" / "g1_pgana.txt"),
        ("ranking v2", data / "final" / "modelos" / "ranking_v2_final.txt"),
    ]:
        b = lgb.Booster(model_file=str(ruta))
        imp = pd.Series(b.feature_importance("gain"), index=b.feature_name())
        imp = 100 * imp / imp.sum()
        d = imp[[f for f in imp.index if dep(f)]].sort_values(ascending=False)
        print(f"  {nombre}: total {d.sum():.1f} % | " + ", ".join(f"{k} {x:.1f}" for k, x in d.head(6).items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--variante", choices=["normal", "sin_unspsc"])
    ap.add_argument("--comparar", action="store_true")
    ap.add_argument("--mes", default="2026-03-01", help="primer día del mes evaluado")
    a = ap.parse_args()
    fijar_mes(a.mes)
    data, salida = Path(a.data), Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    if a.comparar:
        comparar(data, salida)
    else:
        correr(data, salida, a.variante)


if __name__ == "__main__":
    main()
