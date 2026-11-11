"""Métricas de las dos listas (notebook final, secciones 10 y 11), para evaluar un mes ya cerrado.

- Lista 1 (vista por empresa): Precision@10, Recall@10 y NDCG@10 sobre las empresas que disputaron algo en el mes.
- Lista 2: % de oportunidades plausibles (la empresa no se presentó, el proceso se adjudicó y tuvo ≤ 2 oferentes).

Relevancia: ganó = 2, se presentó = 1, no se presentó = 0. Mismas fórmulas del notebook; aquí se calculan a
partir de las filas de las listas (la lista 1 es el top 10 del ranking v2, igual que ``top_n(d, "s_v2")``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DESC = np.r_[0, np.cumsum(1 / np.log2(np.arange(2, 12)))]  # descuentos acumulados del NDCG@10


def participaciones(proponentes: pd.DataFrame) -> pd.DataFrame:
    """``prop_m`` del notebook: una fila por participación con su mes (AAAAMM) y si ganó."""
    pm = proponentes[["proveedor_key", "id_proceso", "fecha_publicacion", "es_ganador"]].copy()
    pm["proveedor_key"] = pm.proveedor_key.astype(str)
    pm["id_proceso"] = pm.id_proceso.astype(str)
    pm["mes"] = pd.to_datetime(pm.fecha_publicacion).dt.strftime("%Y%m")
    pm["gana"] = pm.es_ganador.fillna(False).astype(bool)
    return pm


def relevancia(d: pd.DataFrame, prop_m: pd.DataFrame, mes: str) -> np.ndarray:
    pm = prop_m[prop_m.mes == mes]
    rel_map = pm.groupby(["proveedor_key", "id_proceso"]).gana.max()
    g = rel_map.reindex(pd.MultiIndex.from_arrays([d.proveedor_key.astype(str), d.id_proceso.astype(str)])).values
    return np.where(pd.isna(g), 0, np.where(g == True, 2, 1)).astype(np.int8)  # noqa: E712


def ideal_ndcg(n, n2):
    n, n2 = np.minimum(n, 10).astype(int), np.minimum(n2, 10).astype(int)
    return 2 * DESC[n2] + (DESC[n] - DESC[n2])


def metricas_lista1(lista1: pd.DataFrame, prop_m: pd.DataFrame, mes: str) -> dict[str, float]:
    """``vista_empresa`` del notebook: promedio por empresa que disputó algo en el mes (ideal: todo lo que disputó)."""
    pm = prop_m[prop_m.mes == mes]
    ideal = pm.groupby("proveedor_key").agg(n=("gana", "size"), n2=("gana", "sum"))
    x = lista1.assign(proveedor_key=lista1.proveedor_key.astype(str))
    x = x.assign(rel=relevancia(x, prop_m, mes))
    emp = pd.DataFrame(index=pd.Index(x.proveedor_key.unique(), name="proveedor_key")).join(ideal).dropna(subset=["n"])
    disc = 1 / np.log2(x.pos.values + 1)
    g = (pd.DataFrame({"proveedor_key": x.proveedor_key.values, "dcg": x.rel.values * disc, "ac": x.rel.values > 0})
         .groupby("proveedor_key").agg(dcg=("dcg", "sum"), ac=("ac", "sum")))
    e = emp.join(g).fillna({"dcg": 0, "ac": 0})
    return {"Precision@10": float(np.nanmean(e.ac / 10)), "Recall@10": float(np.nanmean(e.ac / e.n)),
            "NDCG@10": float(np.nanmean(e.dcg / ideal_ndcg(e.n.values, e.n2.values))), "empresas": int(len(e))}


def pct_plausibles(lista2: pd.DataFrame, prop_m: pd.DataFrame, mes: str) -> float:
    """% de oportunidades plausibles de la lista 2 (en porcentaje, como la Tabla 2)."""
    hechos = prop_m.groupby("id_proceso").agg(n_real=("id_proceso", "size"), adjudicado=("gana", "any"))
    x = lista2.assign(proveedor_key=lista2.proveedor_key.astype(str), id_proceso=lista2.id_proceso.astype(str))
    rel = relevancia(x, prop_m, mes)
    n_real = x.id_proceso.map(hechos.n_real).fillna(0).values
    adjudicado = x.id_proceso.map(hechos.adjudicado).eq(True).values
    return float(100 * ((rel == 0) & adjudicado & (n_real <= 2)).mean())


def coincidencia_por_empresa(nuestras: pd.DataFrame, referencia: pd.DataFrame) -> float:
    """Fracción de empresas cuya lista es idéntica (mismos procesos en las mismas posiciones)."""
    k = ["proveedor_key", "pos"]
    a = nuestras.assign(proveedor_key=nuestras.proveedor_key.astype(str), id_proceso=nuestras.id_proceso.astype(str))
    b = referencia.assign(proveedor_key=referencia.proveedor_key.astype(str),
                          id_proceso=referencia.id_proceso.astype(str))
    m = a[k + ["id_proceso"]].merge(b[k + ["id_proceso"]], on=k, how="outer", suffixes=("", "_r"))
    return float((m.id_proceso == m.id_proceso_r).groupby(m.proveedor_key).all().mean())
