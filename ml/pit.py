"""Utilidades point-in-time (notebook final, sección 2), copiadas sin cambios.

``contar_antes`` y ``ultimo_evento`` miran SOLO eventos estrictamente anteriores a la fecha de cada consulta.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd


def a_ns(s):
    """Fechas → enteros en nanosegundos (para comparar rápido con numpy)."""
    return pd.to_datetime(s).values.astype("datetime64[ns]").astype(np.int64)


def contar_antes(qk, qt, ek, et, ew=None):
    """Para cada consulta (llave qk, tiempo qt) suma el peso de los eventos de la MISMA llave con
    tiempo ESTRICTAMENTE anterior. Llaves < 0 (nulas) no cuentan. Vectorizado con un solo ordenamiento."""
    ok = ek >= 0
    ek, et = ek[ok], et[ok]
    ew = np.ones(len(ek)) if ew is None else np.asarray(ew, dtype=float)[ok]
    nq = len(qk)
    k = np.concatenate([qk, ek])
    t = np.concatenate([qt, et])
    tipo = np.concatenate([np.zeros(nq, np.int8), np.ones(len(ek), np.int8)])  # empate: la consulta va primero
    w = np.concatenate([np.zeros(nq), ew])
    o = np.lexsort((tipo, t, k))
    ks, cs = k[o], np.cumsum(w[o])
    ini = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    previo = np.repeat(np.r_[0.0, cs][ini], np.diff(np.r_[ini, len(ks)]))
    out = np.zeros(nq)
    m = o < nq
    out[o[m]] = (cs - previo)[m]
    return out


def ultimo_evento(qk, qt, ek, et):
    """Para cada consulta, posición del último evento de la MISMA llave con tiempo estrictamente anterior (-1 si no hay)."""
    nq, ne_ = len(qk), len(ek)
    k = np.concatenate([qk, np.where(ek >= 0, ek, -2)])
    t = np.concatenate([qt, et])
    tipo = np.concatenate([np.zeros(nq, np.int8), np.ones(ne_, np.int8)])
    pos = np.concatenate([np.full(nq, -1), np.arange(ne_)])
    o = np.lexsort((tipo, t, k))
    ks, es, ps = k[o], tipo[o] == 1, pos[o]
    ult = np.maximum.accumulate(np.where(es, np.arange(len(o)), -1))
    u0 = np.maximum(ult, 0)
    res = np.where((ult >= 0) & (ks[u0] == ks) & (ks >= 0), ps[u0], -1)
    out = np.full(nq, -1)
    m = o < nq
    out[o[m]] = res[m]
    return out


class Llave:
    """Códigos comunes entre eventos y consultas para una llave (una o varias columnas). Nulos → -1."""

    def __init__(self, ev_df, cols):
        self.cols = cols
        df = ev_df[cols]
        nn = ~df.isna().any(axis=1).values
        if len(cols) == 1:
            self.idx = pd.Index(df[cols[0]][nn].unique())
            ev = np.full(len(df), -1, np.int64)
            ev[nn] = self.idx.get_indexer(df[cols[0]][nn])
        else:
            mi = pd.MultiIndex.from_frame(df[nn])
            self.idx = mi.unique()
            ev = np.full(len(df), -1, np.int64)
            ev[nn] = self.idx.get_indexer(mi)
        self.ev = ev

    def q(self, q_df):
        df = q_df[self.cols]
        nn = ~df.isna().any(axis=1).values
        out = np.full(len(df), -1, np.int64)
        if nn.any():
            out[nn] = (self.idx.get_indexer(df[self.cols[0]][nn]) if len(self.cols) == 1
                       else self.idx.get_indexer(pd.MultiIndex.from_frame(df[nn])))
        return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def f32(s):
    return s.astype("Float64").to_numpy(dtype=np.float32, na_value=np.nan)


def norm_txt(s):
    s = s.astype("string").fillna("")
    u = s.unique()
    return s.map({x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().upper().strip() for x in u})


def a_categoria(s, categorias):
    """Columna categórica con categorías fijas; los valores no vistos en entrenamiento quedan nulos."""
    s = s.astype(str)
    return pd.Categorical(s.where(s.isin(categorias)), categories=categorias)
