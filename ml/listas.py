"""Las dos listas de cada empresa (notebook final, sección 9).

- Lista 1 · ``historial`` ("Recomendados según tu historial"): top 10 del ranking v2.
- Lista 2 · ``oportunidad`` ("Oportunidades que normalmente no verías"): de los puestos 11–150 del ranking v2,
  los que pasan los filtros, ordenados por P(gana) del G1; hasta 10.
  Filtros: entidad nueva para la empresa · ≤ MAX_COMP oferentes esperados · tamaño razonable (MAX_PRECIO).

Se quitaron solo las partes de evaluación del notebook (relevancia real, hechos de cada proceso y líneas base
al azar), que no intervienen en la elección de las listas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml import config as C

LISTA_HISTORIAL = "historial"
LISTA_OPORTUNIDAD = "oportunidad"

COLUMNAS_NECESARIAS = ["proveedor_key", "id_proceso", "s_v2", "p_gana",
                       "con_misma_entidad", "perf_ent_n", "comp_entmod_media", "precio_vs_ganados"]


def preparar(candidatos: pd.DataFrame, empresa: str = "proveedor_key") -> pd.DataFrame:
    """Filtros de la lista 2 y posición en el ranking v2 (``preparar`` del notebook, sin los hechos reales)."""
    faltan = [c for c in COLUMNAS_NECESARIAS if c not in candidatos.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en los candidatos: {faltan}")
    d = candidatos.copy()
    d["proveedor_key"] = d.proveedor_key.astype(str)
    d["id_proceso"] = d.id_proceso.astype(str)
    d["entidad_nueva"] = (d.con_misma_entidad.fillna(0) == 0) & (d.perf_ent_n.fillna(0) == 0)
    d["pasa_filtros"] = (d.entidad_nueva & (d.comp_entmod_media <= C.MAX_COMP)
                         & (d.precio_vs_ganados.isna() | (d.precio_vs_ganados.abs() <= C.MAX_PRECIO)))
    d = d.sort_values([empresa, "s_v2"], ascending=[True, False])
    d["pos_v2"] = d.groupby(empresa).cumcount() + 1
    return d


def top_n(x: pd.DataFrame, col: str, n: int = C.N_LISTA, empresa: str = "proveedor_key") -> pd.DataFrame:
    x = x.sort_values([empresa, col], ascending=[True, False])
    return x[x.groupby(empresa).cumcount() < n].assign(pos=lambda t: t.groupby(empresa).cumcount() + 1)


def armar_listas(candidatos: pd.DataFrame, empresa: str = "proveedor_key") -> pd.DataFrame:
    """Interfaz fija del Paso 1: candidatos puntuados → las dos listas.

    ``candidatos``: una fila por (empresa, proceso candidato) con ``s_v2``, ``p_gana`` y las features que usan
    los filtros. ``empresa`` es la columna que identifica a la empresa (``proveedor_key`` por defecto).
    Devuelve una fila por proceso recomendado con ``lista`` (historial | oportunidad), ``pos`` (1–10),
    ``pos_v2``, ``entidad_nueva`` y ``pasa_filtros``, además de las columnas recibidas.
    """
    d = preparar(candidatos, empresa)
    resto = d[d.pos_v2 > C.N_LISTA]  # lo que no entró a la lista 1
    L = pd.concat([top_n(d, "s_v2", empresa=empresa).assign(lista=LISTA_HISTORIAL),
                   top_n(resto[resto.pasa_filtros], "p_gana", empresa=empresa).assign(lista=LISTA_OPORTUNIDAD)],
                  ignore_index=True)
    assert not L.duplicated([empresa, "id_proceso"]).any(), "Un proceso quedó en las dos listas"
    L["pos"] = L["pos"].astype(np.int64)
    return L
