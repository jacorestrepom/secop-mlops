"""Constantes del modelo congelado.

Salen de ``data/final/config.json`` y de la celda de configuración del notebook final (sección 1) y del
notebook 03 (retrieval). ``verificar_config`` compara estas constantes con un ``config.json`` dado.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------- General (notebook final, sección 1) ----------
SEMILLA = 42
DIAS = np.int64(86_400 * 10**9)  # un día en nanosegundos
K = 150  # candidatos del retrieval por empresa
N_LISTA = 10  # tamaño de cada lista
CORTE_AJUSTE = np.datetime64("2024-01-01")  # priors y texto de features: solo información anterior a 2024

# Suavizados de las features
A_ENT, A_EMP, A2, A_PRIOR = 25, 2, 3, 5

# Filtros de la lista 2
MAX_COMP = 3.0  # oferentes esperados máximos (entidad × modalidad)
MAX_PRECIO = 1.4  # |log(precio / precio típico que gana)| ≤ 1,4

# Trozos en que el notebook pasa los candidatos por construir() (sección 6)
TROZO = 300_000

# ---------- Features (notebook final, sección 4) ----------
NUM_P = ["log_precio_base", "log_duracion_dias", "dias_para_ofertar", "numero_lotes", "proc_mes",
         "precio_atipico", "tiene_lotes", "proc_fin_de_anio"]
CAT_P = ["modalidad", "tipo_contrato_grupo", "unspsc_segmento", "unspsc_familia", "departamento_entidad",
         "orden_entidad", "sector_entidad"]
CAT_OUT = set(CAT_P + ["tipo_proveedor", "tipo_empresa_grupo"])  # categóricas de salida
LLAVES_TXT = ["proveedor_key", "nit_entidad", "ciudad_norm", "tipo_proveedor"] + CAT_P
COLS_Q = ["id_proceso", "nit_entidad", "ciudad_norm", "fecha_corte"] + NUM_P + CAT_P

ORDEN_D = (NUM_P + CAT_P + ["tipo_proveedor"]
           + [f"comp_{g}_{s}" for g in ["ent", "moddep", "fam", "entmod", "modprecio"] for s in ["media", "unico", "nproc"]]
           + ["P_ENT", "P_EMP", "emp_dec", "emp_gan", "ent_dec", "ent_gan", "n_part_prev", "es_pyme",
              "emp_dec_365", "emp_gan_365", "dias_ult_part", "n_part_365",
              "con_n", "con_n_365", "con_logvalor_medio", "con_misma_entidad",
              "tipo_empresa_grupo", "mismo_depto", "mismo_municipio", "antiguedad_dias",
              "precio_vs_historial", "precio_vs_ganados"]
           + [f"perf_{d}_{s}" for d in ["fam", "ent", "dep", "mod", "moddep"] for s in ["tasa", "n"]]
           + ["knn_n", "knn_max_sim", "knn_media_sim10", "knn_tasa5", "knn_tasa50",
              "knn_max_sim_ganado", "knn_max_sim_perdido"])
ORDEN_D3 = ([f"riv_{g}_{c}" for g in ["ent", "entmod", "fam", "moddep"] for c in ["f_media", "f_max", "x_media"]]
            + ["ventaja_vs_rivales_ent", "ventaja_vs_rivales_entmod"]
            + [f"ent_{g}_{s}" for g in ["ent", "entmod"] for s in ["repite_ganador", "gana_pyme", "gana_local", "n_ganadores"]]
            + ["emp_ult_resultado", "emp_ent_ult_resultado", "dias_ult_victoria",
               "emp_tasa_90", "emp_dec_90d", "emp_tasa_365", "emp_dec_365d", "emp_tendencia",
               "grafo_part_ent", "grafo_gana_ent", "grafo_gana_fam"])

SOSP = ["dias_ult_part", "n_part_prev", "n_part_365", "precio_vs_historial"]  # features con la fuga
OK3 = ["n_part_prev_ok", "n_part_365_ok", "dias_ult_part_ok"]  # sus reemplazos

# Orden de columnas con que se entrenaron los modelos (secciones 7 y 8)
CAT_G = [c for c in ORDEN_D + ORDEN_D3 if c in CAT_OUT]
FEATS_G1 = CAT_G + [c for c in ORDEN_D + ORDEN_D3 if c not in CAT_OUT and c not in SOSP] + OK3
FEATS_V2 = ["score_retrieval", "rank_retrieval", "es_frio"] + [c for c in ORDEN_D + ORDEN_D3 if c not in SOSP] + OK3

STOP = ("de la el en y a los las del se por con para un una su al lo que o e es como sus le mas este esta "
        "entre sin sobre segun tambien no ser son ha fue cual donde desde hasta ante bajo durante mediante "
        "proceso minima cuantia contratacion contratar contrato objeto prestacion modalidad seleccion "
        "abreviada menor invitacion publica entidad municipio departamento").split()

# ---------- Retrieval M1 (notebook 03, celdas 18 y 19) ----------
CORTE_TFIDF_RETRIEVAL = "2025-01-01"  # vocabulario del retrieval: procesos publicados antes de 2025
VENTANA_DIAS = 730  # historial de 2 años
PESOS_GENERAN = {"unspsc_familia": 1.0, "nit_entidad": 1.0}
PESOS_REFUERZAN = {"unspsc_segmento": 0.5, "departamento_entidad": 0.25}
TOP_TEXTO = 200
TOP_LOCAL = 200
ATTRS = ["unspsc_familia", "unspsc_segmento", "nit_entidad", "departamento_entidad"]
STOP_NOMBRE = "sas s a ltda sa e u eu y de la el del los las limitada sociedad compania cia".split()
W_SIM_VECINOS = (0.60, 0.25, 0.10, 0.05)  # nombre, departamento, tipo de empresa, pyme
K_VECINOS = 20
FEATS_M1 = ["f_fam", "f_ent", "f_seg", "f_dep", "s_afin", "s_texto", "s_afin_n", "s_texto_n", "s_hib",
            "rank_afin", "rank_texto", "n_cand", "n_part_hist", "sim_vecinos", "es_frio",
            "modalidad_cod", "tipo_contrato_cod", "log_precio_base", "dias_para_ofertar", "proc_mes",
            "n_local", "f_local"]
CATS_M1 = ["modalidad_cod", "tipo_contrato_cod"]


def verificar_config(ruta_config: str | Path) -> None:
    """Falla si ``config.json`` del notebook final no coincide con las constantes de este módulo."""
    cfg = json.loads(Path(ruta_config).read_text(encoding="utf-8"))
    esperado = {"K": K, "n_lista": N_LISTA, "suavizados": [A_ENT, A_EMP, A2, A_PRIOR],
                "filtros_lista2": [MAX_COMP, MAX_PRECIO]}
    distintos = {k: (cfg.get(k), v) for k, v in esperado.items() if cfg.get(k) != v}
    if distintos:
        raise ValueError(f"config.json no coincide con ml/config.py (leído, esperado): {distintos}")
