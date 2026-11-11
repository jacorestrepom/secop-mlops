"""Genera la referencia de paridad exacta: ml/ de punta a punta para una muestra fija de empresas de un mes.

    python -m tests.referencia.generar_referencia --data /data --salida /salida [--mes 2026-03-01] [--n 300]

La muestra es fija (semilla 42) y estratificada: incluye empresas con historial y sin historial (arranque en frío
del retrieval). Con el mes completo, recuperar() y construir() necesitan ~14 GB y ~18 GB; con la muestra cabe en
un runner de CI. Las funciones de ml/ son las mismas: solo reciben menos empresas.

Escribe en --salida/AAAAMM/:
    muestra_empresas.parquet  proveedor_key de la muestra (y si es fría)
    candidatos.parquet        recuperar()  (150 por empresa)
    features.parquet          construir()  (106 features)
    puntuados.parquet         rankear()    (llaves + s_v2 + p_gana)
    listas.parquet            armar_listas()
    sha256.txt                huella de cada parquet (dos corridas deben dar lo mismo)

La referencia NO va al repo: se sube a S3 y el CI la descarga (SECOP_REFERENCIA_DIR).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LLAVES = ["pk", "pid", "proveedor_key", "id_proceso"]
COLS_LISTAS = ["lista", "pos", "pos_v2", "s_v2", "p_gana", "pasa_filtros"]
SEMILLA = 42
FRACCION_FRIAS = 0.2  # sobre-representa las empresas sin historial (en marzo son ~11 %)


def elegir_muestra(h, mes, n=300, semilla=SEMILLA) -> pd.DataFrame:
    """Muestra fija de empresas del mes: n·FRACCION_FRIAS sin historial y el resto con historial."""
    empresas, _ = h.entradas_del_mes(mes)
    pk = h.empresas_idx.get_indexer(pd.Index(empresas))
    _, hist = h.historial(pd.Timestamp(mes), pk)
    con_hist = set(hist["pk"].unique())
    frias = np.array(sorted(h.empresas_idx.values[[p for p in pk if p not in con_hist]]))
    normales = np.array(sorted(h.empresas_idx.values[[p for p in pk if p in con_hist]]))
    rng = np.random.default_rng(semilla)
    n_frias = min(len(frias), round(n * FRACCION_FRIAS))
    elegidas = np.r_[rng.choice(frias, n_frias, replace=False), rng.choice(normales, n - n_frias, replace=False)]
    return pd.DataFrame({"proveedor_key": elegidas, "es_frio": [1] * n_frias + [0] * (n - n_frias)})


def correr(data: Path, mes, empresas=None, n=300, log=print) -> dict[str, pd.DataFrame]:
    """recuperar → construir → rankear → armar_listas para la muestra (o para ``empresas`` si se da)."""
    from ml.features import Historia, construir
    from ml.listas import armar_listas
    from ml.ranking import rankear
    from ml.retrieval import HistoriaRetrieval, recuperar

    t0 = time.time()
    h = HistoriaRetrieval.desde_carpeta(data / "clean")
    muestra = elegir_muestra(h, mes, n) if empresas is None else pd.DataFrame({"proveedor_key": list(empresas)})
    _, procesos = h.entradas_del_mes(mes)
    cand = recuperar(muestra.proveedor_key, procesos, mes, historia=h,
                     modelo=data / "retrieval_v2" / "M1_final_unico.txt")
    del h
    gc.collect()
    log(f"recuperar: {len(cand):,} pares de {cand.pk.nunique()} empresas | {time.time() - t0:.0f} s")

    h = Historia.desde_carpeta(data / "clean")
    feats = construir(cand, historia=h)
    del h
    gc.collect()
    log(f"construir: {feats.shape} | {time.time() - t0:.0f} s")

    modelos = data / "final" / "modelos"
    punt = rankear(feats, g1=modelos / "g1_pgana.txt", v2=modelos / "ranking_v2_final.txt")
    listas = armar_listas(punt)
    log(f"listas: {len(listas):,} filas | {time.time() - t0:.0f} s")
    return {"muestra_empresas": muestra, "candidatos": cand, "features": feats,
            "puntuados": punt[LLAVES + ["s_v2", "p_gana"]],
            "listas": listas[LLAVES + COLS_LISTAS].reset_index(drop=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="carpeta data/ con clean/, retrieval_v2/ y final/")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--mes", default="2026-03-01", help="primer día del mes (fecha de corte del notebook)")
    ap.add_argument("--n", type=int, default=300, help="empresas de la muestra")
    a = ap.parse_args()
    mes = pd.Timestamp(a.mes)
    out = Path(a.salida) / mes.strftime("%Y%m")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = correr(Path(a.data), mes, n=a.n, log=lambda s: print(s, flush=True))
    for nombre, df in res.items():
        df.to_parquet(out / f"{nombre}.parquet", index=False)
    lineas = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(out.glob("*.parquet"))]
    (out / "sha256.txt").write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    pico_gb = pico / 1e9 if sys.platform == "darwin" else pico / 1e6  # bytes en macOS, KB en Linux
    print(f"tiempo total: {time.time() - t0:.0f} s | memoria pico (RSS): {pico_gb:.2f} GB")


if __name__ == "__main__":
    main()
