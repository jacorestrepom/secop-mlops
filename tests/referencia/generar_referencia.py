"""Genera la referencia de paridad exacta: ml/ de punta a punta para un mes, dentro del contenedor de pruebas.

    python -m tests.referencia.generar_referencia --data /data --salida /salida [--mes 2026-03-01]

Escribe en --salida (una carpeta por mes, AAAAMM/):
    candidatos.parquet  recuperar()  (150 por empresa)
    features.parquet    construir()  (106 features)
    puntuados.parquet   rankear()    (llaves + s_v2 + p_gana)
    listas.parquet      armar_listas()
    sha256.txt          huella de cada archivo (para confirmar que dos corridas dan lo mismo)

La referencia NO va al repo: se sube a S3 y el CI la descarga (SECOP_REFERENCIA_DIR).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import time
from pathlib import Path

import pandas as pd

from ml.features import Historia, construir
from ml.listas import armar_listas
from ml.ranking import rankear
from ml.retrieval import HistoriaRetrieval, recuperar

LLAVES = ["pk", "pid", "proveedor_key", "id_proceso"]


def _guardar(df: pd.DataFrame, ruta: Path) -> None:
    df.to_parquet(ruta, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="carpeta data/ con clean/, retrieval_v2/ y final/")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--mes", default="2026-03-01", help="primer día del mes (fecha de corte del notebook)")
    a = ap.parse_args()
    data, mes = Path(a.data), pd.Timestamp(a.mes)
    out = Path(a.salida) / mes.strftime("%Y%m")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    h = HistoriaRetrieval.desde_carpeta(data / "clean")
    empresas, procesos = h.entradas_del_mes(mes)
    cand = recuperar(empresas, procesos, mes, historia=h, modelo=data / "retrieval_v2" / "M1_final_unico.txt")
    del h
    gc.collect()
    _guardar(cand, out / "candidatos.parquet")
    print(f"recuperar: {len(cand):,} pares | {time.time() - t0:.0f} s", flush=True)

    h = Historia.desde_carpeta(data / "clean")
    feats = construir(cand, historia=h)
    del h
    gc.collect()
    _guardar(feats, out / "features.parquet")
    print(f"construir: {feats.shape} | {time.time() - t0:.0f} s", flush=True)

    modelos = data / "final" / "modelos"
    punt = rankear(feats, g1=modelos / "g1_pgana.txt", v2=modelos / "ranking_v2_final.txt")
    _guardar(punt[LLAVES + ["s_v2", "p_gana"]], out / "puntuados.parquet")
    listas = armar_listas(punt)
    _guardar(listas[LLAVES + ["lista", "pos", "pos_v2", "s_v2", "p_gana", "pasa_filtros"]], out / "listas.parquet")
    print(f"listas: {len(listas):,} filas | {time.time() - t0:.0f} s", flush=True)

    lineas = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(out.glob("*.parquet"))]
    (out / "sha256.txt").write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
