"""Experimento: ¿se recupera la pérdida por procesos sin UNSPSC infiriendo la familia desde la descripción?

Clasificador simple: TF-IDF (palabras + bigramas, sobre descripcion_limpia) + SGDClassifier(log_loss), que
predice unspsc_familia; el segmento son los 2 primeros dígitos de la familia. Point-in-time: se entrena SOLO con
procesos publicados antes del 2026-03-01 que tienen UNSPSC. Sin ensambles ni ajuste fino.

    python -m tests.experimentos.unspsc_imputado --data <data/> --salida <dir> --entrenar
    python -m tests.experimentos.unspsc_imputado --data <data/> --salida <dir> --variante imputado --umbral 0.5
    python -m tests.experimentos.unspsc_imputado --data <data/> --salida <dir> --comparar --umbral 0.5

--comparar usa también puntuados/listas de las variantes "normal" y "sin_unspsc" de unspsc_faltante.py, en la
misma carpeta de salida. Experimento local: NO está integrado a ml/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import tests.experimentos.unspsc_faltante as F

UMBRALES = [0.0, 0.5, 0.7]
SEMILLA = 42


def entrenar(data: Path, salida: Path, max_features: int = 300_000, sufijo: str = "") -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline

    from ml.config import STOP

    p = pd.read_parquet(
        data / "clean" / "procesos.parquet",
        columns=["id_proceso", "fecha_publicacion", "descripcion_limpia", "unspsc_segmento", "unspsc_familia"],
    )
    p["descripcion_limpia"] = p.descripcion_limpia.fillna("")
    antes = (pd.to_datetime(p.fecha_publicacion) < F.FECHA).values
    tr = p[antes & p.unspsc_familia.notna().values]
    t0 = time.time()
    modelo = make_pipeline(
        TfidfVectorizer(
            min_df=3,
            max_df=0.5,
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words=STOP,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=15, tol=1e-4, n_jobs=4, random_state=SEMILLA),
    )
    modelo.fit(tr.descripcion_limpia.values, tr.unspsc_familia.astype(str).values)
    t_ent = time.time() - t0
    ruta = salida / f"clasificador_unspsc{sufijo}.joblib"
    joblib.dump(modelo, ruta, compress=3)
    print(
        f"entrenamiento: {len(tr):,} procesos (< {F.FECHA.date()}) | {tr.unspsc_familia.nunique()} familias | "
        f"{t_ent:.0f} s | modelo {ruta.stat().st_size / 1e6:.1f} MB"
    )

    mes = p[F._procesos_del_mes(p.fecha_publicacion)].reset_index(drop=True)
    t0 = time.time()
    proba = modelo.predict_proba(mes.descripcion_limpia.values)
    t_inf = time.time() - t0
    print(f"inferencia: {len(mes):,} procesos en {t_inf:.2f} s → ~{2000 * t_inf / len(mes):.2f} s por 2.000 procesos")
    clases = modelo.classes_
    i = proba.argmax(1)
    pred = pd.DataFrame(
        {
            "id_proceso": mes.id_proceso.astype(str),
            "familia_pred": clases[i],
            "prob": proba[np.arange(len(mes)), i],
            "familia_real": mes.unspsc_familia,
            "segmento_real": mes.unspsc_segmento,
        }
    )
    pred["segmento_pred"] = pred.familia_pred.str[:2]
    pred.to_parquet(salida / f"predicciones_unspsc{sufijo}.parquet", index=False)

    real = pred[pred.familia_real.notna()]
    print(
        f"\nexactitud en {F.MES} ({len(real):,} procesos con UNSPSC real; {len(pred) - len(real):,} ya venían nulos):"
    )
    print("umbral | cobertura | top-1 familia | top-1 segmento (cubiertos) | familia correcta / incorrecta / nula")
    for u in UMBRALES:
        c = real.prob >= u
        ok_f = (real.familia_pred == real.familia_real.astype(str))[c]
        ok_s = (real.segmento_pred == real.segmento_real.astype(str))[c]
        print(
            f"{u:.1f} | {c.mean():.1%} | {ok_f.mean():.1%} | {ok_s.mean():.1%} | "
            f"{ok_f.sum():,} / {(~ok_f).sum():,} / {(~c).sum():,}"
        )


def imputacion(salida: Path, umbral: float) -> pd.DataFrame:
    pred = pd.read_parquet(salida / "predicciones_unspsc.parquet")
    pasa = pred.prob >= umbral
    return pd.DataFrame(
        {
            "id_proceso": pred.id_proceso,
            "unspsc_familia": pred.familia_pred.where(pasa),
            "unspsc_segmento": pred.segmento_pred.where(pasa),
        }
    )


def ndcg_por_empresa(lista1: pd.DataFrame, prop_m: pd.DataFrame) -> pd.Series:
    """NDCG@10 de cada empresa (misma fórmula que ml.evaluacion.metricas_lista1, sin promediar)."""
    from ml.evaluacion import ideal_ndcg, relevancia

    pm = prop_m[prop_m.mes == F.MES]
    ideal = pm.groupby("proveedor_key").agg(n=("gana", "size"), n2=("gana", "sum"))
    x = lista1.assign(proveedor_key=lista1.proveedor_key.astype(str))
    x = x.assign(rel=relevancia(x, prop_m, F.MES))
    dcg = (x.rel / np.log2(x.pos + 1)).groupby(x.proveedor_key).sum()
    e = ideal.join(dcg.rename("dcg"), how="inner")
    return e.dcg / ideal_ndcg(e.n.values, e.n2.values)


def comparar(data: Path, salida: Path, umbral: float) -> None:
    from scipy import stats

    from ml.evaluacion import metricas_lista1, participaciones, pct_plausibles, recall_retrieval

    pm = participaciones(pd.read_parquet(data / "clean" / "proponentes.parquet"))
    pos = pd.read_parquet(salida / "positivos.parquet")
    res, ndcg = {}, {}
    for v in ["normal", "sin_unspsc", "imputado"]:
        p = pd.read_parquet(salida / f"puntuados_{v}.parquet", columns=["pk", "pid", "rank_retrieval"])
        L = pd.read_parquet(salida / f"listas_{v}.parquet")
        L1 = L[L.lista == "historial"]
        m = {"Recall@150 retrieval": recall_retrieval(p, pos)}
        m.update({k: x for k, x in metricas_lista1(L1, pm, F.MES).items() if k != "empresas"})
        m["% plausibles lista 2"] = pct_plausibles(L[L.lista == "oportunidad"], pm, F.MES) / 100
        res[v], ndcg[v] = m, ndcg_por_empresa(L1, pm)
    print(f"métrica | normal | sin UNSPSC | imputado (umbral {umbral}) | pérdida recuperada")
    for k in res["normal"]:
        a, b, c = res["normal"][k], res["sin_unspsc"][k], res["imputado"][k]
        rec = f"{100 * (c - b) / (a - b):.0f} %" if abs(a - b) > 1e-9 else "—"
        print(f"{k} | {a:.5f} | {b:.5f} | {c:.5f} | {rec}")

    d = (ndcg["imputado"] - ndcg["sin_unspsc"].reindex(ndcg["imputado"].index)).dropna().values
    rng = np.random.default_rng(SEMILLA)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_boot = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    w = stats.wilcoxon(d[d != 0]) if (d != 0).any() else None
    t = stats.ttest_1samp(d, 0)
    print(f"\nprueba pareada NDCG@10 lista 1 (imputado − sin UNSPSC), {len(d):,} empresas:")
    print(
        f"  diferencia media {d.mean():+.5f} | IC 95 % bootstrap [{lo:+.5f}, {hi:+.5f}] | p bootstrap {p_boot:.4f} | "
        f"p t pareada {t.pvalue:.2e} | p Wilcoxon {w.pvalue if w else float('nan'):.2e} | "
        f"empresas que mejoran {np.mean(d > 0):.1%}, empeoran {np.mean(d < 0):.1%}, igual {np.mean(d == 0):.1%}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--entrenar", action="store_true")
    ap.add_argument("--variante", choices=["imputado"])
    ap.add_argument("--comparar", action="store_true")
    ap.add_argument("--umbral", type=float, default=0.5)
    ap.add_argument("--mes", default="2026-03-01", help="primer día del mes evaluado")
    ap.add_argument("--max-features", type=int, default=300_000)
    ap.add_argument("--sufijo", default="", help="sufijo de los archivos del clasificador (para comparar tamaños)")
    a = ap.parse_args()
    F.fijar_mes(a.mes)
    data, salida = Path(a.data), Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    if a.entrenar:
        entrenar(data, salida, a.max_features, a.sufijo)
    elif a.variante:
        F.correr(data, salida, "imputado", imputacion=imputacion(salida, a.umbral))
    elif a.comparar:
        comparar(data, salida, a.umbral)


if __name__ == "__main__":
    main()
