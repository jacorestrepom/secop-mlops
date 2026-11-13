"""Imputación de la familia UNSPSC desde la descripción del proceso.

Desde octubre de 2026, datos.gov.co publica ``codigo_principal_de_categoria = 'UNSPECIFIED'`` en el 99,6 % de los
procesos nuevos, y silver los deja con ``unspsc_familia`` y ``unspsc_segmento`` nulos. Este módulo infiere la
familia SOLO donde falta, con el clasificador aprobado en el experimento ``tests/experimentos/unspsc_imputado.py``:
TF-IDF de palabras y bigramas sobre ``descripcion_limpia`` (150 mil términos) + ``SGDClassifier(log_loss)``.

- ``entrenar(procesos)``: entrena con los procesos que tienen UNSPSC real.
- ``inferir(procesos, modelo, umbral=0.5)``: llena ``unspsc_familia`` donde es nula y la clase más probable pasa
  el umbral; el segmento son los 2 primeros dígitos de esa familia (siempre coherentes); lo que no pasa el umbral
  queda nulo. Agrega ``unspsc_inferido`` (bool). Nunca cambia un UNSPSC real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UMBRAL = 0.5
MAX_TERMINOS = 150_000
SEMILLA = 42


def nuevo_modelo(max_terminos: int = MAX_TERMINOS):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline

    from ml.config import STOP

    return make_pipeline(
        TfidfVectorizer(
            min_df=3,
            max_df=0.5,
            max_features=max_terminos,
            ngram_range=(1, 2),
            stop_words=STOP,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=15, tol=1e-4, n_jobs=4, random_state=SEMILLA),
    )


def _textos(procesos: pd.DataFrame) -> np.ndarray:
    return procesos.descripcion_limpia.astype("string").fillna("").to_numpy(dtype=object)


def entrenar(procesos: pd.DataFrame, max_terminos: int = MAX_TERMINOS):
    """Ajusta el clasificador con los procesos que traen familia real (los nulos se ignoran)."""
    tr = procesos[procesos.unspsc_familia.notna()]
    modelo = nuevo_modelo(max_terminos)
    modelo.fit(_textos(tr), tr.unspsc_familia.astype(str).to_numpy())
    return modelo


def predecir(procesos: pd.DataFrame, modelo) -> pd.DataFrame:
    """Clase más probable y su probabilidad, para cada fila (con el mismo índice)."""
    if len(procesos) == 0:
        return pd.DataFrame({"familia_pred": pd.Series(dtype="string"), "prob": pd.Series(dtype=float)})
    proba = modelo.predict_proba(_textos(procesos))
    i = proba.argmax(1)
    return pd.DataFrame(
        {"familia_pred": pd.array(modelo.classes_[i].astype(str), dtype="string"), "prob": proba[np.arange(len(i)), i]},
        index=procesos.index,
    )


def inferir(procesos: pd.DataFrame, modelo, umbral: float = UMBRAL) -> pd.DataFrame:
    """Copia de ``procesos`` con la familia inferida donde era nula (y pasa el umbral) y ``unspsc_inferido``."""
    if not procesos.index.is_unique:
        raise ValueError("inferir() necesita un índice único")
    out = procesos.copy()
    for c in ("unspsc_familia", "unspsc_segmento"):
        out[c] = out[c].astype("string")
    nulos = out.unspsc_familia.isna()
    pred = predecir(out[nulos], modelo)
    pasa = pred.index[pred.prob.to_numpy() >= umbral]
    out.loc[pasa, "unspsc_familia"] = pred.loc[pasa, "familia_pred"]
    out.loc[pasa, "unspsc_segmento"] = pred.loc[pasa, "familia_pred"].str[:2]
    out["unspsc_inferido"] = out.index.isin(pasa)
    return out


def evaluar(procesos: pd.DataFrame, modelo, umbral: float = UMBRAL) -> dict[str, float]:
    """Sobre procesos con UNSPSC real: cobertura (pasa el umbral) y acierto top-1 de familia y segmento entre los
    cubiertos; también el top-1 sin umbral."""
    real = procesos[procesos.unspsc_familia.notna()]
    pred = predecir(real, modelo)
    fam = real.unspsc_familia.astype(str).to_numpy()
    ok = pred.familia_pred.astype(str).to_numpy() == fam
    ok_seg = pred.familia_pred.astype(str).str[:2].to_numpy() == pd.Series(fam).str[:2].to_numpy()
    c = pred.prob.to_numpy() >= umbral
    return {
        "n_evaluados": float(len(real)),
        "cobertura": float(c.mean()),
        "top1_familia": float(ok[c].mean()) if c.any() else float("nan"),
        "top1_segmento": float(ok_seg[c].mean()) if c.any() else float("nan"),
        "top1_familia_sin_umbral": float(ok.mean()),
    }
