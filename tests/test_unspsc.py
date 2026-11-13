"""Imputación de la familia UNSPSC (ml/unspsc.py) y su efecto en lo que se publica (jobs/score.py)."""

import numpy as np
import pandas as pd

from ml.unspsc import inferir


class ModeloFijo:
    """Devuelve, para cada texto, la probabilidad que se le dio en ``tabla`` (clase -> prob)."""

    classes_ = np.array(["4311", "7210", "8011"])

    def __init__(self, tabla):
        self.tabla = tabla

    def predict_proba(self, textos):
        out = np.zeros((len(textos), len(self.classes_)))
        for i, t in enumerate(textos):
            clase, p = self.tabla[t]
            j = list(self.classes_).index(clase)
            out[i] = (1 - p) / (len(self.classes_) - 1)
            out[i, j] = p
        return out


def _procesos():
    return pd.DataFrame(
        {
            "id_proceso": ["real", "alta", "baja", "real_sin_seg"],
            "descripcion_limpia": ["obra civil", "consultoria", "algo raro", "obra"],
            "unspsc_familia": pd.array(["9512", None, None, "5013"], dtype="string"),
            "unspsc_segmento": pd.array(["95", None, None, None], dtype="string"),
        }
    )


MODELO = ModeloFijo(
    {"obra civil": ("4311", 0.9), "consultoria": ("8011", 0.8), "algo raro": ("7210", 0.4), "obra": ("4311", 0.9)}
)


def test_no_toca_los_valores_reales():
    out = inferir(_procesos(), MODELO)
    assert out.loc[0, "unspsc_familia"] == "9512" and out.loc[0, "unspsc_segmento"] == "95"
    assert out.loc[3, "unspsc_familia"] == "5013" and pd.isna(out.loc[3, "unspsc_segmento"])
    assert not out.loc[[0, 3], "unspsc_inferido"].any()


def test_segmento_coherente_con_la_familia_inferida():
    out = inferir(_procesos(), MODELO)
    assert out.loc[1, "unspsc_familia"] == "8011" and out.loc[1, "unspsc_segmento"] == "80"
    assert bool(out.loc[1, "unspsc_inferido"])
    inf = out[out.unspsc_inferido]
    assert (inf.unspsc_segmento == inf.unspsc_familia.str[:2]).all()


def test_bajo_el_umbral_queda_nulo():
    out = inferir(_procesos(), MODELO, umbral=0.5)
    assert pd.isna(out.loc[2, "unspsc_familia"]) and pd.isna(out.loc[2, "unspsc_segmento"])
    assert not bool(out.loc[2, "unspsc_inferido"])
    assert inferir(_procesos(), MODELO, umbral=0.3).loc[2, "unspsc_familia"] == "7210"


def test_no_modifica_la_entrada():
    p = _procesos()
    inferir(p, MODELO)
    assert pd.isna(p.loc[1, "unspsc_familia"]) and "unspsc_inferido" not in p.columns
