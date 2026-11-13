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


# ------------------------------------------------------------------ jobs/score.py: abiertos, motivos y publicación
def test_inferir_abiertos_solo_toca_los_abiertos_y_cuenta():
    from jobs.score import inferir_abiertos

    p = _procesos()
    p.loc[len(p)] = ["cerrado", "consultoria", pd.NA, pd.NA]  # nulo pero no abierto: no se infiere
    out, c = inferir_abiertos(p, ["real", "alta", "baja", "real_sin_seg"], MODELO, 0.5)
    assert out.loc[4, "unspsc_familia"] is pd.NA and not out.loc[4, "unspsc_inferido"]
    assert list(out.unspsc_inferido) == [False, True, False, False, False]
    assert (c["abiertos"], c["real"], c["inferido"], c["nulo"]) == (4, 2, 1, 1)
    assert c["unspsc_real_pct"] == 50 and c["unspsc_inferido_pct"] == 25 and c["unspsc_nulo_pct"] == 25


def test_sin_modelo_no_infiere():
    from jobs.score import inferir_abiertos

    out, c = inferir_abiertos(_procesos(), ["alta", "baja"], None, None)
    assert not out.unspsc_inferido.any() and out.unspsc_familia.isna().sum() == 2
    assert (c["inferido"], c["nulo"]) == (0, 2)


def _fila(**k):
    base = dict(con_misma_entidad=2, lista="historial", entidad_nueva=False, comp_entmod_media=2.0,
                precio_vs_ganados=0.5, knn_max_sim_ganado=0.9, unspsc_inferido=False)  # fmt: skip
    return pd.Series({**base, **k})


def test_ningun_motivo_cita_la_categoria():
    from jobs.score import motivos

    for inferido in (False, True):
        for m in motivos(_fila(unspsc_inferido=inferido)):
            assert not any(t in m.lower() for t in ("categor", "unspsc", "familia", "segmento")), m


def test_un_motivo_de_categoria_no_sale_si_fue_inferida(monkeypatch):
    import jobs.score as S

    monkeypatch.setattr(S, "MOTIVOS_DE_CATEGORIA", frozenset({"Ya ha trabajado con esta entidad"}))
    assert "Ya ha trabajado con esta entidad" in S.motivos(_fila(unspsc_inferido=False))
    assert "Ya ha trabajado con esta entidad" not in S.motivos(_fila(unspsc_inferido=True))
