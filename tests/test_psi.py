"""PSI suavizado del reporte de drift (jobs/score.py)."""

import pandas as pd

from jobs.score import categorias_raras, psi_suavizado


def test_misma_distribucion_da_cero():
    x = pd.Series(["a"] * 50 + ["b"] * 50)
    assert abs(psi_suavizado(x, x.sample(frac=1, random_state=0))) < 1e-12


def test_raras_en_las_dos_van_a_otras_y_no_explotan():
    ref = pd.Series(["a"] * 990 + ["r1"] * 5 + ["r2"] * 5)
    hoy = pd.Series(["a"] * 990 + ["r3"] * 5 + ["r4"] * 5)
    assert categorias_raras(ref, hoy) == {"r1", "r2", "r3", "r4"}
    assert psi_suavizado(ref, hoy) < 1e-9  # sin agrupar, cada rara ausente de un lado sumaría ~0,04


def test_categoria_que_desaparece_queda_finita():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    hoy = pd.Series(["a"] * 100)
    v = psi_suavizado(ref, hoy)
    assert 2 < v < 10
