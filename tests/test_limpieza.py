"""Limpieza silver → clean (ml/limpieza.py) frente a data/clean/.

Lee data/raw/*_20261104.parquet (snapshot de silver del 4-nov-2026), aplica ml.limpieza.limpiar y compara con
data/clean/: mismos conteos por tabla y mismos valores en cada columna que produce el notebook 02. Las columnas de
procesos que salen de una celda perdida (ml.limpieza.COLUMNAS_SIN_ORIGEN) se reportan aparte. Se salta si no hay
SECOP_DATA_DIR/raw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import DATA

pytestmark = pytest.mark.skipif(DATA is None or not (DATA / "raw" / "procesos_20261104.parquet").exists(),
                                reason="no hay SECOP_DATA_DIR/raw/*_20261104.parquet")
TABLAS = ["procesos", "proponentes", "contratos", "proveedores"]


@pytest.fixture(scope="module")
def limpias():
    from ml.limpieza import limpiar

    raw = {t: pd.read_parquet(DATA / "raw" / f"{t}_20261104.parquet") for t in TABLAS}
    return limpiar(**raw)


def _iguales(a: pd.Series, b: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(a) or pd.api.types.is_datetime64_any_dtype(b):
        a, b = pd.to_datetime(a).astype("datetime64[ns]"), pd.to_datetime(b).astype("datetime64[ns]")
        return bool(((a == b) | (a.isna() & b.isna())).all())
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
        return bool(np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True))
    return bool(a.astype(object).where(a.notna(), None).tolist() == b.astype(object).where(b.notna(), None).tolist())


def test_controles_de_integridad(limpias):
    from ml.limpieza import controles

    malos = [k for k, ok in controles(limpias["procesos"], limpias["proponentes"], limpias["contratos"],
                                      limpias["proveedores"]).items() if not ok]
    assert not malos, f"controles que fallan: {malos}"


@pytest.mark.parametrize("tabla", TABLAS)
def test_limpieza_igual_a_clean(tabla, limpias):
    from ml.limpieza import COLUMNAS_SIN_ORIGEN

    nuestra = limpias[tabla]
    ref = pd.read_parquet(DATA / "clean" / f"{tabla}.parquet")
    assert len(nuestra) == len(ref), f"{tabla}: {len(nuestra)} filas frente a {len(ref)} en clean"
    faltan = [c for c in ref.columns if c not in nuestra.columns and c not in COLUMNAS_SIN_ORIGEN]
    assert not faltan, f"{tabla}: columnas de clean que la limpieza no produce: {faltan}"
    nuestra = nuestra.reset_index(drop=True)
    if tabla == "procesos":
        # Regla de la celda perdida (no está en el notebook 02): precio_base < 1.000 → nulo
        from ml.limpieza import PRECIO_MINIMO_SIN_ORIGEN

        bajo = nuestra.precio_base < PRECIO_MINIMO_SIN_ORIGEN
        assert ref.precio_base[bajo].isna().all()
        nuestra = nuestra.assign(precio_base=nuestra.precio_base.mask(bajo))
    distintas = [c for c in ref.columns if c in nuestra.columns and not _iguales(nuestra[c], ref[c])]
    assert not distintas, f"{tabla}: columnas con valores distintos: {distintas}"
