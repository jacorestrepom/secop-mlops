"""Limpieza silver → clean (notebook 02, secciones 8 a 10), copiada sin cambios.

Entrada: las cuatro tablas de silver tal como las consulta el notebook 02 (celdas 38, 39, 41 y 42; copia local
en data/raw/*_20261104.parquet). Salida: procesos, proponentes, contratos y proveedores como en data/clean/.

Lo que el notebook 02 NO genera y sí está en data/clean/procesos.parquet (sale de una celda que no quedó en ningún
notebook, igual que base_entrenamiento y los mapas): log_precio_base, log_duracion_dias, tiene_lotes, proc_mes,
proc_fin_de_anio, modalidad_grupo y las columnas *_cat. Este módulo no las inventa: ``COLUMNAS_SIN_ORIGEN``.
Tampoco aplica la única regla de esa celda que se ve en los datos: en data/clean/procesos.parquet, precio_base
< 1.000 queda nulo (353 procesos del snapshot del 4-nov). Ver ``PRECIO_MINIMO_SIN_ORIGEN``.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

ESTADOS_EXPERIENCIA = ["En ejecución", "Cerrado", "Modificado", "terminado",
                       "Aprobado", "Suspendido", "Prorrogado"]  # filtro de contratos (se aplica en la consulta)
TIPOS_CONTRATO = ["Prestación de servicios", "Suministros", "Compraventa", "Obra",
                  "Decreto 092 de 2017", "Consultoría", "Interventoría", "Seguros"]
COLUMNAS_SIN_ORIGEN = ["log_precio_base", "log_duracion_dias", "tiene_lotes", "proc_mes", "proc_fin_de_anio",
                       "modalidad_grupo", "sector_entidad_cat", "unspsc_segmento_cat", "unspsc_familia_cat",
                       "departamento_entidad_cat"]
PRECIO_MINIMO_SIN_ORIGEN = 1_000  # regla observada en data/clean, no está en el notebook 02
COLS_FECHA = {
    "procesos": ["fecha_publicacion", "fecha_adjudicacion", "fecha_limite_ofertas"],
    "proponentes": ["fecha_publicacion"],
    "contratos": ["fecha_firma"],
    "proveedores": ["fecha_registro"],
}


def normalizar(s: pd.Series) -> pd.Series:
    """Minúsculas, sin tildes, sin 'd.c.' ni puntuación. Se calcula sobre los valores únicos (rápido)."""
    def _norm(x):
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii").lower()
        x = re.sub(r"\bd\.?\s?c\.?\b", " ", x)  # quita "d.c.", "dc", "d.c"
        x = re.sub(r"[^a-z0-9 ]", " ", x)
        return re.sub(r"\s+", " ", x).strip()
    mapa = {v: _norm(v) for v in s.dropna().unique()}
    return s.map(mapa)


def agrupar_tipo_empresa(t):
    if t is None or pd.isna(t):
        return None
    t = t.lower()
    if "persona natural" in t or "empresario individual" in t:
        return "persona_natural"
    if any(k in t for k in ["consorcio", "unión temporal", "union temporal", "estructura plural"]):
        return "plural"
    if any(k in t for k in ["entidad estatal", "entidades públicas", "derecho público", "administración",
                            "organismo público", "instituto público", "entidad pública"]):
        return "publica"
    if any(k in t for k in ["sin ánimo", "asociaci", "fundaci", "corporaciones", "cooperativ",
                            "junta de acción", "veeduria", "religios", "federaci",
                            "educación superior", "universidad"]):
        return "sin_animo_lucro"
    if any(k in t for k in ["sociedad", "sociedade", "empresa unipersonal", "sucursal",
                            "corporation", "partnership"]):
        return "sociedad"
    return "otro"


def limpiar_proveedores(proveedores: pd.DataFrame, procesos: pd.DataFrame) -> pd.DataFrame:
    """Celda 45: departamento válido, municipio normalizado y tipo de empresa en 6 grupos."""
    prov = proveedores.copy()
    DEPTOS_VALIDOS = set(procesos["departamento_entidad"].dropna().unique())
    prov["departamento"] = prov["departamento"].where(prov["departamento"].isin(DEPTOS_VALIDOS))
    prov["municipio"] = prov["municipio"].replace("No Provisto", pd.NA)
    prov["municipio_norm"] = normalizar(prov["municipio"])
    mapa_te = {v: agrupar_tipo_empresa(v) for v in prov["tipo_empresa"].dropna().unique()}
    prov["tipo_empresa_grupo"] = prov["tipo_empresa"].map(mapa_te)
    return prov


def limpiar_procesos(procesos: pd.DataFrame) -> pd.DataFrame:
    """Celda 46: ciudad imputada (Bogotá y ciudad única de la entidad) y tipo de contrato en 8 grupos + Otro."""
    proc = procesos.copy()
    es_bogota = (proc["departamento_entidad"] == "Distrito Capital de Bogotá") & proc["ciudad_entidad"].isna()
    proc.loc[es_bogota, "ciudad_entidad"] = "Bogotá"
    ciudades = proc.dropna(subset=["ciudad_entidad"]).groupby("nit_entidad")["ciudad_entidad"].agg(["nunique", "first"])
    ciudad_unica = ciudades.loc[ciudades["nunique"] == 1, "first"]
    faltante = proc["ciudad_entidad"].isna()
    proc["ciudad_imputada"] = faltante & proc["nit_entidad"].isin(ciudad_unica.index)
    proc.loc[faltante, "ciudad_entidad"] = proc.loc[faltante, "nit_entidad"].map(ciudad_unica)
    proc["ciudad_imputada"] = proc["ciudad_imputada"] | es_bogota  # marca de control, no es feature
    proc["ciudad_norm"] = normalizar(proc["ciudad_entidad"])
    proc["tipo_contrato_grupo"] = proc["tipo_contrato"].where(proc["tipo_contrato"].isin(TIPOS_CONTRATO), "Otro")
    return proc


def controles(proc, proponentes, contratos, prov) -> dict[str, bool]:
    """Celda 49: controles de integridad."""
    return {"id_proceso único en procesos": bool(proc["id_proceso"].is_unique),
            "proveedor_key único en proveedores": bool(prov["proveedor_key"].is_unique),
            "(id_proceso, proveedor_key) único en proponentes":
                not proponentes.duplicated(["id_proceso", "proveedor_key"]).any(),
            "todo proponente tiene su proceso": bool(proponentes["id_proceso"].isin(proc["id_proceso"]).all()),
            "contratos sin fecha_firma nula": bool(contratos["fecha_firma"].notna().all())}


def limpiar(procesos, proponentes, contratos, proveedores) -> dict[str, pd.DataFrame]:
    """Secciones 8 a 10: las cuatro tablas limpias, con las fechas convertidas (celda 52)."""
    out = {"procesos": limpiar_procesos(procesos), "proponentes": proponentes.copy(),
           "contratos": contratos.copy(), "proveedores": limpiar_proveedores(proveedores, procesos)}
    for nombre, cols in COLS_FECHA.items():
        for c in cols:
            out[nombre][c] = pd.to_datetime(out[nombre][c], errors="coerce")
    return out
