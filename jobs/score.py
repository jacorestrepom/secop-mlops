"""Job nocturno del recomendador (guía de traspaso, Paso 3): calcula y publica las dos listas de cada empresa.

    python -m jobs.score                          # fecha_corte = hoy en Bogotá
    python -m jobs.score --fecha-corte 2026-11-11
    python -m jobs.score --verificar-lotes 2026-03-01   # prueba: por lotes y sin naturales == de una vez

Etapas: frescura de silver → historia congelada (s3://…/modelos/v1/historia/, verificada con su manifiesto)
→ procesos de silver (Athena) por limpiar() → familia UNSPSC inferida en los abiertos que no la traen
(models:/secop-unspsc@production) → empresas a puntuar → recuperar() por lotes de empresas →
construir() / rankear() / armar_listas() por lotes de pares → motivos → publicar en RDS (atómico) → copia en
gold.recomendaciones.

Decisiones de v1 (README):
- Historial congelado al 27-oct-2026 (base_entrenamiento no se puede regenerar desde silver). Los procesos
  abiertos sí salen frescos de secop_silver.procesos y se agregan a la tabla de procesos de la historia.
- Procesos abiertos: estado 'abierto', publicados hasta la fecha_corte y con fecha límite posterior.
- UNSPSC: desde oct-2026 la fuente publica 'UNSPECIFIED' en casi todos los procesos nuevos. Antes de recuperar()
  se infiere la familia (y su segmento) de los abiertos que no la traen; los procesos inferidos entran a las dos
  historias, así que la familia también genera candidatos. Lo inferido nunca se publica como dato de SECOP: la
  tarjeta (serving.procesos) usa solo el UNSPSC real y serving.recomendaciones lleva unspsc_inferido. Si el modelo
  no carga, la noche sigue sin inferir (métrica unspsc_modelo_cargado = 0 → alarma a secop-alertas).
- Empresas: SOLO personas jurídicas que son usuarios objetivo: con participación en los 12 meses anteriores
  (calientes) o con contrato firmado en esos 12 meses y sin participación en 24 (frías). Ningún dato de
  personas naturales entra al cálculo ni a serving.

Códigos de salida: 0 publicado · 1 error (no publica; la API sigue con la versión anterior) · 2 silver no está
fresco para la fecha_corte (no publica).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BUCKET = "proyecto-integrador-bucket-01"
HISTORIA_S3 = "modelos/v1/historia/"
WORKGROUP = "proyecto-integrador-workgroup"
TZ = ZoneInfo("America/Bogota")
JOBS_FRESCURA = ("secop-silver-procesos", "secop-silver-ganador")  # E3 reescribe procesos; E6 lo completa
MODELOS = {"retrieval": "secop-retrieval", "pgana": "secop-pgana", "ranking": "secop-ranking"}
MODELO_UNSPSC = "secop-unspsc"  # opcional: si no carga, la noche sigue sin inferir
HISTORIA_CONGELADA = "2026-10-27"
SNAPSHOT_HISTORIA = {  # snapshot de silver con el que se armó la historia (tags de los modelos en MLflow)
    "procesos": "5704566866417056960",
    "proponentes": "2911693562262410165",
    "contratos": "528333321308876835",
    "proveedores": "7035817432963350399",
    "fecha": "2026-11-04",
}
SQL_PROCESOS = """
SELECT id_proceso, id_portafolio, fecha_publicacion, modalidad, tipo_contrato,
       CAST(precio_base AS double) AS precio_base, precio_atipico, duracion_dias, dias_para_ofertar, numero_lotes,
       unspsc_segmento, unspsc_familia, unspsc_clase, descripcion_limpia,
       nit_entidad, departamento_entidad, ciudad_entidad, orden_entidad, sector_entidad,
       estado, fecha_adjudicacion, fecha_adjudicacion_es_respaldo, n_proponentes,
       CAST(valor_adjudicacion AS double) AS valor_adjudicacion, valor_atipico, fecha_limite_ofertas,
       nombre_entidad, descripcion, url_proceso
FROM secop_silver.procesos
"""  # columnas de la celda 38 del notebook 02 + las de la tarjeta (nombre_entidad, descripcion, url_proceso)

# Paso 5 · motivos (en este orden; máximo 3 por tarjeta). Umbral de similitud provisional: lo valida ML.
UMBRAL_SIM_GANADO = 0.5
MAX_MOTIVOS = 3

T0 = time.time()


# ---------------------------------------------------------------------------------------------------------- utilidades
def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for linea in f:
                if linea.startswith("VmRSS:"):
                    return int(linea.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def pico_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6  # macOS en bytes, Linux en KB


@contextmanager
def etapa(nombre: str):
    t = time.time()
    log(f"▶ {nombre}")
    yield
    gc.collect()
    log(f"✔ {nombre}: {time.time() - t:.1f} s | RSS {_rss_gb():.1f} GB | pico {pico_gb():.1f} GB")


def sha256_listas(listas: pd.DataFrame) -> str:
    """Huella de las listas (empresa, lista, posición, proceso, puntajes) para comprobar determinismo."""
    cols = ["proveedor_key", "lista", "pos", "id_proceso", "p_gana", "s_v2"]
    d = listas[cols].copy()
    d["p_gana"] = d.p_gana.astype(float).round(7)
    d["s_v2"] = d.s_v2.astype(float).round(7)
    d = d.sort_values(["proveedor_key", "lista", "pos"]).reset_index(drop=True)
    return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()


def hoy_bogota() -> str:
    return datetime.now(TZ).date().isoformat()


# ----------------------------------------------------------------------------------------------- 1 · frescura de silver
def evaluar_frescura(control: pd.DataFrame, fecha_corte: str) -> tuple[bool, str, dict]:
    """``control``: filas de control_corridas (job, tabla, estado, cuadre_ok, fin en UTC, snapshot_id).
    Fresco si E3 y E6 terminaron con éxito y cuadre en la fecha_corte (hora de Bogotá) y E6 después de E3."""
    c = control[control.tabla == "secop_silver.procesos"].copy()
    c["dia"] = pd.to_datetime(c.fin).dt.tz_localize("UTC").dt.tz_convert(TZ).dt.date.astype(str)
    ok = c[(c.estado == "exito") & c.cuadre_ok.fillna(False).astype(bool) & (c.dia == fecha_corte)]
    ultimo = {j: ok[ok.job == j].sort_values("fin").iloc[-1] for j in JOBS_FRESCURA if (ok.job == j).any()}
    faltan = [j for j in JOBS_FRESCURA if j not in ultimo]
    if faltan:
        return False, f"sin corrida exitosa con cuadre el {fecha_corte}: {', '.join(faltan)}", {}
    e3, e6 = ultimo["secop-silver-procesos"], ultimo["secop-silver-ganador"]
    if pd.Timestamp(e6.fin) < pd.Timestamp(e3.fin):
        return False, "secop-silver-ganador no corrió después de secop-silver-procesos", {}
    snap = {"procesos_abiertos_snapshot": str(e6.snapshot_id), "procesos_abiertos_fin_utc": str(e6.fin)}
    return True, f"E3 {e3.fin} y E6 {e6.fin} (UTC) con éxito y cuadre", snap


def leer_control() -> pd.DataFrame:
    import awswrangler as wr

    return wr.athena.read_sql_query(
        """SELECT job, tabla, estado, cuadre_ok, fin, snapshot_id FROM secop_control.control_corridas
           WHERE fin > current_timestamp - interval '4' day""",
        database="secop_control",
        workgroup=WORKGROUP,
        ctas_approach=False,
    )


# ---------------------------------------------------------------------------------------- 2 · historia y procesos
def bajar_historia(destino: Path) -> Path:
    """Copia s3://…/modelos/v1/historia/ y verifica cada archivo contra manifest_historia.sha256."""
    import boto3

    s3 = boto3.client("s3")
    destino.mkdir(parents=True, exist_ok=True)
    s3.download_file(BUCKET, HISTORIA_S3 + "manifest_historia.sha256", str(destino / "manifest_historia.sha256"))
    for linea in (destino / "manifest_historia.sha256").read_text().splitlines():
        esperado, nombre = linea.split(maxsplit=1)
        ruta = destino / nombre
        if not ruta.exists():
            s3.download_file(BUCKET, HISTORIA_S3 + nombre, str(ruta))
        h = hashlib.sha256()
        with open(ruta, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != esperado:
            raise RuntimeError(f"checksum distinto en {nombre}: no se usa la historia")
    return destino


def limpiar_procesos_silver(raw: pd.DataFrame) -> pd.DataFrame:
    """La parte de ml.limpieza.limpiar() que toca procesos (sobre la tabla completa: la ciudad se imputa por
    entidad con todos sus procesos)."""
    from ml.limpieza import COLS_FECHA, limpiar_procesos, reconstruir_columnas

    p = reconstruir_columnas(limpiar_procesos(raw))
    for c in COLS_FECHA["procesos"]:
        p[c] = pd.to_datetime(p[c], errors="coerce")
    return p


def procesos_abiertos(procesos: pd.DataFrame, fecha_corte: str) -> pd.DataFrame:
    c = pd.Timestamp(fecha_corte)
    return procesos[
        (procesos.estado == "abierto") & (procesos.fecha_publicacion <= c) & (procesos.fecha_limite_ofertas > c)
    ]


def unir_procesos(congelados: pd.DataFrame, frescos: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """La tabla de procesos que reciben las historias: la congelada intacta + los procesos nuevos de silver."""
    nuevos = frescos[~frescos.id_proceso.isin(congelados.id_proceso)]
    return pd.concat([congelados, nuevos[congelados.columns.intersection(nuevos.columns)]], ignore_index=True), len(
        nuevos
    )


# ------------------------------------------------------------------------------------------------ 3 · empresas
def empresas_a_puntuar(proveedores, proponentes, contratos, fecha_corte: str) -> tuple[list[str], list[str]]:
    """(calientes, frías), solo personas jurídicas que son usuarios objetivo."""
    c = pd.Timestamp(fecha_corte)
    jur = proveedores[
        (proveedores.es_usuario_objetivo == True)  # noqa: E712
        & proveedores.tipo_proveedor.notna()
        & (proveedores.tipo_proveedor != "persona_natural")
    ]
    jur = set(jur.proveedor_key)
    p = proponentes[proponentes.fecha_publicacion < c]
    p12 = set(p.loc[p.fecha_publicacion >= c - pd.DateOffset(months=12), "proveedor_key"])
    p24 = set(p.loc[p.fecha_publicacion >= c - pd.Timedelta(days=730), "proveedor_key"])
    k = contratos[(contratos.fecha_firma < c) & (contratos.fecha_firma >= c - pd.DateOffset(months=12))]
    calientes = sorted(jur & p12)
    frias = sorted((jur & set(k.proveedor_key)) - p24)
    return calientes, frias


def lotes(empresas: list[str], tam: int) -> list[list[str]]:
    return [empresas[i : i + tam] for i in range(0, len(empresas), tam)]


def centinelas(hr, fecha_corte: str) -> tuple[str, str]:
    """Dos empresas de relleno (usuarios objetivo con llave): una sin participaciones ni contratos en la ventana del
    retrieval (fría) y una con participaciones (caliente). ml.retrieval falla si en un lote no hay ninguna fría o
    ninguna con participaciones (el texto del historial sale solo de participaciones). Se agregan al lote al que le
    falte una y sus filas se descartan: como las empresas son independientes, no cambian a las demás."""
    mes = pd.Timestamp(fecha_corte)
    part, hist = hr.historial(mes, hr.objetivo)
    con_actividad = np.unique(hist.pk.values)
    frio = np.setdiff1d(hr.objetivo, con_actividad)
    con_part = np.unique(part.pk.values)
    return str(hr.empresas_idx.values[frio.min()]), str(hr.empresas_idx.values[con_part.min()])


def recuperar_por_lotes(hr, grupos, abiertos, fecha_corte, modelo, tam) -> pd.DataFrame:
    """recuperar() por lotes de ``tam`` empresas. Cada grupo (p. ej. calientes y frías) se parte por separado: un
    lote nunca mezcla grupos, así que los lotes de solo-frías existen y se prueban (verificar_lotes)."""
    from ml.retrieval import recuperar

    s_frio, s_cal = centinelas(hr, fecha_corte)
    mes = pd.Timestamp(fecha_corte)
    partes = []
    todos = [lote for g in grupos for lote in lotes(list(g), tam)]
    for i, lote in enumerate(todos, 1):
        pks = hr.empresas_idx.get_indexer(pd.Index(lote))
        pks = pks[pks >= 0]  # recuperar() descarta las que no tienen llave
        part, hist = hr.historial(mes, pks)
        n_actividad = len(np.intersect1d(pks, hist.pk.values))  # con participaciones o contratos
        n_part = len(np.intersect1d(pks, part.pk.values))  # con participaciones
        extra = ([s_frio] if n_actividad == len(pks) else []) + ([s_cal] if n_part == 0 else [])
        top = recuperar(lote + extra, abiertos, fecha_corte, historia=hr, modelo=modelo)
        if extra:
            top = top[~top.proveedor_key.isin(set(extra) - set(lote))]
        partes.append(top.drop(columns=["pk", "pid"]))
        log(f"   recuperar lote {i}: {len(lote):,} empresas (+{len(extra)} centinela) → {len(top):,} candidatos")
    return pd.concat(partes, ignore_index=True)


COLS_LISTAS = [
    "proveedor_key",
    "id_proceso",
    "lista",
    "pos",
    "s_v2",
    "p_gana",
    "score_retrieval",
    "es_frio",
    "entidad_nueva",
    "con_misma_entidad",
    "comp_entmod_media",
    "precio_vs_ganados",
    "knn_max_sim_ganado",
]


def puntuar_por_lotes(candidatos, h, g1, v2, max_pares, muestra_drift: list | None = None) -> pd.DataFrame:
    """construir → rankear → armar_listas por lotes de empresas completas (≤ max_pares pares por lote).
    Si se da ``muestra_drift`` (lista), le agrega una muestra fija (~N_DRIFT filas en total) de las features de
    drift de cada lote: así no hay que guardar todas las features."""
    from ml.features import construir
    from ml.listas import armar_listas
    from ml.ranking import rankear

    candidatos = candidatos.sort_values(["proveedor_key", "rank_retrieval"], kind="stable").reset_index(drop=True)
    tam = candidatos.groupby("proveedor_key", sort=False).size()
    corte = (tam.cumsum() // max_pares).rename("lote")
    lote_de = candidatos.proveedor_key.map(corte)
    partes = []
    frac = min(1.0, N_DRIFT / max(len(candidatos), 1))
    for i, (_, idx) in enumerate(candidatos.groupby(lote_de, sort=True).groups.items(), 1):
        pares = candidatos.loc[idx]
        feats = construir(pares, fecha_corte=None, historia=h)
        if muestra_drift is not None:
            muestra_drift.append(feats[COLS_DRIFT].sample(frac=frac, random_state=SEMILLA_DRIFT))
        listas = armar_listas(rankear(feats, g1=g1, v2=v2))
        del feats
        partes.append(listas[COLS_LISTAS])
        log(
            f"   puntuar lote {i}: {pares.proveedor_key.nunique():,} empresas, {len(pares):,} pares → "
            f"{len(listas):,} filas de listas | RSS {_rss_gb():.1f} GB"
        )
        del pares, listas
        gc.collect()
    return pd.concat(partes, ignore_index=True)


# ------------------------------------------------------------------------------------------------- 4 · motivos
# Motivos que se apoyan en la categoría UNSPSC del proceso. Hoy ninguno de los cinco la cita; si se agrega uno,
# va aquí y no se muestra cuando la categoría fue inferida (nunca se presenta como dato de SECOP).
MOTIVOS_DE_CATEGORIA: frozenset[str] = frozenset()


def motivos(fila) -> list[str]:
    """Paso 5 de la guía, en este orden, máximo 3. Ninguno cita la categoría si unspsc_inferido."""
    m = []
    if pd.notna(fila.con_misma_entidad) and fila.con_misma_entidad > 0:
        m.append("Ya ha trabajado con esta entidad")
    if fila.lista == "oportunidad" and bool(fila.entidad_nueva):
        m.append("Entidad con la que no has trabajado")
    if pd.notna(fila.comp_entmod_media) and fila.comp_entmod_media <= 3:
        m.append("Suele tener pocos competidores")
    if pd.notna(fila.precio_vs_ganados) and abs(fila.precio_vs_ganados) <= 1.4:
        m.append("Valor parecido a contratos que has ganado")
    if pd.notna(fila.knn_max_sim_ganado) and fila.knn_max_sim_ganado >= UMBRAL_SIM_GANADO:
        m.append("Parecido a procesos que has ganado")
    if bool(getattr(fila, "unspsc_inferido", False)):
        m = [x for x in m if x not in MOTIVOS_DE_CATEGORIA]
    return m[:MAX_MOTIVOS]


# ------------------------------------------------------------------------------------------------ 5 · publicación
def _fecha(v):
    return None if pd.isna(v) else pd.Timestamp(v).date().isoformat()


def _num(v):
    return None if pd.isna(v) else float(v)


def armar_publicacion(
    listas, procesos_frescos, proveedores, nombres, calientes, frias, fecha_corte, versiones, snapshots, metricas
):
    from jobs.publicar import Publicacion

    recs = [
        {
            "proveedor_key": r.proveedor_key,
            "lista": r.lista,
            "posicion": int(r.pos),
            "id_proceso": r.id_proceso,
            "probabilidad": float(r.p_gana),
            "score_ranking": float(r.s_v2),
            "score_retrieval": float(r.score_retrieval),
            "oferentes_esperados": _num(r.comp_entmod_media),
            "motivos": r.motivos,
            "unspsc_inferido": bool(r.unspsc_inferido),
        }
        for r in listas.itertuples(index=False)
    ]
    usados = listas.drop_duplicates("id_proceso").set_index("id_proceso").comp_entmod_media
    pf = procesos_frescos.drop_duplicates("id_proceso").set_index("id_proceso").reindex(usados.index)
    procs = [
        {
            "id_proceso": i,
            "entidad": r.nombre_entidad if pd.notna(r.nombre_entidad) else f"Entidad {r.nit_entidad}",
            "objeto": r.descripcion if pd.notna(r.descripcion) else (r.descripcion_limpia or ""),
            # Solo el UNSPSC real de silver (procesos_frescos nunca pasa por la inferencia): nulo si la fuente
            # publicó 'UNSPECIFIED'. No hay catálogo de nombres de familias, por eso categoria va nula.
            "categoria": None,
            "unspsc_familia": None if pd.isna(r.unspsc_familia) else str(r.unspsc_familia),
            "departamento": r.departamento_entidad,
            "municipio": r.ciudad_entidad,
            "modalidad": r.modalidad,
            "precio_base": _num(r.precio_base),
            "fecha_publicacion": _fecha(r.fecha_publicacion),
            "fecha_limite_ofertas": _fecha(r.fecha_limite_ofertas),
            "oferentes_esperados": _num(usados[i]),
            "url_proceso": r.url_proceso if pd.notna(r.url_proceso) else None,
        }
        for i, r in pf.iterrows()
    ]
    puntuadas = set(calientes) | set(frias)
    pv = proveedores[proveedores.proveedor_key.isin(puntuadas)].drop_duplicates("proveedor_key")
    pv = pv[pv.tipo_proveedor != "persona_natural"].merge(nombres, on="proveedor_key", how="inner")
    emp = [
        {
            "proveedor_key": r.proveedor_key,
            "nombre": r.nombre_proveedor,
            "departamento": r.departamento,
            "es_pyme": None if pd.isna(r.es_pyme) else bool(r.es_pyme),
            "tiene_historial": r.proveedor_key in set(calientes),
        }
        for r in pv.itertuples(index=False)
        if isinstance(r.nombre_proveedor, str) and r.nombre_proveedor.strip()
    ]
    return Publicacion(
        fecha_corte=fecha_corte,
        versiones=versiones,
        empresas=emp,
        procesos=procs,
        recomendaciones=recs,
        silver_snapshots=snapshots,
        metricas=metricas,
        datos_simulados=False,
    )


NAMESPACE = "Secop"

# Paso 8 · drift: features de hoy frente a una muestra fija de los candidatos de entrenamiento (jul 2024 – jun 2025)
COLS_DRIFT_NUM = ["score_retrieval", "knn_tasa50", "knn_tasa5", "dias_ult_part_ok", "comp_entmod_media"]
COLS_DRIFT_CAT = ["unspsc_familia"]
COLS_DRIFT = COLS_DRIFT_NUM + COLS_DRIFT_CAT
N_DRIFT, SEMILLA_DRIFT = 24_000, 42
# Criterio por tamaño de efecto, fijo (no depende de lo que Evidently elija según el tamaño de la muestra)
METODO_NUM, UMBRAL_NUM = "wasserstein", 0.1  # Wasserstein normalizado por la desviación de la referencia
METODO_CAT, UMBRAL_CAT = "psi", 0.2  # Population Stability Index
MIN_FREC_CAT, EPS_PSI = 0.01, 1e-4  # raras (< 1 % en las dos muestras) a "otras"; épsilon en frecuencias 0
REFERENCIA_DRIFT = "monitoreo/drift/referencia_v1.parquet"


def _como_categoria(x: pd.Series) -> pd.Series:
    return x.astype(str).fillna("sin_dato")


def categorias_raras(referencia: pd.Series, hoy: pd.Series, minimo: float = MIN_FREC_CAT) -> set[str]:
    """Categorías con frecuencia < ``minimo`` en las DOS muestras: se agrupan en "otras" antes del PSI."""
    fr = _como_categoria(referencia).value_counts(normalize=True)
    fh = _como_categoria(hoy).value_counts(normalize=True)
    todas = fr.index.union(fh.index)
    fr, fh = fr.reindex(todas, fill_value=0), fh.reindex(todas, fill_value=0)
    return set(todas[(fr < minimo) & (fh < minimo)])


def psi_suavizado(referencia: pd.Series, hoy: pd.Series, eps: float = EPS_PSI) -> float:
    """PSI con raras en "otras" y épsilon en las frecuencias 0 (renormalizado): sum((h - r) * ln(h / r))."""
    raras = categorias_raras(referencia, hoy)
    agrupar = lambda x: _como_categoria(x).where(~_como_categoria(x).isin(raras), "otras")  # noqa: E731
    fr, fh = agrupar(referencia).value_counts(normalize=True), agrupar(hoy).value_counts(normalize=True)
    todas = fr.index.union(fh.index)
    fr, fh = (f.reindex(todas, fill_value=0).clip(lower=eps) for f in (fr, fh))
    fr, fh = fr / fr.sum(), fh / fh.sum()
    return float(((fh - fr) * np.log(fh / fr)).sum())


def reporte_drift(hoy: pd.DataFrame, referencia: pd.DataFrame) -> tuple[object, dict]:
    """Reporte Evidently (DataDriftPreset). Devuelve (snapshot, resumen con el drift de cada feature)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    dd = DataDefinition(numerical_columns=COLS_DRIFT_NUM, categorical_columns=COLS_DRIFT_CAT)
    raras = {c: categorias_raras(referencia[c], hoy[c]) for c in COLS_DRIFT_CAT}
    prep = lambda d: d[COLS_DRIFT].assign(  # noqa: E731
        **{c: d[c].astype(str).where(~d[c].astype(str).isin(raras[c]), "otras") for c in COLS_DRIFT_CAT}
    )
    preset = DataDriftPreset(
        columns=COLS_DRIFT,
        num_method=METODO_NUM,
        num_threshold=UMBRAL_NUM,
        cat_method=METODO_CAT,
        cat_threshold=UMBRAL_CAT,
    )
    snap = Report([preset]).run(
        current_data=Dataset.from_pandas(prep(hoy), data_definition=dd),
        reference_data=Dataset.from_pandas(prep(referencia), data_definition=dd),
    )
    resumen = {
        "features": {},
        "n_hoy": len(hoy),
        "n_referencia": len(referencia),
        "criterio": {"numericas": f"{METODO_NUM} > {UMBRAL_NUM}", "categoricas": f"{METODO_CAT} > {UMBRAL_CAT}"},
    }
    for m in snap.dict()["metrics"]:
        nombre, valor = str(m.get("metric_id") or m.get("metric_name")), m.get("value")
        if nombre.startswith("DriftedColumnsCount"):
            resumen["drift_features"], resumen["drift_share"] = int(valor["count"]), float(valor["share"])
        elif nombre.startswith("ValueDrift(column="):
            col = nombre.split("column=")[1].split(",")[0]
            metodo = nombre.split("method=")[1].split(",")[0]
            umbral = float(nombre.split("threshold=")[1].rstrip(")"))
            resumen["features"][col] = {
                "distancia": float(valor),
                "metodo": metodo,
                "umbral": umbral,
                "drift": float(valor) > umbral,
            }
    # Categóricas: el PSI se recalcula aquí con suavizado explícito (el de Evidently reemplaza los ceros sin
    # renormalizar y daba 17,97 donde este da 7,03); el HTML de Evidently queda como apoyo visual.
    for c in COLS_DRIFT_CAT:
        v = psi_suavizado(referencia[c], hoy[c])
        resumen["features"][c] = {"distancia": v, "metodo": "psi_suavizado", "umbral": UMBRAL_CAT,
                                  "drift": v > UMBRAL_CAT}  # fmt: skip
    for c in COLS_DRIFT_NUM:
        resumen["features"][c]["media_referencia"] = float(referencia[c].mean())
        resumen["features"][c]["media_hoy"] = float(hoy[c].mean())
    resumen["drift_features"] = sum(f["drift"] for f in resumen["features"].values())
    resumen["drift_share"] = resumen["drift_features"] / len(COLS_DRIFT)
    return snap, resumen


def drift_del_dia(muestra: pd.DataFrame | None, fecha_corte: str, metricas: bool = True) -> dict:
    """Reporte de drift del día. Guarda la muestra de hoy (para poder regenerar el reporte sin recalcular las listas),
    el HTML y resumen.json en s3://…/monitoreo/drift/fecha=<fecha_corte>/. Con ``muestra=None`` la lee de ahí."""
    import tempfile

    import boto3

    s3 = boto3.client("s3")
    base = f"monitoreo/drift/fecha={fecha_corte}/"
    with tempfile.TemporaryDirectory() as tmp:
        if muestra is None:
            s3.download_file(BUCKET, base + "muestra_hoy.parquet", f"{tmp}/hoy.parquet")
            muestra = pd.read_parquet(f"{tmp}/hoy.parquet")
        else:
            muestra.to_parquet(f"{tmp}/hoy.parquet", index=False)
            s3.upload_file(f"{tmp}/hoy.parquet", BUCKET, base + "muestra_hoy.parquet")
        s3.download_file(BUCKET, REFERENCIA_DRIFT, f"{tmp}/ref.parquet")
        snap, resumen = reporte_drift(muestra, pd.read_parquet(f"{tmp}/ref.parquet"))
        snap.save_html(f"{tmp}/drift.html")
        s3.upload_file(f"{tmp}/drift.html", BUCKET, base + "drift.html", ExtraArgs={"ContentType": "text/html"})
        s3.put_object(Bucket=BUCKET, Key=base + "resumen.json", Body=json.dumps(resumen, indent=1).encode())
    if metricas:
        publicar_metricas({"drift_features": resumen["drift_features"], "drift_share": resumen["drift_share"]})
    resumen["s3"] = f"s3://{BUCKET}/{base}drift.html"
    return resumen


UNIDADES = {"duracion_min": "None", "unspsc_modelo_cargado": "None"}


def publicar_metricas(valores: dict[str, float], cliente=None) -> None:
    """Métricas de volumen en CloudWatch (namespace Secop). Solo se envían cuando hubo publicación: una noche
    sin publicar deja la métrica vacía y la alarma secop-sin-publicacion lo detecta."""
    if cliente is None:
        import boto3

        cliente = boto3.client("cloudwatch")
    datos = [
        {"MetricName": k, "Value": float(v), "Unit": UNIDADES.get(k, "Percent" if k.endswith("_pct") else "Count")}
        for k, v in valores.items()
    ]
    cliente.put_metric_data(Namespace=NAMESPACE, MetricData=datos)


def copia_gold(listas: pd.DataFrame, fecha_corte: str, version_id: int) -> None:
    import awswrangler as wr

    d = listas.rename(
        columns={
            "pos": "posicion",
            "p_gana": "probabilidad",
            "s_v2": "score_ranking",
            "comp_entmod_media": "oferentes_esperados",
        }
    )[
        [
            "proveedor_key",
            "lista",
            "posicion",
            "id_proceso",
            "probabilidad",
            "score_ranking",
            "score_retrieval",
            "oferentes_esperados",
            "motivos",
            "unspsc_inferido",
        ]
    ].copy()
    d["motivos"] = d.motivos.map(lambda m: json.dumps(m, ensure_ascii=False))
    d["version_id"] = version_id
    d["fecha_corte"] = fecha_corte
    wr.s3.to_parquet(
        d,
        path=f"s3://{BUCKET}/gold/recomendaciones/",
        dataset=True,
        mode="overwrite_partitions",
        partition_cols=["fecha_corte"],
        database="secop_gold",
        table="recomendaciones",
    )


# ------------------------------------------------------------------------------------------------------ modelos
def conectar_mlflow() -> None:
    """Registro de MLflow leído directo de PostgreSQL (sin el servidor de la EC2)."""
    from urllib.parse import quote

    import boto3

    if os.getenv("MLFLOW_TRACKING_URI"):
        return
    s = json.loads(boto3.client("secretsmanager").get_secret_value(SecretId="secop/mlflow-db")["SecretString"])
    os.environ["MLFLOW_TRACKING_URI"] = (
        f"postgresql+psycopg2://{s['username']}:{quote(s['password'], safe='')}@{s['host']}:{s['port']}/"
        f"{s['dbname']}?sslmode=require"
    )


def cargar_modelos() -> tuple[dict, dict]:
    import mlflow
    from mlflow import MlflowClient

    conectar_mlflow()
    c = MlflowClient()
    boosters, versiones = {}, {}
    for corto, nombre in MODELOS.items():
        v = c.get_model_version_by_alias(nombre, "production")
        boosters[corto] = mlflow.lightgbm.load_model(f"models:/{nombre}@production")
        versiones[corto] = f"{nombre} v{v.version}"
    return boosters, versiones


def cargar_unspsc() -> tuple[object, str, float]:
    """(modelo, versión, umbral) de models:/secop-unspsc@production. El umbral es un parámetro del run."""
    import mlflow
    from mlflow import MlflowClient

    conectar_mlflow()
    c = MlflowClient()
    v = c.get_model_version_by_alias(MODELO_UNSPSC, "production")
    umbral = float(c.get_run(v.run_id).data.params["umbral"])
    return mlflow.sklearn.load_model(f"models:/{MODELO_UNSPSC}@production"), f"{MODELO_UNSPSC} v{v.version}", umbral


def inferir_abiertos(procesos: pd.DataFrame, ids_abiertos, modelo, umbral: float) -> tuple[pd.DataFrame, dict]:
    """Infiere la familia UNSPSC SOLO en los procesos abiertos que no la traen, dentro de la tabla que reciben las
    historias. Devuelve (procesos, conteos). Con ``modelo=None`` no infiere (y lo cuenta todo como real o nulo)."""
    from ml.unspsc import inferir

    es_abierto = procesos.id_proceso.isin(set(ids_abiertos)).to_numpy()
    ab = procesos[es_abierto]
    real = int(ab.unspsc_familia.notna().sum())
    procesos = procesos.assign(unspsc_inferido=False)
    if modelo is not None:
        inf = inferir(ab, modelo, umbral)
        for c in ("unspsc_familia", "unspsc_segmento"):
            procesos[c] = procesos[c].astype("string")
            procesos.loc[inf.index, c] = inf[c]
        procesos.loc[inf.index, "unspsc_inferido"] = inf.unspsc_inferido
    inferido = int(procesos.loc[es_abierto, "unspsc_inferido"].sum())
    n = max(len(ab), 1)
    conteos = {
        "abiertos": len(ab),
        "real": real,
        "inferido": inferido,
        "nulo": len(ab) - real - inferido,
        "unspsc_real_pct": 100 * real / n,
        "unspsc_inferido_pct": 100 * inferido / n,
        "unspsc_nulo_pct": 100 * (len(ab) - real - inferido) / n,
    }
    return procesos, conteos


# ------------------------------------------------------------------------------------------------------ tuberías
def leer_historia(d: Path):
    p = lambda n, **k: pd.read_parquet(d / f"{n}.parquet", **k)  # noqa: E731
    return {
        "procesos": p("procesos"),
        "proponentes": p("proponentes"),
        "contratos": p("contratos"),
        "proveedores": p("proveedores"),
        "nombres": p("proveedores_nombres"),
    }


def calcular_listas(
    tablas, procesos, grupos, abiertos, fecha_corte, modelos, lote_empresas, max_pares, muestra_drift=None
):
    """recuperar (por lotes) → liberar HistoriaRetrieval → construir/rankear/armar_listas (por lotes)."""
    from ml.features import COLS_CONTRATOS, COLS_PROVEEDORES, Historia
    from ml.features import COLS_PROCESOS as CPF
    from ml.retrieval import COLS_CONTRATOS as CCR
    from ml.retrieval import COLS_PROCESOS as CPR
    from ml.retrieval import HistoriaRetrieval

    with etapa("historia del retrieval + recuperar() por lotes"):
        hr = HistoriaRetrieval(
            procesos=procesos[CPR],
            proponentes=tablas["proponentes"],
            contratos=tablas["contratos"][CCR],
            proveedores=tablas["proveedores"],
            nombres=tablas["nombres"],
        )
        cand = recuperar_por_lotes(hr, grupos, abiertos, fecha_corte, modelos["retrieval"], lote_empresas)
        del hr
    log(f"   candidatos: {len(cand):,} pares de {cand.proveedor_key.nunique():,} empresas")
    with etapa("historia de features + construir/rankear/armar_listas por lotes"):
        base = pd.read_parquet(tablas["dir"] / "base_entrenamiento.parquet")
        prov = tablas["proveedores"]
        h = Historia(
            base=base,
            procesos=procesos[CPF],
            proponentes=tablas["proponentes"],
            contratos=tablas["contratos"][COLS_CONTRATOS],
            proveedores=prov[[c for c in COLS_PROVEEDORES if c in prov.columns]],
        )
        del base
        listas = puntuar_por_lotes(cand, h, modelos["pgana"], modelos["ranking"], max_pares, muestra_drift)
        del h
    return listas


def correr(a) -> int:
    fecha_corte = a.fecha_corte or hoy_bogota()
    log(f"score.py · fecha_corte {fecha_corte}")

    with etapa("frescura de silver"):
        fresco, motivo, snap_abiertos = evaluar_frescura(leer_control(), fecha_corte)
    if not fresco:
        log(f"SIN PUBLICAR (código 2): silver no está fresco: {motivo}")
        return 2
    log(f"   {motivo}")

    with etapa("historia congelada desde S3 (checksums)"):
        tablas = leer_historia(bajar_historia(Path(a.dir_trabajo) / "historia"))
        tablas["dir"] = Path(a.dir_trabajo) / "historia"

    with etapa("procesos de silver (Athena) → limpiar()"):
        import awswrangler as wr

        frescos = limpiar_procesos_silver(
            wr.athena.read_sql_query(SQL_PROCESOS, database="secop_silver", workgroup=WORKGROUP, ctas_approach=False)
        )
        procesos, n_nuevos = unir_procesos(tablas["procesos"], frescos)
        abiertos = procesos_abiertos(frescos, fecha_corte)
    log(
        f"   silver: {len(frescos):,} procesos | nuevos frente a la historia: {n_nuevos:,} | "
        f"ABIERTOS a la fecha_corte: {len(abiertos):,}"
    )

    with etapa("familia UNSPSC inferida en los abiertos sin categoría (secop-unspsc)"):
        try:
            mod_unspsc, version_unspsc, umbral_unspsc = cargar_unspsc()
        except Exception as e:  # nunca tumba la noche: sigue sin inferir y la alarma avisa
            mod_unspsc, version_unspsc, umbral_unspsc = None, None, None
            log(f"   AVISO: no cargó {MODELO_UNSPSC} ({type(e).__name__}: {e}); la noche sigue SIN inferir")
        procesos, unspsc = inferir_abiertos(procesos, abiertos.id_proceso, mod_unspsc, umbral_unspsc)
        inferidos = set(procesos.loc[procesos.unspsc_inferido, "id_proceso"])
    log(
        f"   UNSPSC de los {unspsc['abiertos']:,} abiertos ({version_unspsc or 'sin modelo'}, umbral {umbral_unspsc}): "
        f"real {unspsc['real']:,} ({unspsc['unspsc_real_pct']:.1f} %) · inferido {unspsc['inferido']:,} "
        f"({unspsc['unspsc_inferido_pct']:.1f} %) · nulo {unspsc['nulo']:,} ({unspsc['unspsc_nulo_pct']:.1f} %)"
    )

    calientes, frias = empresas_a_puntuar(
        tablas["proveedores"], tablas["proponentes"], tablas["contratos"], fecha_corte
    )
    n_emp_total = len(calientes) + len(frias)
    log(f"   EMPRESAS (solo jurídicas): {len(calientes):,} calientes + {len(frias):,} frías = {n_emp_total:,}")
    if a.muestra:
        calientes, frias = calientes[: a.muestra], frias[: max(1, a.muestra // 4)]
        log(f"   (muestra: {len(calientes)} calientes + {len(frias)} frías)")

    with etapa("modelos de MLflow (models:/…@production, registro en PostgreSQL)"):
        modelos, versiones = cargar_modelos()
    log(f"   {versiones}")

    muestra_drift: list[pd.DataFrame] = []
    listas = calcular_listas(
        tablas,
        procesos,
        [calientes, frias],
        list(abiertos.id_proceso),
        fecha_corte,
        modelos,
        a.lote_empresas,
        a.max_pares,
        muestra_drift,
    )
    listas["unspsc_inferido"] = listas.id_proceso.isin(inferidos)
    listas["motivos"] = [motivos(r) for r in listas.itertuples(index=False)]
    huella = sha256_listas(listas)
    l2 = listas[listas.lista == "oportunidad"].groupby("proveedor_key").size()
    n_emp = listas.proveedor_key.nunique()
    log(f"   listas: {n_emp:,} empresas | sha256 {huella}")

    if a.sin_publicar:
        log("SIN PUBLICAR (--sin-publicar)")
    else:
        from jobs.publicar import dsn_desde_secreto, publicar

        with etapa("publicar en RDS (atómico)"):
            snapshots = {
                **snap_abiertos,
                "historia_congelada": HISTORIA_CONGELADA,
                "historia_s3": f"s3://{BUCKET}/{HISTORIA_S3}",
                **{f"historia_snapshot_{k}": v for k, v in SNAPSHOT_HISTORIA.items()},
                "sha256_listas": huella,
                "unspsc_modelo": version_unspsc,
                "unspsc_umbral": umbral_unspsc,
                **{f"unspsc_{k}" if not k.startswith("unspsc_") else k: v for k, v in unspsc.items()},
            }
            metricas = json.loads((Path(__file__).parent / "metricas_v1.json").read_text(encoding="utf-8"))
            pub = armar_publicacion(
                listas,
                frescos,
                tablas["proveedores"],
                tablas["nombres"],
                calientes,
                frias,
                fecha_corte,
                versiones,
                snapshots,
                metricas,
            )
            version_id = publicar(dsn_desde_secreto("secop/db-job"), pub)
        log(
            f"   publicada la versión {version_id}: {len(pub.empresas):,} empresas, {len(pub.procesos):,} procesos, "
            f"{len(pub.recomendaciones):,} recomendaciones"
        )
        with etapa("copia en gold.recomendaciones"):
            copia_gold(listas, fecha_corte, version_id)

    lista2_media = float(l2.reindex(listas.proveedor_key.unique(), fill_value=0).mean())
    if not a.sin_publicar:
        with etapa("métricas de volumen en CloudWatch"):
            publicar_metricas(
                {
                    "empresas_puntuadas": n_emp,
                    "procesos_abiertos": len(abiertos),
                    "largo_medio_lista2": lista2_media,
                    "duracion_min": (time.time() - T0) / 60,
                    "publicacion": 1,
                    "unspsc_modelo_cargado": int(mod_unspsc is not None),
                    "unspsc_real_pct": unspsc["unspsc_real_pct"],
                    "unspsc_inferido_pct": unspsc["unspsc_inferido_pct"],
                    "unspsc_nulo_pct": unspsc["unspsc_nulo_pct"],
                }
            )
    try:  # también con --sin-publicar (sin métricas en CloudWatch): el reporte es solo de monitoreo
        with etapa("drift (Evidently) frente al entrenamiento"):
            d = drift_del_dia(pd.concat(muestra_drift, ignore_index=True), fecha_corte, metricas=not a.sin_publicar)
        log(f"   drift: {d['drift_features']} de {len(COLS_DRIFT)} features ({d['drift_share']:.0%}) → {d['s3']}")
    except Exception as e:  # monitoreo: no tumba una publicación que ya está hecha
        log(f"   ERROR en el reporte de drift (las listas no dependen de él): {type(e).__name__}: {e}")
    log(
        f"RESUMEN fecha_corte={fecha_corte} empresas={n_emp:,} procesos_abiertos={len(abiertos):,} "
        f"unspsc_real={unspsc['unspsc_real_pct']:.1f}% inferido={unspsc['unspsc_inferido_pct']:.1f}% "
        f"nulo={unspsc['unspsc_nulo_pct']:.1f}% "
        f"lista2_media={lista2_media:.2f} "
        f"duracion={(time.time() - T0) / 60:.1f} min pico={pico_gb():.1f} GB sha256={huella[:16]}"
    )
    return 0


def verificar_lotes(a) -> int:
    """Un mes del notebook (marzo) de una vez con TODAS las empresas (más una muestra de frías jurídicas con la
    regla de producción) vs por lotes, solo jurídicas y con los grupos separados (lotes de solo-frías, como en
    producción): las listas de las jurídicas deben ser idénticas (sha256). Si no, el job por lotes no se usa."""
    from ml.retrieval import COLS_PROCESOS as CPR
    from ml.retrieval import HistoriaRetrieval

    mes = a.verificar_lotes
    with etapa("historia congelada desde S3 (checksums)"):
        tablas = leer_historia(bajar_historia(Path(a.dir_trabajo) / "historia"))
        tablas["dir"] = Path(a.dir_trabajo) / "historia"
    with etapa("modelos de MLflow"):
        modelos, versiones = cargar_modelos()
    procesos = tablas["procesos"]
    with etapa("entradas del mes (como el notebook 03)"):
        hr = HistoriaRetrieval(
            procesos=procesos[CPR],
            proponentes=tablas["proponentes"],
            contratos=tablas["contratos"][
                ["proveedor_key", "fecha_firma", "unspsc_familia", "unspsc_segmento", "nit_entidad", "departamento"]
            ],
            proveedores=tablas["proveedores"],
            nombres=tablas["nombres"],
        )
        empresas, abiertos = hr.entradas_del_mes(mes)
        del hr
    tipo = tablas["proveedores"].drop_duplicates("proveedor_key").set_index("proveedor_key").tipo_proveedor
    juridicas = sorted(k for k in empresas if tipo.get(k) not in (None, "persona_natural") and pd.notna(tipo.get(k)))
    _, frias = empresas_a_puntuar(tablas["proveedores"], tablas["proponentes"], tablas["contratos"], mes)
    frias = [k for k in frias if k not in set(empresas)][: a.muestra_frias]
    log(
        f"   {len(empresas):,} empresas del mes ({len(juridicas):,} jurídicas) + {len(frias):,} frías jurídicas "
        f"| {len(abiertos):,} procesos"
    )
    with etapa("A · de una vez, todas las empresas"):
        A = calcular_listas(
            tablas, procesos, [sorted(set(empresas) | set(frias))], abiertos, mes, modelos, 10**9, 10**12
        )
    with etapa("B · por lotes, solo jurídicas, frías en lotes propios"):
        B = calcular_listas(tablas, procesos, [juridicas, frias], abiertos, mes, modelos, a.lote_empresas, a.max_pares)
    ha, hb = sha256_listas(A[A.proveedor_key.isin(set(juridicas) | set(frias))]), sha256_listas(B)
    log(f"   sha256 A (jurídicas): {ha}\n   sha256 B (por lotes): {hb}")
    if ha != hb:
        log("FALLA: la versión por lotes NO da lo mismo")
        return 1
    log(f"OK: idénticas ({B.proveedor_key.nunique():,} empresas, {len(B):,} filas) | pico {pico_gb():.1f} GB")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fecha-corte", help="AAAA-MM-DD (por defecto, hoy en Bogotá)")
    ap.add_argument("--verificar-lotes", metavar="AAAA-MM-DD", help="prueba de lotes con un mes del notebook")
    ap.add_argument("--regenerar-drift", metavar="AAAA-MM-DD", help="rehace el reporte de drift desde la muestra en S3")
    ap.add_argument("--dir-trabajo", default=os.getenv("SECOP_DIR_TRABAJO", "/tmp/secop"))
    ap.add_argument("--lote-empresas", type=int, default=10_000)
    ap.add_argument("--max-pares", type=int, default=1_000_000)
    ap.add_argument("--muestra", type=int, default=0, help="solo N empresas calientes (pruebas)")
    ap.add_argument("--muestra-frias", type=int, default=3_000, help="frías que agrega --verificar-lotes")
    ap.add_argument("--sin-publicar", action="store_true", help="calcula todo pero no publica")
    a = ap.parse_args(argv)
    try:
        if a.regenerar_drift:
            d = drift_del_dia(None, a.regenerar_drift, metricas=False)
            log(f"drift {a.regenerar_drift}: {d['drift_features']} de {len(COLS_DRIFT)} features → {d['s3']}")
            return 0
        return verificar_lotes(a) if a.verificar_lotes else correr(a)
    except Exception as e:
        log(f"ERROR (código 1, no se publica): {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
