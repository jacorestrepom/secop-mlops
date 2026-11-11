"""Registra en MLflow los tres modelos de la v1 (Paso 2 de la guía de traspaso) y les pone el alias production.

1. Descarga de s3://<bucket>/modelos/v1/ los artefactos y verifica sus checksums contra los manifiestos.
2. Registra secop-retrieval (M1), secop-pgana (G1) y secop-ranking (ranking v2) con parámetros, métricas de
   prueba, features en orden, config.json, commit de git y snapshot de silver.
3. Pone el alias production en los tres.
4. Prueba de equivalencia: models:/<nombre>@production predice igual que el Booster original (dif. < 1e-9).

Uso (con el túnel a MLflow abierto, ver README):
    AWS_PROFILE=secop MLFLOW_TRACKING_URI=http://localhost:5000 python -m jobs.registrar_v1
    MLFLOW_TRACKING_URI=postgresql://... python -m jobs.registrar_v1 --solo-verificar secop-ranking
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

BUCKET = "proyecto-integrador-bucket-01"
PREFIJO = "modelos/v1"
EXPERIMENTO = "secop-registro-v1"
ALIAS = "production"
CORTE_ENTRENAMIENTO = "2025-06"
PERIODO_PRUEBA = "2025-07 a 2026-07"
# Snapshot de silver (Iceberg) con el que se entrenaron y evaluaron los modelos (confirmado por ML, 2026-11-04)
SILVER_SNAPSHOT = {
    "silver_snapshot_procesos": "5704566866417056960",
    "silver_snapshot_proponentes": "2911693562262410165",
    "silver_snapshot_contratos": "528333321308876835",
    "silver_snapshot_proveedores": "7035817432963350399",
    "silver_snapshot_fecha": "2026-11-04",
    "silver_copia": "s3://proyecto-integrador-bucket-01/handoff/raw/",
}
TOLERANCIA = 1e-9

# Recall@150 de prueba del M1 (notebook 03, sección 9; data/retrieval_v2/test/test_AAAAMM.parquet,
# promedio de 13 meses). No están en modelos/v1/, por eso van aquí con su fuente.
METRICAS_M1 = {"recall_at_150": 0.7887, "recall_at_150_normal": 0.8259, "recall_at_150_frio": 0.4332}

MODELOS = {
    "secop-retrieval": {
        "archivo": "retrieval/M1_final_unico.txt",
        "descripcion": "M1 · retrieval: deja los 150 procesos abiertos que mejor encajan con cada empresa",
        "config": None,
    },
    "secop-pgana": {
        "archivo": "modelos/g1_pgana.txt",
        "descripcion": "G1 · P(gana) si la empresa se presenta (LightGBM binario)",
        "config": "g1",
    },
    "secop-ranking": {
        "archivo": "modelos/ranking_v2_final.txt",
        "descripcion": "Ranking v2 · ordena los 150 candidatos de cada empresa (LightGBM LambdaRank)",
        "config": "ranking_v2",
    },
}
ARCHIVOS = [
    "manifest.sha256",
    "config.json",
    "modelos/g1_pgana.txt",
    "modelos/ranking_v2_final.txt",
    "metricas/TABLA1_ranking_v2.parquet",
    "metricas/TABLA2_g1_filtros.parquet",
    "metricas/tablas_finales.xlsx",
    "retrieval/manifest_retrieval.sha256",
    "retrieval/M1_final_unico.txt",
]


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar_y_verificar(destino: Path) -> None:
    """Baja los artefactos de S3 y para si algún checksum no coincide con su manifiesto."""
    s3 = boto3.client("s3")
    for rel in ARCHIVOS:
        (destino / rel).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, f"{PREFIJO}/{rel}", str(destino / rel))
    for manifiesto, base in [
        ("manifest.sha256", destino),
        ("retrieval/manifest_retrieval.sha256", destino / "retrieval"),
    ]:
        for linea in (destino / manifiesto).read_text().splitlines():
            esperado, rel = linea.split(maxsplit=1)
            if (base / rel).exists() and sha256(base / rel) != esperado:
                sys.exit(f"Checksum distinto en {rel}: no se registra nada")
    usados = [r for r in ARCHIVOS if not r.endswith(".sha256")]
    print(f"Descargados {len(ARCHIVOS)} archivos; checksums OK de los {len(usados)} que se usan")


def commit_git() -> str:
    try:
        sucio = subprocess.run(["git", "status", "--porcelain", "--", "jobs"], capture_output=True, text=True).stdout
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-modificado" if sucio.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def metricas_de_prueba(d: Path) -> dict[str, dict[str, float]]:
    """Métricas de la prueba (jul 2025–jul 2026) para cada modelo, con nombres válidos en MLflow."""

    def clave(texto: str) -> str:
        t = texto.lower().replace("@", "_at_").replace("%", "pct").replace("(", "").replace(")", "")
        return "_".join("".join(c if c.isalnum() else " " for c in t).split())

    t1 = pd.read_parquet(d / "metricas/TABLA1_ranking_v2.parquet")
    rank = {}
    for _, f in t1.iterrows():
        vista = "empresa" if f["Vista"] == "Por empresa" else "proceso"
        rank[f"lista1_{vista}_{clave(f['Métrica'])}"] = float(f["Ranking v2"])
        rank[f"referencia_m1_{vista}_{clave(f['Métrica'])}"] = float(f["Referencia: M1 solo"])

    t2 = pd.read_parquet(d / "metricas/TABLA2_g1_filtros.parquet")
    pgana = {}
    for _, f in t2.iterrows():
        nombre = clave(f["Métrica"].split("(≥")[0])
        pgana[f"lista2_{nombre}"] = float(f["G1 + filtros"])
        if pd.notna(f["Referencia"]):
            pgana[f"referencia_{nombre}"] = float(f["Referencia"])
    return {"secop-retrieval": METRICAS_M1, "secop-pgana": pgana, "secop-ranking": rank}


def parametros(nombre: str, booster: lgb.Booster, config: dict) -> dict:
    p = {
        "corte_entrenamiento": CORTE_ENTRENAMIENTO,
        "periodo_prueba": PERIODO_PRUEBA,
        "n_features": booster.num_feature(),
        "n_arboles": booster.num_trees(),
    }
    seccion = MODELOS[nombre]["config"]
    if seccion:
        p.update({f"lgb_{k}": v for k, v in config[seccion].items()})
        p.update(
            {
                "K": config["K"],
                "n_lista": config["n_lista"],
                "filtros_lista2": config["filtros_lista2"],
                "meses_entrena_ranking": ",".join(config["meses_entrena_ranking"]),
            }
        )
    else:  # el M1 no está en config.json: se toman los parámetros guardados en su propio archivo
        p.update(
            {
                f"lgb_{k}": v
                for k, v in booster.params.items()
                if k
                in {
                    "objective",
                    "num_iterations",
                    "learning_rate",
                    "num_leaves",
                    "min_data_in_leaf",
                    "feature_fraction",
                    "bagging_fraction",
                    "bagging_freq",
                    "lambda_l2",
                    "seed",
                }
            }
        )
        p.update(
            {"K": config["K"], "entrenamiento": "candidatos jul 2024 – jun 2025 (positivos + 100 negativos al azar)"}
        )
    return {k: str(v)[:6000] for k, v in p.items()}


def muestra_sintetica(booster: lgb.Booster, n: int = 1000, semilla: int = 42) -> np.ndarray:
    """Filas inventadas con las columnas en el orden del modelo (categóricas como códigos enteros, con nulos)."""
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n, booster.num_feature()))
    infos = booster.dump_model(num_iteration=1)["feature_infos"]
    for i, f in enumerate(booster.feature_name()):
        valores = infos.get(f, {}).get("values")  # solo las categóricas traen la lista de códigos vistos
        if valores:
            X[:, i] = rng.choice(valores, size=n)
    X[rng.random(X.shape) < 0.05] = np.nan
    return X


def verificar(nombre: str, booster_original: lgb.Booster | None = None) -> float:
    """Carga models:/<nombre>@production y compara con el Booster original sobre 1.000 filas sintéticas."""
    cargado = mlflow.lightgbm.load_model(f"models:/{nombre}@{ALIAS}")
    if booster_original is None:
        return float("nan")
    X = muestra_sintetica(booster_original)
    dif = float(np.max(np.abs(cargado.predict(X) - booster_original.predict(X))))
    print(f"   {nombre}@{ALIAS}: diferencia máxima {dif:.3e} sobre {len(X)} filas")
    return dif


def registrar(d: Path) -> None:
    config = json.loads((d / "config.json").read_text())
    metricas = metricas_de_prueba(d)
    commit = commit_git()
    cliente = MlflowClient()
    for nombre in MODELOS:
        try:
            cliente.get_model_version_by_alias(nombre, ALIAS)
            sys.exit(f"{nombre} ya tiene alias {ALIAS}: no se registra nada (v1 no se reescribe)")
        except mlflow.exceptions.MlflowException:
            pass

    mlflow.set_experiment(EXPERIMENTO)
    originales = {}
    for nombre, m in MODELOS.items():
        ruta = d / m["archivo"]
        booster = lgb.Booster(model_file=str(ruta))
        originales[nombre] = booster
        with mlflow.start_run(run_name=f"{nombre}-v1") as run, tempfile.TemporaryDirectory() as tmp:
            mlflow.set_tags(
                {
                    "git_commit": commit,
                    **SILVER_SNAPSHOT,
                    "proyecto": "secop",
                    "fuente": f"s3://{BUCKET}/{PREFIJO}/{m['archivo']}",
                    "sha256_modelo": sha256(ruta),
                    "mlflow.note.content": m["descripcion"],
                }
            )
            mlflow.log_params(parametros(nombre, booster, config))
            mlflow.log_metrics(metricas[nombre])
            features = Path(tmp) / "features_en_orden.txt"
            features.write_text("\n".join(booster.feature_name()) + "\n")
            mlflow.log_artifact(str(features))
            mlflow.log_artifact(str(d / "config.json"))
            if nombre != "secop-retrieval":
                mlflow.log_artifacts(str(d / "metricas"), artifact_path="evaluacion")
            info = mlflow.lightgbm.log_model(booster, name="modelo", registered_model_name=nombre)
            version = info.registered_model_version
            cliente.update_registered_model(nombre, description=m["descripcion"])
            cliente.set_model_version_tag(nombre, version, "git_commit", commit)
            for clave, valor in SILVER_SNAPSHOT.items():
                cliente.set_model_version_tag(nombre, version, clave, valor)
            cliente.set_registered_model_alias(nombre, ALIAS, version)
            print(
                f"{nombre}: versión {version} (run {run.info.run_id}) → alias {ALIAS} | "
                f"{booster.num_feature()} features, {len(metricas[nombre])} métricas"
            )

    print("Prueba de equivalencia:")
    peor = max(verificar(n, b) for n, b in originales.items())
    if not peor < TOLERANCIA:
        sys.exit(f"FALLA: diferencia {peor:.3e} ≥ {TOLERANCIA}")
    print(f"OK: diferencia máxima {peor:.3e} < {TOLERANCIA}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--solo-verificar", metavar="MODELO", help="solo carga MODELO@production (sin registrar)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        if args.solo_verificar:
            descargar_y_verificar(d)
            original = lgb.Booster(model_file=str(d / MODELOS[args.solo_verificar]["archivo"]))
            dif = verificar(args.solo_verificar, original)
            sys.exit(0 if dif < TOLERANCIA else f"FALLA: diferencia {dif:.3e}")
        descargar_y_verificar(d)
        registrar(d)


if __name__ == "__main__":
    main()
