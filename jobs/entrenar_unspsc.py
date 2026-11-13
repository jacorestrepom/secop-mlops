"""Entrena el clasificador de familia UNSPSC (ml/unspsc.py) y lo registra en MLflow como secop-unspsc@production.

    AWS_PROFILE=secop MLFLOW_TRACKING_URI=http://localhost:5001 python -m jobs.entrenar_unspsc
    python -m jobs.entrenar_unspsc --sin-registrar --procesos /ruta/procesos.parquet     # prueba local

Datos: la tabla de procesos de la historia congelada (s3://…/modelos/v1/historia/procesos.parquet, verificada
contra su manifiesto), la misma de la que salen las features del modelo.

- Modelo de producción: todos los procesos con UNSPSC real publicados antes del 2026-10-01 (desde octubre la
  fuente ya casi no lo publica), 150 mil términos.
- Evaluación en septiembre de 2026, point-in-time: un modelo igual entrenado solo con lo publicado antes del
  2026-09-01 se evalúa en los procesos de septiembre con UNSPSC real (top-1 de familia y de segmento, y cobertura
  con el umbral). Evaluar el modelo de producción en septiembre sería evaluar sobre su propio entrenamiento.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from jobs.score import BUCKET, HISTORIA_CONGELADA, HISTORIA_S3, SNAPSHOT_HISTORIA, log
from ml.unspsc import MAX_TERMINOS, UMBRAL, entrenar, evaluar

NOMBRE = "secop-unspsc"
EXPERIMENTO = "secop-unspsc"
ARTEFACTOS = f"s3://{BUCKET}/mlflow/{EXPERIMENTO}"
CORTE = "2026-10-01"  # entrenamiento del modelo de producción: publicados antes de esta fecha
MES_EVAL = ("2026-09-01", "2026-10-01")
COLS = ["id_proceso", "fecha_publicacion", "descripcion_limpia", "unspsc_segmento", "unspsc_familia"]


def bajar_procesos(destino: Path) -> Path:
    """procesos.parquet de la historia congelada, verificado contra manifest_historia.sha256."""
    import boto3

    s3 = boto3.client("s3")
    ruta = destino / "procesos.parquet"
    manifiesto = s3.get_object(Bucket=BUCKET, Key=HISTORIA_S3 + "manifest_historia.sha256")["Body"].read().decode()
    esperado = next(h for h, n in (ln.split(maxsplit=1) for ln in manifiesto.splitlines()) if n == "procesos.parquet")
    s3.download_file(BUCKET, HISTORIA_S3 + "procesos.parquet", str(ruta))
    if hashlib.sha256(ruta.read_bytes()).hexdigest() != esperado:
        raise RuntimeError("checksum distinto en procesos.parquet: no se entrena")
    return ruta


def commit_git() -> str:
    try:
        sucio = subprocess.run(["git", "status", "--porcelain", "--", "ml", "jobs"], capture_output=True, text=True)
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        return c + ("-modificado" if sucio.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--procesos", help="procesos.parquet local (por defecto, el de la historia en S3)")
    ap.add_argument("--max-terminos", type=int, default=MAX_TERMINOS)
    ap.add_argument("--umbral", type=float, default=UMBRAL)
    ap.add_argument("--sin-registrar", action="store_true", help="entrena y evalúa, sin MLflow")
    a = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(a.procesos) if a.procesos else bajar_procesos(Path(tmp))
        p = pd.read_parquet(ruta, columns=COLS)
    f = pd.to_datetime(p.fecha_publicacion)
    con = p.unspsc_familia.notna()
    ev_ini, ev_fin = map(pd.Timestamp, MES_EVAL)

    t0 = time.time()
    tr_eval = p[con & (f < ev_ini)]
    mod_eval = entrenar(tr_eval, a.max_terminos)
    ev = p[(f >= ev_ini) & (f < ev_fin)]
    m = evaluar(ev, mod_eval, a.umbral)
    log(
        f"evaluación sep-2026 (modelo con {len(tr_eval):,} procesos < {MES_EVAL[0]}, {time.time() - t0:.0f} s): "
        f"{int(m['n_evaluados']):,} con UNSPSC real | cobertura {m['cobertura']:.1%} | top-1 familia "
        f"{m['top1_familia']:.1%} | top-1 segmento {m['top1_segmento']:.1%} | "
        f"sin umbral {m['top1_familia_sin_umbral']:.1%}"
    )
    del mod_eval

    t0 = time.time()
    tr = p[con & (f < pd.Timestamp(CORTE))]
    modelo = entrenar(tr, a.max_terminos)
    t_ent = time.time() - t0
    log(f"modelo de producción: {len(tr):,} procesos < {CORTE}, {len(modelo.classes_)} familias, {t_ent:.0f} s")
    if a.sin_registrar:
        return 0

    import mlflow
    from mlflow import MlflowClient

    # Con el registro leído directo de PostgreSQL, un experimento nuevo tomaría ./mlruns (local) como carpeta de
    # artefactos y Fargate no podría cargar el modelo: se crea con su carpeta en S3, como secop-registro-v1.
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        mlflow.create_experiment(EXPERIMENTO, artifact_location=ARTEFACTOS)
    exp = mlflow.set_experiment(EXPERIMENTO)
    if not exp.artifact_location.startswith("s3://"):
        raise RuntimeError(f"el experimento {EXPERIMENTO} guarda artefactos en {exp.artifact_location}, no en S3")
    commit = commit_git()
    with mlflow.start_run(run_name=f"{NOMBRE}-{CORTE}") as run:
        mlflow.log_params(
            {
                "umbral": a.umbral,
                "max_terminos": a.max_terminos,
                "corte_entrenamiento": CORTE,
                "n_entrenamiento": len(tr),
                "n_familias": len(modelo.classes_),
                "mes_evaluacion": "2026-09",
                "corte_modelo_evaluacion": MES_EVAL[0],
                "n_entrenamiento_evaluacion": len(tr_eval),
                "modelo": "TfidfVectorizer(1-2 gramas, min_df=3, max_df=0.5) + SGDClassifier(log_loss, alpha=1e-6)",
            }
        )
        mlflow.log_metrics({k: v for k, v in m.items()} | {"segundos_entrenamiento": t_ent})
        mlflow.set_tags(
            {
                "git_commit": commit,
                "historia_congelada": HISTORIA_CONGELADA,
                "historia_s3": f"s3://{BUCKET}/{HISTORIA_S3}procesos.parquet",
                **{f"silver_snapshot_{k}": v for k, v in SNAPSHOT_HISTORIA.items()},
                "experimento": "tests/experimentos/unspsc_imputado.py",
                "aprobado_por": "ciencia de datos (David), 2026-11-12",
            }
        )
        info = mlflow.sklearn.log_model(modelo, name="modelo", registered_model_name=NOMBRE)
    c = MlflowClient()
    v = c.search_model_versions(f"name='{NOMBRE}' and run_id='{run.info.run_id}'")[0].version
    c.update_model_version(NOMBRE, v, description=f"Familia UNSPSC desde la descripción; umbral {a.umbral}")
    c.set_registered_model_alias(NOMBRE, "production", v)
    c.set_model_version_tag(NOMBRE, v, "umbral", str(a.umbral))
    log(f"registrado {NOMBRE} v{v} @production (run {run.info.run_id}, {info.model_uri}, commit {commit})")

    cargado = mlflow.sklearn.load_model(f"models:/{NOMBRE}@production")
    muestra = p[~con].head(2000)
    iguales = (cargado.predict_proba(muestra.descripcion_limpia.fillna("").to_numpy(dtype=object))
               == modelo.predict_proba(muestra.descripcion_limpia.fillna("").to_numpy(dtype=object))).all()  # fmt: skip
    log(f"models:/{NOMBRE}@production predice igual que el modelo entrenado: {bool(iguales)}")
    return 0 if iguales else 1


if __name__ == "__main__":
    sys.exit(main())
