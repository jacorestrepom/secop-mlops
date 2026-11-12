"""Compuerta de promoción (guía de traspaso, Paso 9): staging → production solo si el candidato es mejor.

    python -m jobs.promover --modelo secop-pgana          # compara secop-pgana@staging con @production
    python -m jobs.promover --modelo secop-pgana --sin-alertas

Regla (en el mismo periodo de validación, para los dos modelos): promueve solo si el candidato MEJORA la métrica
principal y NO EMPEORA ninguna otra, tampoco en pymes. Un empate no se promueve: un modelo igual al actual no
justifica cambiarlo.
  secop-pgana     (G1)        principal: log loss · además: ECE · y las dos solo en pymes. Se recalculan
                              aquí, con los dos modelos sobre la misma muestra fija de la prueba
                              (s3://…/monitoreo/validacion/g1_v1.parquet: 60 mil participaciones jul 2025–jul 2026).
  secop-ranking   (ranking v2) principal: NDCG@10 de la lista 1 · pymes: lista1_pymes_ndcg_at_10
  secop-retrieval (M1)        principal: Recall@150 · pymes: recall_at_150_pymes
                              Para estos dos se comparan las métricas registradas en el run de cada versión (las
                              calcula el script de entrenamiento sobre el mismo periodo). Si falta alguna, la
                              compuerta no puede verificar y el candidato se queda en staging.
Si pasa, el alias production pasa al candidato. Si no, se queda en staging, se etiqueta con el motivo y se avisa a
SNS secop-alertas. Código de salida: 0 promovido · 3 se queda en staging.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile

import numpy as np
import pandas as pd

BUCKET = "proyecto-integrador-bucket-01"
VALIDACION_G1 = "monitoreo/validacion/g1_v1.parquet"
TEMA_ALERTAS = "secop-alertas"
METRICAS_REGISTRADAS = {
    "secop-ranking": ("lista1_empresa_ndcg_at_10", "lista1_pymes_ndcg_at_10"),
    "secop-retrieval": ("recall_at_150", "recall_at_150_pymes"),
}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Error de calibración esperado por deciles de probabilidad (como el notebook)."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"q": q, "y": y, "p": p}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((t.n * (t.y - t.p).abs()).sum() / t.n.sum())


def metricas_g1(booster, v: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import log_loss

    from ml.ranking import predecir_g1

    p = predecir_g1(v, booster).astype(float)
    y = v.gano.values.astype(float)
    pyme = (v.es_pyme == 1).values
    return {
        "log_loss": log_loss(y, p),
        "ece": ece(y, p),
        "log_loss_pymes": log_loss(y[pyme], p[pyme]),
        "ece_pymes": ece(y[pyme], p[pyme]),
    }


def decidir(cand: dict[str, float], prod: dict[str, float], principal: str, menor_es_mejor: bool) -> tuple[bool, str]:
    """Mejora estricta en la principal y ninguna otra peor (con tolerancia numérica)."""
    mejor = (lambda a, b: a < b - 1e-12) if menor_es_mejor else (lambda a, b: a > b + 1e-12)
    peor = (lambda a, b: a > b + 1e-12) if menor_es_mejor else (lambda a, b: a < b - 1e-12)
    faltan = [k for k in prod if k not in cand or cand[k] is None or prod[k] is None]
    if faltan:
        return False, f"no se puede verificar: faltan métricas {faltan}"
    if not mejor(cand[principal], prod[principal]):
        return False, f"{principal} no mejora ({cand[principal]:.6f} vs {prod[principal]:.6f} en production)"
    empeoran = [k for k in prod if k != principal and peor(cand[k], prod[k])]
    if empeoran:
        return False, "empeora " + ", ".join(f"{k} ({cand[k]:.6f} vs {prod[k]:.6f})" for k in empeoran)
    return True, f"{principal} mejora ({cand[principal]:.6f} vs {prod[principal]:.6f}) y nada empeora"


def avisar(asunto: str, mensaje: str) -> None:
    import boto3

    sns = boto3.client("sns")
    arn = next(t["TopicArn"] for t in sns.list_topics()["Topics"] if t["TopicArn"].endswith(":" + TEMA_ALERTAS))
    sns.publish(TopicArn=arn, Subject=asunto[:100], Message=mensaje)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modelo", required=True, choices=["secop-pgana", "secop-ranking", "secop-retrieval"])
    ap.add_argument("--sin-alertas", action="store_true")
    a = ap.parse_args(argv)

    import boto3
    import mlflow
    from mlflow import MlflowClient

    from jobs.score import conectar_mlflow

    conectar_mlflow()
    c = MlflowClient()
    vp, vs = c.get_model_version_by_alias(a.modelo, "production"), c.get_model_version_by_alias(a.modelo, "staging")
    print(f"{a.modelo}: production = v{vp.version} · staging (candidato) = v{vs.version}")
    if vp.version == vs.version:
        print("staging y production son la misma versión: nada que comparar")
        return 3

    if a.modelo == "secop-pgana":
        with tempfile.TemporaryDirectory() as tmp:
            boto3.client("s3").download_file(BUCKET, VALIDACION_G1, f"{tmp}/v.parquet")
            v = pd.read_parquet(f"{tmp}/v.parquet")
        prod = metricas_g1(mlflow.lightgbm.load_model(f"models:/{a.modelo}@production"), v)
        cand = metricas_g1(mlflow.lightgbm.load_model(f"models:/{a.modelo}@staging"), v)
        pasa, motivo = decidir(cand, prod, "log_loss", menor_es_mejor=True)
        periodo = f"muestra fija de la prueba ({len(v):,} participaciones, s3://{BUCKET}/{VALIDACION_G1})"
    else:
        principal, pymes = METRICAS_REGISTRADAS[a.modelo]
        m = lambda ver: {k: c.get_run(ver.run_id).data.metrics.get(k) for k in (principal, pymes)}  # noqa: E731
        prod, cand = m(vp), m(vs)
        pasa, motivo = decidir(cand, prod, principal, menor_es_mejor=False)
        periodo = "métricas registradas en el run de cada versión"
    print(f"periodo de validación: {periodo}")
    for k in prod:
        print(f"   {k:16s} production {prod[k]} | candidato {cand.get(k)}")

    resultado = "promovido" if pasa else "rechazado"
    c.set_model_version_tag(a.modelo, vs.version, "compuerta", resultado)
    c.set_model_version_tag(a.modelo, vs.version, "compuerta_motivo", motivo[:500])
    c.set_model_version_tag(
        a.modelo, vs.version, "compuerta_metricas", json.dumps({"candidato": cand, "production": prod})[:5000]
    )
    if pasa:
        c.set_registered_model_alias(a.modelo, "production", vs.version)
        print(f"PROMOVIDO: {a.modelo} v{vs.version} es production ({motivo})")
        return 0
    print(f"SE QUEDA EN STAGING: {a.modelo} v{vs.version} ({motivo}); production sigue en v{vp.version}")
    if not a.sin_alertas:
        metricas_txt = json.dumps({"candidato": cand, "production": prod}, indent=1)
        avisar(
            f"SECOP: {a.modelo} v{vs.version} no pasó la compuerta",
            f"El candidato {a.modelo} v{vs.version} (alias staging) no se promovió a production.\n"
            f"Motivo: {motivo}\nPeriodo de validación: {periodo}\n"
            f"production sigue en v{vp.version}.\nMétricas: {metricas_txt}",
        )
        print("aviso enviado a SNS secop-alertas")
    return 3


if __name__ == "__main__":
    sys.exit(main())
