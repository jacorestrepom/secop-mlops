"""Genera la referencia de paridad de ml/ en Fargate y la sube a S3 (tests/referencia/README.md).

    python -m jobs.referencia [--n 300] [--mes 2026-03-01]

Arma la carpeta data/ que espera tests/referencia/generar_referencia.py con los artefactos verificados de
s3://…/modelos/v1/ (historia = clean/, M1 en retrieval_v2/, G1 y ranking v2 en final/modelos/), corre el
generador DOS veces y solo sube a s3://…/ci/referencia/ si los dos sha256.txt son idénticos (determinismo).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import boto3

from jobs.score import BUCKET, bajar_historia, log

DESTINO = "ci/referencia/"


def armar_data(base: Path) -> Path:
    s3 = boto3.client("s3")
    clean = bajar_historia(base / "data" / "clean")  # verifica el manifiesto de la historia
    (clean.parent / "retrieval_v2").mkdir(exist_ok=True)
    (clean.parent / "final" / "modelos").mkdir(parents=True, exist_ok=True)
    s3.download_file(
        BUCKET, "modelos/v1/retrieval/M1_final_unico.txt", str(clean.parent / "retrieval_v2" / "M1_final_unico.txt")
    )
    for m in ("g1_pgana.txt", "ranking_v2_final.txt"):
        s3.download_file(BUCKET, f"modelos/v1/modelos/{m}", str(clean.parent / "final" / "modelos" / m))
    return clean.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--mes", default="2026-03-01")
    ap.add_argument("--dir-trabajo", default="/tmp/secop")
    a = ap.parse_args(argv)
    base = Path(a.dir_trabajo)
    data = armar_data(base)
    huellas = []
    for i in (1, 2):
        salida = base / f"referencia_{i}"
        log(f"corrida {i} de generar_referencia")
        subprocess.run([sys.executable, "-m", "tests.referencia.generar_referencia", "--data", str(data),
                        "--salida", str(salida), "--mes", a.mes, "--n", str(a.n)], check=True)  # fmt: skip
        huellas.append(next(salida.rglob("sha256.txt")).read_text())
    if huellas[0] != huellas[1]:
        log("FALLA: las dos corridas no dieron el mismo sha256.txt (no se sube nada)")
        return 1
    log("OK: sha256.txt idéntico en las dos corridas")
    s3 = boto3.client("s3")
    total = 0
    for f in sorted((base / "referencia_1").rglob("*")):
        if f.is_file():
            clave = DESTINO + str(f.relative_to(base / "referencia_1"))
            s3.upload_file(str(f), BUCKET, clave)
            total += f.stat().st_size
            log(f"   s3://{BUCKET}/{clave} ({f.stat().st_size:,} B)")
    log(f"referencia subida: {total:,} bytes ({total / 1e6:.1f} MB)\n{huellas[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
