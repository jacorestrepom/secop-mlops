"""Prueba de paridad de ml/ para el CI (job ``paridad`` de .github/workflows/ci.yml).

Corre DENTRO de la imagen secop-jobs (la misma en la que se generó la referencia en Fargate: linux/amd64, mismas
librerías y hilos fijos). Con credenciales de solo lectura:

1. baja la referencia de s3://…/ci/referencia/ y verifica su sha256.txt;
2. arma la carpeta data/ con s3://…/modelos/v1/ (historia verificada contra su manifiesto, M1, G1 y ranking v2,
   config.json) — la misma función que usó jobs/referencia.py para generarla;
3. corre tests/test_paridad.py con SECOP_REFERENCIA_DIR y SECOP_DATA_DIR, e informa tiempo y memoria pico.

Las pruebas de test_paridad.py contra el notebook (puntajes y listas sobre cand_/pred_ de marzo) necesitan
archivos que no están en modelos/v1/: se excluyen aquí y siguen corriendo en local con data/final completo.

    python -m jobs.paridad_ci [--dir-trabajo /tmp/secop]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PRUEBAS = "tests/test_paridad.py"
FILTRO = "contra_referencia or config_coincide"  # las del notebook necesitan data/final/cand y pred (no en S3)


def bajar_referencia(destino: Path) -> Path:
    import boto3

    from jobs.referencia import DESTINO
    from jobs.score import BUCKET

    s3 = boto3.client("s3")
    claves = [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=DESTINO)["Contents"]]
    for k in claves:
        ruta = destino / k.removeprefix(DESTINO)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, k, str(ruta))
    huellas = next(destino.rglob("sha256.txt"))
    for linea in huellas.read_text().splitlines():
        esperado, nombre = linea.split(maxsplit=1)
        ruta = huellas.parent / nombre
        if hashlib.sha256(ruta.read_bytes()).hexdigest() != esperado:
            raise RuntimeError(f"checksum distinto en la referencia: {nombre}")
    return destino


def main(argv=None) -> int:
    from jobs.referencia import armar_data
    from jobs.score import log

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir-trabajo", default=os.environ.get("SECOP_DIR_TRABAJO", "/tmp/secop"))
    a = ap.parse_args(argv)
    base = Path(a.dir_trabajo)
    t0 = time.time()
    referencia = bajar_referencia(base / "referencia")
    data = armar_data(base)
    (data / "final").mkdir(exist_ok=True)
    import boto3

    from jobs.score import BUCKET

    boto3.client("s3").download_file(BUCKET, "modelos/v1/config.json", str(data / "final" / "config.json"))
    t_bajar = time.time() - t0
    log(f"datos listos en {t_bajar:.0f} s")

    env = {**os.environ, "SECOP_REFERENCIA_DIR": str(referencia), "SECOP_DATA_DIR": str(data)}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", PRUEBAS, "-k", FILTRO],
                       env=env, check=False)  # fmt: skip
    pico = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2  # KB → GB (Linux)
    log(
        f"RESUMEN paridad salida={r.returncode} bajada={t_bajar:.0f} s pruebas={time.time() - t0 - t_bajar:.0f} s "
        f"total={time.time() - t0:.0f} s pico_pytest={pico:.1f} GB"
    )
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
