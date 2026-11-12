# Imagen secop-jobs: job nocturno (jobs/score.py) y generación de la referencia de paridad en Fargate.
# Mismo entorno que tests/referencia/Dockerfile (base, hilos fijos y PYTHONHASHSEED): la paridad exacta de ml/
# solo vale dentro de este entorno.
#
#   docker build --platform linux/amd64 --provenance=false -f infra/jobs.Dockerfile -t secop-jobs .
#   docker run --rm secop-jobs python -m jobs.score --help
FROM --platform=linux/amd64 python:3.11.11-slim

ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    SECOP_DIR_TRABAJO=/tmp/secop

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-ml.txt requirements-jobs.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt -r requirements-jobs.txt

COPY pyproject.toml .
COPY ml ml
COPY jobs jobs
COPY tests tests

RUN useradd --create-home --uid 1000 job && mkdir -p /tmp/secop && chown job /tmp/secop
USER job

CMD ["python", "-m", "jobs.score"]
