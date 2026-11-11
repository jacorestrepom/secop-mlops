# Imagen de pruebas de ml/: genera la referencia de paridad exacta y corre tests/test_paridad.py contra ella.
# La paridad exacta solo vale DENTRO de esta imagen (linux/amd64, mismas librerías, hilos fijos): los empates y
# la numérica del SVD cambian entre plataformas.
#
#   docker build --platform linux/amd64 -f tests/referencia/Dockerfile -t secop-ml-pruebas .
#   docker run --rm --platform linux/amd64 -v <data>:/data:ro -v <salida>:/salida secop-ml-pruebas \
#       python -m tests.referencia.generar_referencia --data /data --salida /salida
FROM --platform=linux/amd64 python:3.11.11-slim

ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt pytest==8.3.5
COPY pyproject.toml .
COPY ml ml
COPY tests tests
