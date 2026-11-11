# secop-mlops

Servicio del recomendador SECOP: cada noche calcula las dos listas de cada empresa, las publica en PostgreSQL y las sirve por API a la web. El plan está en la sección 5 de la *Guía de modelo y traspaso a MLOps*.

```
api/     FastAPI: rutas, esquemas Pydantic (contrato con la web) y orígenes de datos
jobs/    score.py, registrar_v1.py, drift.py (pendiente)
ml/      paquete del modelo: features, retrieval, ranking, listas (pendiente, con Camilo)
infra/   docker-compose (plan B), Step Functions y despliegue
tests/   pruebas de la API y prueba de paridad
```

## Correr la API en local

Con Python 3.11 y [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/uvicorn api.main:app --reload
```

O con Docker: `docker compose -f infra/docker-compose.yml up --build`. La documentación queda en http://localhost:8000/docs.

Hoy la API sirve **datos simulados** (`api/datos_simulados.json`, inventados, con el formato exacto del contrato; `/metadata` trae `datos_simulados: true`). Las métricas de `/metadata` sí son las reales de la prueba jul 2025–jul 2026, salvo la tabla de calibración por deciles, que es de ejemplo. Cuando exista RDS solo cambia `FUENTE_DATOS`.

## Pruebas

```bash
.venv/bin/ruff check . && .venv/bin/python -m pytest
```

## Estado (checklist de la guía)

- [x] Credenciales configuradas (perfil `secop`); falta la prueba de Athena
- [x] Artefactos del modelo en `s3://proyecto-integrador-bucket-01/modelos/v1/` (G1, ranking v2, config, listas, métricas y `manifest.sha256`); el M1 está en `modelos/v1/retrieval/`. El bucket no tiene versionado: `v1/` no se reescribe, una versión nueva va a `v2/`
- [ ] Paquete `ml/` con prueba de paridad contra `listas_202603.parquet`
- [ ] MLflow con los tres modelos y alias `production`
- [x] API simulada con el contrato de la sección 6 y `/docs`
- [ ] `score.py` de punta a punta para una `fecha_corte`
- [ ] Publicación atómica en RDS y copia en `gold.recomendaciones`
- [ ] API leyendo de RDS en menos de 300 ms
- [ ] Step Functions programado con reintentos y alarma
- [ ] GitHub Actions: lint, tests, build y despliegue
- [ ] Reporte de drift diario en S3
- [ ] Reentrenamiento de punta a punta con compuerta de promoción
- [ ] Plan B con docker-compose probado en un portátil
