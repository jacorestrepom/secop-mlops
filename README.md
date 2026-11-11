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

O con Docker: `docker compose -f infra/docker-compose.yml up --build` (es la misma imagen que corre en Lambda). La documentación queda en http://localhost:8000/docs.

Hoy la API sirve **datos simulados** (`api/datos_simulados.json`, inventados, con el formato exacto del contrato; `/metadata` trae `datos_simulados: true`). Las métricas de `/metadata` sí son las reales de la prueba jul 2025–jul 2026, salvo la tabla de calibración por deciles, que es de ejemplo. Cuando exista RDS solo cambia `FUENTE_DATOS`.

## API publicada (simulada)

**URL:** https://i5rev77icbkeissqixiw6h5soi0npkxh.lambda-url.us-east-1.on.aws/ (documentación en [`/docs`](https://i5rev77icbkeissqixiw6h5soi0npkxh.lambda-url.us-east-1.on.aws/docs)).

Corre en AWS Lambda (función `secop-api`, x86_64, 512 MB, 30 s) detrás de una Function URL pública (`AuthType=NONE`). Es la misma imagen de Docker: la extensión [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) le pasa las peticiones a uvicorn por el puerto 8000. El CORS lo maneja solo FastAPI (la Function URL no tiene configuración CORS). Rol de ejecución `secop-api-lambda`, solo con `AWSLambdaBasicExecutionRole`.

Desplegar una versión nueva (hasta que lo haga GitHub Actions), desde la raíz del repo con el árbol limpio:

```bash
TAG=$(git rev-parse --short HEAD); R=938917087333.dkr.ecr.us-east-1.amazonaws.com/secop-api
docker build --platform linux/amd64 --provenance=false --sbom=false -f api/Dockerfile -t $R:$TAG .
aws ecr get-login-password --profile secop | docker login -u AWS --password-stdin ${R%/*}
docker push $R:$TAG
aws lambda update-function-code --function-name secop-api --image-uri $R:$TAG --profile secop
```

`--provenance=false` es obligatorio: Lambda no acepta las imágenes con atestación que `docker buildx` agrega por defecto.

Tiempos medidos el 2026-11-11 desde Colombia: en caliente la API tarda ~3 ms dentro de Lambda y ~100 ms por petición con la conexión abierta (~300 ms si cada petición abre una conexión HTTPS nueva). La primera petición tras un rato sin uso tarda ~5 s (arranque en frío).

> **Decisión:** API en Lambda + Function URL en vez de App Runner por presupuesto (USD 50–130 en total para el proyecto); además, la cuenta no puede crear servicios de App Runner. Calentar con una petición antes de la demo. Cuando lea de RDS irá en VPC con endpoint de S3, sin NAT.
>
> Concurrencia: la cuenta tiene un límite de 10 ejecuciones simultáneas, así que `secop-api` no tiene concurrencia reservada. Se pidió subir la cuota a 1000 (2026-11-11); cuando la aprueben, se le reservan 5.

## Pruebas

```bash
.venv/bin/ruff check . && .venv/bin/python -m pytest
```

## Estado de los datos (2026-11-11)

- **Athena** funciona con el workgroup `proyecto-integrador-workgroup` (resultados en `s3://proyecto-integrador-bucket-01/athena-results/`).
- **`secop_gold` existe pero está vacía.** Decisión: el job de v1 lee de `secop_silver` (Iceberg) y aplica la limpieza del notebook 02 dentro de `ml/`. Si gold llega después, solo cambia el origen.

  | Tabla de silver | Filas |
  |---|---:|
  | `procesos` | 618.676 |
  | `proponentes` | 1.964.565 |
  | `contratos` | 5.677.850 |
  | `proveedores` | 1.607.713 |

- **Frescura:** silver se actualizó a mano por última vez el 2026-11-07. La ingesta a bronze corre con Lambda + EventBridge Scheduler a las 06:00 de Bogotá.
- **Marca de frescura antes de publicar:** `secop_control.control_corridas` (columnas `job`, `tabla`, `estado`, `fin`, `cuadre_ok`). El job solo publica si los jobs de silver tienen `estado = 'exito'` y `cuadre_ok` del día.
- **Orquestación propuesta:** una Step Functions propia de MLOps, lanzada por Scheduler, cuyo primer paso verifica la frescura.

## Estado (checklist de la guía)

- [x] Credenciales configuradas y prueba de Athena (perfil `secop`)
- [x] Artefactos del modelo en `s3://proyecto-integrador-bucket-01/modelos/v1/` (G1, ranking v2, config, listas, métricas y `manifest.sha256`); el M1 está en `modelos/v1/retrieval/`. El bucket no tiene versionado: `v1/` no se reescribe, una versión nueva va a `v2/`
- [ ] Paquete `ml/` con prueba de paridad contra `listas_202603.parquet`
- [ ] MLflow con los tres modelos y alias `production`
- [x] API simulada publicada con el contrato de la sección 6 y `/docs` (Lambda + Function URL)
- [ ] `score.py` de punta a punta para una `fecha_corte`
- [ ] Publicación atómica en RDS y copia en `gold.recomendaciones`
- [ ] API leyendo de RDS en menos de 300 ms
- [ ] Step Functions programado con reintentos y alarma
- [ ] GitHub Actions: lint, tests, build y despliegue
- [ ] Reporte de drift diario en S3
- [ ] Reentrenamiento de punta a punta con compuerta de promoción
- [ ] Plan B con docker-compose probado en un portátil
