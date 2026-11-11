# secop-mlops

Servicio del recomendador SECOP: cada noche calcula las dos listas de cada empresa, las publica en PostgreSQL y las sirve por API a la web. El plan está en la sección 5 de la *Guía de modelo y traspaso a MLOps*.

```
api/     FastAPI: rutas, esquemas Pydantic (contrato con la web) y orígenes de datos (simulada / rds)
jobs/    registrar_v1.py (registro en MLflow), publicar.py (publicación atómica); score.py y drift.py pendientes
ml/      paquete del modelo: features, retrieval, ranking, listas (PR de ML)
infra/   sql/serving.sql (esquema de la base), docker-compose (plan B)
tests/   API (simulada y RDS), publicación atómica y paridad del modelo
.github/ CI: lint, pruebas y despliegue de la API
```

## Correr la API en local

Con Python 3.11 y [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/uvicorn api.main:app --reload
```

O con Docker: `docker compose -f infra/docker-compose.yml up --build` (es la misma imagen que corre en Lambda). La documentación queda en http://localhost:8000/docs.

`FUENTE_DATOS` elige de dónde salen los datos: `simulada` (por defecto, `api/datos_simulados.json`, inventados con el formato exacto del contrato) o `rds` (el esquema `serving`; conexión con `DATABASE_URL` o `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD`).

## API publicada

**URL:** https://i5rev77icbkeissqixiw6h5soi0npkxh.lambda-url.us-east-1.on.aws/ (documentación en [`/docs`](https://i5rev77icbkeissqixiw6h5soi0npkxh.lambda-url.us-east-1.on.aws/docs)).

Corre en AWS Lambda (función `secop-api`, x86_64, 512 MB, 30 s) detrás de una Function URL pública (`AuthType=NONE`), con `FUENTE_DATOS=rds`. Es la misma imagen de Docker: la extensión [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) le pasa las peticiones a uvicorn por el puerto 8000. El CORS lo maneja solo FastAPI (la Function URL no tiene configuración CORS).

- **Lee RDS desde la VPC por defecto**, sin internet ni NAT: security group `secop-api-lambda` sin entrada y con salida solo al 5432 de `secop-rds`. Usuario `api_lectura` (solo `SELECT` en `serving`); como la Lambda no llega a Secrets Manager, su contraseña va en una variable de entorno de la Lambda (cifrada en reposo por Lambda con KMS).
- Hoy sirve la **publicación 0**: los datos simulados cargados en RDS con `publicar.py` (`/metadata` dice `datos_simulados: true`). Cuando `score.py` publique, la API sirve los datos reales sin cambiar nada.
- **Despliegue automático**: cada push a `main` que pasa las pruebas construye la imagen, la sube a ECR con el hash del commit, actualiza la Lambda y comprueba `/health` (ver [CI](#ci-github-actions)).

Tiempos medidos el 2026-11-11 desde Colombia, ya leyendo RDS: dentro de Lambda una recomendación tarda ~9 ms; por la red, ~110 ms con la conexión abierta y ~300 ms si cada petición abre una conexión HTTPS nueva. **El arranque en frío tarda ~5 s**: calentar con una petición antes de la demo.

> **Decisión:** API en Lambda + Function URL en vez de App Runner por presupuesto (USD 50–130 en total para el proyecto); además, la cuenta no puede crear servicios de App Runner. Calentar con una petición antes de la demo.
>
> Concurrencia: la cuenta tiene un límite de 10 ejecuciones simultáneas, así que `secop-api` no tiene concurrencia reservada. Se pidió subir la cuota a 1000 (2026-11-11); cuando la aprueben, se le reservan 5.

## Base de serving y publicación atómica

Esquema `serving` en la base `secop` de RDS `secop-db` ([`infra/sql/serving.sql`](infra/sql/serving.sql), idempotente):

| Tabla | Qué guarda |
|---|---|
| `publicaciones` | Una fila por corrida: `fecha_corte`, versiones de los 3 modelos, snapshots de silver, `estado` (`en_progreso` / `publicada` / `fallida`), `es_actual`, `metricas` (lo que sirve `/metadata`) y `datos_simulados`. Un índice único parcial impide que haya dos `es_actual` |
| `recomendaciones` | Las dos listas (`lista` = `historial` u `oportunidad`, `posicion` 1–10) con `probabilidad`, `score_ranking`, `score_retrieval`, `oferentes_esperados` y `motivos` |
| `procesos`, `empresas` | Lo que pintan las tarjetas y el buscador (sin personas naturales) |
| `estadisticas` | Agregados del dashboard |

Todas las tablas llevan `version_id`: cada publicación es una versión completa y la API siempre filtra por la vigente. [`jobs/publicar.py`](jobs/publicar.py) la escribe así:

1. Crea la publicación `en_progreso`.
2. Carga las filas con `COPY`.
3. En **una** transacción: la anterior deja de ser actual y la nueva pasa a `publicada` + `es_actual`; se borran otras versiones de la misma `fecha_corte` (republicar el mismo día reemplaza, no duplica) y las publicadas más viejas que las últimas 7.

Si algo falla, la nueva queda `fallida`, la vigente no cambia y el proceso sale con código 1. `score.py` solo tiene que armar un `Publicacion` y llamar a `publicar()`.

Usuarios de la base: `job_escritura` (escribe en `serving`; secreto `secop/db-job`) y `api_lectura` (solo `SELECT`; secreto `secop/db-api`). El usuario `mlflow` no puede entrar a la base `secop`.

## MLflow

```
portátil ──SSM (sin puertos abiertos)──► EC2 secop-mlflow (t4g.small, se apaga si no se usa)
                                           └ mlflow server 3.18.0 en 127.0.0.1:5000 (systemd)
                                                 │ metadatos               │ artefactos
                                                 ▼                         ▼
job nocturno ──postgresql+psycopg2──► RDS secop-db (base mlflow)    s3://proyecto-integrador-bucket-01/mlflow/
             └───────────────────── lee los modelos directo de S3 ───────────┘
```

- **RDS `secop-db`**: PostgreSQL 16.15, db.t4g.micro, 20 GB gp3, privada y cifrada. Bases `mlflow` y `secop`. Acepta el 5432 solo desde los security groups `secop-mlflow` y `secop-api-lambda`. La contraseña maestra la gestiona AWS en Secrets Manager; el usuario `mlflow` tiene la suya en `secop/mlflow-db`.
- **El servidor no es intermediario de artefactos** (`--no-serve-artifacts`): los clientes leen y escriben S3 directamente. Por eso el job nocturno carga `models:/secop-ranking@production` con `MLFLOW_TRACKING_URI=postgresql+psycopg2://…` sin que la EC2 esté encendida. Hay que escribir `+psycopg2`: sin él, SQLAlchemy busca psycopg 3.
- La EC2 tiene 1 worker, 2 GB de swap y `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`: con los valores por defecto MLflow 3.18 usaba 1,6 GB de los 1,8 GB de RAM y se caía por falta de memoria.
- Modelos registrados con [`jobs/registrar_v1.py`](jobs/registrar_v1.py): verifica los checksums de `s3://…/modelos/v1/`, registra los tres con parámetros, métricas de prueba, features en orden, `config.json`, `git_commit` y el snapshot de silver (`silver_snapshot_procesos`, `_proponentes`, `_contratos`, `_proveedores`, `silver_snapshot_fecha=2026-11-04`, `silver_copia`), pone el alias `production` y comprueba que el modelo registrado predice igual que el original (diferencia 0 sobre 1.000 filas).

**Abrir la interfaz de MLflow** (requiere `session-manager-plugin`: `brew install --cask session-manager-plugin`):

```bash
ID=$(aws ec2 describe-instances --profile secop --filters Name=tag:Name,Values=secop-mlflow --query 'Reservations[0].Instances[0].InstanceId' --output text)
aws ec2 start-instances --instance-ids $ID --profile secop && aws ec2 wait instance-running --instance-ids $ID --profile secop
# esperar ~1 min a que aparezca en SSM; luego:
aws ssm start-session --profile secop --target $ID --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["5000"],"localPortNumber":["5001"]}'
```

Abrir http://localhost:5001. Se usa el puerto local **5001** porque en macOS el 5000 lo ocupa el Receptor AirPlay. Al terminar, cerrar el túnel (Ctrl+C) y **apagar la EC2**: `aws ec2 stop-instances --instance-ids $ID --profile secop`. Con la misma EC2 se llega a RDS desde el portátil: `AWS-StartPortForwardingSessionToRemoteHost` con `host` = endpoint de `secop-db` y `portNumber` 5432.

## Job nocturno (preparado; falta `score.py`)

Ya existen: repositorio ECR `secop-jobs`, cluster ECS `secop` (solo Fargate), log group `/secop/jobs` (14 días), rol de ejecución `secop-jobs-ejecucion` y rol de la tarea `secop-jobs-tarea` (Athena en `proyecto-integrador-workgroup`, lectura de Glue y S3 de `secop_silver` / `secop_control` / `secop_gold`, lectura de `mlflow/`, escritura en `athena-results/`, secretos `secop/db-job` y `secop/mlflow-db`). Faltan la imagen `secop-jobs`, la task definition y la Step Functions: dependen de `score.py`.

## CI (GitHub Actions)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **En cada pull request:** `ruff`, `pytest` (con un PostgreSQL 16 de servicio para las pruebas de publicación y de la fuente RDS; la de paridad se salta si no hay `SECOP_DATA_DIR`) y build de la imagen sin push.
- **En push a `main`:** lo mismo y, si pasa, build `linux/amd64` → ECR `secop-api:<hash corto>` → `update-function-code` de la Lambda → comprueba que `/health` responda `base_datos: ok`.
- Credenciales por **OIDC**: el rol `secop-github-deploy` solo lo puede asumir `jacorestrepom/secop-mlops` desde `refs/heads/main`, y solo puede subir imágenes a `secop-api`/`secop-jobs` y actualizar el código de `secop-api`. No hay llaves en GitHub; el workflow solo usa la variable del repo `AWS_ACCOUNT_ID`.

## Pruebas

```bash
.venv/bin/ruff check . && .venv/bin/python -m pytest
```

Las pruebas de publicación y de la fuente RDS necesitan un PostgreSQL (se saltan si no está):

```bash
docker run -d --name secop-pg-prueba -e POSTGRES_PASSWORD=prueba -e POSTGRES_DB=secop -p 55432:5432 postgres:16
TEST_DATABASE_URL=postgresql://postgres:prueba@localhost:55432/secop .venv/bin/python -m pytest
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
- [ ] Paquete `ml/` con prueba de paridad contra `listas_202603.parquet` (PR de ML)
- [x] MLflow con los tres modelos y alias `production` (`secop-retrieval`, `secop-pgana`, `secop-ranking`, versión 1, con el snapshot de silver)
- [x] API simulada publicada con el contrato de la sección 6 y `/docs` (Lambda + Function URL)
- [ ] `score.py` de punta a punta para una `fecha_corte`
- [x] Publicación atómica en RDS (`jobs/publicar.py`, probada con fallas, doble corrida y retención de 7)
- [ ] Copia de las listas en `gold.recomendaciones`
- [x] API leyendo de RDS en menos de 300 ms (~110 ms con la conexión abierta; frío ~5 s)
- [ ] Step Functions programado con reintentos y alarma
- [x] GitHub Actions: lint, tests, build y despliegue
- [ ] Reporte de drift diario en S3
- [ ] Reentrenamiento de punta a punta con compuerta de promoción
- [ ] Plan B con docker-compose probado en un portátil
