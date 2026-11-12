# secop-mlops

Servicio del recomendador SECOP: cada noche calcula las dos listas de cada empresa, las publica en PostgreSQL y las sirve por API a la web. El plan está en la sección 5 de la *Guía de modelo y traspaso a MLOps*.

```
api/     FastAPI: rutas, esquemas Pydantic (contrato con la web) y orígenes de datos (simulada / rds)
jobs/    score.py (job nocturno + métricas + drift), ingesta_recientes.py, publicar.py, registrar_v1.py, promover.py (compuerta), referencia.py
ml/      paquete del modelo: features, retrieval, ranking, listas (PR de ML)
infra/   sql/serving.sql, jobs.Dockerfile (imagen secop-jobs), stepfunctions/secop-nocturno.asl.json, docker-compose (plan B)
tests/   API (simulada y RDS), publicación atómica y paridad del modelo
.github/ CI: lint, pruebas y despliegue de la API
```

## Correr la API en local

Con Python 3.11 y [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/uvicorn api.main:app --reload
```

Con Docker y datos reales: ver [Plan B de la demo](#plan-b-de-la-demo-sin-aws). La documentación queda en http://localhost:8000/docs.

`FUENTE_DATOS` elige de dónde salen los datos: `simulada` (por defecto, `api/datos_simulados.json`, inventados con el formato exacto del contrato) o `rds` (el esquema `serving`; conexión con `DATABASE_URL` o `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD`).

## API publicada

**URL:** https://i5rev77icbkeissqixiw6h5soi0npkxh.lambda-url.us-east-1.on.aws/ (documentación en [`/docs`](https://i5rev77icbkeissqixiw6h5soi0npkxh.lambda-url.us-east-1.on.aws/docs)).

Corre en AWS Lambda (función `secop-api`, x86_64, 512 MB, 30 s) detrás de una Function URL pública (`AuthType=NONE`), con `FUENTE_DATOS=rds`. Es la misma imagen de Docker: la extensión [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) le pasa las peticiones a uvicorn por el puerto 8000. El CORS lo maneja solo FastAPI (la Function URL no tiene configuración CORS).

- **Lee RDS desde la VPC por defecto**, sin internet ni NAT: security group `secop-api-lambda` sin entrada y con salida solo al 5432 de `secop-rds`. Usuario `api_lectura` (solo `SELECT` en `serving`); como la Lambda no llega a Secrets Manager, su contraseña va en una variable de entorno de la Lambda (cifrada en reposo por Lambda con KMS).
- **Sirve datos reales** desde el 2026-11-11: las listas que publica cada noche `jobs/score.py` (`/metadata` dice `datos_simulados: false`). La publicación 0 (datos simulados) quedó como versión anterior y se borra sola por la retención de 7.
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

## Job nocturno

```
EventBridge Scheduler secop-nocturno-diario (07:30 America/Bogota)
  └► Step Functions secop-nocturno
       0. ECS Fargate IngestaRecientes (jobs/ingesta_recientes.py, 1 vCPU / 2 GB): procesos de los últimos 45 días
          a bronze. Si falla tras 2 reintentos → aviso a SNS secop-alertas y la noche SIGUE con lo que hay en bronze
       1. Glue secop-silver-procesos (E3: reescribe secop_silver.procesos desde bronze)
       2. Glue secop-silver-ganador  (E6: completa ganador, estado, sector_entidad, n_proponentes)
       3. ECS Fargate secop-score:3  (jobs/score.py, 4 vCPU / 30 GB / 50 GB de disco)
     2 reintentos por paso con backoff; si E3, E6 o score se agotan → estado Falla, sin publicar
  └► si termina en FAILED / TIMED_OUT / ABORTED: regla secop-nocturno-fallo → SNS secop-alertas (correo)
Alarma secop-api-errores (Errors de la Lambda > 0 en 5 min) → SNS secop-alertas
```

[`jobs/score.py`](jobs/score.py) hace, en orden:

1. **Frescura:** en `secop_control.control_corridas`, `secop-silver-procesos` **y** `secop-silver-ganador` con `estado='exito'` y `cuadre_ok` en la `fecha_corte` (hora de Bogotá), y E6 después de E3. Si no, sale con **código 2 sin publicar**. Se exigen los dos porque E3 reescribe la tabla completa y deja sin llenar lo que completa E6 (`sector_entidad` es una feature del modelo).
2. **Historia congelada** desde `s3://…/modelos/v1/historia/`, verificada contra `manifest_historia.sha256`.
3. **Procesos de silver** (Athena, la consulta de la celda 38 del notebook 02 más los campos de la tarjeta) → `limpiar()`. Los procesos que no están en la historia se agregan a la tabla de procesos de `Historia` y `HistoriaRetrieval`.
4. **Empresas** (solo personas jurídicas que son usuarios objetivo): con participación en los 12 meses anteriores (calientes) o con contrato firmado en esos 12 meses y sin participación en 24 (frías).
5. `recuperar()` **por lotes de 10.000 empresas** (calientes y frías en lotes separados) → libera la historia del retrieval → `construir()` / `rankear()` / `armar_listas()` **por lotes de ~1 millón de pares**, con los modelos `models:/secop-*@production` leídos del registro de MLflow directo en PostgreSQL (sin la EC2).
6. **Motivos** (Paso 5 de la guía, en orden y máximo 3): misma entidad · entidad nueva (lista 2) · ≤ 3 oferentes esperados · precio dentro del rango que suele ganar (\|log\| ≤ 1,4) · similitud con procesos ganados (`knn_max_sim_ganado` ≥ 0,5, umbral provisional que valida ML).
7. **Publica** en RDS con `publicar.py` (atómico; sin personas naturales; métricas de `/metadata` desde [`jobs/metricas_v1.json`](jobs/metricas_v1.json)) y deja una copia en Parquet en `s3://…/gold/recomendaciones/` (tabla `secop_gold.recomendaciones`, partición `fecha_corte`, `overwrite_partitions`).
8. Envía las métricas de volumen a CloudWatch, genera el reporte de drift (ver [Monitoreo](#monitoreo)) y registra tiempo y memoria de cada etapa con una línea `RESUMEN` en `/secop/jobs`.

**Verificado (2026-11-11):**
- Por lotes da **exactamente** lo mismo que de una vez: marzo de 2026, 10.607 jurídicas, incluidos lotes de solo-frías con centinelas; sha256 idéntico. Quitar las personas naturales no cambia las listas de las jurídicas.
- La misma `fecha_corte` dos veces da las mismas listas (sha256 idéntico), una sola publicación por fecha y un único `es_actual`.
  Si en medio se vuelve a correr silver (E3 + E6), las listas pueden cambiar un poco: E3 imputa la ciudad de la entidad con `F.first`, que en Spark no es determinista (pendiente de ingeniería de datos).
- Corrida real del 11-nov: 39.964 empresas (24.462 calientes y 15.502 frías), 29.994 con listas, 436 procesos abiertos, largo medio de la lista 2 de 6,6, **19 min y 15 GB de pico**.

> **Decisión · historial congelado al 27-oct-2026.** `base_entrenamiento.parquet` no se puede regenerar desde silver: la celda que lo creaba se perdió. El historial (participaciones, contratos, proveedores y nombres) sale de la copia en `modelos/v1/historia/` y queda fijo en el 27-oct-2026. Los procesos abiertos sí son frescos. Consecuencia: las features de una empresa no ven sus participaciones posteriores a esa fecha, hasta que ML reconstruya la celda y se publique una historia nueva (`v2/`).
>
> **Decisión · ejecución a las 07:30 (Bogotá).** La ingesta a bronze corre a las 06:00. El flujo corre después y reescribe silver antes de puntuar.
>
> **Filtro de servicio (no cambia el modelo):** solo se recomiendan procesos con `estado='abierto'`; los cancelados y los cerrados sin adjudicar no se ofrecen aunque su fecha límite no haya pasado.

**Costo por noche** (ejecución completa de `secop-nocturno` del 2026-11-12, con la ingesta: **38 min** de punta a punta, ingesta 5,2 min · E3 6,1 · E6 3,9 · score 22,9): ingesta en Fargate (1 vCPU / 2 GB) ≈ USD 0,004; Glue E3 + E6 ≈ 5.400 DPU-segundos ≈ **USD 0,66**; `score.py` en Fargate ≈ 23 min con 4 vCPU / 30 GB ≈ **USD 0,11**; Step Functions, Scheduler, SNS y EventBridge entran en el nivel gratuito. **Total ≈ USD 0,78 por noche (≈ USD 23 al mes)**, más RDS (≈ USD 14 al mes, todo el tiempo) y las métricas y alarmas (≈ USD 2,40 al mes).

**Correr `score.py` a mano** (la misma tarea que lanza la Step Functions):

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --profile secop --query Account --output text)
SG=$(aws ec2 describe-security-groups --profile secop --filters Name=group-name,Values=secop-jobs --query 'SecurityGroups[0].GroupId' --output text)
SUBNETS=$(aws ec2 describe-subnets --profile secop --filters Name=default-for-az,Values=true --query 'Subnets[?AvailabilityZone!=`us-east-1e`].SubnetId' --output text | tr '\t' ',')
aws ecs run-task --profile secop --cluster secop --task-definition secop-score --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=ENABLED}" \
  --overrides '{"containerOverrides":[{"name":"score","command":["python","-m","jobs.score","--fecha-corte","2026-11-11"]}]}'
```

Opciones de `score.py`: `--sin-publicar` (calcula todo y no publica), `--muestra N` (solo N empresas), `--verificar-lotes 2026-03-01` (la prueba de lotes). Correr el flujo completo: `aws stepfunctions start-execution --profile secop --state-machine-arn arn:aws:states:us-east-1:$AWS_ACCOUNT_ID:stateMachine:secop-nocturno`. Los logs quedan en CloudWatch, grupo `/secop/jobs`.

**Referencia de paridad:** `python -m jobs.referencia` (en Fargate, con la misma imagen) corre `tests/referencia/generar_referencia.py` dos veces y, si el `sha256.txt` es idéntico, la sube a `s3://…/ci/referencia/` (9 MB, 300 empresas de marzo). Todavía no la usa el CI.

## Monitoreo

Todo avisa al tema SNS `secop-alertas` (correo confirmado).

| Qué | Cómo | Alarma |
|---|---|---|
| El flujo nocturno terminó | Regla EventBridge `secop-nocturno-fallo` sobre la Step Functions | `FAILED`, `TIMED_OUT` o `ABORTED` |
| La API responde | Alarma `secop-api-errores` | `Errors` de la Lambda > 0 en 5 min |
| Volumen | `score.py` envía a CloudWatch (namespace `Secop`) `empresas_puntuadas`, `procesos_abiertos`, `largo_medio_lista2`, `duracion_min` y `publicacion` (solo cuando publica) | `secop-procesos-abiertos-bajos`: `procesos_abiertos` < 300 · `secop-sin-publicacion`: ninguna publicación en 26 h (métrica ausente = alarma) |
| Drift (Paso 8) | Al final de `score.py`, reporte [Evidently](https://github.com/evidentlyai/evidently) 0.7.23 de ~24 mil pares de hoy frente a una muestra fija de 24 mil candidatos del entrenamiento (jul 2024 – jun 2025, `s3://…/monitoreo/drift/referencia_v1.parquet`) en `score_retrieval`, `knn_tasa50`, `knn_tasa5`, `unspsc_familia`, `dias_ult_part_ok` y `comp_entmod_media`. HTML y `resumen.json` en `s3://…/monitoreo/drift/fecha=<fecha_corte>/`; `drift_features` y `drift_share` en CloudWatch | — (se lee; si falla, no tumba la publicación) |

Primer reporte (2026-11-11): **6 de 6 features con drift**. Es esperable y no indica un error: hoy hay solo 436 procesos abiertos, muy concentrados en pocas categorías (`unspsc_familia` 0,74; `comp_entmod_media` 0,61); entran 15 mil empresas frías que no estaban en el entrenamiento (`score_retrieval` 0,46); y el historial congelado hace crecer los días desde la última participación (`dias_ult_part_ok` 0,36). El reporte refleja justo los dos pendientes conocidos: la historia congelada y la ingesta.

Costo: 7 métricas propias ≈ USD 2,10 al mes y 3 alarmas ≈ USD 0,30 al mes.

## Reentrenamiento y compuerta de promoción (Paso 9)

[`jobs/promover.py`](jobs/promover.py) decide si un modelo candidato (alias `staging`) reemplaza al de `production`, comparándolos **en el mismo periodo de validación**:

| Modelo | Métrica principal (debe mejorar) | No puede empeorar | Cómo se mide |
|---|---|---|---|
| `secop-pgana` (G1) | log loss | ECE, y log loss y ECE en pymes | Se recalculan con los dos modelos sobre la misma muestra fija de la prueba: 60 mil participaciones jul 2025 – jul 2026 en `s3://…/monitoreo/validacion/g1_v1.parquet` |
| `secop-ranking` | NDCG@10 de la lista 1 | NDCG@10 en pymes | Métricas registradas en el run de cada versión |
| `secop-retrieval` (M1) | Recall@150 | Recall@150 en pymes | Métricas registradas en el run de cada versión |

**Un empate no se promueve:** el candidato tiene que mejorar la métrica principal y no empeorar ninguna otra. Si solo se exigiera "no empeorar", un modelo idéntico al actual pasaría. Si pasa, el alias `production` se mueve al candidato. Si no, se queda en `staging` con las etiquetas `compuerta=rechazado` y `compuerta_motivo`, y se avisa a `secop-alertas`. Si a un candidato de ranking o retrieval le falta alguna métrica (por ejemplo la de pymes), la compuerta no puede verificar y no promueve.

**Demostrado una vez (2026-11-12):** se registró una copia exacta del G1 actual como `secop-pgana` v2 en `staging`. La compuerta midió métricas idénticas (log loss 0,3825, ECE 0,0053; en pymes 0,4044 y 0,0055), lo dejó en `staging` ("log_loss no mejora"), `production` siguió en la v1 y llegó el aviso por SNS. Correrlo: `python -m jobs.promover --modelo secop-pgana` (con acceso al registro de MLflow; ver [MLflow](#mlflow)).

**Pendiente: el reentrenamiento real.** Los scripts `train_retrieval.py`, `train_g1.py` y `train_ranking.py` no existen todavía: el código de entrenamiento sigue en el notebook final y el del retrieval en el notebook 03. Además necesita más de 18 GB (construir las features de 12 meses de candidatos), y la celda que crea `base_entrenamiento` se perdió. Cuando ML los entregue, el ciclo mensual sería: entrenar en Fargate → registrar con alias `staging` → `promover.py`. Para ranking y retrieval, el script de entrenamiento tendrá que registrar también las métricas de pymes.

## Ingesta de procesos recientes (paso 0 de la noche)

**Por qué existe:** hasta el 26-oct entraban ~2.400 procesos nuevos por semana a silver, y después cayeron a 346, 189 y 5. La Lambda `proyecto-integrador-ingesta` (de ingeniería de datos) pide `procesos-contratacion` por `:updated_at >= watermark`. SECOP reescribe `:updated_at` en millones de filas al día y la Lambda se corta en ~200 mil, casi todas de procesos viejos (en un archivo suyo del 11-nov, las fechas de publicación van de 2015 a 2026). Así casi nunca llegaba a los procesos nuevos.

**Qué hace:** [`jobs/ingesta_recientes.py`](jobs/ingesta_recientes.py) pide a la API de Socrata (dataset `p6dx-8zbt`) solo lo publicado en los últimos 45 días (`fecha_de_publicacion_del`), con la misma paginación keyset (`$order=:updated_at,:id`) y el mismo formato que la Lambda (NDJSON en gzip, todas las columnas). Escribe `part-recientes-*.json.gz` en `bronze/procesos-contratacion/ingesta/fecha=<hoy UTC>/`, junto a los archivos de la Lambda, que no se tocan. `secop-silver-procesos` lee toda esa carpeta y su regla R4 deja una fila por `id_proceso` (gana la versión más reciente), así que no duplica. Autorizado por ingeniería de datos (Juan Diego) el 2026-11-12.

**Permisos:** el rol `secop-jobs-tarea` puede leer el token `/secop/app-token` (SSM) y escribir **solo** `bronze/procesos-contratacion/ingesta/*/part-recientes-*`.

**Primera corrida (2026-11-12, a mano en Fargate):** 216.165 filas en 22 archivos, 5,7 min. Después de E3 + E6: `cuadre_ok = true`, `secop_silver.procesos` pasó de 618.850 a **623.592** filas, **0 `id_proceso` duplicados**, y las semanas desde el 26-oct pasaron de 346 / 189 / 5 a **3.037 / 2.603 / 780** procesos.

**Primera noche completa con la ingesta (2026-11-12, Step Functions a mano):** `SUCCEEDED` en 38 min. **2.089 procesos abiertos** (antes 436 el 11-nov y 369 en la corrida de las 07:30 del 12-nov), 6.270 procesos nuevos frente a la historia, 35.642 empresas con lista (antes 29.994), 508.269 recomendaciones (versión 8), drift en 4 de 6 features (antes 6 de 6).

Mejora posible: el 89 % de lo que baja son modalidades que E3 descarta por alcance (sobre todo régimen especial y contratación directa). Filtrarlas en la consulta bajaría el tiempo, sin cambiar el resultado.

## CI (GitHub Actions)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **En cada pull request:** `ruff`, `pytest` (con un PostgreSQL 16 de servicio para las pruebas de publicación y de la fuente RDS; la de paridad se salta si no hay `SECOP_DATA_DIR`) y build de las dos imágenes (`secop-api` y `secop-jobs`) sin push.
- **En push a `main`:** lo mismo y, si pasa, build `linux/amd64` → ECR `secop-api:<hash corto>` y `secop-jobs:<hash corto>` (una etiqueta existente de `secop-jobs` no se reescribe) → `update-function-code` de la Lambda → comprueba que `/health` responda `base_datos: ok`. La task definition `secop-score` apunta a una etiqueta fija de `secop-jobs`: cambiar la imagen del job es registrar una revisión nueva, probarla y actualizar la Step Functions.
- Credenciales por **OIDC**: el rol `secop-github-deploy` solo lo puede asumir `jacorestrepom/secop-mlops` desde `refs/heads/main`, y solo puede subir imágenes a `secop-api`/`secop-jobs` y actualizar el código de `secop-api`. No hay llaves en GitHub; el workflow solo usa la variable del repo `AWS_ACCOUNT_ID`.

## Plan B de la demo (sin AWS)

Si la nube falla el día de la presentación, la API corre en un portátil con una copia de la publicación vigente. Requiere Docker y el perfil `secop` de AWS solo para bajar el volcado.

```bash
./infra/exportar_demo.sh     # (con internet, antes de la demo) vuelca la publicación vigente a s3://…/demo/
./infra/cargar_demo.sh       # baja el volcado más reciente y levanta PostgreSQL + API en local
open http://localhost:8000/docs
```

- [`infra/exportar_demo.sh`](infra/exportar_demo.sh) enciende la EC2 `secop-mlflow` (la única que llega a RDS), hace allí un `pg_dump` del esquema `serving` con **solo la publicación vigente** (`COPY … WHERE version_id = <actual>`) y lo sube a `s3://proyecto-integrador-bucket-01/demo/serving_<fecha_corte>.sql.gz`. Apaga la EC2 al terminar, aunque falle.
- [`infra/cargar_demo.sh`](infra/cargar_demo.sh) deja el volcado en `infra/demo/` (no va a Git) y levanta [`infra/docker-compose.yml`](infra/docker-compose.yml): PostgreSQL 16 lo carga al crear su volumen, y la API (misma imagen que Lambda) lee de ahí con `FUENTE_DATOS=rds`. Sin internet, se le puede pasar un volcado ya bajado: `./infra/cargar_demo.sh serving_2026-11-11.sql.gz`.
- Probado el 2026-11-12: volcado de 24 MB (39.923 empresas, 436 procesos, 454.476 recomendaciones), carga en ~1 min, y la misma recomendación que la API en la nube. Detener: `docker compose -f infra/docker-compose.yml down`.

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
- [x] Paquete `ml/` con prueba de paridad (PR #1 de ML); referencia de paridad generada en Fargate y subida a `s3://…/ci/referencia/` (falta conectarla al CI)
- [x] MLflow con los tres modelos y alias `production` (`secop-retrieval`, `secop-pgana`, `secop-ranking`, versión 1, con el snapshot de silver)
- [x] API simulada publicada con el contrato de la sección 6 y `/docs` (Lambda + Function URL)
- [x] `score.py` de punta a punta para una `fecha_corte` (por lotes verificados contra marzo; idempotente y determinista)
- [x] Publicación atómica en RDS (`jobs/publicar.py`, probada con fallas, doble corrida y retención de 7)
- [x] Copia de las listas en `gold.recomendaciones` (Parquet particionado por `fecha_corte`)
- [x] API leyendo de RDS en menos de 300 ms (~110 ms con la conexión abierta; frío ~5 s)
- [x] Step Functions programado con reintentos y alarma (`secop-nocturno` a las 07:30 de Bogotá; alertas por SNS al correo)
- [x] GitHub Actions: lint, tests, build y despliegue (API y job)
- [x] Reporte de drift diario en S3 (Evidently en `score.py`) y métricas de volumen con alarmas
- [ ] Reentrenamiento de punta a punta con compuerta de promoción: **la compuerta está hecha y demostrada** (`jobs/promover.py`); faltan los `train_*.py` (el entrenamiento sigue en los notebooks y necesita más de 18 GB)
- [x] Ingesta de procesos recientes (`jobs/ingesta_recientes.py`), paso 0 de la Step Functions; si falla, avisa y la noche sigue
- [x] Plan B con docker-compose probado en un portátil (volcado de la publicación vigente desde RDS + API local)
