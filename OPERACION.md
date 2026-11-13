# Operación del recomendador SECOP

Manual corto para operar el sistema sin quien lo construyó. Los detalles de diseño están en el [README](README.md).

Todos los comandos usan estas variables. Las credenciales son las del perfil de AWS CLI de quien opere; no hay llaves en este repo:

```bash
export AWS_PROFILE=secop            # o el perfil que tenga acceso a la cuenta
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SM=arn:aws:states:$AWS_REGION:$AWS_ACCOUNT_ID:stateMachine:secop-nocturno
API=$(aws lambda get-function-url-config --function-name secop-api --query FunctionUrl --output text)
```

Las alertas llegan por correo desde el tema SNS `secop-alertas`: avisos de la Step Functions, la regla `secop-nocturno-fallo` y las alarmas de CloudWatch. La noche la lanza el Scheduler `secop-nocturno-diario` a las **07:30 de Bogotá**, después de la ingesta de ingeniería de datos de las 06:00.

## 1. Alertas

**Qué detiene la noche:** solo tres pasos son obligatorios, Glue `secop-silver-procesos`, Glue `secop-silver-ganador` y `secop-score`. Si uno falla tras sus reintentos, la ejecución termina en `FAILED`, **no se publica nada y la API sigue sirviendo la versión anterior**. Los demás pasos solo avisan y la noche sigue.

Dónde mirar:
- **Step Functions:** la consola de `secop-nocturno` → ejecución → pestaña *Graph* o *Events*. La máquina no tiene log group propio: el error está en el historial de la ejecución.
- **Fargate** (ingestas y score): log group `/secop/jobs`, stream `score/score/<id de la tarea>`. El id de la tarea sale en el evento `TaskSubmitted` del historial.
- **Glue:** log groups `/aws-glue/jobs/output` y `/aws-glue/jobs/error`, stream = `JobRunId` (en el historial de la ejecución o en la consola de Glue → job → *Runs*).
- **API:** log group `/aws/lambda/secop-api`.

| Alerta (asunto del correo o alarma) | Qué significa | Qué revisar | Qué hacer |
|---|---|---|---|
| `SECOP: falló la ingesta de recientes de procesos` | La tarea de Fargate `jobs.ingesta_recientes --tabla procesos` falló 2 veces (Socrata caído, token `/secop/app-token`, permisos de S3). **Solo avisa**: silver se arma con lo que ya hay en bronze, incluida la ingesta de la Lambda de ingeniería de datos | `/secop/jobs`, stream de esa tarea; busca `RESUMEN tabla=procesos` o el traceback | Si es la API de datos.gov.co, nada: la próxima noche vuelve a pedir 45 días. Si se repite, relanzar la noche completa (§2) |
| `SECOP: falló la ingesta de recientes de proponentes` / `…de contratos` / `…de proveedores` | Igual, para `--tabla proponentes`, `contratos` o `proveedores`. **Solo avisan** | `/secop/jobs` (`RESUMEN tabla=…`) | Igual que procesos |
| `SECOP: falló el paso de ingestas` | Falló el bloque paralelo completo (raro: error de la propia Step Functions). **Solo avisa** | Historial de la ejecución | Revisar si la noche siguió; si no, relanzarla |
| Ejecución `FAILED` en `SilverProcesos` (correo de `secop-nocturno-fallo`) | Glue `secop-silver-procesos` (E3) falló 3 veces. **Detiene la noche**: sin procesos frescos no se puntúa | `/aws-glue/jobs/error` y `/output` con el `JobRunId`; `secop_control.control_corridas` (Athena) para ver `estado` y `cuadre_ok` | Es un job de ingeniería de datos: avisarles con el `JobRunId`. Cuando lo arreglen, relanzar la noche completa |
| `SECOP: falló secop-silver-proponentes` / `…-proveedores` / `…-contratos` | Glue E4, proveedores o E5 falló 3 veces. **Solo avisa**: la tabla queda con su versión anterior. El modelo no las usa (trabaja con la historia congelada); afecta al dashboard | `/aws-glue/jobs/error` con el `JobRunId` | Avisar a ingeniería de datos. No hace falta relanzar la noche |
| Ejecución `FAILED` en `SilverGanador` | Glue `secop-silver-ganador` (E6) falló 3 veces. **Detiene la noche**: completa `estado` y `sector_entidad`, que el modelo usa | Igual que E3. Ojo: E6 se niega a correr si procesos ya estaba enriquecido; hay que correr E3 antes | Avisar a ingeniería de datos; relanzar la noche completa (E3 → … → E6), no solo E6 |
| Ejecución `FAILED` en `Score` | `jobs/score.py` falló 3 veces (reintentos cada 5 min). **Detiene la noche** | `/secop/jobs`: la última línea `ERROR (código 1…)` o `SIN PUBLICAR (código 2…)` | Código 1: leer el traceback (RDS, MLflow, memoria). Código 2: ver la fila siguiente. Corregido, relanzar solo el score (§2) |
| Score con **código 2** (`SIN PUBLICAR: silver no está fresco`) | La frescura falló: en `secop_control.control_corridas` no hay `secop-silver-procesos` **y** `secop-silver-ganador` con `estado='exito'` y `cuadre_ok` en la fecha de hoy (Bogotá), con E6 después de E3. El score no publica y **la API sigue con la versión anterior** (es lo correcto: no se publican listas con datos viejos). La tarea termina con código distinto de 0, así que la integración ECS de la Step Functions marca el paso como fallido y llega el correo de `secop-nocturno-fallo` | La línea `SIN PUBLICAR` dice qué falta; en Athena: `SELECT * FROM secop_control.control_corridas ORDER BY fin DESC LIMIT 20` | Si silver no corrió o no cuadró, es de ingeniería de datos. Si la noche cruzó la medianoche de Bogotá, la fecha del score ya no es la de silver: relanzar la noche completa |
| `secop-api-errores` | La Lambda `secop-api` tuvo `Errors` > 0 en 5 minutos | `/aws/lambda/secop-api` | Verificar la API (§2). Si `/health` dice `base_datos` distinto de `ok`, revisar que RDS `secop-db` esté encendida (se enciende sola a los 7 días si se detuvo) |
| `secop-procesos-abiertos-bajos` | La última publicación tuvo menos de 300 procesos abiertos (lo normal son 2.000–3.000) | `/secop/jobs`: `ABIERTOS a la fecha_corte`; los avisos de ingesta de la misma noche | Casi siempre es la ingesta: la de recientes falló o la fuente no publicó. Si la ingesta está bien, avisar a ingeniería de datos |
| `secop-sin-publicacion` | No hubo ninguna publicación en 26 horas (métrica ausente = alarma) | Lista de ejecuciones de `secop-nocturno` (§2) y si el Scheduler está `ENABLED` | Si la noche falló, ver su correo. Si no corrió, revisar el Scheduler `secop-nocturno-diario`. Después del apagado (§3) esta alarma se borra |
| `secop-unspsc-sin-modelo` | `score.py` no pudo cargar `models:/secop-unspsc@production`. **La noche siguió y publicó SIN inferir la familia UNSPSC**: las listas salen como antes de la imputación (peores, ver README) | `/secop/jobs`: línea `AVISO: no cargó secop-unspsc` con el error | Ver que el alias `production` exista en el registro de MLflow y que el artefacto esté en `s3://…/mlflow/secop-unspsc/`. No hay que relanzar nada: la próxima noche lo vuelve a intentar |
| `secop-unspsc-inferido-bajo` | El % de procesos abiertos con familia inferida cayó por debajo de 15,6 % (la mitad del 31,2 % de la primera corrida) | `/secop/jobs`: línea `UNSPSC de los N abiertos: real … · inferido … · nulo …` | Si el % real subió (la fuente volvió a publicar el UNSPSC), es buena noticia: ajustar el umbral de la alarma. Si subió el nulo, las descripciones cambiaron y el clasificador se quedó viejo: reentrenar con `jobs/entrenar_unspsc.py` (README) |

## 2. Comandos

**Relanzar la Step Functions completa y ver su estado** (≈ 60 min; no la lances cerca de la medianoche de Bogotá):

```bash
E=$(aws stepfunctions start-execution --state-machine-arn $SM --name manual-$(date +%Y%m%d-%H%M) --query executionArn --output text)
aws stepfunctions describe-execution --execution-arn $E --query '[status,startDate,stopDate]' --output text
aws stepfunctions list-executions --state-machine-arn $SM --max-results 5 --query 'executions[].[name,status,startDate]' --output text
```

**Relanzar solo `score.py`** (≈ 25 min). Usa la misma task definition y la misma red que la noche. Necesita que silver esté fresco **para esa fecha** (E3 y E6 con éxito ese día): en la práctica sirve para hoy, después de una falla del score. Agrega `"--sin-publicar"` al comando para calcular sin publicar.

```bash
FECHA=$(TZ=America/Bogota date +%F)
read TD SUBNETS SG <<<$(aws stepfunctions describe-state-machine --state-machine-arn $SM --query definition --output text | python3 -c '
import json,sys; p=json.load(sys.stdin)["States"]["Score"]["Parameters"]; n=p["NetworkConfiguration"]["AwsvpcConfiguration"]
print(p["TaskDefinition"], ",".join(n["Subnets"]), ",".join(n["SecurityGroups"]))')
T=$(aws ecs run-task --cluster secop --task-definition $TD --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=ENABLED}" \
  --overrides "{\"containerOverrides\":[{\"name\":\"score\",\"command\":[\"python\",\"-m\",\"jobs.score\",\"--fecha-corte\",\"$FECHA\"]}]}" \
  --tags key=proyecto,value=secop --query 'tasks[0].taskArn' --output text)
aws ecs wait tasks-stopped --cluster secop --tasks $T     # espera hasta 10 min; repetirlo si sigue corriendo
aws ecs describe-tasks --cluster secop --tasks $T --query 'tasks[0].[lastStatus,containers[0].exitCode]' --output text
aws logs tail /secop/jobs --log-stream-names score/score/${T##*/} --since 1h | grep -E "RESUMEN|ERROR|SIN PUBLICAR|AVISO"
```

**Volver a la publicación anterior.** El SQL está en [`infra/sql/volver_version_anterior.sql`](infra/sql/volver_version_anterior.sql) y corre en una sola transacción:
- pone `es_actual` en la publicación `publicada` más reciente con `fecha_corte` anterior a la vigente (la de ayer);
- si no hay una, falla sin cambiar nada.

RDS es privada, así que se corre por SSM desde la EC2 `secop-mlflow`, que hay que encender y apagar al terminar. La próxima noche que publique vuelve a mover `es_actual`: para quedarse en la versión anterior, deshabilita antes el Scheduler `secop-nocturno-diario` (consola de EventBridge Scheduler → *Disable*).

```bash
ID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=secop-mlflow --query 'Reservations[0].Instances[0].InstanceId' --output text)
aws ec2 start-instances --instance-ids $ID >/dev/null && aws ec2 wait instance-running --instance-ids $ID
until [ "$(aws ssm describe-instance-information --filters Key=InstanceIds,Values=$ID --query 'InstanceInformationList[0].PingStatus' --output text)" = Online ]; do sleep 10; done
SECRETO=$(aws rds describe-db-instances --db-instance-identifier secop-db --query 'DBInstances[0].MasterUserSecret.SecretArn' --output text)
HOST=$(aws rds describe-db-instances --db-instance-identifier secop-db --query 'DBInstances[0].Endpoint.Address' --output text)
PARAMS=$(python3 - "$SECRETO" "$HOST" infra/sql/volver_version_anterior.sql <<'PY'
import base64, json, sys
secreto, host, ruta = sys.argv[1:]
sql = base64.b64encode(open(ruta, "rb").read()).decode()
py = ("import base64,json,boto3,psycopg2; "
      f"s=json.loads(boto3.client('secretsmanager',region_name='us-east-1').get_secret_value(SecretId='{secreto}')['SecretString']); "
      f"c=psycopg2.connect(host='{host}',dbname='secop',user=s['username'],password=s['password'],sslmode='require'); c.autocommit=True; cur=c.cursor(); "
      f"cur.execute(base64.b64decode('{sql}').decode()); "
      "cur.execute('SELECT version_id, fecha_corte FROM serving.publicaciones WHERE es_actual'); print('vigente ahora:', cur.fetchone())")
print(json.dumps({"commands": [f'/opt/mlflow/venv/bin/python -c "{py}"']}))
PY
)
CID=$(aws ssm send-command --instance-ids $ID --document-name AWS-RunShellScript --parameters "$PARAMS" --query Command.CommandId --output text)
sleep 15; aws ssm get-command-invocation --command-id $CID --instance-id $ID --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
aws ec2 stop-instances --instance-ids $ID >/dev/null && echo "EC2 apagada"
```

**Verificar la API:**

```bash
curl -s ${API}health          # {"estado":"ok", ..., "base_datos":"ok"}
curl -s ${API}metadata | python3 -m json.tool | head -30
K=$(curl -s "${API}empresas?q=sas&limite=1" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["proveedor_key"])')
curl -s ${API}empresas/$K/recomendaciones | python3 -m json.tool | head -40
```

**Métricas de la última noche** (namespace `Secop`):

```bash
for m in procesos_abiertos empresas_puntuadas largo_medio_lista2 duracion_min publicacion \
         unspsc_real_pct unspsc_inferido_pct unspsc_nulo_pct unspsc_modelo_cargado drift_features drift_share; do
  printf '%-22s %s\n' $m "$(aws cloudwatch get-metric-statistics --namespace Secop --metric-name $m \
    --start-time $(date -u -v-30H +%FT%TZ 2>/dev/null || date -u -d '-30 hours' +%FT%TZ) --end-time $(date -u +%FT%TZ) \
    --period 3600 --statistics Maximum --query 'sort_by(Datapoints,&Timestamp)[-1].Maximum' --output text)"
done
aws cloudwatch describe-alarms --alarm-name-prefix secop- --query 'MetricAlarms[].[AlarmName,StateValue]' --output text
```

El reporte de drift de cada noche está en `s3://proyecto-integrador-bucket-01/monitoreo/drift/fecha=<AAAA-MM-DD>/drift.html`, junto con `resumen.json`.

## 3. Apagado después de la sustentación

Hacerlo en este orden: primero lo que lanza trabajos, después lo que los sirve y al final los permisos. El costo es el que sigue corriendo cada mes si el recurso se queda encendido.

| # | Recurso | Costo al mes si se queda | Comando |
|---|---|---|---|
| 1 | Scheduler `secop-nocturno-diario` | Lo que dispara: la noche cuesta ≈ USD 1,56 (Glue ≈ 1,42 + Fargate ≈ 0,14), ≈ **USD 47 al mes** | `aws scheduler delete-schedule --name secop-nocturno-diario` (si solo quieres pausarlo, hazlo en la consola de EventBridge Scheduler → *Disable*: `update-schedule` desde la CLI exige repetir todos los campos y borra la descripción si no se pasa) |
| 2 | Step Functions `secop-nocturno` y la regla `secop-nocturno-fallo` | ≈ USD 0 (transiciones en el nivel gratuito) | `aws events remove-targets --rule secop-nocturno-fallo --ids $(aws events list-targets-by-rule --rule secop-nocturno-fallo --query 'Targets[].Id' --output text) && aws events delete-rule --name secop-nocturno-fallo` · `aws stepfunctions delete-state-machine --state-machine-arn $SM` |
| 3 | Lambda `secop-api` y su Function URL | ≈ USD 0 (se cobra por uso) | `aws lambda delete-function-url-config --function-name secop-api && aws lambda delete-function --function-name secop-api` |
| 4 | RDS `secop-db` (db.t4g.micro, 20 GB gp3) | ≈ **USD 14**. Detenida, ≈ USD 2,30 (solo disco), pero AWS la vuelve a encender sola a los 7 días | Detener: `aws rds stop-db-instance --db-instance-identifier secop-db`. Borrar con snapshot final (≈ USD 0,10 por GB al mes): `aws rds delete-db-instance --db-instance-identifier secop-db --final-db-snapshot-identifier secop-db-final` (sin snapshot: `--skip-final-snapshot`). Con la instancia se borra su secreto `rds!db-…` |
| 5 | EC2 `secop-mlflow` (t4g.small + 8 GB gp3) | Apagada ≈ USD 0,65 (disco); encendida ≈ USD 13 | `aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances --filters Name=tag:Name,Values=secop-mlflow --query 'Reservations[].Instances[].InstanceId' --output text)` |
| 6 | ECR `secop-jobs` y `secop-api` (≈ 7,6 GB de imágenes) | ≈ USD 0,76 | `aws ecr delete-repository --repository-name secop-jobs --force && aws ecr delete-repository --repository-name secop-api --force` (`--force` vacía el repo) |
| 7 | Cluster ECS `secop` y task definitions `secop-score` | USD 0 | `for r in $(aws ecs list-task-definitions --family-prefix secop-score --query 'taskDefinitionArns' --output text); do aws ecs deregister-task-definition --task-definition $r >/dev/null; aws ecs delete-task-definitions --task-definitions $r >/dev/null; done && aws ecs delete-cluster --cluster secop` |
| 8 | Alarmas `secop-*` y métricas propias (namespace `Secop`) | 5 alarmas ≈ USD 0,50 + 11 métricas ≈ USD 3,30 (las métricas dejan de cobrarse cuando nadie las publica) | `aws cloudwatch delete-alarms --alarm-names secop-api-errores secop-procesos-abiertos-bajos secop-sin-publicacion secop-unspsc-sin-modelo secop-unspsc-inferido-bajo` |
| 9 | Tema SNS `secop-alertas` (y su suscripción) | USD 0 | `aws sns delete-topic --topic-arn arn:aws:sns:$AWS_REGION:$AWS_ACCOUNT_ID:secop-alertas` |
| 10 | Secretos `secop/mlflow-db`, `secop/db-job`, `secop/db-api` | ≈ USD 1,20 (USD 0,40 cada uno) | `for s in secop/mlflow-db secop/db-job secop/db-api; do aws secretsmanager delete-secret --secret-id $s --force-delete-without-recovery; done` |
| 11 | Security groups `secop-api-lambda`, `secop-rds`, `secop-mlflow`, `secop-jobs` | USD 0 | Después de borrar RDS, la Lambda y la EC2 (las interfaces de red de la Lambda tardan unos minutos en liberarse): `for g in secop-api-lambda secop-jobs secop-mlflow secop-rds; do aws ec2 delete-security-group --group-id $(aws ec2 describe-security-groups --filters Name=group-name,Values=$g --query 'SecurityGroups[0].GroupId' --output text); done` |
| 12 | Log group `/secop/jobs` (retención de 14 días) | Centavos | `aws logs delete-log-group --log-group-name /secop/jobs` |
| 13 | Tabla de Glue `secop_gold.recomendaciones` (la creó MLOps con la copia diaria de las listas) | USD 0 (solo metadatos; los datos se cobran en S3) | `aws glue delete-table --database-name secop_gold --name recomendaciones`. Borra solo la definición de la tabla: sus datos siguen en `s3://…/gold/recomendaciones/`, y si se borran lo decide el dueño del bucket. La base `secop_gold` es de ingeniería de datos y no se toca |
| 14 | Roles `secop-api-lambda`, `secop-github-ci-lectura`, `secop-github-deploy`, `secop-jobs-ejecucion`, `secop-jobs-tarea`, `secop-mlflow-ec2` (y su instance profile), `secop-nocturno-sfn`, `secop-scheduler-sfn` | USD 0 | Por cada rol: borrar sus políticas en línea (`aws iam list-role-policies` → `delete-role-policy`), quitar las administradas (`list-attached-role-policies` → `detach-role-policy`) y `aws iam delete-role`. Para la EC2: `aws iam remove-role-from-instance-profile --instance-profile-name secop-mlflow-ec2 --role-name secop-mlflow-ec2 && aws iam delete-instance-profile --instance-profile-name secop-mlflow-ec2` |
| 15 | Proveedor OIDC de GitHub (`token.actions.githubusercontent.com`) | USD 0 | Solo si ningún otro proyecto de la cuenta lo usa: `aws iam delete-open-id-connect-provider --open-id-connect-provider-arn arn:aws:iam::$AWS_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com`. En GitHub, deshabilitar el workflow `CI` o borrar la variable `AWS_ACCOUNT_ID` del repo |

**No se borra desde aquí. Es de ingeniería de datos y solo se menciona:**
- el bucket `proyecto-integrador-bucket-01` completo. Ahí también quedaron carpetas que escribió MLOps (`modelos/`, `mlflow/`, `monitoreo/`, `gold/recomendaciones/`, `ci/`, `demo/` y los `part-recientes-*` de bronze): que decida el dueño del bucket;
- las bases de Glue `secop_silver`, `secop_control` y `secop_gold` (salvo la tabla `secop_gold.recomendaciones`, que se borra en el paso 13), y sus jobs `secop-silver-*`;
- la Lambda `proyecto-integrador-ingesta`, sus Schedulers `secop-ingesta-*` y los parámetros de SSM `/secop/app-token`, `/secop/hash-salt` y `/secop/watermark/*`;
- el workgroup de Athena `proyecto-integrador-workgroup` y los presupuestos de la cuenta.

## 4. Qué es normal

| | Normal | Fuera de lo normal |
|---|---|---|
| Duración de la noche | **≈ 60 min**: ingestas ≈ 7, cinco jobs de silver ≈ 26, score ≈ 25 | Más de 90 min: Socrata lento o un job de silver con muchos más datos (los de proponentes y contratos crecen con lo que acumula la Lambda de ingesta). El límite de la Step Functions es de 4 h |
| Procesos abiertos | **≈ 2.000–3.000** | Menos de 300 dispara `secop-procesos-abiertos-bajos` |
| UNSPSC de los abiertos | Real ≈ 0–1 % (la fuente publica `'UNSPECIFIED'`) · **inferido ≈ 30 %** · nulo ≈ 70 % | Inferido < 15,6 % dispara `secop-unspsc-inferido-bajo`. Real que sube de golpe: la fuente volvió a publicar la categoría |
| Score (`secop-score`, 4 vCPU / 30 GB) | **≈ 25 min y ≈ 15 GB de pico de 30** | Más de 25 GB: avisar antes de que no quepa (subir la memoria de la task definition) |
| Empresas con listas | ≈ 35.000–39.000 de ≈ 39.800 jurídicas puntuadas | Una caída grande sin caída de procesos abiertos |
| Drift (reporte diario) | 4 de 6 features con drift, todas con causa conocida: **`dias_ult_part_ok`** sube un día por día porque el historial está congelado al 27-oct-2026; **`unspsc_familia`** (PSI ≈ 2) porque la fuente dejó de publicar la categoría y dos tercios de los abiertos quedan sin familia; `score_retrieval` y `comp_entmod_media` por el cambio de población. `knn_tasa5` y `knn_tasa50` sin drift | Que `knn_tasa5` o `knn_tasa50` pasen de 0,1, o un salto brusco en el PSI de `unspsc_familia` |
