#!/bin/bash
# Plan B de la demo · exporta la publicación VIGENTE del esquema serving a
#   s3://proyecto-integrador-bucket-01/demo/serving_<fecha_corte>.sql.gz
# Corre desde el portátil: enciende la EC2 secop-mlflow, ejecuta el volcado allá por SSM (es la única máquina
# que llega a RDS) y la apaga al terminar, aunque algo falle.
#   ./infra/exportar_demo.sh
set -euo pipefail
PERFIL=${AWS_PROFILE_SECOP:-secop}
P=(--profile "$PERFIL")
ID=$(aws ec2 describe-instances "${P[@]}" --filters Name=tag:Name,Values=secop-mlflow \
     --query 'Reservations[0].Instances[0].InstanceId' --output text)

apagar() { aws ec2 stop-instances "${P[@]}" --instance-ids "$ID" >/dev/null && echo "EC2 $ID apagándose"; }
trap apagar EXIT

aws ec2 start-instances "${P[@]}" --instance-ids "$ID" >/dev/null
aws ec2 wait instance-running "${P[@]}" --instance-ids "$ID"
for _ in $(seq 1 30); do
  [ "$(aws ssm describe-instance-information "${P[@]}" --filters Key=InstanceIds,Values="$ID" \
       --query 'InstanceInformationList[0].PingStatus' --output text)" = Online ] && break
  sleep 10
done

# Script que corre en la EC2 (usuario job_escritura, secreto secop/db-job)
REMOTO=$(cat <<'EOS'
set -euo pipefail
command -v pg_dump >/dev/null || dnf install -y -q postgresql16 >/dev/null
eval "$(/opt/mlflow/venv/bin/python - <<'PY'
import json, shlex, boto3
s = json.loads(boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId="secop/db-job")["SecretString"])
for k, v in {"PGHOST": s["host"], "PGPORT": s["port"], "PGDATABASE": s["dbname"], "PGUSER": s["username"],
             "PGPASSWORD": s["password"], "PGSSLMODE": "require"}.items():
    print(f"export {k}={shlex.quote(str(v))}")
PY
)"
V=$(psql -Atc "SELECT version_id FROM serving.publicaciones WHERE es_actual")
F=$(psql -Atc "SELECT fecha_corte FROM serving.publicaciones WHERE es_actual")
[ -n "$V" ] || { echo "no hay publicación vigente"; exit 1; }
OUT=/tmp/serving_${F}.sql
{
  echo "-- Plan B de la demo: publicación vigente $V (fecha_corte $F) del esquema serving"
  pg_dump --section=pre-data --schema=serving --no-owner --no-privileges
  for T in publicaciones empresas procesos recomendaciones estadisticas; do
    COLS=$(psql -Atc "SELECT string_agg(quote_ident(column_name), ', ' ORDER BY ordinal_position)
                      FROM information_schema.columns WHERE table_schema='serving' AND table_name='$T'")
    echo "COPY serving.$T ($COLS) FROM stdin;"
    psql -Atc "\\copy (SELECT $COLS FROM serving.$T WHERE version_id = $V) TO STDOUT"
    echo '\.'
  done
  echo "SELECT setval('serving.publicaciones_version_id_seq', $V);"
  pg_dump --section=post-data --schema=serving --no-owner --no-privileges
} > "$OUT"
gzip -f "$OUT"
aws s3 cp "$OUT.gz" "s3://proyecto-integrador-bucket-01/demo/serving_${F}.sql.gz" --only-show-errors
echo "versión $V · fecha_corte $F · $(stat -c %s "$OUT.gz") bytes → s3://proyecto-integrador-bucket-01/demo/serving_${F}.sql.gz"
for T in publicaciones empresas procesos recomendaciones; do
  echo "  $T: $(psql -Atc "SELECT count(*) FROM serving.$T WHERE version_id = $V") filas"
done
rm -f "$OUT.gz"
EOS
)
PARAMS=$(python3 -I -c 'import json,sys; print(json.dumps({"commands": sys.stdin.read().splitlines(), "executionTimeout": ["1800"]}))' <<< "$REMOTO")
CID=$(aws ssm send-command "${P[@]}" --instance-ids "$ID" --document-name AWS-RunShellScript \
      --parameters "$PARAMS" --query Command.CommandId --output text)
while :; do
  ST=$(aws ssm get-command-invocation "${P[@]}" --command-id "$CID" --instance-id "$ID" --query Status --output text 2>/dev/null || echo Pending)
  case "$ST" in Pending|InProgress|Delayed) sleep 5;; *) break;; esac
done
aws ssm get-command-invocation "${P[@]}" --command-id "$CID" --instance-id "$ID" \
    --query '[StandardOutputContent,StandardErrorContent]' --output text
[ "$ST" = Success ] || { echo "el volcado falló ($ST)"; exit 1; }
