#!/bin/bash
# Plan B de la demo · baja el volcado más reciente de s3://…/demo/ (o usa el que se le pase) y levanta
# API + PostgreSQL en local con esos datos. Desde la raíz del repo:
#   ./infra/cargar_demo.sh                       # el más reciente de S3
#   ./infra/cargar_demo.sh serving_2026-11-12.sql.gz
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p demo
if [ $# -ge 1 ]; then
  cp "$1" demo/serving.sql.gz
else
  ULTIMO=$(aws s3 ls s3://proyecto-integrador-bucket-01/demo/ --profile "${AWS_PROFILE_SECOP:-secop}" \
           | awk '/serving_.*\.sql\.gz/ {print $4}' | sort | tail -1)
  [ -n "$ULTIMO" ] || { echo "no hay volcados en s3://…/demo/ (correr infra/exportar_demo.sh)"; exit 1; }
  aws s3 cp "s3://proyecto-integrador-bucket-01/demo/$ULTIMO" demo/serving.sql.gz \
      --profile "${AWS_PROFILE_SECOP:-secop}" --only-show-errors
  echo "volcado: $ULTIMO"
fi
# El contenedor de PostgreSQL carga demo/serving.sql.gz solo al crear su volumen: se reinicia para recargar
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
docker compose up -d --build
for _ in $(seq 1 90); do
  curl -sf http://localhost:8000/health | grep -q '"base_datos":"ok"' && break
  sleep 2
done
curl -s http://localhost:8000/health; echo
curl -s http://localhost:8000/metadata | python3 -c 'import json,sys; d=json.load(sys.stdin); print("datos de", d["fecha_datos"], "| datos_simulados", d["datos_simulados"])'
echo "API: http://localhost:8000/docs"
