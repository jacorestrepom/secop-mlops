# Imagen secop-api. Se construye desde la raíz del repo:
#   docker build --platform linux/amd64 --provenance=false -f api/Dockerfile -t secop-api .
# La misma imagen corre en Docker (docker-compose) y en AWS Lambda: la extensión Lambda Web Adapter
# solo se activa dentro de Lambda y le pasa las peticiones a uvicorn por el puerto 8000.
FROM python:3.11-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AWS_LWA_PORT=8000 AWS_LWA_READINESS_CHECK_PATH=/health
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ api/

RUN useradd --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
