"""Base PostgreSQL para las pruebas de publicación y de la fuente RDS.

Se activa con TEST_DATABASE_URL (en CI lo pone el servicio postgres del workflow). En local:
    docker run -d --name secop-pg-prueba -e POSTGRES_PASSWORD=prueba -e POSTGRES_DB=secop -p 55432:5432 postgres:16
    export TEST_DATABASE_URL=postgresql://postgres:prueba@localhost:55432/secop
"""

import os
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
SQL_SERVING = RAIZ / "infra" / "sql" / "serving.sql"
SIMULADOS = RAIZ / "api" / "datos_simulados.json"


@pytest.fixture
def pg_dsn():
    """DSN de una base con el esquema serving recién creado (vacío)."""
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL no está definido")
    psycopg2 = pytest.importorskip("psycopg2")
    con = psycopg2.connect(dsn)
    con.autocommit = True
    with con.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS serving CASCADE")
        cur.execute(SQL_SERVING.read_text())
    con.close()
    return dsn
