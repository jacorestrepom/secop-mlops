"""La API con FUENTE_DATOS=rds contra un PostgreSQL real: mismo contrato que la fuente simulada."""

import copy

import pytest
from fastapi.testclient import TestClient

from api.fuentes.simulada import FuenteSimulada
from api.main import app, fuente
from jobs import publicar as P
from tests.conftest import SIMULADOS

RUTAS_IGUALES = [
    "/empresas/900123456/recomendaciones",
    "/empresas/902999888/recomendaciones",  # lista de oportunidades corta
    "/empresas/903111222/recomendaciones",  # sin historial
    "/empresas/000000000/recomendaciones",  # no existe
    "/empresas?q=andinos",
    "/empresas?q=tecnologia",
    "/empresas?q=9012",
    "/estadisticas",
    "/estadisticas/procesos_por_departamento",
]


@pytest.fixture
def rds(pg_dsn):
    from api.fuentes.rds import FuenteRDS

    return FuenteRDS(pg_dsn)


def cliente_con(f):
    app.dependency_overrides[fuente] = lambda: f
    return TestClient(app)


@pytest.fixture(autouse=True)
def limpiar():
    yield
    app.dependency_overrides.clear()


def test_sin_publicacion_responde_503_y_health_ok(rds):
    c = cliente_con(rds)
    assert c.get("/health").json() == {"estado": "ok", "fuente_datos": "rds", "base_datos": "ok"}
    assert c.get("/empresas/900123456/recomendaciones").status_code == 503
    assert c.get("/metadata").status_code == 503


def test_misma_respuesta_que_la_fuente_simulada(pg_dsn, rds):
    P.publicar(pg_dsn, P.desde_simulados(str(SIMULADOS)))
    c_rds, c_sim = cliente_con(rds), None
    respuestas_rds = {r: c_rds.get(r).json() for r in RUTAS_IGUALES}
    c_sim = cliente_con(FuenteSimulada())
    for r in RUTAS_IGUALES:
        assert respuestas_rds[r] == c_sim.get(r).json(), r


def test_proceso_y_abiertos(pg_dsn, rds):
    P.publicar(pg_dsn, P.desde_simulados(str(SIMULADOS)))
    c = cliente_con(rds)
    rec = c.get("/empresas/900123456/recomendaciones").json()
    id_ = rec["listas"]["historial"]["procesos"][0]["id_proceso"]
    assert c.get(f"/procesos/{id_}").json() == cliente_con(FuenteSimulada()).get(f"/procesos/{id_}").json()
    c = cliente_con(rds)
    assert c.get("/procesos/NO.EXISTE").status_code == 404
    ant = c.get("/procesos/abiertos", params={"departamento": "antioquia"}).json()
    assert ant and all(p["departamento"] == "Antioquia" for p in ant)


def test_metadata_desde_publicaciones(pg_dsn, rds):
    P.publicar(pg_dsn, P.desde_simulados(str(SIMULADOS)))
    m = cliente_con(rds).get("/metadata").json()
    s = cliente_con(FuenteSimulada()).get("/metadata").json()
    assert m["datos_simulados"] is True and m["fecha_datos"] == "2026-11-10"
    assert m["version_modelos"] == s["version_modelos"] and m["metricas"] == s["metricas"]
    assert m["ultima_publicacion"]  # la pone la base al publicar


def test_solo_se_ve_la_version_actual(pg_dsn, rds):
    pub = P.desde_simulados(str(SIMULADOS))
    P.publicar(pg_dsn, pub)
    nueva = copy.deepcopy(pub)
    nueva.fecha_corte = "2026-11-11"
    for r in nueva.recomendaciones:
        r["probabilidad"] = 0.5
    P.publicar(pg_dsn, nueva)
    d = cliente_con(rds).get("/empresas/900123456/recomendaciones").json()
    assert d["fecha_corte"] == "2026-11-11"
    assert {p["probabilidad"] for p in d["listas"]["historial"]["procesos"]} == {0.5}


def test_health_degradado_si_la_base_no_responde():
    from api.fuentes.rds import FuenteRDS

    caida = FuenteRDS("postgresql://nadie:x@127.0.0.1:1/secop?connect_timeout=1")
    assert cliente_con(caida).get("/health").json() == {
        "estado": "degradado",
        "fuente_datos": "rds",
        "base_datos": "error",
    }
