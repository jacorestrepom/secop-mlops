"""Pruebas del contrato de la API con la fuente simulada."""

import pytest
from fastapi.testclient import TestClient

from api.esquemas import TITULO_HISTORIAL, TITULO_OPORTUNIDAD
from api.main import app

CAMPOS_TARJETA = {
    "posicion",
    "id_proceso",
    "entidad",
    "objeto",
    "categoria",
    "departamento",
    "municipio",
    "precio_base",
    "fecha_limite_ofertas",
    "probabilidad",
    "oferentes_esperados",
    "motivos",
    "url_proceso",
}


@pytest.fixture(scope="module")
def cliente():
    return TestClient(app)


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "fuente_datos": "simulada", "base_datos": "no_aplica"}


def test_recomendaciones_contrato(cliente):
    r = cliente.get("/empresas/900123456/recomendaciones")
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"proveedor_key", "fecha_corte", "tiene_historial", "version_modelos", "listas"}
    assert set(d["version_modelos"]) == {"retrieval", "pgana", "ranking"}
    assert d["tiene_historial"] is True
    hist, opor = d["listas"]["historial"], d["listas"]["oportunidad"]
    assert hist["titulo"] == TITULO_HISTORIAL and opor["titulo"] == TITULO_OPORTUNIDAD
    assert len(hist["procesos"]) == 10 and len(opor["procesos"]) <= 10
    for lista in (hist, opor):
        assert [p["posicion"] for p in lista["procesos"]] == list(range(1, len(lista["procesos"]) + 1))
        for p in lista["procesos"]:
            assert set(p) == CAMPOS_TARJETA
            assert 0 <= p["probabilidad"] <= 1
    # Ningún proceso aparece en las dos listas
    ids_h = {p["id_proceso"] for p in hist["procesos"]}
    assert not ids_h & {p["id_proceso"] for p in opor["procesos"]}


def test_lista_oportunidad_corta_no_es_error(cliente):
    d = cliente.get("/empresas/902999888/recomendaciones").json()
    assert 0 < len(d["listas"]["oportunidad"]["procesos"]) < 10


@pytest.mark.parametrize("key", ["903111222", "000000000"])
def test_sin_historial_devuelve_listas_vacias(cliente, key):
    r = cliente.get(f"/empresas/{key}/recomendaciones")
    assert r.status_code == 200
    d = r.json()
    assert d["tiene_historial"] is False
    assert d["listas"]["historial"]["procesos"] == [] and d["listas"]["oportunidad"]["procesos"] == []


def test_buscar_empresas(cliente):
    assert [e["proveedor_key"] for e in cliente.get("/empresas", params={"q": "andinos"}).json()] == ["900123456"]
    assert len(cliente.get("/empresas", params={"q": "9012"}).json()) == 1
    assert cliente.get("/empresas", params={"q": "tecnologia"}).json()[0]["nombre"].startswith("Tecnología")
    assert cliente.get("/empresas", params={"q": "x"}).status_code == 422


def test_proceso(cliente):
    id_ = cliente.get("/empresas/900123456/recomendaciones").json()["listas"]["historial"]["procesos"][0]["id_proceso"]
    assert cliente.get(f"/procesos/{id_}").json()["id_proceso"] == id_
    assert cliente.get("/procesos/NO.EXISTE").status_code == 404


def test_procesos_abiertos_filtra(cliente):
    todos = cliente.get("/procesos/abiertos").json()
    ant = cliente.get("/procesos/abiertos", params={"departamento": "antioquia"}).json()
    assert 0 < len(ant) < len(todos)
    assert all(p["departamento"] == "Antioquia" for p in ant)


def test_estadisticas(cliente):
    indicadores = cliente.get("/estadisticas").json()
    assert "procesos_por_departamento" in indicadores
    e = cliente.get(f"/estadisticas/{indicadores[0]}").json()
    assert e["datos"] and {"etiqueta", "valor"} == set(e["datos"][0])
    assert cliente.get("/estadisticas/no_existe").status_code == 404


def test_metadata(cliente):
    m = cliente.get("/metadata").json()
    assert m["datos_simulados"] is True
    nombres = {x["nombre"] for x in m["metricas"]["lista_historial"]}
    assert {"Precision@10", "Recall@10", "NDCG@10"} <= nombres
    assert len(m["metricas"]["calibracion_pgana"]) == 10


def test_cors_solo_dominio_web(cliente):
    ok = cliente.get("/health", headers={"Origin": "http://localhost:5173"})
    otro = cliente.get("/health", headers={"Origin": "https://malicioso.example"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in otro.headers
