"""Publicación atómica (jobs/publicar.py) contra un PostgreSQL real."""

import copy

import psycopg2
import pytest

from jobs import publicar as P
from tests.conftest import SIMULADOS


def estado(dsn):
    with psycopg2.connect(dsn) as con, con.cursor() as cur:
        cur.execute("SELECT version_id, fecha_corte::text, estado, es_actual FROM serving.publicaciones ORDER BY 1")
        filas = cur.fetchall()
        cur.execute("SELECT version_id, count(*) FROM serving.recomendaciones GROUP BY 1 ORDER BY 1")
        recs = dict(cur.fetchall())
    con.close()
    return filas, recs


def actual(dsn):
    return [f for f in estado(dsn)[0] if f[3]]


@pytest.fixture
def pub():
    return P.desde_simulados(str(SIMULADOS))


def test_publica_y_queda_actual(pg_dsn, pub):
    v = P.publicar(pg_dsn, pub)
    filas, recs = estado(pg_dsn)
    assert filas == [(v, "2026-11-10", "publicada", True)]
    assert recs[v] == len(pub.recomendaciones) > 0


def test_falla_en_la_carga_no_cambia_la_actual(pg_dsn, pub):
    v1 = P.publicar(pg_dsn, pub)
    malo = copy.deepcopy(pub)
    malo.fecha_corte = "2026-11-11"
    malo.recomendaciones[5]["probabilidad"] = 1.7  # viola el CHECK a mitad del COPY
    with pytest.raises(P.PublicacionFallida):
        P.publicar(pg_dsn, malo)
    filas, recs = estado(pg_dsn)
    assert actual(pg_dsn) == [(v1, "2026-11-10", "publicada", True)]
    assert filas[-1][2:] == ("fallida", False)
    assert set(recs) == {v1}  # la carga fallida no dejó filas


def test_falla_al_activar_no_cambia_la_actual(pg_dsn, pub, monkeypatch):
    v1 = P.publicar(pg_dsn, pub)

    def explota(cur, version_id):
        raise RuntimeError("se cayó justo antes del cambio de versión")

    monkeypatch.setattr(P, "_antes_de_activar", explota)
    otro = copy.deepcopy(pub)
    otro.fecha_corte = "2026-11-11"
    with pytest.raises(P.PublicacionFallida, match="se cayó"):
        P.publicar(pg_dsn, otro)
    assert actual(pg_dsn) == [(v1, "2026-11-10", "publicada", True)]
    assert estado(pg_dsn)[0][-1][2:] == ("fallida", False)


def test_main_sale_con_codigo_distinto_de_cero(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    monkeypatch.setattr(P, "_antes_de_activar", lambda cur, v: (_ for _ in ()).throw(RuntimeError("falla")))
    assert P.main(["--simulados", str(SIMULADOS)]) == 1
    monkeypatch.setattr(P, "_antes_de_activar", lambda cur, v: None)
    assert P.main(["--simulados", str(SIMULADOS)]) == 0


def test_misma_fecha_reemplaza_sin_duplicar(pg_dsn, pub):
    v1 = P.publicar(pg_dsn, pub)
    v2 = P.publicar(pg_dsn, pub)
    filas, recs = estado(pg_dsn)
    assert v2 > v1
    assert filas == [(v2, "2026-11-10", "publicada", True)]
    assert set(recs) == {v2}


def test_retiene_solo_las_ultimas_7(pg_dsn, pub):
    versiones = []
    for dia in range(1, 11):
        p = copy.deepcopy(pub)
        p.fecha_corte = f"2026-11-{dia:02d}"
        versiones.append(P.publicar(pg_dsn, p))
    filas, recs = estado(pg_dsn)
    assert [f[0] for f in filas] == versiones[-7:]
    assert set(recs) == set(versiones[-7:])  # las filas de las viejas se borran en cascada
    assert actual(pg_dsn) == [(versiones[-1], "2026-11-10", "publicada", True)]


def test_la_base_impide_dos_versiones_actuales(pg_dsn, pub):
    P.publicar(pg_dsn, pub)
    otra = copy.deepcopy(pub)
    otra.fecha_corte = "2026-11-11"
    P.publicar(pg_dsn, otra)
    with psycopg2.connect(pg_dsn) as con, con.cursor() as cur, pytest.raises(psycopg2.errors.UniqueViolation):
        cur.execute("UPDATE serving.publicaciones SET es_actual = true")
    con.close()
    assert len(actual(pg_dsn)) == 1
