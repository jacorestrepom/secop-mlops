"""Piezas de jobs/score.py que no necesitan la historia completa (la corrida real se prueba en Fargate)."""

import pandas as pd
import pytest

from jobs import score as S


def control(*filas):
    return pd.DataFrame(filas, columns=["job", "tabla", "estado", "cuadre_ok", "fin", "snapshot_id"])


E3 = ("secop-silver-procesos", "secop_silver.procesos")
E6 = ("secop-silver-ganador", "secop_silver.procesos")


def test_frescura_ok_con_e3_y_e6_del_dia_en_hora_de_bogota():
    """Las 02:00 UTC del 12 son las 21:00 del 11 en Bogotá: cuenta para el 11."""
    c = control((*E3, "exito", True, "2026-11-12 01:45:00", 1), (*E6, "exito", True, "2026-11-12 02:00:00", 2))
    ok, _, snap = S.evaluar_frescura(c, "2026-11-11")
    assert ok and snap["procesos_abiertos_snapshot"] == "2"


@pytest.mark.parametrize(
    "filas, motivo",
    [
        ([(*E3, "exito", True, "2026-11-11 15:00:00", 1)], "secop-silver-ganador"),  # falta E6
        ([(*E3, "exito", True, "2026-11-12 06:00:00", 1), (*E6, "exito", True, "2026-11-12 07:00:00", 2)], "sin"),
        ([(*E3, "exito", False, "2026-11-11 15:00:00", 1), (*E6, "exito", True, "2026-11-11 16:00:00", 2)], "sin"),
        ([(*E3, "fallo", True, "2026-11-11 15:00:00", 1), (*E6, "exito", True, "2026-11-11 16:00:00", 2)], "sin"),
        ([(*E3, "exito", True, "2026-11-11 16:00:00", 1), (*E6, "exito", True, "2026-11-11 15:00:00", 2)], "después"),
    ],
)
def test_frescura_falla(filas, motivo):
    ok, texto, _ = S.evaluar_frescura(control(*filas), "2026-11-11")
    assert not ok and motivo in texto


def test_procesos_abiertos_solo_estado_abierto_y_fecha_limite_posterior():
    p = pd.DataFrame(
        {
            "id_proceso": list("abcde"),
            "estado": ["abierto", "abierto", "abierto", "cancelado", "abierto"],
            "fecha_publicacion": pd.to_datetime(["2026-11-01", "2026-11-12", "2026-11-01", "2026-11-01", "2026-11-01"]),
            "fecha_limite_ofertas": pd.to_datetime(["2026-11-20", "2026-11-20", "2026-11-11", "2026-11-20", None]),
        }
    )
    assert list(S.procesos_abiertos(p, "2026-11-11").id_proceso) == ["a"]


def test_unir_procesos_no_toca_los_congelados():
    cong = pd.DataFrame({"id_proceso": ["a", "b"], "x": [1, 2]})
    fres = pd.DataFrame({"id_proceso": ["b", "c"], "x": [99, 3], "extra": [0, 0]})
    u, n = S.unir_procesos(cong, fres)
    assert n == 1 and list(u.id_proceso) == ["a", "b", "c"] and list(u.x) == [1, 2, 3] and "extra" not in u


def test_empresas_solo_juridicas_objetivo():
    prov = pd.DataFrame(
        {
            "proveedor_key": ["j1", "j2", "j3", "n1", "j4"],
            "es_usuario_objetivo": [True, True, True, True, False],
            "tipo_proveedor": ["empresa", "consorcio", "empresa", "persona_natural", "empresa"],
        }
    )
    prop = pd.DataFrame(
        {
            "proveedor_key": ["j1", "n1", "j4", "j3"],
            "fecha_publicacion": pd.to_datetime(["2026-06-01", "2026-06-01", "2026-06-01", "2025-06-01"]),
        }
    )
    con = pd.DataFrame(
        {"proveedor_key": ["j2", "j3", "n1"], "fecha_firma": pd.to_datetime(["2026-05-01", "2026-05-01", "2026-05-01"])}
    )
    cal, fr = S.empresas_a_puntuar(prov, prop, con, "2026-11-11")
    assert cal == ["j1"]  # n1 es natural, j4 no es objetivo
    assert fr == ["j2"]  # j3 participó hace menos de 24 meses


def test_motivos_en_orden_y_maximo_3():
    fila = pd.Series(
        {
            "lista": "oportunidad",
            "entidad_nueva": True,
            "con_misma_entidad": 0.7,
            "comp_entmod_media": 2.1,
            "precio_vs_ganados": -0.3,
            "knn_max_sim_ganado": 0.8,
        }
    )
    assert S.motivos(fila) == [
        "Ya ha trabajado con esta entidad",
        "Entidad con la que no has trabajado",
        "Suele tener pocos competidores",
    ]
    fila = pd.Series(
        {
            "lista": "historial",
            "entidad_nueva": False,
            "con_misma_entidad": 0.0,
            "comp_entmod_media": 5.0,
            "precio_vs_ganados": float("nan"),
            "knn_max_sim_ganado": 0.6,
        }
    )
    assert S.motivos(fila) == ["Parecido a procesos que has ganado"]


def test_sha256_no_depende_del_orden_de_las_filas():
    d = pd.DataFrame(
        {
            "proveedor_key": ["a", "a", "b"],
            "lista": ["historial"] * 3,
            "pos": [1, 2, 1],
            "id_proceso": ["x", "y", "z"],
            "p_gana": [0.1, 0.2, 0.3],
            "s_v2": [1.0, 0.5, 2.0],
        }
    )
    assert S.sha256_listas(d) == S.sha256_listas(d.iloc[::-1])
    assert S.sha256_listas(d) != S.sha256_listas(d.assign(p_gana=[0.1, 0.2, 0.31]))


def test_publicacion_sin_personas_naturales(pg_dsn):
    from jobs.publicar import publicar

    listas = pd.DataFrame(
        {
            "proveedor_key": ["j1", "j1"],
            "id_proceso": ["P1", "P2"],
            "lista": ["historial", "oportunidad"],
            "pos": [1, 1],
            "s_v2": [1.2, 0.4],
            "p_gana": [0.4, 0.2],
            "score_retrieval": [0.9, 0.5],
            "comp_entmod_media": [2.0, 5.0],
            "motivos": [["Suele tener pocos competidores"], []],
            "unspsc_inferido": [False, True],  # P2 llegó 'UNSPECIFIED' y su familia se infirió para el modelo
        }
    )
    frescos = pd.DataFrame(
        {
            "id_proceso": ["P1", "P2"],
            "nombre_entidad": ["Alcaldía X", None],
            "nit_entidad": ["1", "2"],
            "descripcion": ["Compra de sillas", None],
            "descripcion_limpia": ["compra sillas", "obra"],
            "unspsc_familia": ["5612", None],
            "departamento_entidad": ["Antioquia", "Cauca"],
            "ciudad_entidad": ["Bello", None],
            "modalidad": ["Mínima cuantía", "Licitación pública"],
            "precio_base": [1e6, None],
            "fecha_publicacion": pd.to_datetime(["2026-11-01", "2026-11-02"]),
            "fecha_limite_ofertas": pd.to_datetime(["2026-11-20", "2026-11-25"]),
            "url_proceso": ["https://x", None],
        }
    )
    prov = pd.DataFrame(
        {
            "proveedor_key": ["j1", "n1"],
            "tipo_proveedor": ["empresa", "persona_natural"],
            "departamento": ["A", "B"],
            "es_pyme": [True, False],
        }
    )
    nombres = pd.DataFrame({"proveedor_key": ["j1", "n1"], "nombre_proveedor": ["Sillas SAS", "Juan Pérez"]})
    pub = S.armar_publicacion(
        listas,
        frescos,
        prov,
        nombres,
        ["j1", "n1"],
        [],
        "2026-11-11",
        {"retrieval": "r v1", "pgana": "p v1", "ranking": "k v1"},
        {},
        {},
    )
    assert [e["proveedor_key"] for e in pub.empresas] == ["j1"]
    assert {p["entidad"] for p in pub.procesos} == {"Alcaldía X", "Entidad 2"}
    assert not pub.datos_simulados
    # La tarjeta usa solo el UNSPSC real: P2 sale sin categoría aunque el modelo haya usado una inferida
    tarjeta = {p["id_proceso"]: p for p in pub.procesos}
    assert tarjeta["P1"]["unspsc_familia"] == "5612" and tarjeta["P2"]["unspsc_familia"] is None
    assert all(p["categoria"] is None for p in pub.procesos)
    v = publicar(pg_dsn, pub)
    assert v > 0
    import psycopg2

    with psycopg2.connect(pg_dsn) as con, con.cursor() as cur:
        cur.execute("SELECT id_proceso, unspsc_inferido FROM serving.recomendaciones WHERE version_id = %s", (v,))
        assert dict(cur.fetchall()) == {"P1": False, "P2": True}


def test_metricas_de_volumen_en_cloudwatch():
    class Falso:
        def put_metric_data(self, **kw):
            self.kw = kw

    c = Falso()
    S.publicar_metricas({"empresas_puntuadas": 29994, "procesos_abiertos": 436, "duracion_min": 21.2}, cliente=c)
    assert c.kw["Namespace"] == "Secop"
    assert {d["MetricName"]: d["Value"] for d in c.kw["MetricData"]} == {
        "empresas_puntuadas": 29994.0,
        "procesos_abiertos": 436.0,
        "duracion_min": 21.2,
    }


def test_reporte_drift_detecta_la_feature_que_se_movio():
    pytest.importorskip("evidently")
    import numpy as np

    rng = np.random.default_rng(0)

    def datos(desplazamiento):
        n = 3000
        return pd.DataFrame(
            {
                "score_retrieval": rng.normal(desplazamiento, 1, n),
                "knn_tasa50": rng.uniform(0, 1, n),
                "knn_tasa5": rng.uniform(0, 1, n),
                "dias_ult_part_ok": rng.exponential(30, n),
                "comp_entmod_media": rng.normal(3, 1, n),
                "unspsc_familia": rng.choice(["4312", "8011", "7210"], n),
            }
        )

    _, r = S.reporte_drift(datos(3.0), datos(0.0))
    assert r["drift_features"] == 1 and r["features"]["score_retrieval"]["distancia"] > 0.1
    assert set(r["features"]) == set(S.COLS_DRIFT)
