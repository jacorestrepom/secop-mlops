"""Origen RDS: lee el esquema serving de PostgreSQL, siempre la publicación vigente (es_actual).

Conexión por variables de entorno: DATABASE_URL, o DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
(DB_SSLMODE, por defecto require). El pool (Conexiones) se crea una sola vez por proceso: la fuente se cachea
en main.py.
"""

import os
import threading
import time
import unicodedata
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

from api.esquemas import (
    TITULO_HISTORIAL,
    TITULO_OPORTUNIDAD,
    Empresa,
    Estadistica,
    Lista,
    Listas,
    Metadata,
    Proceso,
    ProcesoRecomendado,
    Recomendaciones,
    VersionModelos,
)
from api.fuentes.base import SinPublicacion

ACTUAL = "(SELECT version_id FROM serving.publicaciones WHERE es_actual)"
SIN_TILDES = "translate(lower({}), 'áéíóúüñ', 'aeiouun')"


def _normalizar(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower().strip()


def dsn_desde_entorno() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    return psycopg2.extensions.make_dsn(
        host=os.environ["DB_HOST"],
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "secop"),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        sslmode=os.getenv("DB_SSLMODE", "require"),
        connect_timeout=5,
        application_name="secop-api",
    )


class Conexiones:
    """Pool pequeño: abre conexiones solo cuando hacen falta (la API arranca aunque la base esté caída) y las
    guarda para reutilizarlas. Si una lleva más de PING_TRAS segundos sin usarse (p. ej. la Lambda estuvo
    congelada), la comprueba antes de entregarla y la cambia si murió."""

    PING_TRAS = 30.0

    def __init__(self, dsn: str, maximo: int = 4):
        self._dsn, self._maximo = dsn, maximo
        self._libres: list[tuple[object, float]] = []  # (conexión, último uso)
        self._lock = threading.Lock()

    def _nueva(self):
        con = psycopg2.connect(self._dsn)
        con.autocommit = True
        return con

    def tomar(self):
        while True:
            with self._lock:
                con, ultimo = self._libres.pop() if self._libres else (None, 0.0)
            if con is None:
                return self._nueva()
            if con.closed:
                continue
            if time.monotonic() - ultimo < self.PING_TRAS:
                return con
            try:
                with con.cursor() as cur:
                    cur.execute("SELECT 1")
                return con
            except psycopg2.Error:
                con.close()

    def devolver(self, con, rota: bool = False) -> None:
        if rota or con.closed:
            con.close()
            return
        with self._lock:
            if len(self._libres) < self._maximo:
                self._libres.append((con, time.monotonic()))
                return
        con.close()


class FuenteRDS:
    nombre = "rds"

    def __init__(self, dsn: str | None = None, maxconn: int = 4):
        self._pool = Conexiones(dsn or dsn_desde_entorno(), maxconn)

    @contextmanager
    def _cursor(self):
        con = self._pool.tomar()
        rota = False
        try:
            with con.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            rota = True
            raise
        finally:
            self._pool.devolver(con, rota)

    def _consulta(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def salud_base(self) -> str:
        try:
            self._consulta("SELECT 1")
            return "ok"
        except psycopg2.Error:
            return "error"

    def _publicacion(self) -> dict:
        filas = self._consulta(
            """SELECT version_id, fecha_corte, publicada_en, version_retrieval, version_pgana, version_ranking,
                      metricas, datos_simulados
               FROM serving.publicaciones WHERE es_actual"""
        )
        if not filas:
            raise SinPublicacion("todavía no hay una publicación vigente")
        return filas[0]

    @staticmethod
    def _versiones(p: dict) -> VersionModelos:
        return VersionModelos(retrieval=p["version_retrieval"], pgana=p["version_pgana"], ranking=p["version_ranking"])

    def buscar_empresas(self, q: str, limite: int) -> list[Empresa]:
        qn = f"%{_normalizar(q)}%"
        filas = self._consulta(
            f"""SELECT proveedor_key, nombre, departamento, es_pyme, tiene_historial FROM serving.empresas
                WHERE version_id = {ACTUAL} AND ({SIN_TILDES.format("nombre")} LIKE %s OR proveedor_key LIKE %s)
                ORDER BY nombre LIMIT %s""",
            (qn, qn, limite),
        )
        return [Empresa(**f) for f in filas]

    def recomendaciones(self, proveedor_key: str) -> Recomendaciones:
        p = self._publicacion()
        emp = self._consulta(
            "SELECT tiene_historial FROM serving.empresas WHERE version_id = %s AND proveedor_key = %s",
            (p["version_id"], proveedor_key),
        )
        filas = self._consulta(
            """SELECT r.lista, r.posicion, r.id_proceso, r.probabilidad, r.motivos,
                      coalesce(r.oferentes_esperados, pr.oferentes_esperados) AS oferentes_esperados,
                      pr.entidad, pr.objeto, pr.categoria, pr.departamento, pr.municipio, pr.precio_base,
                      pr.fecha_limite_ofertas, pr.url_proceso
               FROM serving.recomendaciones r
               JOIN serving.procesos pr ON pr.version_id = r.version_id AND pr.id_proceso = r.id_proceso
               WHERE r.version_id = %s AND r.proveedor_key = %s
               ORDER BY r.lista, r.posicion""",
            (p["version_id"], proveedor_key),
        )
        por_lista = {"historial": [], "oportunidad": []}
        for f in filas:
            lista = f.pop("lista")
            f["probabilidad"] = float(f["probabilidad"])
            for c in ("precio_base", "oferentes_esperados"):
                f[c] = None if f[c] is None else float(f[c])
            por_lista[lista].append(ProcesoRecomendado(**f))
        return Recomendaciones(
            proveedor_key=proveedor_key,
            fecha_corte=p["fecha_corte"],
            tiene_historial=bool(emp and emp[0]["tiene_historial"]),
            version_modelos=self._versiones(p),
            listas=Listas(
                historial=Lista(titulo=TITULO_HISTORIAL, procesos=por_lista["historial"]),
                oportunidad=Lista(titulo=TITULO_OPORTUNIDAD, procesos=por_lista["oportunidad"]),
            ),
        )

    _COLS_PROCESO = """id_proceso, entidad, objeto, categoria, unspsc_familia, departamento, municipio, modalidad,
                       precio_base, fecha_publicacion, fecha_limite_ofertas, url_proceso"""

    @staticmethod
    def _proceso(f: dict) -> Proceso:
        f["precio_base"] = None if f["precio_base"] is None else float(f["precio_base"])
        return Proceso(**f)

    def proceso(self, id_proceso: str) -> Proceso | None:
        filas = self._consulta(
            f"SELECT {self._COLS_PROCESO} FROM serving.procesos WHERE version_id = {ACTUAL} AND id_proceso = %s",
            (id_proceso,),
        )
        return self._proceso(filas[0]) if filas else None

    def procesos_abiertos(self, familia: str | None, departamento: str | None, limite: int) -> list[Proceso]:
        filas = self._consulta(
            f"""SELECT {self._COLS_PROCESO} FROM serving.procesos
                WHERE version_id = {ACTUAL}
                  AND (fecha_limite_ofertas IS NULL
                       OR fecha_limite_ofertas >= (SELECT fecha_corte FROM serving.publicaciones WHERE es_actual))
                  AND (%s::text IS NULL OR unspsc_familia = %s)
                  AND (%s::text IS NULL OR {SIN_TILDES.format("departamento")} = %s)
                ORDER BY fecha_limite_ofertas NULLS LAST, id_proceso LIMIT %s""",
            (familia, familia, departamento, departamento and _normalizar(departamento), limite),
        )
        return [self._proceso(f) for f in filas]

    def estadistica(self, indicador: str) -> Estadistica | None:
        filas = self._consulta(
            f"""SELECT indicador, descripcion, unidad, datos FROM serving.estadisticas
                WHERE version_id = {ACTUAL} AND indicador = %s""",
            (indicador,),
        )
        return Estadistica(**filas[0]) if filas else None

    def indicadores(self) -> list[str]:
        filas = self._consulta(
            f"SELECT indicador FROM serving.estadisticas WHERE version_id = {ACTUAL} ORDER BY indicador"
        )
        return [f["indicador"] for f in filas]

    def metadata(self) -> Metadata:
        p = self._publicacion()
        return Metadata(
            fecha_datos=p["fecha_corte"],
            ultima_publicacion=p["publicada_en"],
            version_modelos=self._versiones(p),
            metricas=p["metricas"],
            datos_simulados=p["datos_simulados"],
        )
