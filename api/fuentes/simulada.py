"""Origen simulado: lee api/datos_simulados.json (datos inventados con la forma exacta del contrato).
Sirve para que la web avance antes de que exista RDS y como plan B de la demo."""

import json
import unicodedata
from pathlib import Path

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

RUTA_DATOS = Path(__file__).resolve().parent.parent / "datos_simulados.json"


def _normalizar(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()


class FuenteSimulada:
    nombre = "simulada"

    def __init__(self, ruta: Path = RUTA_DATOS):
        d = json.loads(ruta.read_text(encoding="utf-8"))
        self._fecha_corte = d["fecha_corte"]
        self._ultima_publicacion = d["ultima_publicacion"]
        self._versiones = VersionModelos(**d["version_modelos"])
        self._empresas = {e["proveedor_key"]: Empresa(**e) for e in d["empresas"]}
        self._procesos = {p["id_proceso"]: p for p in d["procesos"]}
        self._recs = d["recomendaciones"]
        self._estadisticas = {k: Estadistica(**v) for k, v in d["estadisticas"].items()}
        self._metricas = d["metricas"]

    def salud_base(self) -> str:
        return "no_aplica"

    def buscar_empresas(self, q: str, limite: int) -> list[Empresa]:
        qn = _normalizar(q.strip())
        return [e for e in self._empresas.values() if qn in _normalizar(e.nombre) or qn in e.proveedor_key][:limite]

    def _tarjeta(self, r: dict) -> ProcesoRecomendado:
        p = self._procesos[r["id_proceso"]]
        return ProcesoRecomendado(
            posicion=r["posicion"],
            id_proceso=p["id_proceso"],
            entidad=p["entidad"],
            objeto=p["objeto"],
            categoria=p.get("categoria"),
            departamento=p.get("departamento"),
            municipio=p.get("municipio"),
            precio_base=p.get("precio_base"),
            fecha_limite_ofertas=p.get("fecha_limite_ofertas"),
            probabilidad=r["probabilidad"],
            oferentes_esperados=p.get("oferentes_esperados"),
            motivos=r["motivos"],
            url_proceso=p.get("url_proceso"),
        )

    def recomendaciones(self, proveedor_key: str) -> Recomendaciones:
        recs = self._recs.get(proveedor_key, {"historial": [], "oportunidad": []})
        empresa = self._empresas.get(proveedor_key)
        return Recomendaciones(
            proveedor_key=proveedor_key,
            fecha_corte=self._fecha_corte,
            tiene_historial=bool(empresa and empresa.tiene_historial),
            version_modelos=self._versiones,
            listas=Listas(
                historial=Lista(titulo=TITULO_HISTORIAL, procesos=[self._tarjeta(r) for r in recs["historial"]]),
                oportunidad=Lista(titulo=TITULO_OPORTUNIDAD, procesos=[self._tarjeta(r) for r in recs["oportunidad"]]),
            ),
        )

    def proceso(self, id_proceso: str) -> Proceso | None:
        p = self._procesos.get(id_proceso)
        return Proceso(**p) if p else None

    def procesos_abiertos(self, familia: str | None, departamento: str | None, limite: int) -> list[Proceso]:
        out = []
        for p in self._procesos.values():
            if familia and p.get("unspsc_familia") != familia:
                continue
            if departamento and _normalizar(p.get("departamento") or "") != _normalizar(departamento):
                continue
            out.append(Proceso(**p))
        return out[:limite]

    def estadistica(self, indicador: str) -> Estadistica | None:
        return self._estadisticas.get(indicador)

    def indicadores(self) -> list[str]:
        return sorted(self._estadisticas)

    def metadata(self) -> Metadata:
        return Metadata(
            fecha_datos=self._fecha_corte,
            ultima_publicacion=self._ultima_publicacion,
            version_modelos=self._versiones,
            metricas=self._metricas,
            datos_simulados=True,
        )
