"""Interfaz que debe cumplir cualquier origen de datos de la API (simulado hoy, RDS después).
Las rutas solo hablan con esta interfaz: cambiar de origen es cambiar FUENTE_DATOS."""

from typing import Protocol

from api.esquemas import Empresa, Estadistica, Metadata, Proceso, Recomendaciones


class FuenteDatos(Protocol):
    nombre: str

    def salud_base(self) -> str:
        """ "ok", "error" o "no_aplica"."""
        ...

    def buscar_empresas(self, q: str, limite: int) -> list[Empresa]: ...

    def recomendaciones(self, proveedor_key: str) -> Recomendaciones: ...

    def proceso(self, id_proceso: str) -> Proceso | None: ...

    def procesos_abiertos(self, familia: str | None, departamento: str | None, limite: int) -> list[Proceso]: ...

    def estadistica(self, indicador: str) -> Estadistica | None: ...

    def indicadores(self) -> list[str]: ...

    def metadata(self) -> Metadata: ...
