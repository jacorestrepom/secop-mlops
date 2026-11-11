import os

from api.fuentes.base import FuenteDatos, SinPublicacion  # noqa: F401


def crear_fuente() -> FuenteDatos:
    """Elige el origen de datos según FUENTE_DATOS (por defecto, simulado)."""
    tipo = os.getenv("FUENTE_DATOS", "simulada")
    if tipo == "simulada":
        from api.fuentes.simulada import FuenteSimulada

        return FuenteSimulada()
    if tipo == "rds":
        from api.fuentes.rds import FuenteRDS

        return FuenteRDS()
    raise ValueError(f"FUENTE_DATOS desconocida: {tipo!r} (usa 'simulada' o 'rds')")
