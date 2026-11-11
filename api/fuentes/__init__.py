import os

from api.fuentes.base import FuenteDatos


def crear_fuente() -> FuenteDatos:
    """Elige el origen de datos según FUENTE_DATOS (por defecto, simulado)."""
    tipo = os.getenv("FUENTE_DATOS", "simulada")
    if tipo == "simulada":
        from api.fuentes.simulada import FuenteSimulada

        return FuenteSimulada()
    raise ValueError(f"FUENTE_DATOS desconocida: {tipo!r} (por ahora solo 'simulada')")
