"""Modelos de respuesta de la API. Son el contrato con la web (guía de traspaso, sección 6):
los nombres de campo no se cambian sin acordarlo con el equipo de web."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

TITULO_HISTORIAL = "Recomendados según tu historial"
TITULO_OPORTUNIDAD = "Oportunidades que normalmente no verías"


class VersionModelos(BaseModel):
    retrieval: str = Field(examples=["secop-retrieval v1"])
    pgana: str = Field(examples=["secop-pgana v1"])
    ranking: str = Field(examples=["secop-ranking v1"])


class ProcesoRecomendado(BaseModel):
    posicion: int = Field(ge=1, le=10, description="1 a 10 dentro de su lista")
    id_proceso: str
    entidad: str
    objeto: str
    categoria: str | None = None
    departamento: str | None = None
    municipio: str | None = None
    precio_base: float | None = None
    fecha_limite_ofertas: date | None = None
    probabilidad: float = Field(ge=0, le=1, description="P(gana) del G1 si la empresa se presenta")
    oferentes_esperados: float | None = Field(None, description="Competencia típica en esa entidad y modalidad")
    motivos: list[str] = Field(default_factory=list, description="2 o 3 frases para la tarjeta")
    url_proceso: str | None = None


class Lista(BaseModel):
    titulo: str
    procesos: list[ProcesoRecomendado] = Field(default_factory=list, max_length=10)


class Listas(BaseModel):
    historial: Lista
    oportunidad: Lista


class Recomendaciones(BaseModel):
    proveedor_key: str
    fecha_corte: date | None
    tiene_historial: bool
    version_modelos: VersionModelos
    listas: Listas


class Empresa(BaseModel):
    """Solo personas jurídicas: la API nunca devuelve nombres ni documentos de personas naturales."""

    proveedor_key: str
    nombre: str
    departamento: str | None = None
    es_pyme: bool | None = None
    tiene_historial: bool


class Proceso(BaseModel):
    id_proceso: str
    entidad: str
    objeto: str
    categoria: str | None = None
    unspsc_familia: str | None = None
    departamento: str | None = None
    municipio: str | None = None
    modalidad: str | None = None
    precio_base: float | None = None
    fecha_publicacion: date | None = None
    fecha_limite_ofertas: date | None = None
    url_proceso: str | None = None


class PuntoEstadistica(BaseModel):
    etiqueta: str
    valor: float


class Estadistica(BaseModel):
    indicador: str
    descripcion: str
    unidad: str
    datos: list[PuntoEstadistica]


class MetricaLista(BaseModel):
    nombre: str
    valor: float
    referencia: float | None = None
    nombre_referencia: str | None = None


class DecilCalibracion(BaseModel):
    decil: int = Field(ge=1, le=10)
    probabilidad_media: float
    tasa_real: float


class Metricas(BaseModel):
    periodo_prueba: str
    lista_historial: list[MetricaLista]
    lista_oportunidad: list[MetricaLista]
    calibracion_pgana: list[DecilCalibracion]


class Metadata(BaseModel):
    fecha_datos: date | None
    ultima_publicacion: datetime | None
    version_modelos: VersionModelos
    metricas: Metricas
    datos_simulados: bool = Field(description="true mientras la API sirve respuestas simuladas")


class Salud(BaseModel):
    estado: Literal["ok", "degradado"]
    fuente_datos: str
    base_datos: Literal["ok", "error", "no_aplica"]
