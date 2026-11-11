"""API del recomendador SECOP. La web solo habla con esta API; nunca lee S3, Athena ni RDS."""

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.esquemas import Empresa, Estadistica, Metadata, Proceso, Recomendaciones, Salud
from api.fuentes import FuenteDatos, SinPublicacion, crear_fuente

app = FastAPI(
    title="Recomendador SECOP",
    version="0.1.0",
    description=(
        "Recomendaciones de procesos abiertos de SECOP II para cada empresa. "
        "`probabilidad` es la P(gana) **si la empresa se presenta**; `score_*` solo ordenan. "
        "Fuente: SECOP II (datos.gov.co)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(SinPublicacion)
def sin_publicacion(_: Request, e: SinPublicacion) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(e)})


@lru_cache
def fuente() -> FuenteDatos:
    return crear_fuente()


Fuente = Annotated[FuenteDatos, Depends(fuente)]


@app.get("/health", response_model=Salud, tags=["sistema"])
def health(f: Fuente) -> Salud:
    base = f.salud_base()
    return Salud(estado="degradado" if base == "error" else "ok", fuente_datos=f.nombre, base_datos=base)


@app.get("/empresas", response_model=list[Empresa], tags=["empresas"])
def buscar_empresas(f: Fuente, q: Annotated[str, Query(min_length=2, description="Nombre o NIT")]) -> list[Empresa]:
    """Buscador: hasta 20 empresas (solo personas jurídicas)."""
    return f.buscar_empresas(q, limite=20)


@app.get("/empresas/{proveedor_key}/recomendaciones", response_model=Recomendaciones, tags=["empresas"])
def recomendaciones(f: Fuente, proveedor_key: str) -> Recomendaciones:
    """Las dos listas vigentes de la empresa. Sin recomendaciones: listas vacías y `tiene_historial: false`."""
    return f.recomendaciones(proveedor_key)


@app.get("/procesos/abiertos", response_model=list[Proceso], tags=["procesos"])
def procesos_abiertos(
    f: Fuente,
    familia: Annotated[str | None, Query(description="Familia UNSPSC")] = None,
    departamento: str | None = None,
    limite: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[Proceso]:
    """Para empresas sin historial: procesos abiertos filtrados por categoría y lugar."""
    return f.procesos_abiertos(familia, departamento, limite)


@app.get("/procesos/{id_proceso}", response_model=Proceso, tags=["procesos"])
def proceso(f: Fuente, id_proceso: str) -> Proceso:
    p = f.proceso(id_proceso)
    if p is None:
        raise HTTPException(404, f"No existe el proceso {id_proceso}")
    return p


@app.get("/estadisticas", response_model=list[str], tags=["estadisticas"])
def indicadores(f: Fuente) -> list[str]:
    """Indicadores disponibles para el dashboard."""
    return f.indicadores()


@app.get("/estadisticas/{indicador}", response_model=Estadistica, tags=["estadisticas"])
def estadistica(f: Fuente, indicador: str) -> Estadistica:
    e = f.estadistica(indicador)
    if e is None:
        raise HTTPException(404, f"Indicador desconocido: {indicador}. Disponibles: {', '.join(f.indicadores())}")
    return e


@app.get("/metadata", response_model=Metadata, tags=["sistema"])
def metadata(f: Fuente) -> Metadata:
    """Fecha de los datos, última publicación, versión de los modelos y métricas para «Cómo funciona»."""
    return f.metadata()
