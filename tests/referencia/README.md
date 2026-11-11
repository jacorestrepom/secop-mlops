# Referencia de paridad exacta de `ml/`

`test_paridad.py` compara `ml/` de punta a punta contra una referencia generada **con el mismo `ml/`**, sobre una
muestra fija de 300 empresas de marzo de 2026 (semilla 42, 20 % sin historial). La comparación es exacta, así que
solo vale dentro del mismo entorno: linux/amd64, las versiones de `requirements-ml.txt` y los hilos fijos de
`Dockerfile`. Los empates y la numérica del SVD cambian entre plataformas.

## Dónde se genera

En **Fargate (4 vCPU / 30 GB)**, durante la ronda de `score.py`, con la imagen de producción:

```bash
python -m tests.referencia.generar_referencia --data <data/> --salida <salida/> --n 300
```

Hay que correrlo **dos veces** y comprobar que `sha256.txt` sale idéntico en las dos: esa es la prueba de
determinismo, junto con la idempotencia del job (misma `fecha_corte`, mismo resultado). La salida se sube a S3 y
el CI la descarga en `SECOP_REFERENCIA_DIR`. No va al repo.

## Por qué no en Docker local

Lo que pesa no es la muestra, es la historia, que se carga completa sin importar cuántas empresas se puntúen
(medido en macOS con `footprint`):

| Etapa | Memoria |
|---|---|
| `HistoriaRetrieval.desde_carpeta` | 6,5 GB de pico |
| `Historia.desde_carpeta` (features) | 6,7 GB estable, 9,2 GB de pico al construirse |
| Mes completo (9.770 empresas): `recuperar()` | ~14 GB de pico |
| Mes completo: `construir()` | ~18 GB de pico |

La VM de Docker Desktop de un Mac de 16 GB (8,3 GB) no alcanza ni para cargar la historia.
