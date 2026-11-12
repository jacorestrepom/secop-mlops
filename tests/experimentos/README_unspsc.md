# Propuesta: imputar la familia UNSPSC cuando SECOP no la publica

Para: Camilo (ciencia de datos) · De: MLOps · 12-nov-2026 · Estado: **experimento local, no integrado a `ml/`**

## Problema

Desde octubre de 2026, SECOP publica `codigo_principal_de_categoria = 'UNSPECIFIED'` en el 99,6 % de los procesos
nuevos (10.588 de 10.626 entre el 15-oct y el 12-nov). En marzo era el 8,5 %. En silver y en `data/clean`, esos
procesos quedan con `unspsc_segmento` y `unspsc_familia` **nulos**. En el snapshot del 4-nov, los nulos por mes
fueron 9,4 % en marzo, 58 % en septiembre y 99,4 % en octubre. Los procesos abiertos que se puntúan cada noche
ya llegan sin categoría.

## Impacto medido

Simulación: UNSPSC nulo **solo** en los procesos del mes evaluado (el historial queda intacto) y `ml/` de punta
a punta. Script: `unspsc_faltante.py`.

| métrica | mar-2026 normal | sin UNSPSC | oct-2025 normal | sin UNSPSC |
|---|---|---|---|---|
| Recall@150 retrieval | 0,8034 | 0,7661 (−4,6 %) | 0,7337 | 0,6951 (−5,3 %) |
| Precision@10 lista 1 | 0,1224 | 0,1107 (−9,5 %) | 0,1053 | 0,0944 (−10,3 %) |
| Recall@10 lista 1 | 0,6093 | 0,5635 (−7,5 %) | 0,5052 | 0,4635 (−8,2 %) |
| NDCG@10 lista 1 | 0,5114 | 0,4760 (−6,9 %) | 0,4063 | 0,3717 (−8,5 %) |

La pérdida viene de que `unspsc_familia` es la feature categórica más pesada del G1 (8,6 % de la ganancia) y del
ranking v2 (6,9 %). Además, en el retrieval la familia es una de las dos llaves que generan candidatos. La lista 2
casi no cambia: depende de entidad, competencia y precio.

## Mitigación propuesta

Antes de `ml/`, y **solo cuando el proceso no trae UNSPSC**, se infiere la familia a partir de `descripcion_limpia`:

- **Modelo:** TF-IDF de palabras y bigramas (150 mil términos, `min_df=3`) más un `SGDClassifier(log_loss)`. El
  segmento son los 2 primeros dígitos de la familia. Sin ensambles ni ajuste fino.
- **Umbral fijo de 0,5:** si la probabilidad de la clase más probable no llega a 0,5, el UNSPSC **queda nulo**, igual
  que hoy. Se eligió en marzo con la regla "aciertos − errores máximo" y se usó **tal cual** en octubre.
- **Entrenamiento point-in-time:** solo procesos publicados antes del mes evaluado que tienen UNSPSC. En producción
  se reentrenaría junto con los modelos, con todo el histórico que trae categoría (hasta sep-2026).
- **Tamaño:** se probaron 50k, 100k, 150k, 200k y 300k términos. Se eligió el más pequeño que no pierde más de
  1 punto de acierto frente al de 300k:

| términos | tamaño | acierto top-1 familia (umbral 0,5, marzo) | cobertura |
|---|---|---|---|
| 300k | 448 MB | 79,3 % | 40,2 % |
| 200k | 299 MB | 79,0 % | 40,5 % |
| **150k** | **225 MB** | **78,6 %** | **40,7 %** |
| 100k | 150 MB | 78,2 % | 40,8 % |
| 50k | 75 MB | 77,3 % | 40,9 % |

## Resultados (modelo 150k, umbral 0,5, `ml/` de punta a punta)

Exactitud del clasificador en los procesos con UNSPSC real, con umbral 0,5. Familia: 78,6 % en marzo y 74,4 % en
octubre. Segmento: 82,8 % y 80,0 %. Cobertura: 40,7 % y 29,4 %.

| métrica | mar-2026: sin UNSPSC → imputado | recuperado | oct-2025: sin UNSPSC → imputado | recuperado |
|---|---|---|---|---|
| Recall@150 retrieval | 0,7661 → 0,7830 | 45 % | 0,6951 → 0,7135 | 48 % |
| Precision@10 lista 1 | 0,1107 → 0,1171 | 55 % | 0,0944 → 0,0994 | 46 % |
| Recall@10 lista 1 | 0,5635 → 0,5912 | 60 % | 0,4635 → 0,4853 | 52 % |
| **NDCG@10 lista 1** | 0,4760 → 0,4954 | **55 %** | 0,3717 → 0,3896 | **52 %** |
| % plausibles lista 2 | 0,6885 → 0,6796 | — | 0,6735 → 0,6691 | — |

**Prueba pareada de NDCG@10 por empresa (imputado − sin UNSPSC):**

| mes | empresas | diferencia media | IC 95 % (bootstrap, 10.000) | p (Wilcoxon) | mejoran / empeoran |
|---|---|---|---|---|---|
| mar-2026 | 9.734 | +0,0208 | [+0,0176, +0,0241] | 7e-41 | 24,0 % / 15,5 % |
| oct-2025 | 9.647 | +0,0193 | [+0,0162, +0,0225] | 2e-33 | 23,8 % / 16,5 % |

**Criterio acordado** (recuperar ≥ 50 % del NDCG@10 y que el IC no cruce cero): **se cumple en los dos meses.** En
octubre se cumple por poco (52 %). El retrieval recupera algo menos de la mitad (45–48 %).

**Riesgo:** el % de plausibles de la lista 2 baja 0,4–0,9 puntos frente a "sin UNSPSC". El efecto es pequeño, pero
va en contra.

## Costos

| | |
|---|---|
| Entrenamiento | ~80–95 s con 4 hilos (~490–520 mil procesos) |
| Modelo | 225 MB (joblib comprimido) |
| Inferencia | ~0,04 s por 2.000 procesos |
| Memoria | despreciable frente a `ml/` (~18 GB de pico en `construir()`) |

## Decisión que se pide

1. ¿Se aprueba imputar la familia UNSPSC (umbral 0,5, nulo si no pasa) solo para los procesos que llegan sin
   categoría?
2. Si se aprueba: el clasificador entra como un paso de `limpiar()`/`score.py` y se registra en MLflow como un cuarto
   modelo, con su propia prueba de exactitud. Los tres modelos actuales no se tocan.
3. Alternativa de fondo, a decidir más adelante: reentrenar G1, ranking v2 y M1 con procesos sin UNSPSC, porque
   desde oct-2026 ese es el caso normal.

Reproducir: `tests/experimentos/unspsc_faltante.py` y `unspsc_imputado.py` (ver los docstrings).
