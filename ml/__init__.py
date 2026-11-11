"""Paquete de ML del recomendador SECOP.

La lógica viene, sin cambios, de los notebooks de ciencia de datos:

- ``retrieval.recuperar``  → notebook 03 (``candidatos_mes2`` + M1), 150 candidatos por empresa.
- ``features.construir``   → notebook final, sección 4 (``construir`` + ``feats_participacion_ok``).
- ``ranking.rankear``      → notebook final, secciones 7 y 8 (G1 → ``p_gana``, ranking v2 → ``s_v2``).
- ``listas.armar_listas``  → notebook final, sección 9 (lista ``historial`` y lista ``oportunidad``).

Ningún módulo tiene rutas fijas: las tablas, los modelos y las carpetas se reciben como argumentos.
La prueba de paridad (``tests/test_paridad.py``) es el árbitro de que la lógica no cambió.
"""
