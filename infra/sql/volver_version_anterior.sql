-- Vuelve a la publicación anterior (OPERACION.md). Una sola transacción: si algo falla, no cambia nada.
-- Marca es_actual en la publicación 'publicada' más reciente con fecha_corte ANTERIOR a la vigente (la de ayer).
-- Se corre con el usuario maestro en la base "secop". La próxima noche que publique vuelve a mover es_actual:
-- para quedarse en la versión anterior hay que deshabilitar antes el Scheduler secop-nocturno-diario.
BEGIN;

CREATE TEMP TABLE destino ON COMMIT DROP AS
SELECT p.version_id, p.fecha_corte
FROM serving.publicaciones p
WHERE p.estado = 'publicada'
  AND p.fecha_corte < (SELECT fecha_corte FROM serving.publicaciones WHERE es_actual)
ORDER BY p.fecha_corte DESC, p.version_id DESC
LIMIT 1;

DO $$
BEGIN
    IF (SELECT count(*) FROM destino) <> 1 THEN
        RAISE EXCEPTION 'no hay una publicación anterior en estado publicada (retención: 7 versiones)';
    END IF;
END
$$;

SELECT 'antes' AS momento, version_id, fecha_corte FROM serving.publicaciones WHERE es_actual;

UPDATE serving.publicaciones SET es_actual = false WHERE es_actual;
UPDATE serving.publicaciones SET es_actual = true WHERE version_id = (SELECT version_id FROM destino);

SELECT 'después' AS momento, version_id, fecha_corte FROM serving.publicaciones WHERE es_actual;

COMMIT;
