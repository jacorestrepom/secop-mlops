-- Esquema serving: lo que la API lee y el job nocturno publica (guía de traspaso, sección 5, Paso 4).
-- Idempotente: se puede correr varias veces. Se aplica conectado a la base "secop" con el usuario maestro.
-- Las contraseñas de job_escritura y api_lectura NO van aquí: las pone jobs/... o infra desde Secrets Manager.
--
-- Publicación atómica: cada corrida escribe una versión nueva (version_id) en todas las tablas y, en una sola
-- transacción, la marca es_actual. La API siempre filtra por la versión actual, así nunca ve una a medio escribir.

CREATE SCHEMA IF NOT EXISTS serving;

CREATE TABLE IF NOT EXISTS serving.publicaciones (
    version_id        bigserial PRIMARY KEY,
    fecha_corte       date        NOT NULL,
    creada_en         timestamptz NOT NULL DEFAULT now(),
    publicada_en      timestamptz,
    estado            text        NOT NULL DEFAULT 'en_progreso'
                      CHECK (estado IN ('en_progreso', 'publicada', 'fallida')),
    es_actual         boolean     NOT NULL DEFAULT false,
    version_retrieval text        NOT NULL,
    version_pgana     text        NOT NULL,
    version_ranking   text        NOT NULL,
    silver_snapshots  jsonb       NOT NULL DEFAULT '{}'::jsonb,
    n_empresas        integer,
    metricas          jsonb       NOT NULL DEFAULT '{}'::jsonb,   -- lo que sirve GET /metadata
    datos_simulados   boolean     NOT NULL DEFAULT false,
    error             text,
    CHECK (NOT es_actual OR estado = 'publicada')
);
-- Garantiza que haya UNA sola versión vigente
CREATE UNIQUE INDEX IF NOT EXISTS publicaciones_una_actual ON serving.publicaciones (es_actual) WHERE es_actual;
CREATE INDEX IF NOT EXISTS publicaciones_fecha ON serving.publicaciones (fecha_corte);

-- Solo personas jurídicas: nunca nombre ni documento de personas naturales
CREATE TABLE IF NOT EXISTS serving.empresas (
    version_id      bigint  NOT NULL REFERENCES serving.publicaciones ON DELETE CASCADE,
    proveedor_key   text    NOT NULL,
    nombre          text    NOT NULL,
    departamento    text,
    es_pyme         boolean,
    tiene_historial boolean NOT NULL,
    PRIMARY KEY (version_id, proveedor_key)
);

CREATE TABLE IF NOT EXISTS serving.procesos (
    version_id           bigint NOT NULL REFERENCES serving.publicaciones ON DELETE CASCADE,
    id_proceso           text   NOT NULL,
    entidad              text   NOT NULL,
    objeto               text   NOT NULL,
    categoria            text,
    unspsc_familia       text,
    departamento         text,
    municipio            text,
    modalidad            text,
    precio_base          numeric,
    fecha_publicacion    date,
    fecha_limite_ofertas date,
    oferentes_esperados  numeric,
    url_proceso          text,
    PRIMARY KEY (version_id, id_proceso)
);

CREATE TABLE IF NOT EXISTS serving.recomendaciones (
    version_id          bigint   NOT NULL REFERENCES serving.publicaciones ON DELETE CASCADE,
    proveedor_key       text     NOT NULL,
    lista               text     NOT NULL CHECK (lista IN ('historial', 'oportunidad')),
    posicion            smallint NOT NULL CHECK (posicion BETWEEN 1 AND 10),
    id_proceso          text     NOT NULL,
    probabilidad        numeric  NOT NULL CHECK (probabilidad BETWEEN 0 AND 1),  -- P(gana) del G1
    score_ranking       numeric,                                                 -- solo ordena
    score_retrieval     numeric,                                                 -- solo ordena
    oferentes_esperados numeric,
    motivos             jsonb    NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY (version_id, proveedor_key, lista, posicion),
    UNIQUE (version_id, proveedor_key, id_proceso),               -- un proceso nunca está en las dos listas
    FOREIGN KEY (version_id, id_proceso) REFERENCES serving.procesos (version_id, id_proceso) ON DELETE CASCADE
);

-- Migración (2026-11-12): la familia UNSPSC del proceso fue inferida por secop-unspsc (la fuente publicó
-- 'UNSPECIFIED'). Solo marca: la categoría inferida nunca se publica en serving.procesos ni la ve la web.
ALTER TABLE serving.recomendaciones ADD COLUMN IF NOT EXISTS unspsc_inferido boolean NOT NULL DEFAULT false;

-- Agregados del dashboard (GET /estadisticas/{indicador})
CREATE TABLE IF NOT EXISTS serving.estadisticas (
    version_id  bigint NOT NULL REFERENCES serving.publicaciones ON DELETE CASCADE,
    indicador   text   NOT NULL,
    descripcion text   NOT NULL,
    unidad      text   NOT NULL,
    datos       jsonb  NOT NULL,          -- [{"etiqueta": ..., "valor": ...}]
    PRIMARY KEY (version_id, indicador)
);

-- Usuarios (sin LOGIN hasta que se les ponga contraseña desde Secrets Manager)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'job_escritura') THEN
        CREATE ROLE job_escritura NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api_lectura') THEN
        CREATE ROLE api_lectura NOLOGIN;
    END IF;
END
$$;

REVOKE ALL ON SCHEMA serving FROM PUBLIC;
GRANT USAGE ON SCHEMA serving TO job_escritura, api_lectura;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA serving TO job_escritura;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA serving TO job_escritura;
GRANT SELECT ON ALL TABLES IN SCHEMA serving TO api_lectura;

ALTER DEFAULT PRIVILEGES IN SCHEMA serving GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO job_escritura;
ALTER DEFAULT PRIVILEGES IN SCHEMA serving GRANT USAGE, SELECT ON SEQUENCES TO job_escritura;
ALTER DEFAULT PRIVILEGES IN SCHEMA serving GRANT SELECT ON TABLES TO api_lectura;

-- La API no tiene nada que hacer fuera de serving
ALTER ROLE api_lectura SET search_path = serving;
ALTER ROLE job_escritura SET search_path = serving;
ALTER ROLE api_lectura SET statement_timeout = '5s';
