-- ============================================================================
-- ESQUEMA RELACIONAL ESTANDAR (PostgreSQL / Supabase / Neon)
-- Proyecto: Accesibilidad de Vivienda para Jovenes (18-34) en el Reino Unido
-- Generado automaticamente por script/proyecto/prediccion_dual.py
--
-- Nivel 1 (tablas operativas y analiticas: ~3 MB total en data/integrado/*.csv)
--   Listo para subir en < 5 segundos a cualquier plan de BD en la nube.
-- Nivel 2 (tabla opcional masiva: 11.369.413 filas de F1 via COPY por lotes)
--   Activar cuando la BD remota disponga de almacenamiento sin limite de GB.
-- ============================================================================

-- 1. Cuadro maestro integrado (60 filas: 10 regiones x 6 anios, 2015-2020)
CREATE TABLE IF NOT EXISTS maestro_regional (
    anio INTEGER NOT NULL,
    region VARCHAR(64) NOT NULL,
    n_transacciones INTEGER NOT NULL,
    precio_medio DOUBLE PRECISION NOT NULL,
    precio_mediana DOUBLE PRECISION NOT NULL,
    salario_mediana_real DOUBLE PRECISION NOT NULL,
    salario_joven_nacional DOUBLE PRECISION NOT NULL,
    factor_regional DOUBLE PRECISION NOT NULL,
    salario_joven_region DOUBLE PRECISION NOT NULL,
    tmax_anom DOUBLE PRECISION,
    tmin_anom DOUBLE PRECISION,
    lluvia_anom DOUBLE PRECISION,
    sol_anom DOUBLE PRECISION,
    af_anom DOUBLE PRECISION,
    precio_var_pct DOUBLE PRECISION,
    anos_salario DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (anio, region)
);

-- 2. Agregado granular por ciudad y anio (14.050 filas, 1.153 ciudades UK)
CREATE TABLE IF NOT EXISTS precios_ciudad (
    anio INTEGER NOT NULL,
    region VARCHAR(64) NOT NULL,
    ciudad VARCHAR(96) NOT NULL,
    n_transacciones INTEGER NOT NULL,
    precio_medio DOUBLE PRECISION NOT NULL,
    precio_mediana DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (anio, region, ciudad)
);
CREATE INDEX IF NOT EXISTS idx_precios_ciudad_region ON precios_ciudad (region, anio);

-- 3. Catalogo hipotecario limpio F2 (7.796 productos, snapshot 2022)
CREATE TABLE IF NOT EXISTS hipotecas_catalogo (
    sku VARCHAR(32) NOT NULL,
    banco VARCHAR(128) NOT NULL,
    subtitulo VARCHAR(64),
    tipo_crudo VARCHAR(64),
    tipo VARCHAR(32),
    plazo_anios INTEGER,
    tasa_inicial_pct DOUBLE PRECISION NOT NULL,
    apr_pct DOUBLE PRECISION,
    tasa_reversion DOUBLE PRECISION,
    comisiones_total DOUBLE PRECISION,
    meses_inicial INTEGER,
    fecha_escaneo TIMESTAMP,
    tid VARCHAR(32) NOT NULL,
    es_atipico_tasa BOOLEAN NOT NULL,
    PRIMARY KEY (sku, tid)
);
CREATE INDEX IF NOT EXISTS idx_hipotecas_tipo_sub ON hipotecas_catalogo (tipo, subtitulo);

-- 4. Corte hipotecario regional 2022 (10 regiones)
CREATE TABLE IF NOT EXISTS hipoteca_2022 (
    region VARCHAR(64) PRIMARY KEY,
    precio_mediana DOUBLE PRECISION,
    precio_real_2020 DOUBLE PRECISION,
    tasa_inicial_pct DOUBLE PRECISION,
    comisiones_media DOUBLE PRECISION,
    salario_joven_region DOUBLE PRECISION,
    cuota_mensual DOUBLE PRECISION,
    cuota_pct_salario DOUBLE PRECISION,
    accesible BOOLEAN
);

-- 5. Serie historica nacional (22 anios: 1999-2020)
CREATE TABLE IF NOT EXISTS historico_nacional (
    anio INTEGER PRIMARY KEY,
    precio_promedio_real DOUBLE PRECISION,
    salario_mediana_real DOUBLE PRECISION,
    salario_joven_nacional DOUBLE PRECISION
);

-- 6. Segmentos descubiertos por Clustering (territoriales, hipotecarios y gran escala F1)
CREATE TABLE IF NOT EXISTS segmentos_territoriales (
    anio INTEGER NOT NULL,
    region VARCHAR(64) NOT NULL,
    cluster INTEGER NOT NULL,
    segmento_nombre VARCHAR(128) NOT NULL,
    accion_politica TEXT,
    precio_mediana DOUBLE PRECISION,
    salario_joven_region DOUBLE PRECISION,
    n_transacciones INTEGER,
    esfuerzo DOUBLE PRECISION,
    anos_salario DOUBLE PRECISION,
    PRIMARY KEY (anio, region)
);

CREATE TABLE IF NOT EXISTS segmentos_transacciones_f1 (
    cluster INTEGER PRIMARY KEY,
    segmento_f1 VARCHAR(128) NOT NULL,
    n_transacciones BIGINT NOT NULL,
    pct_mercado DOUBLE PRECISION,
    precio_medio_gbp DOUBLE PRECISION,
    latitud_media DOUBLE PRECISION,
    longitud_media DOUBLE PRECISION,
    pct_freehold DOUBLE PRECISION,
    pct_obra_nueva DOUBLE PRECISION,
    pct_flat DOUBLE PRECISION,
    pct_london DOUBLE PRECISION
);

-- 7. Reglas de asociacion (Apriori) y benchmarks de modelos / telemetria
CREATE TABLE IF NOT EXISTS reglas_asociacion_inmobiliario (
    id SERIAL PRIMARY KEY,
    antecedente TEXT NOT NULL,
    consecuente TEXT NOT NULL,
    soporte DOUBLE PRECISION,
    confianza DOUBLE PRECISION,
    lift DOUBLE PRECISION,
    score DOUBLE PRECISION,
    categoria_analitica VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS reglas_asociacion_hipotecas (
    id SERIAL PRIMARY KEY,
    antecedente TEXT NOT NULL,
    consecuente TEXT NOT NULL,
    soporte DOUBLE PRECISION,
    confianza DOUBLE PRECISION,
    lift DOUBLE PRECISION,
    score DOUBLE PRECISION
);

-- 8. Bitacora de consultas en vivo del futuro Frontend + Backend API
CREATE TABLE IF NOT EXISTS historial_predicciones (
    id SERIAL PRIMARY KEY,
    creado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tipo_consulta VARCHAR(32) NOT NULL,   -- 'regional' o 'hipoteca'
    region VARCHAR(64),
    entrada_json JSONB NOT NULL,
    prediccion_clasificacion VARCHAR(32),
    probabilidad DOUBLE PRECISION,
    prediccion_regresion DOUBLE PRECISION,
    umbral_aplicado VARCHAR(32),
    latencia_ms DOUBLE PRECISION
);

-- ============================================================================
-- NIVEL 2 (OPCIONAL · NUBE SIN LIMITE DE GB): Transacciones F1 completas (11,37 M)
-- Cargar en lotes desde data/limpia/uk_property_price_limpio.parquet via Python/COPY:
-- ============================================================================
CREATE TABLE IF NOT EXISTS transacciones_f1 (
    transaction_id VARCHAR(64) PRIMARY KEY,
    price DOUBLE PRECISION NOT NULL,
    date_of_transfer DATE NOT NULL,
    year SMALLINT NOT NULL,
    month SMALLINT NOT NULL,
    quarter SMALLINT NOT NULL,
    postcode VARCHAR(16),
    postcode_outward VARCHAR(8),
    property_type_full VARCHAR(32),
    duration_full VARCHAR(16),
    is_new_build BOOLEAN,
    town_city VARCHAR(96),
    district VARCHAR(96),
    county VARCHAR(96),
    ppd_category_type VARCHAR(32),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    region VARCHAR(64),
    country VARCHAR(32),
    admin_district VARCHAR(96),
    es_sin_geo BOOLEAN NOT NULL,
    es_atipico_precio BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tx_f1_region_year ON transacciones_f1 (region, year);
CREATE INDEX IF NOT EXISTS idx_tx_f1_city_year ON transacciones_f1 (town_city, year);
