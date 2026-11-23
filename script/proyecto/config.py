"""Configuracion unica del modulo proyecto: rutas, fuentes y reglas de negocio.

Rutas resueltas desde este archivo hacia la raiz del proyecto (rutas
relativas portables), por lo que el pipeline corre desde cualquier carpeta.

Esquema de fuentes (4 CSV):
  F1 transacciones   -> uk_property_price (ya limpio en data/limpia/*.parquet)
  F2 hipotecas       -> UK_Mortgage_Rate.csv            (Kaggle, snapshot 2022)
  F3 clima           -> MET_Office_Weather_Data.csv     (Met Office, 36 estaciones)
  F4 precios/salario -> Average_UK_houseprices_and_salary.csv + Income_by_age_and_gender.csv
"""
from __future__ import annotations

import os
from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[2]

DATA_CRUDO = PROYECTO / "data" / "crudo"
DATA_LIMPIA = PROYECTO / "data" / "limpia"
DATA_INTEGRADO = PROYECTO / "data" / "integrado"
DATA_MODELOS = PROYECTO / "data" / "modelos"
MD_FIGURAS = PROYECTO / "md" / "figuras"

# ----- rutas de origen (intocables, crudo) -----
RUTA_HIPOTECAS = DATA_CRUDO / "UK_Mortgage_Rate.csv"
RUTA_CLIMA = DATA_CRUDO / "MET_Office_Weather_Data.csv"
RUTA_PRECIOS_SALARIO = DATA_CRUDO / "Average_UK_houseprices_and_salary.csv"
RUTA_INGRESO_EDAD = DATA_CRUDO / "Income_by_age_and_gender.csv"

# F1 ya fue limpiada en el modulo limpieza (partes 1-7); se reutiliza:
RUTA_TRANSACCIONES_LIMPIO = DATA_LIMPIA / "uk_property_price_limpio.parquet"
# Muestra liviana (2.000 filas, ~135 KB) para modelado, clustering y asociacion sin saturar RAM:
RUTA_TRANSACCIONES_MUESTRA = DATA_LIMPIA / "uk_property_price_limpio_muestra.parquet"

# ----- rutas de salida (la etapa NO toca los origenes) -----
RUTA_HIPOTECAS_LIMPIO = DATA_LIMPIA / "hipotecas_limpio.parquet"
RUTA_CLIMA_LIMPIO = DATA_LIMPIA / "clima_limpio.parquet"
RUTA_PRECIOS_SALARIO_LIMPIO = DATA_LIMPIA / "precios_salario_limpio.parquet"

RUTA_PRECIOS_REGION = DATA_LIMPIA / "precios_region.parquet"
RUTA_PRECIOS_CIUDAD = DATA_LIMPIA / "precios_ciudad.parquet"

RUTA_ANOMALIAS_REGION = DATA_INTEGRADO / "anomalias_region.parquet"
RUTA_ANOMALIAS_NACIONAL = DATA_INTEGRADO / "anomalias_nacional.parquet"
RUTA_MAESTRO = DATA_INTEGRADO / "maestro_joven_uk.parquet"
RUTA_MAESTRO_MUESTRA = DATA_INTEGRADO / "maestro_joven_uk_muestra.csv"
RUTA_HIPOTECA_2022 = DATA_INTEGRADO / "hipoteca_2022.parquet"
RUTA_HIPOTECA_2022_MUESTRA = DATA_INTEGRADO / "hipoteca_2022.csv"
RUTA_HISTORICO = DATA_INTEGRADO / "historico_nacional.parquet"
RUTA_HISTORICO_MUESTRA = DATA_INTEGRADO / "historico_nacional.csv"

# ----- salidas analiticas livianas (preparadas para futura BD y backend) -----
RUTA_TORNEO_RESUMEN = DATA_INTEGRADO / "torneo_clasificadores_resumen.csv"
RUTA_SEGMENTOS_TERRITORIALES = DATA_INTEGRADO / "segmentos_territoriales.csv"
RUTA_SEGMENTOS_HIPOTECAS = DATA_INTEGRADO / "segmentos_hipotecas_muestra.csv"
RUTA_SEGMENTOS_TRANSACCIONES_1M = DATA_INTEGRADO / "segmentos_transacciones_1m.csv"
RUTA_CLUSTERING_24GB = DATA_INTEGRADO / "clustering_benchmark_24gb.csv"
RUTA_REGLAS_INMOBILIARIO = DATA_INTEGRADO / "reglas_asociacion_inmobiliario.csv"
RUTA_REGLAS_HIPOTECAS = DATA_INTEGRADO / "reglas_asociacion_hipotecas.csv"
RUTA_EVALUACION_UMBRALES = DATA_INTEGRADO / "evaluacion_umbrales_resumen.csv"
RUTA_EVALUACION_CV5 = DATA_INTEGRADO / "evaluacion_cv5_pliegues.csv"
RUTA_COMPARATIVA_GLOBAL = DATA_INTEGRADO / "comparativa_global_modelos.csv"
RUTA_COMPARATIVA_REGRESION = DATA_INTEGRADO / "comparativa_regresion_dual.csv"
RUTA_TELEMETRIA_RENDIMIENTO = DATA_INTEGRADO / "telemetria_rendimiento.csv"
RUTA_AUDITORIA_DATOS = DATA_INTEGRADO / "auditoria_fuentes_calidad.csv"
RUTA_HIPOTECAS_EXPORT_CSV = DATA_INTEGRADO / "hipotecas_catalogo_limpio.csv"
RUTA_PRECIOS_CIUDAD_EXPORT_CSV = DATA_INTEGRADO / "precios_ciudad_limpio.csv"

RUTA_MODELO_GANADOR = DATA_MODELOS / "modelo_ganador_clasificacion.joblib"
RUTA_MODELO_REGRESION_ESFUERZO = DATA_MODELOS / "modelo_regresion_esfuerzo.joblib"
RUTA_MODELO_REGRESION_TASA_F2 = DATA_MODELOS / "modelo_regresion_tasa_f2.joblib"
RUTA_MODELO_CLASIFICACION_TASA_F2 = DATA_MODELOS / "modelo_clasificacion_tasa_f2.joblib"
RUTA_CONTRATO_PREDICCION = DATA_MODELOS / "contrato_prediccion.json"
RUTA_ESQUEMA_BD_SQL = DATA_MODELOS / "esquema_bd.sql"

RUTA_BITACORA = PROYECTO / "md" / "BITACORA_DECISIONES.md"

# ----- perfiles de recursos: proteccion local, equipo 24 GB y escala nube -----
MODO_EJECUCION = os.getenv("MINERIA_MODO", "local").strip().lower()

PERFILES_RECURSOS: dict[str, dict[str, int | None]] = {
    "local": {
        "max_filas_kmeans": 3000,            # tope estratificado en PC conservador para F2 (7.796)
        "max_filas_minibatch_f1": 1_000_000, # 1 millon de transacciones reales F1 en streaming
        "batch_size_streaming": 100_000,     # ~2,4 MB por lote en float32
        "max_filas_jerarquico": 60,          # tope duro O(n^2) para dendrograma (~28 KB RAM)
        "max_transacciones_apriori": 3000,
        "pico_ram_mb_alerta": 220,
        "presupuesto_ram_sistema_gb": 8,
    },
    "local_24gb": {
        "max_filas_kmeans": None,            # 100 % de F2 (7.796 productos hipotecarios)
        "max_filas_minibatch_f1": None,      # 100 % de F1 (11.369.413 ventas) en streaming float32
        "batch_size_streaming": 250_000,     # ~6 MB por lote en float32
        "max_filas_jerarquico": 250,         # dendrograma Ward amplio y legible (< 1 MB)
        "max_transacciones_apriori": None,   # 100 % de F2 en Apriori
        "pico_ram_mb_alerta": 4096,          # tope operativo seguro (4 GB dentro de 24 GB)
        "presupuesto_ram_sistema_gb": 24,
    },
    "cloud": {
        "max_filas_kmeans": None,            # sin limite: procesa 100 % de F2 en nube
        "max_filas_minibatch_f1": None,      # sin limite: 11.369.413 transacciones F1 en streaming
        "batch_size_streaming": 250_000,
        "max_filas_jerarquico": 300,         # muestra visualmente legible para dendrograma Ward
        "max_transacciones_apriori": None,
        "pico_ram_mb_alerta": 4096,
        "presupuesto_ram_sistema_gb": 32,
    },
}


def obtener_limites_recursos(modo: str | None = None) -> dict[str, int | None]:
    """Devuelve los limites de filas y RAM segun perfil ('local' o 'cloud')."""
    clave = (modo or MODO_EJECUCION).strip().lower()
    return PERFILES_RECURSOS.get(clave, PERFILES_RECURSOS["local"]).copy()


KMEANS_K_MIN = 2
KMEANS_K_MAX = 8
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
APRIORI_MAX_K = 4
APRIORI_SOPORTE_PISO = 0.03

# ----- reglas de negocio del analisis -----
ANIO_MIN_PRECIOS = 2015            # inicio de transacciones georreferenciadas
ANIO_HIPOTECA = 2022               # snapshot de productos hipotecarios
VENTANA_CUADRO = (2015, 2020)      # anios donde conviven transacciones+clima+salario
LTV = 0.90                         # prestamo tipico: 90 % del precio
PLAZO_ANIOS = 25                   # MTG_PRODUCT_YEARS constante en F2
UMBRAL_ACCESIBILIDAD = 0.30        # cuota hipotecaria <= 30 % del salario (regla 30 %)
SEED = 42                          # reproducibilidad del analisis

# Deflactor nominal 2022 -> real 2020 (Banco de Inglaterra, calculadora CPI,
# dic 2020 -> dic 2022). El salario de F4 esta ajustado a GBP 2020.
DEFLACTOR_2022_A_2020 = 1.137

# Joven = 18 a 34 anios. La fuente por edad usa bandas; se aproxima con las
# bandas 18-21, 22-29 y 30-39 (la inferior de 30-39 cubre 30-34).
GRUPOS_JOVEN = ["18 to 21", "22 to 29", "30 to 39"]

# Factores regionales aprox. de salario medio (referencia ASHE 2021, razon
# ingreso medio anual de la region / Reino Unido; redondeado a 2 decimales,
# citado en el informe como aproximacion para regionalizar la mediana nacional).
FACTOR_REGIONAL = {
    "London": 1.25,
    "South East": 1.06,
    "East of England": 1.01,
    "South West": 0.95,
    "East Midlands": 0.92,
    "West Midlands": 0.91,
    "North West": 0.91,
    "Yorkshire and The Humber": 0.88,
    "North East": 0.85,
    "Wales": 0.90,
}

# Estacion Met Office -> region ONS (solo Inglaterra y Gales cruzan con F1;
# Escocia e Irlanda del Norte quedan para las anomalias nacionales).
ESTACION_REGION = {
    "aberporth": "Wales",
    "bradford": "Yorkshire and The Humber",
    "camborne": "South West",
    "cambridge": "East of England",
    "cardiff": "Wales",
    "chivenor": "South West",
    "cwmystwyth": "Wales",
    "durham": "North East",
    "eastbourne": "South East",
    "heathrow": "London",
    "hurn": "South West",
    "lowestoft": "East of England",
    "manston": "South East",
    "newtonrigg": "North West",
    "oxford": "South East",
    "ringway": "North West",
    "rossonwye": "West Midlands",
    "shawbury": "West Midlands",
    "sheffield": "Yorkshire and The Humber",
    "southampton": "South East",
    "valley": "Wales",
    "waddington": "East Midlands",
    "whitby": "Yorkshire and The Humber",
    "yeovilton": "South West",
}

# Columnas esperadas por fuente (contrato de entrada).
COLS_HIPOTECAS = [
    "sku", "banco", "subtitulo", "tipo_crudo", "tipo", "plazo_anios",
    "tasa_inicial_pct", "apr_pct", "tasa_reversion", "comisiones_total",
    "meses_inicial", "fecha_escaneo", "tid",
]