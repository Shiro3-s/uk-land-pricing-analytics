"""Configuracion unica del proyecto: rutas relativas, dtypes y reglas de negocio.

Rutas resueltas desde este archivo (config.py) hacia la raiz del proyecto, por lo
que el pipeline corre desde cualquier carpeta (rutas relativas portables).
"""
from __future__ import annotations

from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[2]

RUTA_ORIGEN = PROYECTO / "db" / "uk_property_price.csv"
RUTA_SALIDA = PROYECTO / "data" / "limpia" / "uk_property_price_limpio.parquet"

# ----- cifras verificadas del diagnostico real (consenso del proyecto) -----
FILAS_VERIFICADAS = 11_369_413
NULOS_GEO_VERIFICADOS = 22_913          # lat/lon/region/country/admin_district juntos (0,20 %)
FUERA_IQR_VERIFICADOS = 718_461         # price por arriba del limite superior (6,32 %)

# ----- reglas de negocio -----
ANIO_CORTE = 2026                       # no se aceptan transacciones futuras
PRECIO_Q1 = 158_000
PRECIO_Q3 = 385_000
PRECIO_IQR = PRECIO_Q3 - PRECIO_Q1
PRECIO_LIM_INF = PRECIO_Q1 - 1.5 * PRECIO_IQR
PRECIO_LIM_SUP = PRECIO_Q3 + 1.5 * PRECIO_IQR
PRECIO_MIN_OBSERVADO = 100

# ----- contrato de columnas -----
DTYPE_CATEGORICOS = {
    "property_type_full": "category",
    "duration_full": "category",
    "is_new_build": "category",
    "ppd_category_type": "category",
    "region": "category",
    "country": "category",
    "town_city": "category",
    "district": "category",
    "county": "category",
    "postcode_outward": "category",
}

COLUMNAS_CRITICAS = ["transaction_id", "price", "date_of_transfer"]
CLAVE_PRIMARIA = ["transaction_id"]
COLUMNAS_GEO = ["latitude", "longitude", "region", "country", "admin_district"]
COLUMNAS_DERIVADAS = ["year", "month", "quarter"]
COLUMNAS_BANDERAS = ["es_sin_geo", "es_atipico_precio"]
COLUMNAS_FOCO = ["price", "latitude", "longitude", "town_city", "county", "country"]
COLUMNAS_CATEGORIAS_FOCO = ["town_city", "county", "country"]

COLUMNA_FECHA = "date_of_transfer"