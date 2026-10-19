"""Diagnostico del dataset: perfil de columnas, IQR, nulos geo, categorias y fechas.

Cada funcion devuelve un resultado claro (DataFrame o dict) sin efectos secundarios
sobre ``df``; la visualizacion vive en ``graficos.py``.
"""
from __future__ import annotations

import unicodedata

import pandas as pd

from . import config


def perfil_foco(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    """Resumen por columna: tipo, nulos (n y %), unicos y min/max (solo numericas)."""
    columnas = columnas or config.COLUMNAS_FOCO
    filas = []
    for col in columnas:
        serie = df[col]
        es_numerica = pd.api.types.is_numeric_dtype(serie)
        minmax = (serie.min(), serie.max()) if es_numerica else ("-", "-")
        filas.append({
            "columna": col,
            "tipo": str(serie.dtype),
            "nulos": int(serie.isna().sum()),
            "nulos_%": round(100 * serie.isna().mean(), 2),
            "unicos": int(serie.nunique()),
            "min": minmax[0],
            "max": minmax[1],
        })
    return pd.DataFrame(filas)


def limites_iqr(valores: pd.Series) -> dict:
    """Limites de la regla IQR (Tukey): Q1, Q2 (mediana), Q3, IQR, limites."""
    q1, q2, q3 = valores.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    return {
        "q1": q1, "q2": q2, "q3": q3, "iqr": iqr,
        "limite_inferior": q1 - 1.5 * iqr,
        "limite_superior": q3 + 1.5 * iqr,
    }


def tabla_cuartiles(valores: pd.Series) -> pd.DataFrame:
    """Tabla de los 4 cuartiles de una serie: rango, transacciones y % del total.

    Cada cuartil agrupa (por definicion) ~25 % de los valores. Q1+Q2 (hasta la
    mediana) concentran el mercado tipico; Q4 recoge la cola extrema.
    """
    v = valores.dropna()
    n = len(v)
    b = [v.min()] + list(v.quantile([0.25, 0.5, 0.75])) + [v.max()]
    filas = []
    for i in range(4):
        lo, hi = int(b[i]), int(b[i + 1])
        dentro = (v >= lo) & (v <= hi) if i == 3 else (v >= lo) & (v < hi)
        filas.append({
            "cuartil": f"Q{i + 1}",
            "rango (GBP)": f"{lo:,.0f} .. {hi:,.0f}",
            "transacciones": int(dentro.sum()),
            "% datos": round(100 * int(dentro.sum()) / n, 2),
        })
    return pd.DataFrame(filas)


def contar_fuera_de_iqr(valores: pd.Series, limites: dict) -> int:
    """Cuenta valores fuera de los limites IQR dados."""
    return int(((valores < limites["limite_inferior"]) | (valores > limites["limite_superior"])).sum())


def nulos_geo(df: pd.DataFrame, columnas: list[str] | None = None) -> dict:
    """Verifica que los nulos geo sean simultaneos y entrega las metricas clave."""
    columnas = columnas or config.COLUMNAS_GEO
    cualquier = int(df[columnas].isna().any(axis=1).sum())
    todas = int(df[columnas].isna().all(axis=1).sum())
    return {"cualquiera": cualquier, "todas": todas, "simultaneo": cualquier == todas}


def resumen_categorias(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    """Cardinalidad y nulos de columnas categoricas/elegidas."""
    columnas = columnas or config.COLUMNAS_CATEGORIAS_FOCO
    filas = [{
        "columna": col,
        "categorias": int(df[col].nunique()),
        "nulos": int(df[col].isna().sum()),
    } for col in columnas]
    return pd.DataFrame(filas)


def _sin_tildes(texto: str) -> str:
    """Quita acentos para comparar escrituras de forma uniforme."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def normalizar_texto(serie: pd.Series) -> pd.Series:
    """Strip + minusculas + sin tildes."""
    return serie.str.strip().str.lower().map(_sin_tildes)


def colisiones_al_normalizar(df: pd.DataFrame, columnas: list[str] | None = None) -> dict:
    """Indica si alguna escritura distinta colapsa al normalizar (strip + tildes)."""
    columnas = columnas or config.COLUMNAS_CATEGORIAS_FOCO
    resultado = {}
    for col in columnas:
        base = df[col].dropna()
        faltan = int(base.nunique() != normalizar_texto(base).nunique())
        todo_mayus = bool(base.str.isupper().all())
        resultado[col] = {"colisiones": faltan, "todo_en_mayusculas": todo_mayus}
    return resultado


def analizar_fechas(
    df: pd.DataFrame,
    anio_corte: int | None = None,
    columnas_derivadas: list[str] | None = None,
) -> dict:
    """Rango del periodo, fechas futuras e inconsistencias year/month/quarter."""
    anio_corte = anio_corte or config.ANIO_CORTE
    columnas_derivadas = columnas_derivadas or config.COLUMNAS_DERIVADAS
    fecha = df[config.COLUMNA_FECHA]
    esperado = pd.DataFrame({
        "year": fecha.dt.year,
        "month": fecha.dt.month,
        "quarter": fecha.dt.quarter,
    })
    incoherencias = int((df[columnas_derivadas] != esperado).any(axis=1).sum())
    return {
        "min": fecha.min(),
        "max": fecha.max(),
        "futuras": int(fecha.dt.year.gt(anio_corte).sum()),
        "incoherencias_derivadas": incoherencias,
    }