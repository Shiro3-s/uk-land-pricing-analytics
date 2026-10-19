"""Validacion post-limpieza: criterios reales contra los valores verificados.

Los asserts embebidos por fase cuentan como pruebas automaticas
y cortan la corrida con mensaje claro.
"""
from __future__ import annotations

import pandas as pd

from . import config


def construir_criterios(
    df_limpio: pd.DataFrame,
    anio_corte: int | None = None,
    filas_esperadas: int | None = None,
    nulos_geo_esperados: int | None = None,
    fuera_iqr_esperados: int | None = None,
    precio_limite_superior: float | None = None,
    precio_minimo: int | None = None,
) -> pd.DataFrame:
    """Tabla criterio vs real vs esperado para auditarla.

    Los esperados proceden de ``config`` (diagnostico verificado) salvo que se
    inyecten para pruebas.
    """
    anio_corte = anio_corte if anio_corte is not None else config.ANIO_CORTE
    filas_esperadas = filas_esperadas if filas_esperadas is not None else config.FILAS_VERIFICADAS
    nulos_geo_esperados = nulos_geo_esperados if nulos_geo_esperados is not None else config.NULOS_GEO_VERIFICADOS
    fuera_iqr_esperados = fuera_iqr_esperados if fuera_iqr_esperados is not None else config.FUERA_IQR_VERIFICADOS
    precio_limite_superior = (
        precio_limite_superior if precio_limite_superior is not None else config.PRECIO_LIM_SUP
    )
    precio_minimo = precio_minimo if precio_minimo is not None else config.PRECIO_MIN_OBSERVADO

    reales = {
        "duplicados exactos": int(df_limpio.duplicated().sum()),
        "duplicados por transaction_id": int(df_limpio[config.CLAVE_PRIMARIA].duplicated().sum()),
        "nulos en criticas": int(df_limpio[config.COLUMNAS_CRITICAS].isna().sum().sum()),
        "fechas futuras": int(df_limpio[config.COLUMNA_FECHA].dt.year.gt(anio_corte).sum()),
        "filas es_sin_geo": int(df_limpio["es_sin_geo"].sum()),
        "filas es_atipico_precio": int(df_limpio["es_atipico_precio"].sum()),
        "price minimo": int(df_limpio["price"].min()),
        "price maximo": int(df_limpio["price"].max()),
    }
    esperados = {
        "duplicados exactos": 0,
        "duplicados por transaction_id": 0,
        "nulos en criticas": 0,
        "fechas futuras": 0,
        "filas es_sin_geo": nulos_geo_esperados,
        "filas es_atipico_precio": fuera_iqr_esperados,
        "price minimo": precio_minimo,
        "price maximo": int(precio_limite_superior),
    }
    return pd.DataFrame({
        "criterio": list(reales),
        "real": list(reales.values()),
        "esperado": [esperados[k] for k in reales],
    })


def validar_criterios(criterios: pd.DataFrame) -> None:
    """Corta la corrida si algun criterio no coincide con lo esperado."""
    errores = criterios[criterios["real"] != criterios["esperado"]]
    if not errores.empty:
        texto = "; ".join(f"{fila.criterio}: real={fila.real} esperado={fila.esperado}"
                          for fila in errores.itertuples())
        raise AssertionError(f"Revisar criterios no alineados: {texto}")
    print("VALIDACION OK: todos los criterios en verde")