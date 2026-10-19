"""Limpieza por fuente (F2 hipotecas, F3 clima, F4 precios/salario).

Sigue el orden: duplicados -> tipos/categorias -> nulos 
(MCAR/MAR/MNAR) -> outliers. F1 ya fue limpiada por el modulo ``limpieza``.
Devuelve (df_limpio, decisiones) donde decisiones es una lista de dicts de la
bitacora: (fuente, problema, cantidad, metodo, razon).
"""
from __future__ import annotations

import pandas as pd


def limpiar_hipotecas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """F2: base unica, numericas consistentes y nulos de tasa."""
    d: list[dict] = []
    n_inicial = len(df)

    df = df.drop_duplicates(subset=["sku", "tid"], keep="first")
    d.append({
        "fuente": "F2 hipotecas", "problema": "duplicados por sku+tid",
        "cantidad": n_inicial - len(df), "metodo": "keep='first'",
        "razon": "cada producto unico por escaneo (clave natural sku+tid)"})

    for col in ["tasa_inicial_pct", "apr_pct", "tasa_reversion", "comisiones_total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_sin_tasa = int(df["tasa_inicial_pct"].isna().sum())
    df = df.dropna(subset=["tasa_inicial_pct"])
    d.append({
        "fuente": "F2 hipotecas", "problema": "productos sin tasa inicial (MNAR)",
        "cantidad": n_sin_tasa, "metodo": "eliminar filas",
        "razon": "sin tasa no hay cuota que estimar; son casos puntuales"})

    df["tipo"] = df["tipo"].astype("category")
    df["banco"] = df["banco"].astype("category")
    df["plazo_anios"] = df["plazo_anios"].astype("int")
    d.append({
        "fuente": "F2 hipotecas", "problema": "tipos y plazos textuales",
        "cantidad": 0, "metodo": "categoria + int",
        "razon": "contrato de tipos para el cruce por ano de escaneo"})
    return df, d


def limpiar_clima(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """F3: corruptos sin fecha (MNAR de registro), duplicados y numericas."""
    d: list[dict] = []
    n_inicial = len(df)

    corruptos = int(df["anio"].isna().sum())
    df = df.dropna(subset=["anio"])
    if corruptos:
        d.append({
            "fuente": "F3 clima", "problema": "filas sin fecha (corruptas/footer)",
            "cantidad": corruptos, "metodo": "eliminar fila",
            "razon": "no son mediciones: no tienen anio ni variables (MNAR de registro)"})

    df = df.drop_duplicates(subset=["anio", "mes", "estacion"], keep="first")
    d.append({
        "fuente": "F3 clima", "problema": "filas temporales duplicadas",
        "cantidad": n_inicial - corruptos - len(df), "metodo": "keep='first'",
        "razon": "una medicion mensual por estacion"})

    for col in ["tmax", "tmin", "af", "lluvia", "sol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    d.append({
        "fuente": "F3 clima", "problema": "valores NA/*/# en tmax/tmin/af/lluvia/sol",
        "cantidad": 0, "metodo": "NaN numerico",
        "razon": "los simbolos Met Office (---, *, #) se trataron como faltantes MCAR"})
    return df, d


def limpiar_precios_salario(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """F4a: serie anual limpia (anio unico, numericas coherentes)."""
    d: list[dict] = []
    df = df.sort_values("anio").reset_index(drop=True)
    for col in ["precio_promedio_real", "salario_mediana_real"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    d.append({
        "fuente": "F4 precios/salario", "problema": "salario 'null' 1975-1998",
        "cantidad": int(df["salario_mediana_real"].isna().sum()),
        "metodo": "dejar NaN (no imputar)",
        "razon": "el salario solo esta publicado desde 1999; no se inventa historia"})
    return df, d


def limpiar_ingreso_edad(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """F4b: grupos de edad sin duplicados y salario numerico."""
    d: list[dict] = []
    df = df.drop_duplicates(subset=["grupo_edad", "genero"], keep="first")
    df["salario_mediana"] = pd.to_numeric(df["salario_mediana"], errors="coerce")
    df["grupo_edad"] = df["grupo_edad"].astype("category")
    df["genero"] = df["genero"].astype("category")
    d.append({
        "fuente": "F4 ingreso por edad", "problema": "duplicados y tipos",
        "cantidad": 0, "metodo": "clave grupo+genero",
        "razon": "una mediana salarial por banda y genero"})
    return df, d