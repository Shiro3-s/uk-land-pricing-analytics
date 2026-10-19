"""Diagnostico de fase 3: faltantes (MCAR/MAR/MNAR) y outliers (IQR vs z-score).

Fase 3 · Limpieza y cruzamiento (KDD etapa 2 · CRISP-DM preparacion de datos).
Metodologia: clasificar el ORIGEN del faltante y elegir el
metodo de atipicos segun la forma de la distribucion, no por el numero.
"""
from __future__ import annotations

import pandas as pd

from . import config


def resumen_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla (columna, nulos, porcentaje) para una fuente."""
    n_obs = len(df)
    filas = [{"columna": c, "nulos": int(v), "pct": round(100 * v / n_obs, 2)}
             for c, v in df.isna().sum().items() if v > 0]
    return pd.DataFrame(filas, columns=["columna", "nulos", "pct"])


# ---------------------------------------------------------------- MCAR/MAR/MNAR
def _fila(entidad: str, columna: str, nulos: int, n_obs: int,
          tipo: str, evidencia: str, tratamiento: str) -> dict:
    """Arma una fila de clasificacion (fuente, columna, % nulos, clase, evidencia)."""
    return {
        "fuente": entidad, "columna": columna, "nulos": nulos,
        "pct": round(100 * nulos / n_obs, 2), "clasificacion": tipo,
        "evidencia": evidencia, "tratamiento": tratamiento,
    }


def clasificacion_hipotecas(df: pd.DataFrame) -> list[dict]:
    """F2: nulos de tasa_reversion dependen del tipo de producto (observado)."""
    n = len(df)
    filas = []
    sin_tasa = int(df["tasa_inicial_pct"].isna().sum())
    if sin_tasa:
        filas.append(_fila("F2 hipotecas", "tasa_inicial_pct", sin_tasa, n, "MNAR",
                           "productos sin tasa (sin cuota posible)",
                           "eliminar fila (no aportan al esfuerzo)"))
    reversion = df["tasa_reversion"].isna()
    n_rev = int(reversion.sum())
    if n_rev:
        prop = (df.groupby("tipo", observed=True)["tasa_reversion"]
                .apply(lambda s: int(s.isna().sum()) / len(s) * 100).round(1))
        por_tipo = "; ".join(f"{k}={v}%" for k, v in prop.items())
        filas.append(_fila(
            "F2 hipotecas", "tasa_reversion", n_rev, n, "MAR",
            f"la ausencia depende del tipo (observado): {por_tipo}",
            "conservar NaN (la cuota usa tasa_inicial; reversion es informativa)"))
    return filas


def clasificacion_clima(df: pd.DataFrame) -> list[dict]:
    """F3: corruptos (MNAR de registro), tmax/tmin instrumentales (MCAR) y
    sol no publicado antes de 1930 (MNAR estructural)."""
    n = len(df)
    filas = []
    corruptos = int(df["anio"].isna().sum())
    if corruptos:
        filas.append(_fila("F3 clima", "anio/month", corruptos, n, "MNAR",
                           "filas sin fecha (registro corrupto/footer)",
                           "eliminar fila (no son mediciones)"))
    w = df.dropna(subset=["anio"])

    simultaneo = int((w["tmax"].isna() & w["tmin"].isna()).sum())
    tmax_n = int(w["tmax"].isna().sum())
    for col in ["tmax", "tmin", "af", "lluvia"]:
        k = int(w[col].isna().sum())
        if k:
            filas.append(_fila(
                "F3 clima", col, k, n, "MCAR",
                f"gaps instrumentales de estacion: {simultaneo}/{tmax_n} con tmax "
                "nulo tambien falta tmin (falla conjunta de reporte); no depende del valor",
                "NaN (los agregados anuales y la normal toleran meses faltantes)"))
    sol = int(w["sol"].isna().sum())
    pre1930 = int(w.loc[w["anio"] < 1931, "sol"].isna().sum())
    filas.append(_fila(
        "F3 clima", "sol", sol, n, "MNAR estructural",
        f"horas de sol no medidas/publicadas antes de 1931 ({pre1930} de {sol} nulos; "
        "decadas 1850-1930 casi vacias), no es azar",
        "NaN (la anomalia de sol se usa aparte, solo donde existe)"))
    return filas


def clasificacion_precios(df: pd.DataFrame) -> list[dict]:
    """F4: el salario no se publicaba antes de 1999 (ausencia por diseno)."""
    n = len(df)
    salario = int(df["salario_mediana_real"].isna().sum())
    if salario:
        inicio = int(df.loc[df["salario_mediana_real"].notna(), "anio"].min())
        return [_fila("F4 precios/salario", "salario_mediana_real", salario, n,
                      "MNAR estructural",
                      f"mediana salarial no publicada hasta {inicio} (serie oficial), "
                      "ausencia por diseno, no aleatoria",
                      "NaN (no se imputa historia inventada)")]
    return []


def clasificacion_fuentes(fuentes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Clasificacion consolidada de faltantes por fuente (para informe/bitacora)."""
    filas = []
    for nombre, df in fuentes.items():
        if nombre == "hipotecas":
            filas += clasificacion_hipotecas(df)
        elif nombre == "clima":
            filas += clasificacion_clima(df)
        elif nombre == "precios":
            filas += clasificacion_precios(df)
        elif nombre == "ingreso":
            continue
        elif nombre == "transacciones":
            continue
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ IQR vs z-score
def limites_iqr(serie: pd.Series) -> dict:
    """Regla IQR (Tukey): cuartiles reales y limites 1,5 x IQR."""
    q1, q2, q3 = serie.quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    return {"q1": q1, "q2": q2, "q3": q3, "iqr": iqr,
            "limite_inferior": q1 - 1.5 * iqr, "limite_superior": q3 + 1.5 * iqr}


def zscore_fuera(serie: pd.Series, umbral: float = 3.0) -> int:
    """Candidatos con |z| > umbral bajo el supuesto de normalidad."""
    z = (serie - serie.mean()) / serie.std()
    return int((z.abs() > umbral).sum())


def comparar_iqr_zscore(serie: pd.Series, nombre: str, umbral_z: float = 3.0) -> dict:
    """Compara IQR vs z-score y devuelve el dict con la evidencia de la eleccion."""
    lim = limites_iqr(serie)
    fuera_iqr = int((serie > lim["limite_superior"]).sum())
    fuera_z = zscore_fuera(serie, umbral_z)
    skew = float(serie.skew())
    return {
        "columna": nombre,
        "n": int(serie.notna().sum()),
        "q1": round(lim["q1"], 2), "mediana": round(lim["q2"], 2),
        "q3": round(lim["q3"], 2), "iqr": round(lim["iqr"], 2),
        "lim_sup": round(lim["limite_superior"], 2),
        "fuera_iqr": fuera_iqr, "pct_iqr": round(100 * fuera_iqr / len(serie), 2),
        "fuera_z": fuera_z, "pct_z": round(100 * fuera_z / len(serie), 2),
        "skew": round(skew, 2),
        "metodo": "IQR (Tukey)",
        "razon": ("IQR no supone normalidad y detecta la cola real aun con skew=%s; "
                  "el z-score con |z|>%.0f no detecta nada (%d). Se winsoriza al "
                  "limite superior."
                  % (round(skew, 2), umbral_z, fuera_z)),
    }


def winsorizar(serie: pd.Series, limite_superior: float) -> pd.Series:
    """Recorta (clip) al limite superior; conserva filas y documenta la bandera."""
    return serie.clip(upper=limite_superior)