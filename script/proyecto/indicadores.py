"""Indicadores de accesibilidad: anos de salario, cuota hipotecaria (PMT) y
relacion clima-precio."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def anos_salario(precio: pd.Series, salario: pd.Series) -> pd.Series:
    """Anios de salario completo necesarios para comprar (precio / salario)."""
    return precio / salario


def cuota_mensual_pmt(precio: float, tasa_pct: float, plazo_anios: int = config.PLAZO_ANIOS,
                      ltv: float = config.LTV, comisiones: float = 0.0) -> float:
    """Cuota mensual francesa de un prestamo LTV sobre el precio.

    Formula de anualidad constante (amortizacion francesa):

        C = P · r / (1 − (1 + r)^(−n))

    con P = precio × LTV (capital prestado), r = (tasa_pct / 100) / 12 (tasa
    mensual) y n = plazo_anios × 12 (numero de cuotas). Si r = 0 el caso se
    reduce a la amortizacion lineal C = P / n.

    Nota sobre las comisiones: se suman FUERA de la formula, prorrateadas de
    forma lineal (comisiones / n). Es una aproximacion al cuadro de
    amortizacion, no un recargo compuesto: las comisiones reales de una
    hipoteca britanica pueden financiarse y por tanto pagar interes. La
    diferencia es pequena frente a la cuota (GBP 1.116 sobre 300 meses = GBP
    3,72 al mes) y se documenta como limitacion conocida, no como error.
    """
    principal = precio * ltv
    if principal <= 0 or plazo_anios <= 0:
        return np.nan
    r = (tasa_pct / 100.0) / 12.0
    n = int(plazo_anios * 12)
    if r == 0:
        cuota = principal / n
    else:
        cuota = principal * r / (1 - (1 + r) ** -n)
    return float(cuota + comisiones / n)


def hipoteca_2022(precios_region: pd.DataFrame, hipotecas: pd.DataFrame,
                  salario_region: pd.DataFrame) -> pd.DataFrame:
    """Corte 2022: para cada region, cuota mensual sobre la mediana 2022 al LTV,
    plazo y tasa inicial media de la oferta; % del salario joven mensual y
    bandera de accesibilidad (umbral 30 %)."""
    tasa = float(hipotecas["tasa_inicial_pct"].mean())
    comisiones = float(hipotecas["comisiones_total"].mean())
    salario_2022 = salario_region[salario_region["anio"] == config.ANIO_HIPOTECA]
    if salario_2022.empty:
        salario_2022 = salario_region[salario_region["anio"] == salario_region["anio"].max()]

    tb = (precios_region[precios_region["anio"] == config.ANIO_HIPOTECA]
          [["region", "precio_mediana"]]
          .merge(salario_2022[["region", "salario_joven_region"]], on="region", how="inner"))
    tb["precio_real_2020"] = tb["precio_mediana"] / config.DEFLACTOR_2022_A_2020
    tb["tasa_inicial_pct"] = tasa
    tb["cuota_mensual"] = tb.apply(
        lambda r: cuota_mensual_pmt(r["precio_real_2020"], tasa, config.PLAZO_ANIOS,
                                    config.LTV, comisiones), axis=1)
    tb["salario_joven_mensual"] = tb["salario_joven_region"] / 12
    tb["cuota_pct_salario"] = tb["cuota_mensual"] / tb["salario_joven_mensual"]
    tb["accesible"] = tb["cuota_pct_salario"] <= config.UMBRAL_ACCESIBILIDAD
    return tb.sort_values("cuota_pct_salario").reset_index(drop=True)


def correlacion_clima_precio(maestro: pd.DataFrame) -> pd.DataFrame:
    """Pearson entre la variacion anual del precio y las anomalias climaticas."""
    cols = ["lluvia_anom", "tmax_anom", "tmin_anom", "af_anom", "sol_anom"]
    filas = []
    for col in cols:
        sub = maestro[["precio_var_pct", col]].dropna()
        if len(sub) >= 3:
            filas.append({"clima": col, "pearson": sub[col].corr(sub["precio_var_pct"]),
                          "n": len(sub)})
    return pd.DataFrame(filas)