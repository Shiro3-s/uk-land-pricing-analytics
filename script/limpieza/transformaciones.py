"""Transformaciones sobre una copia: banderas y winsorizacion. Ninguna fila se elimina."""
from __future__ import annotations

import pandas as pd

from . import config
from .bitacora import Bitacora


def winsorizar_price(price: pd.Series, limite_superior: float) -> pd.Series:
    """Recorta (clip) el precio al limite superior, teniendo 0 como piso."""
    return price.clip(lower=0, upper=limite_superior).astype("int64")


def crear_bandera_sin_geo(df: pd.DataFrame, bitacora: Bitacora | None = None) -> pd.DataFrame:
    """Copia con ``es_sin_geo``: nulos geo MCAR marcados, nunca imputados."""
    limpio = df.copy()
    bitacora = bitacora or Bitacora()
    limpio["es_sin_geo"] = limpio[config.COLUMNAS_GEO[0]].isna()
    bitacora.anotar(
        "lat/long/region/country/admin_district", "nulos geo (MCAR)",
        int(limpio["es_sin_geo"].sum()), "bandera es_sin_geo",
        "ausencia de geocodificacion; inventar coordenadas falsearia el clustering",
        f"{int(limpio['es_sin_geo'].sum()):,} filas marcadas, 0 eliminadas",
    )
    return limpio


def aplicar_bandera_atipicos(
    limpio: pd.DataFrame,
    price_original: pd.Series,
    limites: dict,
    bitacora: Bitacora | None = None,
) -> pd.DataFrame:
    """Agrega ``es_atipico_precio`` y winsoriza ``price`` a partir del precio original.

    ``limpio`` debe venir de ``crear_bandera_sin_geo``; ``price_original`` es el precio
    sin transformar (IQR siempre se calcula sobre el original, regla clásica).
    """
    bitacora = bitacora or Bitacora()
    limpio = limpio.copy()
    limpio["es_atipico_precio"] = (
        price_original.gt(limites["limite_superior"]) | price_original.lt(limites["limite_inferior"])
    )
    limpio[config.COLUMNAS_CRITICAS[1]] = winsorizar_price(price_original, limites["limite_superior"])
    bitacora.anotar(
        "price", "valores atipicos (IQR)", int(limpio["es_atipico_precio"].sum()),
        "winsorizar (clip) + bandera es_atipico_precio",
        "6,32 % por arriba; price entra a un modelo predictivo",
        f"{int(limpio['es_atipico_precio'].sum()):,} filas ajustadas al limite, 0 eliminadas",
    )
    return limpio