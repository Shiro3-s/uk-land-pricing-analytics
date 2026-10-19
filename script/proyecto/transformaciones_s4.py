"""Preparacion de la matriz X·y para la fase 4 (transformacion y reduccion).

Aplica el metodo del proyecto sobre el cuadro maestro (60 filas):
  - objetivo y: esfuerzo hipotecario por region y anio (cuota PMT sobre el
    precio mediano / salario joven mensual; tasa winsorizada y comisiones de
    la oferta 2022, 90 % LTV y 25 anios);
  - rasgos: 9 numericos estandarizados con StandardScaler (media 0, desv 1);
  - region nominal: one-hot con OneHotEncoder (10 columnas);
  - nulos neutrales: sol_anom y precio_var_pct (primer anio) a 0;
  - fuera de X: anio y region (agrupadores), n_transacciones, precio_medio y
    las escalas derivadas del salario;
  - salida: data/integrado/matriz_features.parquet (+ muestra de 10 filas).

El pipeline usa ColumnTransformer (escala + one-hot en un solo estimador), el
mismo patron de la fase 4 donde el ajuste ira dentro de cada pliegue (sin fuga
de datos). Insumo: config.RUTA_MAESTRO y RUTA_HIPOTECAS_LIMPIO.
"""
# Ejecutar:  python script/proyecto/transformaciones_s4.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in (None, ""):  # lanzado como script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys, Path

from proyecto import config, indicadores

RASGOS_NUMERICOS = [
    "precio_mediana", "salario_joven_region", "tmax_anom", "tmin_anom",
    "lluvia_anom", "sol_anom", "af_anom", "precio_var_pct", "anos_salario",
]


def cargar_insumo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el cuadro maestro (60 filas) y la oferta hipotecaria ya limpia."""
    maestro = pd.read_parquet(config.RUTA_MAESTRO).sort_values(["region", "anio"])
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    return maestro, hipotecas


def agregar_esfuerzo(maestro: pd.DataFrame, hipotecas: pd.DataFrame
                     ) -> tuple[pd.DataFrame, float, float]:
    """Objetivo y: cuota mensual PMT sobre el precio mediano / salario joven
    mensual. Usa la tasa inicial media winsorizada y las comisiones promedio
    de la oferta 2022 (config: 90 % LTV, 25 anios)."""
    tasa = float(hipotecas["tasa_inicial_pct"].mean())
    comisiones = float(hipotecas["comisiones_total"].mean())
    df = maestro.copy()
    cuota = df["precio_mediana"].apply(
        lambda p: indicadores.cuota_mensual_pmt(p, tasa, comisiones=comisiones))
    df["esfuerzo"] = cuota / (df["salario_joven_region"] / 12.0)
    return df, tasa, comisiones


def completar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 0 como valor neutral: anomalia nula (sin anomalia) o variacion
    nula en el primer anio de la serie (no hay anio previo que comparar)."""
    m = df.copy()
    for col in ("sol_anom", "precio_var_pct"):
        m[col] = m[col].fillna(0.0)
    return m


def construir_matriz(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Arma X con ColumnTransformer (StandardScaler + OneHotEncoder) y y
    (esfuerzo): escala y codifica en un solo estimador, el mismo patron que
    se reajustara dentro de cada pliegue en la fase 4."""
    preprocesador = ColumnTransformer(
        transformers=[
            ("escala", StandardScaler(), RASGOS_NUMERICOS),
            ("region", OneHotEncoder(sparse_output=False), ["region"]),
        ],
        remainder="drop",
    )
    salida = preprocesador.fit_transform(df[RASGOS_NUMERICOS + ["region"]])
    col_region = [f"region_{v}" for v in
                  preprocesador.named_transformers_["region"].categories_[0]]
    x = pd.DataFrame(salida,
                     columns=[f"z_{c}" for c in RASGOS_NUMERICOS] + col_region,
                     index=df.index)
    y = df["esfuerzo"].astype(float).rename("esfuerzo")
    return x, y


def guardar_matriz(x: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> None:
    """Valida la matriz y la guarda en parquet (completa y muestra de 10)."""
    assert len(x) == len(y) == len(df) == 60, "la matriz no tiene 60 filas"
    assert int(x.isna().sum().sum()) == 0, "la matriz X tiene nulos"
    assert bool(y.gt(0).all()), "el esfuerzo debe ser positivo"
    assert x.shape[1] == len(RASGOS_NUMERICOS) + df["region"].nunique(), \
        "numero de columnas (rasgos + one-hot) incorrecto"
    ruta = config.DATA_INTEGRADO / "matriz_features.parquet"
    ruta_muestra = config.DATA_INTEGRADO / "matriz_features_muestra.parquet"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    matriz = pd.concat([df[["region", "anio"]], x, y], axis=1)
    matriz.to_parquet(ruta, index=False)
    matriz.head(10).to_parquet(ruta_muestra, index=False)
    print(f"  guardado {str(ruta.relative_to(config.PROYECTO))} "
          f"({matriz.shape})")
    print(f"  guardado {str(ruta_muestra.relative_to(config.PROYECTO))} "
          f"({matriz.head(10).shape})")


def main() -> None:
    """Ejecuta la preparacion de la matriz X·y (metodo usado en la fase 4)."""
    print("=" * 72)
    print("PREPARACION DE LA MATRIZ X·y (fase 4: transformacion y reduccion)")
    print("=" * 72)

    # 1. insumo y nulos
    maestro, hipotecas = cargar_insumo()
    print("\n[1] Insumo maestro_joven_uk")
    print(f"  filas {len(maestro)} | anios {maestro['anio'].min()}-"
          f"{maestro['anio'].max()} | regiones {maestro['region'].nunique()}")
    columnas = ["tmax_anom", "tmin_anom", "lluvia_anom", "sol_anom",
                "af_anom", "precio_var_pct", "anos_salario"]
    print("  nulos en candidatos a rasgos:")
    print(maestro[columnas].isna().sum().to_string())

    df = completar_nulos(maestro)
    df, tasa, comisiones = agregar_esfuerzo(df, hipotecas)
    print(f"  tasa winsorizada {tasa:.2f} % | comisiones GBP {comisiones:,.0f} | "
          f"90 % LTV, 25 anios")

    # 2. objetivo y
    print("\n[2] Objetivo y = esfuerzo (cuota mensual / salario joven mensual)")
    for region, valor in (df.groupby("region", observed=True)["esfuerzo"]
                          .mean().sort_values().items()):
        estado = ("accesible" if valor <= config.UMBRAL_ACCESIBILIDAD
                  else "no accesible")
        print(f"  {region:<26} esfuerzo medio {valor:.3f} -> {estado}")
    print(f"  umbral {config.UMBRAL_ACCESIBILIDAD:.0%} "
          f"(config.UMBRAL_ACCESIBILIDAD)")

    # 3. metodo: escalado + one-hot
    print("\n[3] Metodo: ColumnTransformer (StandardScaler + OneHotEncoder)")
    x, y = construir_matriz(df)
    z = x[[f"z_{c}" for c in RASGOS_NUMERICOS]].to_numpy(dtype=float)
    print(f"  StandardScaler tras ajustar: media {z.mean():.4f} / "
          f"desv {z.std():.3f} (media 0, desv 1)")
    print(f"  one-hot de region: {int(x.shape[1]) - len(RASGOS_NUMERICOS)} "
          f"columnas (OneHotEncoder)")

    # 4. matriz X·y
    print("\n[4] Matriz X·y construida")
    print(f"  X {x.shape} | y {y.shape[0]} | nulos en X: "
          f"{int(x.isna().sum().sum())}")
    print(f"  esfuerzo: media {y.mean():.3f} | rango "
          f"{y.min():.3f}-{y.max():.3f}")
    print("\n  Matriz y (esfuerzo) por region y anio: 10 regiones x 6 anios")
    tab = y.to_frame().assign(region=df["region"], anio=df["anio"].astype(int))
    print(tab.pivot(index="region", columns="anio", values="esfuerzo")
          .round(3).to_string())
    print("\n  Distribucion del esfuerzo (y):")
    tramos_y = pd.cut(y, bins=[0, 0.3, 0.5, 0.7, 1.0, 2.0], right=False)
    print(tramos_y.value_counts(normalize=True).sort_index().round(3).to_string())

    guardar_matriz(x, y, df)


if __name__ == "__main__":
    main()