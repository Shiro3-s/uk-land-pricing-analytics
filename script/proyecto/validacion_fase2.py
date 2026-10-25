"""Validacion fase 2: revision de datos antes de clasificacion.

Revisa consistencia temporal, nulos, coherencia del esfuerzo, cruces
y define un set limpio de variables X para modelado.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto import config
from proyecto import indicadores
import pandas as pd


def regla_iqr(s: pd.Series):
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {}
    q1 = s_clean.quantile(0.25)
    q3 = s_clean.quantile(0.75)
    iqr = q3 - q1
    lim_sup = q3 + 1.5 * iqr
    lim_inf = q1 - 1.5 * iqr
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lim_inf": float(lim_inf),
        "lim_sup": float(lim_sup),
        "out_sup": int((s_clean > lim_sup).sum()),
        "out_inf": int((s_clean < lim_inf).sum()),
    }


def main() -> None:
    print("=" * 72)
    print("FASE 2: VALIDACION Y DEPURACION DE DATOS")
    print("=" * 72)

    maestro = pd.read_parquet(config.RUTA_MAESTRO).copy()
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).copy()
    hipoteca_2022 = pd.read_parquet(config.RUTA_HIPOTECA_2022).copy()

    print("\n[1] CONSISTENCIA TEMPORAL")
    print(f"Maestro: {maestro.shape}")
    print("Anios por region:")
    print(maestro.groupby("region")["anio"].nunique().to_string())
    print(f"Rango: {maestro['anio'].min()} - {maestro['anio'].max()}")

    # Esfuerzo
    tasa = float(hipotecas["tasa_inicial_pct"].mean())
    comisiones = float(hipotecas["comisiones_total"].mean())
    df = maestro.copy()
    df["cuota_mensual"] = df.apply(
        lambda r: indicadores.cuota_mensual_pmt(r["precio_mediana"], tasa, config.PLAZO_ANIOS, config.LTV, comisiones),
        axis=1,
    )
    df["salario_joven_mensual"] = df["salario_joven_region"] / 12.0
    df["esfuerzo"] = df["cuota_mensual"] / df["salario_joven_mensual"]

    print("\n[2] NULOS Y DUPLICADOS")
    print(f"Nulos totales: {int(df.isna().sum().sum())}")
    print(f"Duplicados (anio,region): {int(df.duplicated(['anio','region']).sum())}")

    print("\n[3] COHERENCIA DEL ESFUERZO")
    print(df["esfuerzo"].describe())
    print(f"\nFilas con esfuerzo > 0.30: {int((df['esfuerzo']>0.30).sum())} / {len(df)}")
    print("Esfuerzo medio por region:")
    print(df.groupby("region")["esfuerzo"].mean().sort_values(ascending=False).round(3).to_string())

    print("\n[4] CRUCE CON OUTLIERS TRATADOS")
    print(f"Tasa hipotecas limpia (media): {tasa:.3f}%")
    print(f"Comisiones media: {comisiones:.2f}")
    print(f"hipoteca_2022 filas: {len(hipoteca_2022)}")
    print("hipoteca_2022 cuota_pct_salario:")
    print(hipoteca_2022[["region","cuota_pct_salario","accesible"]].sort_values("cuota_pct_salario").to_string(index=False))

    print("\n[5] REGLA IQR EN ESFUERZO")
    iq = regla_iqr(df["esfuerzo"])
    print(iq)

    print("\n[6] SET DE VARIABLES PARA X (propuesto)")
    num_vars = ["precio_mediana", "salario_joven_region", "anos_salario", "precio_var_pct",
                "tmax_anom", "tmin_anom", "lluvia_anom", "sol_anom", "af_anom"]
    print("Numericas:", num_vars)
    print("Categorica:", ["region"])
    print("Identificadores a excluir:", ["anio"])
    print("Nulos en variables propuestas:", int(df[num_vars + ["region"]].isna().sum().sum()))

    # validar
    ok = True
    if df.duplicated(['anio','region']).sum() != 0:
        ok = False
    if df.isna().sum().sum() != 0:
        ok = False
    if df.groupby('region')['anio'].nunique().min() != 6:
        ok = False
    if (df['esfuerzo'] <= 0).sum() != 0:
        ok = False

    print("\n" + "=" * 72)
    if ok:
        print("VALIDACION: DATOS VALIDADOS OK")
    else:
        print("VALIDACION: REVISAR")
    print("=" * 72)


if __name__ == "__main__":
    main()
