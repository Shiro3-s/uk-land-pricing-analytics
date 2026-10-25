"""Analisis exploratorio de datos (EDA) de la matriz de accesibilidad.

Espejo en terminal del libro 07_EDA_Accesibilidad.ipynb.
Sigue la estructura de un informe EDA:
  [1] insumo y objetivo (y = esfuerzo);
  [2] calidad de datos (herencia de los libros 01-02);
  [3] univariado: describe + tendencia central/dispersión/forma + outliers IQR;
  [4] bivariado: correlaciones precio/salario vs esfuerzo y por región;
  [5] multivariado: heatmap de correlaciones de los 9 rasgos + y;
  [6] figuras (histograma, boxplot, barras, dispersión+recta, heatmap,
      matriz de dispersión) guardadas en md/figuras/;
  [7] hallazgos: mínimo 3 en formato evidencia -> acción.

Insumo: config.RUTA_MAESTRO y RUTA_HIPOTECAS_LIMPIO (mismo cruce que el
libro 06 y matriz_xy.py, sin fuga de datos).
"""
# Ejecutar:  python script/proyecto/eda.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

if __package__ in (None, ""):  # lanzado como script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys, Path

from proyecto import config, indicadores

plt.rcParams["figure.dpi"] = 140
PALETA = {"azul": "#4C72B0", "rojo": "#C44E52", "verde": "#55A868"}

RASGOS = [
    "precio_mediana", "salario_joven_region", "tmax_anom", "tmin_anom",
    "lluvia_anom", "sol_anom", "af_anom", "precio_var_pct", "anos_salario",
]


def cargar_insumo() -> tuple[pd.DataFrame, pd.DataFrame]:
    maestro = pd.read_parquet(config.RUTA_MAESTRO).sort_values(["region", "anio"])
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    return maestro, hipotecas


def agregar_esfuerzo(maestro: pd.DataFrame, hipotecas: pd.DataFrame
                     ) -> tuple[pd.DataFrame, float, float]:
    tasa = float(hipotecas["tasa_inicial_pct"].mean())
    comisiones = float(hipotecas["comisiones_total"].mean())
    df = maestro.copy()
    cuota = df["precio_mediana"].apply(
        lambda p: indicadores.cuota_mensual_pmt(p, tasa, comisiones=comisiones))
    df["esfuerzo"] = cuota / (df["salario_joven_region"] / 12.0)
    return df, tasa, comisiones


def estadisticos_univariados(df: pd.DataFrame) -> pd.DataFrame:
    """Tendencia central, dispersión y forma por columna numérica + outliers IQR."""
    columnas = RASGOS + ["esfuerzo"]
    filas = {}
    for c in columnas:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        filas[c] = {
            "media": float(s.mean()), "mediana": float(s.median()),
            "desv": float(s.std()), "rango": float(s.max() - s.min()),
            "asimetria": float(stats.skew(s)),
            "curtosis": float(stats.kurtosis(s)),
            "outliers_iqr": n_out, "pct_outliers": 100 * n_out / len(s),
        }
    return pd.DataFrame(filas).T


def guardar_figura(fig, nombre: str) -> Path:
    config.MD_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = config.MD_FIGURAS / f"eda_{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def fig_histograma(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["esfuerzo"], bins=12, color=PALETA["azul"], edgecolor="white")
    ax.axvline(df["esfuerzo"].mean(), color=PALETA["rojo"], ls="--",
               label=f"media {df['esfuerzo'].mean():.3f}")
    ax.axvline(df["esfuerzo"].median(), color=PALETA["verde"], ls="--",
               label=f"mediana {df['esfuerzo'].median():.3f}")
    ax.axvline(config.UMBRAL_ACCESIBILIDAD, color="black", lw=1, ls=":",
               label=f"umbral {config.UMBRAL_ACCESIBILIDAD:.0%}")
    ax.set_title("El esfuerzo hipotecario supera el umbral del 30 % en las 60 filas")
    ax.set_xlabel("esfuerzo (cuota PMT / salario joven mensual)")
    ax.set_ylabel("frecuencia")
    ax.legend(fontsize=8)
    guardar_figura(fig, "histograma_esfuerzo")


def fig_boxplot_region(df: pd.DataFrame) -> None:
    orden = (df.groupby("region", observed=True)["esfuerzo"]
             .median().sort_values().index)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    datos = [df.loc[df["region"] == r, "esfuerzo"] for r in orden]
    bp = ax.boxplot(datos, tick_labels=orden, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETA["azul"])
        patch.set_alpha(0.6)
    ax.axhline(config.UMBRAL_ACCESIBILIDAD, color=PALETA["rojo"], ls="--",
               lw=1, label=f"umbral {config.UMBRAL_ACCESIBILIDAD:.0%}")
    ax.set_title("Ninguna región queda bajo el umbral del 30 % del salario")
    ax.set_xlabel("región")
    ax.set_ylabel("esfuerzo (ratio)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=8)
    guardar_figura(fig, "boxplot_esfuerzo_region")


def fig_barras_region(df: pd.DataFrame) -> None:
    resumen = (df.groupby("region", observed=True)["esfuerzo"]
               .median().sort_values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colores = np.where(resumen.values >= 1.0, PALETA["rojo"], PALETA["azul"])
    ax.barh(resumen.index, resumen.values, color=colores)
    for i, v in enumerate(resumen.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(config.UMBRAL_ACCESIBILIDAD, color="black", ls="--", lw=1)
    ax.set_title("Mediana del esfuerzo por región: London supera 1,0 (todo el salario)")
    ax.set_xlabel("esfuerzo mediano (ratio)")
    guardar_figura(fig, "barras_esfuerzo_region")


def fig_dispersion_precio(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["precio_mediana"], df["esfuerzo"], color=PALETA["azul"],
               alpha=0.7)
    m, b = np.polyfit(df["precio_mediana"], df["esfuerzo"], 1)
    xs = np.linspace(df["precio_mediana"].min(), df["precio_mediana"].max(), 50)
    ax.plot(xs, m * xs + b, color=PALETA["rojo"], lw=1.5,
            label=f"recta: y = {m:.2e}·x + {b:.2f}")
    r = df["precio_mediana"].corr(df["esfuerzo"])
    ax.set_title(f"El precio explica el esfuerzo (r = {r:.3f}, fuerte y positivo)")
    ax.set_xlabel("precio mediano de vivienda (GBP)")
    ax.set_ylabel("esfuerzo (ratio)")
    ax.legend(fontsize=8)
    guardar_figura(fig, "dispersion_precio_esfuerzo")


def fig_heatmap(df: pd.DataFrame) -> None:
    cols = RASGOS + ["esfuerzo"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels=cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), labels=cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlaciones de Pearson: los 9 rasgos y el esfuerzo")
    guardar_figura(fig, "heatmap_correlaciones")


def fig_matriz_dispersion(df: pd.DataFrame) -> None:
    cols = ["precio_mediana", "salario_joven_region", "anos_salario", "esfuerzo"]
    fig, ax = plt.subplots(figsize=(8, 8), ncols=len(cols), nrows=len(cols),
                           sharex="col", sharey="row")
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            a = ax[i, j]
            if i == j:
                a.hist(df[c1], bins=10, color=PALETA["azul"], edgecolor="white")
            else:
                a.scatter(df[c2], df[c1], s=8, color=PALETA["azul"], alpha=0.6)
            if i == len(cols) - 1:
                a.set_xlabel(c2, fontsize=7)
            if j == 0:
                a.set_ylabel(c1, fontsize=7)
            a.tick_params(labelsize=6)
    fig.suptitle("Matriz de dispersión: precio, salario, años de salario y esfuerzo",
                 y=1.0)
    guardar_figura(fig, "matriz_dispersion")


def seccion_hallazgos(df: pd.DataFrame) -> None:
    r_precio = df["precio_mediana"].corr(df["esfuerzo"])
    r_salario = df["salario_joven_region"].corr(df["esfuerzo"])
    med = df.groupby("region", observed=True)["esfuerzo"].median()
    ratio = med.max() / med.min()
    print(f"  H1: el precio domina el esfuerzo (r = {r_precio:.3f}) frente al")
    print(f"      salario (r = {r_salario:.3f}): el numerador pesa más.")
    print(f"      Acción: la política de accesibilidad debe atacar el precio,")
    print(f"      no solo el salario.")
    print(f"  H2: la brecha entre la mediana más alta ({med.idxmax()},")
    print(f"      {med.max():.2f}) y la más baja ({med.idxmin()}, {med.min():.2f})")
    print(f"      es {ratio:.1f}×: la región es el predictor categórico clave.")
    print(f"      Acción: segmentar cualquier intervención por región, no nacional.")
    n_sobre1 = int((df["esfuerzo"] >= 1.0).sum())
    print(f"  H3: {n_sobre1} de {len(df)} filas ({100 * n_sobre1 / len(df):.1f} %)")
    print(f"      superan esfuerzo 1,0 (la cuota se come todo el salario joven).")
    print(f"      Acción: priorizar esas filas como foco de la fase de modelado.")


def main() -> None:
    print("=" * 70)
    print("EDA · Accesibilidad de vivienda joven")
    print("=" * 70)

    maestro, hipotecas = cargar_insumo()
    df, tasa, comisiones = agregar_esfuerzo(maestro, hipotecas)
    for col in ("sol_anom", "precio_var_pct"):
        df[col] = df[col].fillna(0.0)

    # 1. insumo y objetivo
    print(f"\n[1] Insumo: maestro {df.shape[0]} filas | "
          f"{df['region'].nunique()} regiones | {df['anio'].min()}-{df['anio'].max()}")
    print(f"    tasa winsorizada {tasa:.2f} % | comisiones GBP {comisiones:,.0f}")
    print(f"    y = esfuerzo: media {df['esfuerzo'].mean():.3f} | "
          f"mediana {df['esfuerzo'].median():.3f} | "
          f"rango {df['esfuerzo'].min():.3f}-{df['esfuerzo'].max():.3f}")

    # 2. calidad de datos (herencia)
    print("\n[2] Calidad de datos (herencia libros 01-02)")
    print(f"    nulos en sol_anom: {int(maestro['sol_anom'].isna().sum())} "
          f"(imputados a 0)")
    print(f"    nulos en precio_var_pct: {int(maestro['precio_var_pct'].isna().sum())} "
          f"(primer anio, imputados a 0)")
    print(f"    outliers de tasa por IQR: 9,77 % winsorizados (libro 02)")

    # 3. univariado
    print("\n[3] Univariado: tendencia central, dispersión y forma")
    tabla = estadisticos_univariados(df)
    print(tabla.round(3).to_string())
    asim = tabla["asimetria"].abs()
    print(f"\n    asimetria media |skew|: {asim.mean():.2f} | "
          f"columnas con |skew|>1: {int((asim > 1).sum())}")
    print(f"    outliers IQR en esfuerzo: "
          f"{int(tabla.loc['esfuerzo', 'outliers_iqr'])} "
          f"({tabla.loc['esfuerzo', 'pct_outliers']:.1f} %)")

    # 4. bivariado
    print("\n[4] Bivariado: precio y salario vs esfuerzo")
    for col in ("precio_mediana", "salario_joven_region", "anos_salario"):
        r = df[col].corr(df["esfuerzo"])
        print(f"    {col:<24} r = {r:+.3f}")
    print("\n    Mediana del esfuerzo por region:")
    med = df.groupby("region", observed=True)["esfuerzo"].median().sort_values()
    print(med.round(3).to_string())

    # 5. multivariado
    print("\n[5] Multivariado: heatmap de correlaciones (resumen)")
    cols = RASGOS + ["esfuerzo"]
    corr = df[cols].corr()
    pares = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pares.append((cols[i], cols[j], corr.iloc[i, j]))
    pares.sort(key=lambda t: abs(t[2]), reverse=True)
    print("    top 5 correlaciones entre rasgos (|r|):")
    for a, b, r in pares[:5]:
        print(f"      {a} <-> {b}: r = {r:+.3f}")

    # 6. figuras
    print("\n[6] Figuras guardadas en md/figuras/:")
    fig_histograma(df)
    fig_boxplot_region(df)
    fig_barras_region(df)
    fig_dispersion_precio(df)
    fig_heatmap(df)
    fig_matriz_dispersion(df)
    for nombre in ("histograma_esfuerzo", "boxplot_esfuerzo_region",
                   "barras_esfuerzo_region", "dispersion_precio_esfuerzo",
                   "heatmap_correlaciones", "matriz_dispersion"):
        print(f"    eda_{nombre}.png")

    # 7. hallazgos
    print("\n[7] Hallazgos (evidencia -> accion)")
    seccion_hallazgos(df)


if __name__ == "__main__":
    main()
