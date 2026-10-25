"""EDA exploratorio para revision de datos (fase 1).

Lee maestro_joven_uk y calcula el esfuerzo hipotecario usando la logica
del proyecto para validar distribuciones, outliers, correlaciones y
relaciones por region.

Genera figuras en md/figuras con prefijo rev_ sin sobreescribir las existentes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto import config
from proyecto import indicadores


CARPETA_FIGURAS = config.MD_FIGURAS


def figura(nombre: str) -> Path:
    return CARPETA_FIGURAS / nombre


def resumen_numerico(s: pd.Series) -> dict:
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {"n": 0}
    return {
        "n": int(len(s_clean)),
        "media": float(s_clean.mean()),
        "mediana": float(s_clean.median()),
        "min": float(s_clean.min()),
        "max": float(s_clean.max()),
        "desv": float(s_clean.std()),
        "asimetria": float(s_clean.skew()),
        "curtosis": float(s_clean.kurtosis()),
    }


def regla_iqr(s: pd.Series) -> dict:
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {}
    q1 = s_clean.quantile(0.25)
    q3 = s_clean.quantile(0.75)
    iqr = q3 - q1
    lim_sup = q3 + 1.5 * iqr
    lim_inf = q1 - 1.5 * iqr
    out_sup = int((s_clean > lim_sup).sum())
    out_inf = int((s_clean < lim_inf).sum())
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lim_inf": float(lim_inf),
        "lim_sup": float(lim_sup),
        "outliers_sup": out_sup,
        "outliers_inf": out_inf,
        "outliers_tot": out_sup + out_inf,
    }


def main() -> None:
    print("=" * 72)
    print("EDA EXPLORATORIO - FASE 1")
    print("=" * 72)

    # Carga
    maestro = pd.read_parquet(config.RUTA_MAESTRO).copy()
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).copy()

    print(f"\nMaestro: {maestro.shape[0]} filas x {maestro.shape[1]} columnas")
    print(f"Regiones: {sorted(maestro['region'].unique().tolist())}")
    print(f"Anios: {maestro['anio'].min()} - {maestro['anio'].max()}")
    print(f"Clave unica (anio,region): {(maestro.duplicated(['anio','region'])).sum() == 0}")

    # Calculo de esfuerzo
    tasa = float(hipotecas["tasa_inicial_pct"].mean())
    comisiones = float(hipotecas["comisiones_total"].mean())
    print(f"\nTasa media hipotecas (limpia): {tasa:.3f}% | Comisiones media: {comisiones:.2f}")

    df = maestro.copy()
    df["cuota_mensual"] = df.apply(
        lambda r: indicadores.cuota_mensual_pmt(r["precio_mediana"], tasa, config.PLAZO_ANIOS, config.LTV, comisiones),
        axis=1,
    )
    df["salario_joven_mensual"] = df["salario_joven_region"] / 12.0
    df["esfuerzo"] = df["cuota_mensual"] / df["salario_joven_mensual"]

    print(f"\nEsfuerzo calculado: min={df['esfuerzo'].min():.3f} | mediana={df['esfuerzo'].median():.3f} | max={df['esfuerzo'].max():.3f}")
    print(f"Regiones con esfuerzo > 0.30: {int((df['esfuerzo']>0.30).sum())} de {len(df)} filas")

    # Resumen numerico
    for col in ["esfuerzo", "precio_mediana", "salario_joven_region", "anos_salario"]:
        r = resumen_numerico(df[col])
        print(f"\n{col}: n={r['n']} | media={r['media']:.2f} | mediana={r['mediana']:.2f} | min={r['min']:.2f} | max={r['max']:.2f} | desv={r['desv']:.2f} | asim={r['asimetria']:.2f} | curt={r['curtosis']:.2f}")

    # IQR
    print("\n" + "=" * 72)
    print("REGLA IQR")
    print("=" * 72)
    for col in ["esfuerzo", "precio_mediana"]:
        iq = regla_iqr(df[col])
        if iq:
            print(f"{col}: Q1={iq['q1']:.4f} Q3={iq['q3']:.4f} IQR={iq['iqr']:.4f} | lim_inf={iq['lim_inf']:.4f} lim_sup={iq['lim_sup']:.4f} | out_sup={iq['outliers_sup']} out_inf={iq['outliers_inf']} tot={iq['outliers_tot']}")

    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)

    # 1. Histograma esfuerzo
    plt.figure(figsize=(8, 4.6))
    datos = df["esfuerzo"].dropna()
    plt.hist(datos, bins=20, color="#0e7490", edgecolor="white")
    plt.axvline(datos.mean(), color="#dc2626", ls="--", lw=2, label=f"Media {datos.mean():.3f}")
    plt.axvline(datos.median(), color="#059669", ls="-", lw=2, label=f"Mediana {datos.median():.3f}")
    plt.axvline(0.30, color="#7c3aed", ls=":", lw=2, label="Umbral 0.30")
    plt.xlabel("Esfuerzo (cuota/salario joven mensual)")
    plt.ylabel("Frecuencia")
    plt.title("Distribucion del esfuerzo hipotecario")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figura("rev_histograma_esfuerzo.png"), dpi=110)
    plt.close()
    print("\nGuardado: rev_histograma_esfuerzo.png")

    # 2. Histograma precio_mediana
    plt.figure(figsize=(8, 4.6))
    datos = df["precio_mediana"].dropna() / 1000.0
    plt.hist(datos, bins=20, color="#2563eb", edgecolor="white")
    plt.axvline(datos.mean(), color="#dc2626", ls="--", lw=2, label=f"Media {datos.mean():.0f} mil")
    plt.axvline(datos.median(), color="#059669", ls="-", lw=2, label=f"Mediana {datos.median():.0f} mil")
    plt.xlabel("Precio mediana (miles GBP)")
    plt.ylabel("Frecuencia")
    plt.title("Distribucion del precio mediana")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figura("rev_histograma_precio_mediana.png"), dpi=110)
    plt.close()
    print("Guardado: rev_histograma_precio_mediana.png")

    # 3. Boxplot esfuerzo
    plt.figure(figsize=(8, 2.8))
    plt.boxplot(df["esfuerzo"].dropna(), vert=False, widths=0.55)
    plt.axvline(0.30, color="#7c3aed", ls=":", lw=2, label="Umbral 0.30")
    plt.xlabel("Esfuerzo (cuota/salario joven mensual)")
    plt.title("Boxplot del esfuerzo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figura("rev_boxplot_esfuerzo.png"), dpi=110)
    plt.close()
    print("Guardado: rev_boxplot_esfuerzo.png")

    # 4. Dispersión precio_mediana vs esfuerzo
    plt.figure(figsize=(7.8, 5.0))
    sub = df.dropna(subset=["precio_mediana", "esfuerzo"])
    plt.scatter(sub["precio_mediana"] / 1000.0, sub["esfuerzo"], s=30, alpha=0.7, color="#0e7490")
    x = sub["precio_mediana"] / 1000.0
    y = sub["esfuerzo"]
    if len(x) > 1:
        pend, inter = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        plt.plot(xx, pend * xx + inter, color="#dc2626", lw=2.5, label=f"y = {pend:.3f}x + {inter:.3f}")
    plt.axhline(0.30, color="#7c3aed", ls=":", lw=2, label="Umbral 0.30")
    plt.xlabel("Precio mediana (miles GBP)")
    plt.ylabel("Esfuerzo")
    plt.title("Precio mediana vs esfuerzo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figura("rev_dispersion_precio_esfuerzo.png"), dpi=110)
    plt.close()
    print("Guardado: rev_dispersion_precio_esfuerzo.png")

    # 5. Por region
    agg_reg = df.groupby("region", observed=True).agg(
        esfuerzo_medio=("esfuerzo", "mean"),
        esfuerzo_mediana=("esfuerzo", "median"),
        anos_salario_medio=("anos_salario", "mean"),
        precio_mediana_medio=("precio_mediana", "mean"),
        n=("esfuerzo", "count"),
    ).reset_index().sort_values("esfuerzo_medio", ascending=False)

    print("\n" + "=" * 72)
    print("POR REGION (promedio 2015-2020)")
    print("=" * 72)
    print(agg_reg.round(3).to_string(index=False))

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8))
    agg_reg.plot.barh(x="region", y="esfuerzo_medio", ax=ejes[0], color="#dc2626")
    ejes[0].axvline(0.30, color="#7c3aed", ls=":", lw=2)
    ejes[0].set_title("Esfuerzo medio por region")
    ejes[0].set_xlabel("Esfuerzo")
    agg_reg.plot.barh(x="region", y="anos_salario_medio", ax=ejes[1], color="#059669")
    ejes[1].set_title("Años de salario medio por region")
    ejes[1].set_xlabel("Años de salario")
    plt.tight_layout()
    plt.savefig(figura("rev_barras_esfuerzo_anos_salario_region.png"), dpi=110)
    plt.close()
    print("Guardado: rev_barras_esfuerzo_anos_salario_region.png")

    # 6. Mapa de calor correlaciones
    cols_corr = ["precio_mediana", "salario_joven_region", "anos_salario", "precio_var_pct",
                 "tmax_anom", "tmin_anom", "lluvia_anom", "sol_anom", "af_anom", "esfuerzo"]
    corr = df[cols_corr].corr()
    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlacion Pearson")
    plt.xticks(range(len(cols_corr)), cols_corr, rotation=45, ha="right")
    plt.yticks(range(len(cols_corr)), cols_corr)
    for i in range(len(cols_corr)):
        for j in range(len(cols_corr)):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Mapa de calor de correlaciones")
    plt.tight_layout()
    plt.savefig(figura("rev_heatmap_correlaciones.png"), dpi=110)
    plt.close()
    print("Guardado: rev_heatmap_correlaciones.png")

    # 7. Matriz de dispersion
    cols_sm = ["precio_mediana", "salario_joven_region", "anos_salario", "esfuerzo"]
    eje = scatter_matrix(df[cols_sm].dropna(), figsize=(9, 9), diagonal="hist", alpha=0.5, color="#0e7490")
    plt.suptitle("Matriz de dispersion", y=0.92)
    plt.tight_layout()
    plt.savefig(figura("rev_matriz_dispersion.png"), dpi=110)
    plt.close()
    print("Guardado: rev_matriz_dispersion.png")

    print("\n" + "=" * 72)
    print("FASE 1 COMPLETADA - Figuras en md/figuras con prefijo rev_")
    print("=" * 72)


if __name__ == "__main__":
    main()
