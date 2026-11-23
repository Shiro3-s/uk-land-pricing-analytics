"""Figuras del analisis (una figura por archivo, guardadas en md/figuras)."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 140


def guardar_figura(fig, ruta: Path) -> Path:
    """Guarda la figura en PNG (bbox compacto) y devuelve la ruta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def fig_historico_nacional(historico: pd.DataFrame) -> plt.Figure:
    """Precio real nacional vs salario joven 1999-2020."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(historico["anio"], historico["precio_promedio_real"], marker="o",
            ms=3, label="precio medio nacional (GBP 2020)")
    ax.plot(historico["anio"], historico["salario_joven_nacional"], marker="s",
            ms=3, label="salario joven 18-34 (GBP 2020)")
    ax.set_title("Precio de vivienda vs salario joven (Reino Unido, 1999-2020)")
    ax.set_xlabel("anio"); ax.set_ylabel("GBP ajustados a 2020")
    ax.legend(); ax.grid(alpha=0.3)
    return fig


def fig_precios_region(precios_region: pd.DataFrame) -> plt.Figure:
    """Evolucion del precio medio por region (2015-2026)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for region, g in precios_region.groupby("region"):
        ax.plot(g["anio"], g["precio_medio"], marker=".", ms=4, label=region)
    ax.set_title("Precio medio de venta por region (GBP)")
    ax.set_xlabel("anio"); ax.set_ylabel("GBP")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    return fig


def fig_anos_salario(maestro: pd.DataFrame) -> plt.Figure:
    """Anios de salario completo para comprar, promedio 2015-2020, por region."""
    resumen = (maestro.groupby("region", observed=True)["anos_salario"]
               .mean().sort_values())
    colores = np.where(resumen.values >= 10, "#c0392b", "#2e86c1")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(resumen.index, resumen.values, color=colores)
    ax.axvline(10, color="gray", ls="--", lw=1)
    ax.text(10.02, -0.6, "10 anios de salario", fontsize=8, color="gray")
    ax.set_title("Anios de salario joven (18-34) necesarios para comprar (prom. 2015-2020)")
    ax.set_xlabel("precio mediana / salario joven regional")
    for i, v in enumerate(resumen.values):
        ax.text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=8)
    return fig


def fig_anomalia_lluvia(anomalias_nacional: pd.DataFrame) -> plt.Figure:
    """Anomalia anual de lluvia nacional (mm) en la ventana del cuadro."""
    a = anomalias_nacional
    x = a["anio"].astype(int).tolist()
    y = a["lluvia_anom"]
    colores = np.where(y.values > 0, "#2980b9", "#c0392b")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x, y, color=colores, alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + (0.02 if yi >= 0 else -0.12), f"{yi:+.1f}",
                ha="center", fontsize=8)
    ax.set_title("Anomalia de lluvia frente a la normal historica por estacion (Reino Unido)")
    ax.set_xlabel("anio"); ax.set_ylabel("mm (estaciones promediadas)")
    ax.grid(axis="y", alpha=0.3)
    return fig


def fig_clima_precio(maestro: pd.DataFrame) -> plt.Figure:
    """Variacion anual del precio vs anomalia de lluvia (una recta global)."""
    sub = maestro.dropna(subset=["lluvia_anom", "precio_var_pct"]).copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for region, g in sub.groupby("region"):
        ax.scatter(g["lluvia_anom"], g["precio_var_pct"], s=16, label=region)
    b = np.polyfit(sub["lluvia_anom"], sub["precio_var_pct"], 1)
    xx = np.linspace(sub["lluvia_anom"].min(), sub["lluvia_anom"].max(), 50)
    ax.plot(xx, np.polyval(b, xx), color="black", lw=1.5, ls="--",
            label=f"regresion (pendiente {b[0]:+.2f} %/mm)")
    ax.set_title("Variacion anual del precio vs anomalia de lluvia (2016-2020, por region)")
    ax.set_xlabel("anomalia de lluvia (mm)"); ax.set_ylabel("variacion del precio mediana (%)")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
    return fig


def fig_hipoteca_2022(hipoteca: pd.DataFrame) -> plt.Figure:
    """Cuota hipotecaria como % del salario joven mensual por region (corte 2022)."""
    ord = hipoteca.sort_values("cuota_pct_salario")["region"].tolist()
    y = hipoteca.set_index("region").loc[ord, "cuota_pct_salario"] * 100
    acc = hipoteca.set_index("region").loc[ord, "accesible"]
    colores = np.where(acc.values, "#27ae60", "#c0392b")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(y.index, y.values, color=colores)
    ax.axhline(30, color="black", ls="--", lw=1.3)
    ax.text(9.2, 30.6, "umbral 30 % del salario", fontsize=8)
    for i, v in enumerate(y.values):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("% del salario joven mensual")
    ax.set_title("Esfuerzo hipotecario 2022 por region (mediana, 90 % LTV, 25 anios)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    return fig


def fig_nulos_fuentes(clasificacion: pd.DataFrame) -> plt.Figure:
    """% de faltantes por fuente y columna, con la clase MCAR/MAR/MNAR."""
    tabla = clasificacion.sort_values("pct", ascending=False)
    colores = {"MCAR": "#2e86c1", "MAR": "#f1c40f", "MNAR": "#c0392b",
               "MNAR estructural": "#c0392b", "MAR (depende de observadas)": "#f1c40f"}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    etiquetas = [f"{r['fuente']} · {r['columna']}" for r in tabla.to_dict("records")]
    bar_col = [colores.get(r["clasificacion"], "gray") for r in tabla.to_dict("records")]
    ax.barh(etiquetas, tabla["pct"], color=bar_col)
    for i, r in enumerate(tabla.to_dict("records")):
        ax.text(r["pct"] + 0.4, i, f"{r['clasificacion']} ({r['nulos']:,})",
                va="center", fontsize=8)
    ax.set_xlabel("% de la fuente")
    ax.set_title("Faltantes por fuente y clasificacion (Fase 3: MCAR / MAR / MNAR)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    return fig


def fig_outliers_tasa(serie_ori: pd.Series, serie_winsor: pd.Series,
                      evidencia: dict) -> plt.Figure:
    """Boxplot de la tasa inicial antes/despues de winsorizar con IQR (F2)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    datos = [serie_ori.dropna(), serie_winsor.dropna()]
    bp = ax.boxplot(datos, vert=True, patch_artist=True,
                    tick_labels=["tasa inicial 2022\n(original)",
                                 "tasa inicial 2022\n(winsorizada, IQR)"])
    for p in bp["boxes"]:
        p.set_facecolor("#bdc3c7")
    ax.axhline(evidencia["lim_sup"], color="#c0392b", ls="--", lw=1.2)
    ax.text(2.42, evidencia["lim_sup"], f"limite IQR {evidencia['lim_sup']:.2f} %",
            va="center", fontsize=8, color="#c0392b")
    ax.set_title(f"Outliers en tasa inicial (F2): IQR detecta {evidencia['fuera_iqr']:,} "
                 f"({evidencia['pct_iqr']} %) vs z-score {evidencia['fuera_z']}")
    ax.set_ylabel("tasa inicial (%)")
    ax.grid(axis="y", alpha=0.3)
    return fig


def fig_fases() -> plt.Figure:
    """Mapa completo de fases KDD + CRISP-DM del proyecto analitico."""
    fases = [
        ("1", "Problema, objetivos y marcos", "Seleccion", "Comprension del negocio", "Completada"),
        ("2", "Ingesta y contratos de datos (5 fuentes)", "Seleccion + prep. I", "Comprension de datos", "Completada"),
        ("3", "Limpieza (MCAR/MAR/MNAR, IQR, bitacora)", "Preprocesamiento II", "Preparacion de datos", "Completada"),
        ("4", "Integracion, PMT y anomalias climaticas", "Preprocesamiento II", "Preparacion de datos", "Completada"),
        ("5", "Transformacion, escalado, VIF y matriz X-y", "Transformacion", "Preparacion de datos", "Completada"),
        ("6", "Analisis exploratorio (EDA) y hallazgos", "Interpretacion I", "Exploracion", "Completada"),
        ("7", "Regresion tabular comparativa (CV-5)", "Mineria (supervisada)", "Modelado", "Completada"),
        ("8", "Clasificacion I: Arbol CART interpretativo", "Mineria (supervisada)", "Modelado y evaluacion", "Completada"),
        ("9", "Torneo: Arbol vs Naive Bayes vs KNN", "Mineria (supervisada)", "Evaluacion y seleccion", "Completada"),
        ("10", "Clustering (K-Means + Ward + 1M) y Apriori", "Mineria (no supervisada)", "Modelado y patrones", "Completada"),
        ("11", "Evaluacion forense, CV-5 y umbrales", "Evaluacion avanzada", "Evaluacion", "Completada"),
        ("12", "Despliegue Web (Front + Back + BD)", "Operacionalizacion", "Despliegue", "Proxima etapa"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.axis("off")
    tabla = ax.table(
        cellText=[list(f) for f in fases],
        colLabels=["#", "Fase del proyecto", "KDD", "CRISP-DM", "Estado"],
        loc="center", cellLoc="left")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.5)
    tabla.scale(1, 1.38)
    for i, fila in enumerate(fases):
        estado = fila[4]
        celda = tabla[i + 1, 4]
        if estado == "Completada":
            celda.set_facecolor("#27ae60")
            celda.set_text_props(color="white", weight="bold")
        else:
            celda.set_facecolor("#2980b9")
            celda.set_text_props(color="white", weight="bold")
    ax.set_title("Mapa de fases del proyecto (ciclo integrado KDD + CRISP-DM)",
                 fontsize=10.5, pad=12)
    return fig