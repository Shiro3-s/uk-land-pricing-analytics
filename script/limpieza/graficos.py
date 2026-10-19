"""Graficas listas para exponer (matplotlib). Cada funcion devuelve una Figure.

Separacion de responsabilidad A2/A3: la viz vive aqui, no en los notebooks.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

from . import config

COLOR_PRINCIPAL = "#2563eb"
COLOR_SEGUNDARIO = "#f59e0b"
COLOR_MEDIANA = "#7c3aed"
COLOR_OJO = "#dc2626"
COLOR_VERDE = "#16a34a"


def establecer_estilo() -> None:
    """Estilo consistente para todas las figuras."""
    for nombre in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(nombre)
            break
        except OSError:
            continue
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid.axis": "y",
        "figure.autolayout": True,
    })


def presentar(fig: plt.Figure) -> None:
    """Renderiza la figura a PNG y la muestra en el notebook (una sola imagen).

    No se devuelve la figura: si una celda termina en ``presentar(...)``, Jupyter
    re-mostraria su valor (segunda imagen). Rasterizar antes de cerrar evita el
    doble render del backend inline.
    """
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    display(Image(data=buffer.getvalue()))


def guardar_figura(fig: plt.Figure, ruta: str | Path) -> Path:
    """Guarda la figura en PNG (para scripts/consola) y libera memoria."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return ruta


# --------------------------- Lectura ---------------------------
def transacciones_por_anio(df: pd.DataFrame) -> plt.Figure:
    """Barras: volumen de transacciones por anio."""
    conteos = df["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(conteos.index.astype(str), conteos.values / 1e6, color=COLOR_PRINCIPAL, alpha=0.85)
    ax.set_title("UK Property · transacciones por año (millones)")
    ax.set_xlabel("año")
    ax.set_ylabel("transacciones (millones)")
    ax.tick_params(axis="x", rotation=45)
    return fig


# --------------------------- Diagnóstico ---------------------------
def tasa_nulos_geo_por_anio(df: pd.DataFrame) -> plt.Figure:
    """Linea: tasa de nulos geo por anio (evidencia de MCAR)."""
    tasa = (
        df.groupby(df[config.COLUMNA_FECHA].dt.year)["latitude"]
        .apply(lambda s: s.isna().mean())
        .mul(100)
        .round(3)
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tasa.index.astype(str), tasa.values, marker="o", color=COLOR_SEGUNDARIO, linewidth=2)
    ax.axhline(float(tasa.mean()), color=COLOR_OJO, linestyle="--", linewidth=1,
               label=f"media {tasa.mean():.2f} %")
    ax.set_title("Nulos geo por año (%) · ausencia estable ≈ MCAR")
    ax.set_xlabel("año")
    ax.set_ylabel("nulos de georreferencia (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return fig


def top_categorias(df: pd.DataFrame, columna: str, top_n: int = 10) -> plt.Figure:
    """Barras horizontales de las top categorias de una columna."""
    conteos = df[columna].value_counts().head(top_n).iloc[::-1]
    etiquetas = [str(v) for v in conteos.index]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.42 * len(etiquetas) + 2)))
    ax.barh(etiquetas, conteos.values / 1e6, color=COLOR_PRINCIPAL, alpha=0.85)
    ax.set_title(f"{columna}: top {len(etiquetas)} (millones de registros)")
    ax.set_xlabel("transacciones (millones)")
    return fig


def pais_donut(df: pd.DataFrame) -> plt.Figure:
    """Donut: distribucion de country (England vs Wales)."""
    conteos = df["country"].value_counts()
    etiquetas = list(conteos.index)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pie(
        conteos.values,
        labels=etiquetas,
        autopct=lambda p: f"{p:.1f} %",
        colors=[COLOR_PRINCIPAL, COLOR_SEGUNDARIO],
        startangle=90,
        wedgeprops={"width": 0.35, "edgecolor": "white"},
    )
    ax.set_title("country: England vs Wales")
    return fig


def cuartiles_barras(resumen: pd.DataFrame, mediana: float) -> plt.Figure:
    """Barras del % de transacciones por cuartil (Q1..Q4) de price.

    Cada cuartil pesa 25 %; el realce visual deja claro que el mercado tipico
    (Q1 + Q2) queda por debajo de la mediana.
    """
    colores = [COLOR_VERDE, COLOR_PRINCIPAL, COLOR_SEGUNDARIO, COLOR_OJO]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(resumen["cuartil"], resumen["% datos"], color=colores, alpha=0.85)
    for x, (p, rango) in enumerate(zip(resumen["% datos"], resumen["rango (GBP)"])):
        hasta = rango.split("..")[-1].strip()
        ax.text(x, p + 0.6, f"{p:.1f} %\nhasta {hasta}", ha="center",
                va="bottom", fontsize=8.5)
    ax.axhline(25, color="#334155", linestyle="--", linewidth=1, label="25 % (un cuartil)")
    ax.set_ylim(0, 33)
    ax.set_ylabel("% de transacciones")
    ax.set_title("price: 4 cuartiles, cada uno ≈ 25 % de las transacciones")
    ax.text(0.01, 0.94,
            f"Q1 + Q2 (hasta la mediana {mediana:,.0f} £) = mercado típico",
            transform=ax.transAxes, fontsize=9.5, color="#334155",
            bbox={"boxstyle": "round", "facecolor": "#e2e8f0", "alpha": 0.9})
    ax.legend(loc="lower right")
    return fig


def hist_price_iqr(price: pd.Series, limites: dict) -> plt.Figure:
    """Histograma de price (escala log) con Q1/Q2/Q3 y la zona fuera del limite."""
    vmin, vmax = 100.0, 1_000_000_000.0
    bins = np.logspace(np.log10(vmin), np.log10(vmax), 80)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.hist(price, bins=bins, color=COLOR_PRINCIPAL, alpha=0.75)
    for etiqueta, valor, color in (
        (f"Q1 (p25) = {limites['q1']:,.0f}", limites["q1"], COLOR_VERDE),
        (f"Q2 mediana = {limites['q2']:,.0f}", limites["q2"], COLOR_MEDIANA),
        (f"Q3 (p75) = {limites['q3']:,.0f}", limites["q3"], COLOR_SEGUNDARIO),
    ):
        ax.axvline(valor, color=color, linestyle="--", linewidth=1.6, label=etiqueta)
    ax.axvline(
        limites["limite_superior"], color=COLOR_OJO, linestyle="--", linewidth=2,
        label=f"límite superior IQR = {limites['limite_superior']:,.0f}",
    )

    fuera = int(((price < limites["limite_inferior"]) | (price > limites["limite_superior"])).sum())
    ax.text(
        0.985, 0.90, f"{100 * fuera / len(price):.2f} % fuera\n(por arriba)",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, color=COLOR_OJO,
        bbox={"boxstyle": "round", "facecolor": "#fee2e2", "alpha": 0.9},
    )
    ax.set_xscale("log")
    ax.set_xlabel("price (GBP, escala log)")
    ax.set_ylabel("transacciones")
    ax.set_title("Distribución de price con cuartiles (Q1, Q2, Q3) y límite IQR (Tukey)")
    ax.legend()
    return fig


def box_antes_despues(original: pd.Series, winsorizado: pd.Series, seed: int = 7) -> plt.Figure:
    """Boxplot comparado original vs winsorizado (muestra fija para reproducibilidad)."""
    rng = np.random.default_rng(seed)
    n = min(2_000_000, len(original))
    idx = rng.choice(len(original), size=n, replace=False)
    datos = [original.iloc[idx], winsorizado.iloc[idx]]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    cajas = ax.boxplot(
        datos, tick_labels=["original", "winsorizado"],
        showfliers=True, flierprops={"marker": ".", "markersize": 2},
    )
    for caja, color in zip(cajas["boxes"], [COLOR_PRINCIPAL, COLOR_VERDE]):
        caja.set_color(color)
    ax.set_yscale("log")
    ax.set_ylabel("price (GBP, escala log)")
    ax.set_title("price: original vs winsorizado (clip al límite IQR)")
    ax.grid(axis="y")
    return fig


def temporal_price(df: pd.DataFrame) -> plt.Figure:
    """Combinadas: volumen anual (barras) y mediana de price anual (linea)."""
    agrupado = df.groupby("year")["price"].agg(n="size", mediana="median")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(agrupado.index.astype(str), agrupado["n"] / 1e6, color=COLOR_PRINCIPAL, alpha=0.85,
           label="transacciones (millones)")
    ax.set_xlabel("año")
    ax.set_ylabel("transacciones (millones)")

    ax2 = ax.twinx()
    ax2.plot(agrupado.index.astype(str), agrupado["mediana"] / 1e3, marker="o",
             color=COLOR_SEGUNDARIO, linewidth=2, label="mediana price (miles £)")
    ax2.set_ylabel("mediana de price (miles de GBP)")
    ax2.tick_params(axis="y", labelcolor=COLOR_SEGUNDARIO)

    ax.set_title("Volumen y mediana de price por año")
    lineas1, etiquetas1 = ax.get_legend_handles_labels()
    lineas2, etiquetas2 = ax2.get_legend_handles_labels()
    ax.legend(lineas1 + lineas2, etiquetas1 + etiquetas2, loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    return fig