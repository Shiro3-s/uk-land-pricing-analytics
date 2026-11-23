"""Aprendizaje no supervisado II: reglas de asociacion con algoritmo Apriori
(implementacion transparente en Python puro + pandas) y comparativa FP-Growth.

Con perfil dual de proteccion de recursos (local vs. cloud):
  - Modo 'local' (por defecto): acota transacciones a max 3.000, profundidad max k=4 y
    soporte piso >= 0,03, verificando consumo de RAM con tracemalloc.
  - Modo 'cloud' (MINERIA_MODO=cloud o --modo cloud): procesa 100 % de las transacciones
    disponibles manteniendo la poda Apriori nivel por nivel.

Aplica el motor a dos conjuntos transaccionales reales del proyecto UK Housing:
  1. F1 Transacciones inmobiliarias (uk_property_price_limpio_muestra.parquet, n=2.000):
     tipo de propiedad, tenencia, obra nueva/usada, macro-zona y tramo de esfuerzo joven.
  2. F2 Catalogo hipotecario (hipotecas_limpio.parquet, 7.796 productos; 3.000 en local):
     tipo de tasa, perfil comprador (First time buyer, etc.), plazo inicial, nivel de tasa,
     comisiones y declaracion de reversion.

Incluye verificacion manual sobre 10 tickets reales, poda nivel por nivel, sensibilidad
de umbrales, asimetria direccional (A->B vs B->A), filtro critico de reglas triviales,
puntaje compuesto (lift * soporte) y comparativa tecnica Apriori vs. FP-Growth.

Ejecutar:
  python script/proyecto/reglas_asociacion.py
  python script/proyecto/reglas_asociacion.py --modo cloud
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc
from itertools import combinations
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys

from proyecto import config, indicadores
from proyecto.clustering_mercados import muestrear_con_limite

SEED = config.SEED


def _guardar_png(fig: plt.Figure, nombre: str) -> Path:
    ruta = config.MD_FIGURAS / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    return ruta


# ---------------------------------------------------------- motor Apriori puro
def a_formato_largo(df_trans: pd.DataFrame, col_id: str, cols_items: list[str]) -> pd.DataFrame:
    """Convierte tabla ancha de atributos discretos en formato largo (id_transaccion, item)."""
    largo = df_trans[[col_id] + cols_items].melt(
        id_vars=[col_id],
        value_vars=cols_items,
        value_name="item",
    )[[col_id, "item"]].dropna()
    return largo.rename(columns={col_id: "id_transaccion"})


def agrupar_transacciones(df_largo: pd.DataFrame) -> list[frozenset[str]]:
    """Agrupa el formato largo por id_transaccion en tickets-conjunto (frozenset)."""
    return (
        df_largo.groupby("id_transaccion")["item"]
        .apply(lambda s: frozenset(s.astype(str)))
        .tolist()
    )


def items_frecuentes_1(transacciones: list[frozenset[str]], soporte_min: float) -> dict[frozenset[str], float]:
    """Nivel k=1: calcula soporte individual y aplica poda por soporte minimo."""
    n = len(transacciones)
    conteo: dict[str, int] = {}
    for t in transacciones:
        for item in t:
            conteo[item] = conteo.get(item, 0) + 1
    return {frozenset([i]): c / n for i, c in conteo.items() if (c / n) >= soporte_min}


def combinar(nivel_anterior: dict[frozenset[str], float], k: int) -> set[frozenset[str]]:
    """Genera candidatos de tamano k aplicando propiedad Apriori (todos sus subconjuntos k-1 vivos)."""
    lista = list(nivel_anterior.keys())
    set_prev = set(lista)
    candidatos: set[frozenset[str]] = set()
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            union = lista[i] | lista[j]
            if len(union) == k:
                # Propiedad de poda previa: todo subconjunto (k-1) debe estar vivo
                if all(frozenset(sub) in set_prev for sub in combinations(union, k - 1)):
                    candidatos.add(union)
    return candidatos


def _indice_tidsets(transacciones: list[frozenset[str]]) -> dict[str, set[int]]:
    """Indice invertido item -> conjunto de IDs de transaccion (Python puro, conteo O(|interseccion|))."""
    idx: dict[str, set[int]] = {}
    for tid, t in enumerate(transacciones):
        for item in t:
            idx.setdefault(item, set()).add(tid)
    return idx


def filtrar(
    candidatos: set[frozenset[str]],
    transacciones: list[frozenset[str]],
    soporte_min: float,
    tidsets: dict[str, set[int]] | None = None,
) -> dict[frozenset[str], float]:
    """Poda Apriori: conserva unicamente candidatos con soporte >= soporte_min."""
    n = len(transacciones)
    if tidsets is None:
        tidsets = _indice_tidsets(transacciones)
    vivos: dict[frozenset[str], float] = {}
    for c in candidatos:
        sets = [tidsets[it] for it in c if it in tidsets]
        if len(sets) != len(c):
            continue
        sets.sort(key=len)
        inter = sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0]
        soporte = len(inter) / n
        if soporte >= soporte_min:
            vivos[c] = soporte
    return vivos


def ejecutar_apriori(
    transacciones: list[frozenset[str]],
    soporte_min: float = 0.05,
    max_k: int = config.APRIORI_MAX_K,
) -> tuple[dict[frozenset[str], float], pd.DataFrame]:
    """Corre Apriori nivel por nivel (k=1..max_k) y devuelve itemsets frecuentes + traza de poda."""
    assert soporte_min >= config.APRIORI_SOPORTE_PISO or len(transacciones) <= 50, (
        f"soporte_min ({soporte_min}) por debajo del piso seguro ({config.APRIORI_SOPORTE_PISO})"
    )
    tidsets = _indice_tidsets(transacciones)
    items_unicos = len(tidsets)
    n = len(transacciones)
    nivel = {frozenset([it]): len(ids) / n for it, ids in tidsets.items() if (len(ids) / n) >= soporte_min}
    todos = dict(nivel)
    traza = [
        {
            "nivel_k": 1,
            "candidatos": items_unicos,
            "tras_poda": len(nivel),
            "poda_pct": round(100.0 * (1.0 - len(nivel) / max(items_unicos, 1)), 1),
        }
    ]

    k = 2
    while k <= max_k:
        candidatos = combinar(nivel, k)
        if not candidatos:
            break
        nivel = filtrar(candidatos, transacciones, soporte_min, tidsets=tidsets)
        traza.append(
            {
                "nivel_k": k,
                "candidatos": len(candidatos),
                "tras_poda": len(nivel),
                "poda_pct": round(100.0 * (1.0 - len(nivel) / max(len(candidatos), 1)), 1),
            }
        )
        if not nivel:
            break
        todos.update(nivel)
        k += 1

    return todos, pd.DataFrame(traza)


def generar_reglas(
    itemsets: dict[frozenset[str], float],
    confianza_min: float = 0.45,
    lift_min: float = 1.0,
) -> pd.DataFrame:
    """Extrae reglas A -> B con soporte, confianza, lift, leverage, conviccion y score=lift*soporte."""
    reglas = []
    for itemset, sup_ab in itemsets.items():
        if len(itemset) < 2:
            continue
        for r_len in range(1, len(itemset)):
            for ant in combinations(sorted(itemset), r_len):
                a = frozenset(ant)
                b = itemset - a
                if a in itemsets and b in itemsets:
                    sup_a = itemsets[a]
                    sup_b = itemsets[b]
                    conf = sup_ab / sup_a
                    lift = conf / sup_b
                    if conf >= confianza_min and lift >= lift_min:
                        leverage = sup_ab - (sup_a * sup_b)
                        conv = (1.0 - sup_b) / (1.0 - conf) if conf < 1.0 else np.inf
                        reglas.append(
                            {
                                "antecedente": " + ".join(sorted(a)),
                                "consecuente": " + ".join(sorted(b)),
                                "tam_itemset": len(itemset),
                                "soporte": round(sup_ab, 4),
                                "soporte_A": round(sup_a, 4),
                                "soporte_B": round(sup_b, 4),
                                "confianza": round(conf, 4),
                                "lift": round(lift, 3),
                                "leverage": round(leverage, 4),
                                "conviccion": round(float(conv), 3) if np.isfinite(conv) else 999.0,
                                "score": round(lift * sup_ab, 4),
                            }
                        )
    df_r = pd.DataFrame(reglas)
    if df_r.empty:
        return df_r
    return df_r.sort_values(["score", "lift", "confianza"], ascending=[False, False, False]).reset_index(drop=True)


# ----------------------------------------- construccion de canastas UK Housing
def _macro_zona(region: str) -> str:
    if region == "London":
        return "Zona:London"
    if region in ("South East", "South West", "East of England"):
        return "Zona:Sur_y_Este"
    if region in ("East Midlands", "West Midlands"):
        return "Zona:Midlands"
    return "Zona:Norte_y_Gales"


def preparar_transacciones_inmobiliarias(modo: str | None = None) -> tuple[list[frozenset[str]], pd.DataFrame]:
    """Construye canastas discretas desde la muestra limpia F1 (2.000 ventas, 135 KB)."""
    limites = config.obtener_limites_recursos(modo)
    f1 = pd.read_parquet(config.RUTA_TRANSACCIONES_MUESTRA).copy()
    f1 = muestrear_con_limite(f1, limites["max_transacciones_apriori"], col_estrato="region", seed=SEED)

    hip = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    tasa = float(hip["tasa_inicial_pct"].mean())
    comisiones = float(hip["comisiones_total"].mean())
    maestro = pd.read_parquet(config.RUTA_MAESTRO)
    sal_por_region = maestro.groupby("region", observed=True)["salario_joven_region"].median().to_dict()
    sal_med_global = float(maestro["salario_joven_region"].median())

    cuota = f1["price"].apply(lambda p: indicadores.cuota_mensual_pmt(float(p), tasa, comisiones=comisiones))
    sal_mensual = f1["region"].map(lambda r: sal_por_region.get(r, sal_med_global) / 12.0)
    f1["esfuerzo_est"] = cuota / sal_mensual

    f1["item_prop"] = "Prop:" + f1["property_type_full"].astype(str)
    f1["item_tenencia"] = "Tenencia:" + f1["duration_full"].astype(str)
    f1["item_obra"] = np.where(f1["is_new_build"].astype(str) == "True", "Obra:Nueva", "Obra:Usada")
    f1["item_zona"] = f1["region"].astype(str).map(_macro_zona)
    f1["item_acceso"] = pd.cut(
        f1["esfuerzo_est"],
        bins=[-np.inf, 0.50, 0.80, 1.00, np.inf],
        labels=[
            "Acceso:Moderado_<=50%",
            "Acceso:Alto_50_80%",
            "Acceso:Critico_80_100%",
            "Acceso:Extremo_>100%",
        ],
    ).astype(str)
    f1["item_atipico"] = np.where(f1["es_atipico_precio"].astype(bool), "Precio:Atipico_IQR", None)

    cols_items = ["item_prop", "item_tenencia", "item_obra", "item_zona", "item_acceso", "item_atipico"]
    largo = a_formato_largo(f1, "transaction_id", cols_items)
    trans = agrupar_transacciones(largo)
    return trans, f1


def preparar_transacciones_hipotecas(modo: str | None = None) -> tuple[list[frozenset[str]], pd.DataFrame]:
    """Construye canastas discretas desde F2 (7.796 productos; 3.000 en local, 100 % en cloud)."""
    limites = config.obtener_limites_recursos(modo)
    hip_full = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).copy()
    hip = muestrear_con_limite(hip_full, limites["max_transacciones_apriori"], col_estrato="tipo", seed=SEED)
    del hip_full
    gc.collect()

    hip["id_prod"] = hip["sku"].astype(str) + "_" + hip["tid"].astype(str)
    hip["item_tipo"] = "Tipo:" + hip["tipo"].astype(str)
    hip["item_perfil"] = "Perfil:" + hip["subtitulo"].astype(str).str.replace(" ", "_")

    # Periodo inicial agrupado en bandas comerciales
    hip["item_inicial"] = np.select(
        [hip["meses_inicial"] <= 30, hip["meses_inicial"] <= 48, hip["meses_inicial"] <= 72],
        ["Inicial:2_anios", "Inicial:3_anios", "Inicial:5_anios"],
        default="Inicial:Largo_o_Total",
    )

    q1_t, q3_t = hip["tasa_inicial_pct"].quantile([0.25, 0.75])
    hip["item_tasa"] = np.select(
        [hip["es_atipico_tasa"].astype(bool), hip["tasa_inicial_pct"] <= q1_t, hip["tasa_inicial_pct"] >= q3_t],
        ["Tasa:Atipica_IQR", "Tasa:Baja_Q1", "Tasa:Alta_Q4"],
        default="Tasa:Media",
    )

    hip["item_comision"] = np.select(
        [hip["comisiones_total"] <= 0, hip["comisiones_total"] <= 1200],
        ["Comision:Cero", "Comision:Media"],
        default="Comision:Alta",
    )
    hip["item_reversion"] = np.where(hip["tasa_reversion"].isna(), "Reversion:No_Declarada", "Reversion:Declarada")

    cols_items = ["item_tipo", "item_perfil", "item_inicial", "item_tasa", "item_comision", "item_reversion"]
    largo = a_formato_largo(hip, "id_prod", cols_items)
    trans = agrupar_transacciones(largo)
    return trans, hip


def verificacion_manual_10_tickets(transacciones: list[frozenset[str]]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Toma 10 transacciones reales, verifica a mano soporte/confianza/lift y valida contra Apriori."""
    mini = transacciones[:10]
    tabla_mini = pd.DataFrame(
        [{"ticket": i + 1, "items": ", ".join(sorted(t))} for i, t in enumerate(mini)]
    )

    # Elegimos un par real presente en al menos 2 de los 10 tickets
    itemsets_mini, _ = ejecutar_apriori(mini, soporte_min=0.20, max_k=2)
    reglas_mini = generar_reglas(itemsets_mini, confianza_min=0.50, lift_min=1.0)
    mejor = reglas_mini.iloc[0]
    a = mejor["antecedente"]
    b = mejor["consecuente"]

    n = len(mini)
    c_a = sum(1 for t in mini if a in t)
    c_b = sum(1 for t in mini if b in t)
    c_ab = sum(1 for t in mini if (a in t and b in t))

    sup_ab_manual = c_ab / n
    conf_manual = c_ab / c_a
    lift_manual = conf_manual / (c_b / n)

    assert np.isclose(sup_ab_manual, mejor["soporte"], atol=1e-4)
    assert np.isclose(conf_manual, mejor["confianza"], atol=1e-4)
    assert np.isclose(lift_manual, mejor["lift"], atol=1e-3)

    verif = {
        "antecedente": a,
        "consecuente": b,
        "freq_A": c_a,
        "freq_B": c_b,
        "freq_AB": c_ab,
        "soporte_manual": round(sup_ab_manual, 4),
        "confianza_manual": round(conf_manual, 4),
        "lift_manual": round(lift_manual, 3),
    }
    return tabla_mini, verif


def clasificar_y_filtrar_reglas_inmobiliarias(reglas: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta reglas triviales estructurales (ej. Flat <-> Leasehold) vs. hallazgos accionables."""
    if reglas.empty:
        return reglas
    df = reglas.copy()

    def _etiquetar(row: pd.Series) -> str:
        ant = row["antecedente"]
        con = row["consecuente"]
        par = f"{ant} -> {con}"
        # En el derecho inmobiliario britanico, Flat/Maisonette casi siempre es Leasehold
        # y Detached/Semi-Detached casi siempre es Freehold (reglas estructurales conocidas).
        if (
            ("Prop:Flat/Maisonette" in ant and "Tenencia:Leasehold" in con)
            or ("Tenencia:Leasehold" in ant and "Prop:Flat/Maisonette" in con)
            or ("Prop:Detached" in ant and "Tenencia:Freehold" in con)
            or ("Prop:Semi-Detached" in ant and "Tenencia:Freehold" in con)
            or ("Obra:Usada" in con and row["lift"] < 1.08)
        ):
            return "Trivial / estructural"
        if "Acceso:" in par or "Zona:" in par:
            return "Accionable (accesibilidad/territorio)"
        return "Informativa"

    df["categoria_analitica"] = df.apply(_etiquetar, axis=1)
    return df


def sensibilidad_umbrales(
    transacciones: list[frozenset[str]],
    soportes: list[float] | None = None,
    confianzas: list[float] | None = None,
) -> pd.DataFrame:
    """Evalua cuantas reglas sobreviven en una rejilla de soporte_min x confianza_min."""
    if soportes is None:
        soportes = [0.03, 0.05, 0.08, 0.10]
    if confianzas is None:
        confianzas = [0.35, 0.50, 0.65, 0.75]

    filas = []
    for s in soportes:
        itemsets, _ = ejecutar_apriori(transacciones, soporte_min=s, max_k=config.APRIORI_MAX_K)
        for c in confianzas:
            reglas = generar_reglas(itemsets, confianza_min=c, lift_min=1.05)
            filas.append(
                {
                    "soporte_min": s,
                    "confianza_min": c,
                    "n_itemsets": len(itemsets),
                    "n_reglas": len(reglas),
                    "lift_max": float(reglas["lift"].max()) if not reglas.empty else 0.0,
                }
            )
    return pd.DataFrame(filas)


def tabla_comparativa_apriori_fpgrowth() -> pd.DataFrame:
    """Resumen tecnico comparativo entre Apriori y FP-Growth."""
    return pd.DataFrame(
        [
            {"Criterio": "Generacion de candidatos", "Apriori (implementado)": "Si, nivel por nivel (k=1, 2, 3...)", "FP-Growth (referencia industrial)": "No: crece patrones desde arbol FP"},
            {"Criterio": "Barridos sobre la base", "Apriori (implementado)": "Uno por cada nivel k", "FP-Growth (referencia industrial)": "Solo 2 barridos"},
            {"Criterio": "Uso de memoria RAM", "Apriori (implementado)": "Bajo con soporte piso y max_k", "FP-Growth (referencia industrial)": "Mayor en RAM (mantiene arbol FP)"},
            {"Criterio": "Escalabilidad", "Apriori (implementado)": "Ideal para muestras pequenas/medianas", "FP-Growth (referencia industrial)": "Optimo para millones de tickets densos"},
            {"Criterio": "Transparencia / auditoria", "Apriori (implementado)": "Total (traza explicita de poda)", "FP-Growth (referencia industrial)": "Disponible en mlxtend.frequent_patterns"},
        ]
    )


# ------------------------------------------------------------------ figuras
def fig_poda_niveles(traza_f1: pd.DataFrame, traza_f2: pd.DataFrame) -> plt.Figure:
    """Muestra la eficacia de la poda Apriori por nivel k en F1 y F2."""
    fig, ejes = plt.subplots(1, 2, figsize=(11.8, 4.1))
    for ax, tr, titulo in (
        (ejes[0], traza_f1, "A. Transacciones Inmobiliarias F1 (n=2.000)"),
        (ejes[1], traza_f2, "B. Catalogo Hipotecario F2"),
    ):
        xs = np.arange(len(tr))
        ax.bar(xs - 0.18, tr["candidatos"], 0.34, color="#95a5a6", label="Candidatos evaluados")
        ax.bar(xs + 0.18, tr["tras_poda"], 0.34, color="#2e86c1", label="Sobrevivientes tras poda")
        for i, r in tr.iterrows():
            ax.text(i - 0.18, r["candidatos"] + 1, str(int(r["candidatos"])), ha="center", fontsize=8)
            ax.text(i + 0.18, r["tras_poda"] + 1, str(int(r["tras_poda"])), ha="center", fontsize=8, weight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"k={int(k)}" for k in tr["nivel_k"]])
        ax.set_ylabel("Cantidad de itemsets")
        ax.set_title(titulo)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Poda Apriori nivel por nivel: reduccion del espacio de busqueda", fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "apriori_poda_niveles.png")
    return fig


def fig_mapa_reglas(reglas: pd.DataFrame, titulo: str, archivo: str) -> plt.Figure:
    """Dispersion Soporte vs Confianza coloreada por Lift y dimensionada por score."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sc = ax.scatter(
        reglas["soporte"],
        reglas["confianza"],
        c=reglas["lift"],
        s=np.clip(reglas["score"] * 900, 35, 320),
        cmap="viridis",
        alpha=0.82,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Lift (fuerza sobre el azar)")
    top5 = reglas.head(5)
    for _, r in top5.iterrows():
        etiq = f"{r['antecedente'][:24]} -> {r['consecuente'][:20]}"
        ax.annotate(etiq, (r["soporte"], r["confianza"]), fontsize=7.0, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Soporte (frecuencia conjunta)")
    ax.set_ylabel("Confianza P(B | A)")
    ax.set_title(titulo, fontsize=10.5, weight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _guardar_png(fig, archivo)
    return fig


def fig_top_reglas(reglas: pd.DataFrame, titulo: str, archivo: str, n_top: int = 8) -> plt.Figure:
    """Barras horizontales de las top reglas ordenadas por puntaje compuesto (lift * soporte)."""
    top = reglas.head(n_top).iloc[::-1]
    etiquetas = [f"{r['antecedente']}  ->  {r['consecuente']}" for _, r in top.iterrows()]

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    barras = ax.barh(etiquetas, top["score"], color="#117a65", edgecolor="white")
    for barra, (_, r) in zip(barras, top.iterrows()):
        v = barra.get_width()
        ax.text(
            v + top["score"].max() * 0.015,
            barra.get_y() + barra.get_height() / 2,
            f"score={r['score']:.3f} | lift={r['lift']:.2f} | conf={r['confianza']:.2f} | sup={r['soporte']:.2f}",
            va="center",
            fontsize=7.6,
        )
    ax.set_xlim(0, top["score"].max() * 1.48)
    ax.set_xlabel("Puntaje compuesto (Lift x Soporte)")
    ax.set_title(titulo, fontsize=10.5, weight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _guardar_png(fig, archivo)
    return fig


def fig_sensibilidad(sens_df: pd.DataFrame) -> plt.Figure:
    """Mapa de calor de reglas sobrevivientes segun rejilla soporte_min x confianza_min."""
    piv = sens_df.pivot(index="soporte_min", columns="confianza_min", values="n_reglas")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{s:.2f}" for s in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = int(piv.iloc[i, j])
            color_txt = "white" if v > piv.values.max() * 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color_txt, fontsize=11, weight="bold")
    ax.set_xlabel("Confianza minima")
    ax.set_ylabel("Soporte minimo")
    ax.set_title("Sensibilidad de umbrales: reglas sobrevivientes (lift > 1,05)")
    fig.colorbar(im, ax=ax, label="Numero de reglas")
    fig.tight_layout()
    _guardar_png(fig, "apriori_sensibilidad_umbrales.png")
    return fig


def main(modo: str | None = None) -> None:
    modo_activo = (modo or config.MODO_EJECUCION).strip().lower()
    limites = config.obtener_limites_recursos(modo_activo)

    print("=" * 76)
    print(f"REGLAS DE ASOCIACION (APRIORI PURO + FP-GROWTH) · PERFIL: {modo_activo.upper()}")
    print("=" * 76)

    tracemalloc.start()
    t0 = time.perf_counter()

    # 1. F1 Inmobiliario
    trans_f1, _ = preparar_transacciones_inmobiliarias(modo_activo)
    tabla_mini, verif = verificacion_manual_10_tickets(trans_f1)
    print("\n[1] Verificacion manual sobre 10 transacciones inmobiliarias reales:")
    print(f"    Regla verificada: {verif['antecedente']} -> {verif['consecuente']} | "
          f"freq(A)={verif['freq_A']}, freq(B)={verif['freq_B']}, freq(AB)={verif['freq_AB']}")
    print(f"    Soporte={verif['soporte_manual']} | Confianza={verif['confianza_manual']} | "
          f"Lift={verif['lift_manual']} (assert manual == Apriori OK)")

    sop_f1, conf_f1 = 0.05, 0.45
    itemsets_f1, traza_f1 = ejecutar_apriori(trans_f1, soporte_min=sop_f1, max_k=config.APRIORI_MAX_K)
    reglas_f1 = clasificar_y_filtrar_reglas_inmobiliarias(
        generar_reglas(itemsets_f1, confianza_min=conf_f1, lift_min=1.05)
    )
    print(f"\n[2] Aplicacion 1 · Transacciones Inmobiliarias F1 (n={len(trans_f1):,}, sup>={sop_f1}, conf>={conf_f1}):")
    print(traza_f1.to_string(index=False))
    print(f"    Itemsets frecuentes totales: {len(itemsets_f1)} | Reglas (lift >= 1.05): {len(reglas_f1)}")
    accionables_f1 = reglas_f1[reglas_f1["categoria_analitica"] == "Accionable (accesibilidad/territorio)"]
    print("\n    Top 5 reglas accionables (descartando trivialidades estructurales):")
    cols_r = ["antecedente", "consecuente", "soporte", "confianza", "lift", "score"]
    print(accionables_f1[cols_r].head(5).to_string(index=False))

    # Contraste de asimetria direccional A -> B vs B -> A
    if not accionables_f1.empty:
        r0 = accionables_f1.iloc[0]
        inv = reglas_f1[(reglas_f1["antecedente"] == r0["consecuente"]) & (reglas_f1["consecuente"] == r0["antecedente"])]
        if not inv.empty:
            ri = inv.iloc[0]
            print(f"\n    Asimetria direccional demostrada:")
            print(f"      {r0['antecedente']} -> {r0['consecuente']} : confianza = {r0['confianza']:.3f}")
            print(f"      {ri['antecedente']} -> {ri['consecuente']} : confianza = {ri['confianza']:.3f}")

    # 2. F2 Catalogo Hipotecario
    trans_f2, _ = preparar_transacciones_hipotecas(modo_activo)
    sop_f2, conf_f2 = 0.05, 0.45
    itemsets_f2, traza_f2 = ejecutar_apriori(trans_f2, soporte_min=sop_f2, max_k=config.APRIORI_MAX_K)
    reglas_f2 = generar_reglas(itemsets_f2, confianza_min=conf_f2, lift_min=1.05)
    print(f"\n[3] Aplicacion 2 · Catalogo Hipotecario F2 (n={len(trans_f2):,}, sup>={sop_f2}, conf>={conf_f2}):")
    print(traza_f2.to_string(index=False))
    print(f"    Itemsets frecuentes totales: {len(itemsets_f2)} | Reglas (lift >= 1.05): {len(reglas_f2)}")
    print("\n    Top 5 reglas hipotecarias por puntaje compuesto (lift * soporte):")
    print(reglas_f2[cols_r].head(5).to_string(index=False))

    # 3. Sensibilidad de umbrales
    sens_f1 = sensibilidad_umbrales(trans_f1)
    print("\n[4] Sensibilidad de umbrales (F1):")
    print(sens_f1.to_string(index=False))

    # 4. Exportar tablas livianas
    config.DATA_INTEGRADO.mkdir(parents=True, exist_ok=True)
    reglas_f1.to_csv(config.RUTA_REGLAS_INMOBILIARIO, index=False)
    reglas_f2.to_csv(config.RUTA_REGLAS_HIPOTECAS, index=False)

    # 5. Figuras
    for fig in (
        fig_poda_niveles(traza_f1, traza_f2),
        fig_mapa_reglas(reglas_f1, "Mapa de reglas inmobiliarias F1 (Soporte vs. Confianza vs. Lift)", "apriori_mapa_reglas_inmobiliario.png"),
        fig_top_reglas(accionables_f1 if not accionables_f1.empty else reglas_f1, "Top reglas accionables · Mercado Inmobiliario Joven (F1)", "apriori_top_reglas_inmobiliario.png"),
        fig_top_reglas(reglas_f2, "Top reglas de condiciones crediticias · Catalogo Hipotecario (F2)", "apriori_top_reglas_hipotecas.png"),
        fig_sensibilidad(sens_f1),
    ):
        plt.close(fig)

    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pico_mb = pico_bytes / (1024.0**2)
    duracion_s = time.perf_counter() - t0
    print(f"\n[5] Telemetria Apriori ({modo_activo}): tiempo = {duracion_s:.2f} s | pico RAM Python = {pico_mb:.2f} MB")
    alerta_mb = limites["pico_ram_mb_alerta"]
    if alerta_mb is not None:
        assert pico_mb < alerta_mb, f"Pico de RAM ({pico_mb:.1f} MB) excedio el limite ({alerta_mb} MB)"
        print(f"    -> assert OK: consumo acotado ({pico_mb:.2f} MB < {alerta_mb} MB).")
    print("[6] Reglas CSV y 5 figuras guardadas en data/integrado/ y md/figuras/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reglas de asociacion Apriori con perfil local / local_24gb / cloud")
    parser.add_argument("--modo", choices=["local", "local_24gb", "cloud"], default=None, help="Perfil de limites de recursos")
    args = parser.parse_args()
    main(modo=args.modo)
