"""Clasificacion supervisada I: arbol de decision interpretativo (CART, max_depth=4).

Espejo en terminal y biblioteca de apoyo del cuaderno 09_Arbol_Decision_Clasificacion.ipynb.
Construye una matriz X libre de fuga aritmetica (sin precio ni salario) sobre el cuadro
maestro (60 filas = 10 regiones x 6 anios, 2015-2020) para clasificar si un joven (18-34)
enfrenta un esfuerzo hipotecario Alto (>= mediana) o Moderado (< mediana):
  1. Preparacion anti-fuga de X e y (7 rasgos numericos + 9 dummies de region).
  2. Verificacion de invarianza monotona ante 4 esquemas de escalado.
  3. Curva sesgo-varianza por profundidad (1..5 y sin limite) con CV-5 estratificada.
  4. Extraccion de reglas interpretables (export_text), pureza y particion espacial.
  5. Diagnostico de elegibilidad predictiva (taxonomia regional vs. validacion temporal).

Ejecutar:  python script/proyecto/arbol_clasificacion.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=False)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys

from proyecto import config
from proyecto.modelado import agregar_esfuerzo, cargar_insumo

SEED = config.SEED
TEST_SIZE = 0.30
PROFUNDIDAD = 4

# 7 rasgos numericos libres de fuga aritmetica (orden del cuadro maestro; sin precio ni salario)
RASGOS_NUMERICOS = [
    "n_transacciones",
    "tmax_anom",
    "tmin_anom",
    "lluvia_anom",
    "sol_anom",
    "af_anom",
    "precio_var_pct",
]

PALETA_CLASE = {"Moderado": "#2e86c1", "Alto": "#e67e22"}


def _guardar_png(fig: plt.Figure, nombre: str) -> Path:
    ruta = config.MD_FIGURAS / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    return ruta


def preparar_xy() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, float]:
    """Arma X (16 rasgos sin fuga), y (Alto/Moderado por mediana), df completo y umbral."""
    maestro, hipotecas = cargar_insumo()
    df = agregar_esfuerzo(maestro, hipotecas).sort_values(["region", "anio"]).reset_index(drop=True)
    umbral = float(df["esfuerzo"].median())
    df["clase"] = np.where(df["esfuerzo"] >= umbral, "Alto", "Moderado")

    X = pd.get_dummies(
        df[RASGOS_NUMERICOS + ["region"]],
        columns=["region"],
        drop_first=True,
    ).astype(float)
    y = df["clase"].copy()
    return X, y, df, umbral


def _cv_estratificada() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def entrenar_arboles(
    X: pd.DataFrame,
    y: pd.Series,
    profundidades: list[int | None] | None = None,
) -> tuple[dict[int | None, DecisionTreeClassifier], pd.DataFrame]:
    """Entrena arboles CART para varias profundidades y mide train, test, brecha y CV-5."""
    if profundidades is None:
        profundidades = [1, 2, 3, 4, 5, None]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    cv = _cv_estratificada()
    modelos: dict[int | None, DecisionTreeClassifier] = {}
    filas = []

    for prof in profundidades:
        arbol = DecisionTreeClassifier(max_depth=prof, random_state=SEED)
        arbol.fit(X_train, y_train)
        acc_tr = float(arbol.score(X_train, y_train))
        acc_te = float(arbol.score(X_test, y_test))
        cv_scores = cross_val_score(
            DecisionTreeClassifier(max_depth=prof, random_state=SEED),
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )
        etiqueta = f"profundidad_{prof}" if prof is not None else "sin_limite"
        modelos[prof] = arbol
        filas.append(
            {
                "arbol": etiqueta,
                "max_depth_param": prof if prof is not None else 6,
                "train": acc_tr,
                "test": acc_te,
                "brecha_train_test": acc_tr - acc_te,
                "cv5_estratificada": float(cv_scores.mean()),
                "cv5_desv": float(cv_scores.std()),
                "nodos": int(arbol.tree_.node_count),
                "profundidad_real": int(arbol.tree_.max_depth),
                "hojas": int(arbol.get_n_leaves()),
            }
        )

    return modelos, pd.DataFrame(filas)


def comparar_escalamiento(
    X: pd.DataFrame,
    y: pd.Series,
    profundidad: int = PROFUNDIDAD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compara 4 esquemas de escalado ajustados SOLO en train y demuestra invarianza CART."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    cv = _cv_estratificada()
    esquemas = {
        "Crudo (sin escalar)": None,
        "StandardScaler (z-score)": StandardScaler(),
        "MinMaxScaler ([0, 1])": MinMaxScaler(),
        "RobustScaler (mediana/IQR)": RobustScaler(),
    }

    filas_esc = []
    matrices_tr: dict[str, pd.DataFrame] = {}
    escaladores_ajustados: dict[str, object | None] = {}
    arbol_crudo: DecisionTreeClassifier | None = None

    for nombre, esc in esquemas.items():
        xtr = X_train.copy()
        xte = X_test.copy()
        if esc is not None:
            esc.fit(xtr[RASGOS_NUMERICOS])
            xtr[RASGOS_NUMERICOS] = esc.transform(xtr[RASGOS_NUMERICOS])
            xte[RASGOS_NUMERICOS] = esc.transform(xte[RASGOS_NUMERICOS])

        matrices_tr[nombre] = xtr[RASGOS_NUMERICOS].copy()
        escaladores_ajustados[nombre] = esc

        arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=SEED).fit(xtr, y_train)
        if nombre.startswith("Crudo"):
            arbol_crudo = arbol

        # CV-5 sin fuga (escalador ajustado dentro de cada pliegue)
        cv_vals = []
        for idx_tr, idx_va in cv.split(X, y):
            xf_tr = X.iloc[idx_tr].copy()
            xf_va = X.iloc[idx_va].copy()
            if esc is not None:
                clon = esc.__class__()
                xf_tr[RASGOS_NUMERICOS] = clon.fit_transform(xf_tr[RASGOS_NUMERICOS])
                xf_va[RASGOS_NUMERICOS] = clon.transform(xf_va[RASGOS_NUMERICOS])
            clf = DecisionTreeClassifier(max_depth=profundidad, random_state=SEED)
            clf.fit(xf_tr, y.iloc[idx_tr])
            cv_vals.append(float(clf.score(xf_va, y.iloc[idx_va])))

        filas_esc.append(
            {
                "escalador": nombre,
                "train_acc": float(arbol.score(xtr, y_train)),
                "test_acc": float(arbol.score(xte, y_test)),
                "cv5_acc": float(np.mean(cv_vals)),
                "nodos": int(arbol.tree_.node_count),
                "profundidad": int(arbol.tree_.max_depth),
            }
        )

    # Traduccion exacta de los umbrales numericos del arbol crudo a cada escala
    assert arbol_crudo is not None
    cols = list(X.columns)
    umbrales_dict: dict[str, float] = {}
    for i in range(arbol_crudo.tree_.node_count):
        feat_idx = int(arbol_crudo.tree_.feature[i])
        if feat_idx >= 0:
            col = cols[feat_idx]
            if col in RASGOS_NUMERICOS and col not in umbrales_dict:
                umbrales_dict[col] = float(arbol_crudo.tree_.threshold[i])

    filas_umb = []
    for nombre in esquemas:
        fila_u: dict[str, object] = {"escalador": nombre}
        esc = escaladores_ajustados[nombre]
        for col, val_crudo in umbrales_dict.items():
            if esc is None:
                fila_u[col] = val_crudo
            else:
                df_tmp = pd.DataFrame(np.zeros((1, len(RASGOS_NUMERICOS))), columns=RASGOS_NUMERICOS)
                df_tmp.loc[0, col] = val_crudo
                j = RASGOS_NUMERICOS.index(col)
                fila_u[col] = float(esc.transform(df_tmp)[0, j])  # type: ignore[union-attr]
        filas_umb.append(fila_u)

    return pd.DataFrame(filas_esc), pd.DataFrame(filas_umb), matrices_tr


def diagnostico_elegibilidad_predictiva(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    profundidad: int = PROFUNDIDAD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contrasta taxonomia regional estatica vs. validacion temporal (con y sin rezago t-1)."""
    cruce_reg = pd.crosstab(df["region"], df["clase"])
    cruce_reg["total"] = cruce_reg.sum(axis=1)
    cruce_reg["cambia_en_2015_2020"] = (cruce_reg["Alto"] > 0) & (cruce_reg["Moderado"] > 0)

    # 1) Split aleatorio estratificado (referencia del cuaderno)
    xtr, xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    arb_rand = DecisionTreeClassifier(max_depth=profundidad, random_state=SEED).fit(xtr, ytr)

    # 2) Validacion temporal out-of-time sin rezago: entrenar 2015-2018, evaluar 2019-2020
    mask_tr = df["anio"] <= 2018
    mask_te = df["anio"] >= 2019
    arb_temp = DecisionTreeClassifier(max_depth=profundidad, random_state=SEED).fit(
        X.loc[mask_tr], y.loc[mask_tr]
    )

    # 3) Validacion temporal con rezago legitimo t-1 (esfuerzo del anio previo por region)
    df_lag = df.sort_values(["region", "anio"]).copy()
    df_lag["esfuerzo_lag1"] = df_lag.groupby("region", observed=True)["esfuerzo"].shift(1)
    valid = df_lag["esfuerzo_lag1"].notna()
    X_lag = pd.get_dummies(
        df_lag.loc[valid, RASGOS_NUMERICOS + ["esfuerzo_lag1", "region"]],
        columns=["region"],
        drop_first=True,
    ).astype(float)
    y_lag = df_lag.loc[valid, "clase"]
    anios_lag = df_lag.loc[valid, "anio"]
    m_tr_l = anios_lag <= 2018
    m_te_l = anios_lag >= 2019
    arb_lag = DecisionTreeClassifier(max_depth=profundidad, random_state=SEED).fit(
        X_lag.loc[m_tr_l], y_lag.loc[m_tr_l]
    )

    comp_temp = pd.DataFrame(
        [
            {
                "protocolo": "Split aleatorio estratificado 70/30",
                "n_train": int(len(xtr)),
                "n_test": int(len(xte)),
                "acc_train": float(arb_rand.score(xtr, ytr)),
                "acc_test": float(arb_rand.score(xte, yte)),
                "lectura": "Taxonomia regional explicable en periodo observado",
            },
            {
                "protocolo": "Temporal out-of-time (2015-2018 -> 2019-2020, sin rezago)",
                "n_train": int(mask_tr.sum()),
                "n_test": int(mask_te.sum()),
                "acc_train": float(arb_temp.score(X.loc[mask_tr], y.loc[mask_tr])),
                "acc_test": float(arb_temp.score(X.loc[mask_te], y.loc[mask_te])),
                "lectura": "Caida por choque de volumen inmobiliario en 2020",
            },
            {
                "protocolo": "Temporal out-of-time con rezago t-1 (esfuerzo_lag1)",
                "n_train": int(m_tr_l.sum()),
                "n_test": int(m_te_l.sum()),
                "acc_train": float(arb_lag.score(X_lag.loc[m_tr_l], y_lag.loc[m_tr_l])),
                "acc_test": float(arb_lag.score(X_lag.loc[m_te_l], y_lag.loc[m_te_l])),
                "lectura": "Prediccion prospectiva solida aprovechando memoria regional",
            },
        ]
    )
    return cruce_reg, comp_temp


# ------------------------------------------------------------------ figuras
def fig_diagrama_flujo() -> plt.Figure:
    """Diagrama de arquitectura de 5 etapas del pipeline de clasificacion CART."""
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 3.0)
    ax.axis("off")

    etapas = [
        ("1. Cuadro Maestro", "60 filas (10 reg x 6 anios)\nEsfuerzo PMT 90% LTV\nCorte mediana = 0,5636", "#1b4f72"),
        ("2. Filtro Anti-Fuga", "Excluye precio, salario\ny anos_salario.\n7 rasgos + 9 dummies", "#2874a6"),
        ("3. Escalamiento", "Crudo / Standard /\nMinMax / Robust\n(ajuste solo en train)", "#117a65"),
        ("4. Validacion", "Split 70/30 estratificado\nCV-5 + Temporal 19-20\nCurva max_depth 1..5", "#b9770e"),
        ("5. Arbol CART (d=4)", "15 nodos · 8 hojas\nReglas export_text\nAcciones regionales", "#922b21"),
    ]
    xs = [0.2, 2.45, 4.7, 6.95, 9.2]
    for i, ((titulo, cuerpo, color), x) in enumerate(zip(etapas, xs)):
        caja = mpatches.FancyBboxPatch(
            (x, 0.55),
            1.95,
            1.85,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(caja)
        ax.text(x + 0.975, 2.10, titulo, ha="center", va="center", color="white", fontsize=8.8, weight="bold")
        ax.text(x + 0.975, 1.25, cuerpo, ha="center", va="center", color="white", fontsize=7.4, linespacing=1.35)
        if i < len(etapas) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.05, 1.48),
                xytext=(x + 2.02, 1.48),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.0),
            )

    ax.set_title("Arquitectura metodologica del pipeline de clasificacion (control anti-fuga y validacion)", fontsize=10.5, weight="bold")
    _guardar_png(fig, "flujo_clasificacion.png")
    return fig


def fig_eda_esfuerzo_clases(df: pd.DataFrame, umbral: float) -> plt.Figure:
    """Distribucion del esfuerzo con umbral teorico (0,30) vs mediana (0,5636) y conteo por region."""
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Panel A: histograma por clase
    mod = df.loc[df["clase"] == "Moderado", "esfuerzo"]
    alt = df.loc[df["clase"] == "Alto", "esfuerzo"]
    bins = np.linspace(df["esfuerzo"].min() - 0.02, df["esfuerzo"].max() + 0.02, 14)
    ejes[0].hist(mod, bins=bins, color=PALETA_CLASE["Moderado"], alpha=0.8, edgecolor="white", label=f"Moderado (n={len(mod)})")
    ejes[0].hist(alt, bins=bins, color=PALETA_CLASE["Alto"], alpha=0.8, edgecolor="white", label=f"Alto (n={len(alt)})")
    ejes[0].axvline(config.UMBRAL_ACCESIBILIDAD, color="#c0392b", ls="--", lw=1.8, label="Umbral teorico 0,30 (0 casos)")
    ejes[0].axvline(umbral, color="#1e8449", ls="-", lw=2.0, label=f"Mediana = {umbral:.4f} (30/30)")
    ejes[0].set_title("A. Etiqueta binaria balanceada por mediana")
    ejes[0].set_xlabel("Esfuerzo hipotecario (cuota / salario joven mensual)")
    ejes[0].set_ylabel("Frecuencia (region-anio)")
    ejes[0].legend(fontsize=7.5)
    ejes[0].grid(alpha=0.25)

    # Panel B: apilado por region
    ct = pd.crosstab(df["region"], df["clase"]).reindex(
        df.groupby("region", observed=True)["esfuerzo"].median().sort_values().index
    )
    ct[["Moderado", "Alto"]].plot.barh(
        stacked=True,
        ax=ejes[1],
        color=[PALETA_CLASE["Moderado"], PALETA_CLASE["Alto"]],
        edgecolor="white",
    )
    ejes[1].set_title("B. Distribucion de clases por region (2015-2020)")
    ejes[1].set_xlabel("Numero de anios observados")
    ejes[1].set_ylabel("")
    ejes[1].legend(title="Clase", fontsize=8)
    ejes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    _guardar_png(fig, "eda_esfuerzo_clases.png")
    return fig


def fig_escalamiento_comparacion(matrices_tr: dict[str, pd.DataFrame]) -> plt.Figure:
    """Boxplots comparativos de las 7 variables numericas bajo los 4 esquemas de escalado."""
    fig, ejes = plt.subplots(2, 2, figsize=(11.5, 6.4))
    ejes_flat = ejes.ravel()
    colores = ["#2e86c1", "#117a65", "#b9770e", "#7d3c98"]

    for ax, (nombre, mat), color in zip(ejes_flat, matrices_tr.items(), colores):
        bp = ax.boxplot(
            [mat[c].values for c in RASGOS_NUMERICOS],
            orientation="horizontal",
            patch_artist=True,
            tick_labels=RASGOS_NUMERICOS,
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.65)
        ax.set_title(f"Esquema: {nombre} (n_train={len(mat)})", fontsize=9.5, weight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Comparacion de magnitudes de los 7 predictores bajo 4 esquemas de escalamiento", fontsize=11, y=1.01)
    fig.tight_layout()
    _guardar_png(fig, "escalamiento_comparacion.png")
    return fig


def fig_curva_profundidad_y_cm(
    tabla: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    arbol_sel: DecisionTreeClassifier,
) -> plt.Figure:
    """Panel A: curva sesgo-varianza (train/test/CV-5). Panel B: matriz de confusion en test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    y_pred = arbol_sel.predict(X_test)
    etiquetas = ["Moderado", "Alto"]
    cm = confusion_matrix(y_test, y_pred, labels=etiquetas)

    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Panel A: curva por profundidad
    xs = np.arange(len(tabla))
    xlabels = [a.replace("profundidad_", "d=").replace("sin_limite", "Sin lim.") for a in tabla["arbol"]]
    ejes[0].plot(xs, tabla["train"], marker="o", lw=2, color="#2e86c1", label="Train")
    ejes[0].plot(xs, tabla["test"], marker="s", lw=2, color="#e67e22", label="Test (30 %)")
    ejes[0].plot(xs, tabla["cv5_estratificada"], marker="^", lw=2, ls="--", color="#1e8449", label="CV-5 estratificada")
    idx_sel = int(np.where(tabla["arbol"] == f"profundidad_{PROFUNDIDAD}")[0][0])
    ejes[0].axvline(idx_sel, color="#7f8c8d", ls=":", lw=1.4, label=f"Elegido (max_depth={PROFUNDIDAD})")
    ejes[0].set_xticks(xs)
    ejes[0].set_xticklabels(xlabels)
    ejes[0].set_ylim(0.55, 1.05)
    ejes[0].set_ylabel("Accuracy")
    ejes[0].set_xlabel("Profundidad maxima del arbol")
    ejes[0].set_title("A. Curva sesgo-varianza por profundidad")
    ejes[0].legend(fontsize=8)
    ejes[0].grid(alpha=0.3)

    # Panel B: matriz de confusion
    im = ejes[1].imshow(cm, cmap="Blues", vmin=0, vmax=max(int(cm.max()), 1))
    ejes[1].set_xticks([0, 1])
    ejes[1].set_yticks([0, 1])
    ejes[1].set_xticklabels([f"Pred: {e}" for e in etiquetas])
    ejes[1].set_yticklabels([f"Real: {e}" for e in etiquetas])
    for i in range(2):
        for j in range(2):
            color_txt = "white" if cm[i, j] > cm.max() / 2 else "black"
            ejes[1].text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color_txt, fontsize=14, weight="bold")
    ejes[1].set_title(f"B. Matriz de confusion en Test (n={len(y_test)}, max_depth={PROFUNDIDAD})")
    fig.colorbar(im, ax=ejes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    _guardar_png(fig, "curva_profundidad_cm.png")
    return fig


def guardar_figura(arbol: DecisionTreeClassifier, X: pd.DataFrame) -> plt.Figure:
    """Dibuja el arbol de decision legible, lo guarda en md/figuras y devuelve la figura."""
    fig, ax = plt.subplots(figsize=(17, 8.5))
    plot_tree(
        arbol,
        feature_names=list(X.columns),
        class_names=list(arbol.classes_),
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        f"Arbol de decision CART (max_depth={arbol.max_depth}) · Clasificacion de esfuerzo hipotecario joven (UK)",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    fig.tight_layout()
    _guardar_png(fig, "arbol_decision_esfuerzo.png")
    return fig


def fig_particion_e_importancia(
    df: pd.DataFrame,
    arbol: DecisionTreeClassifier,
    X: pd.DataFrame,
) -> plt.Figure:
    """Panel A: particion geometrica raiz/nivel 2. Panel B: importancia Gini de variables."""
    fig, ejes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # Panel A: dispersion n_transacciones vs sol_anom con lineas de corte del arbol
    for cl, color in PALETA_CLASE.items():
        sub = df[df["clase"] == cl]
        ejes[0].scatter(
            sub["n_transacciones"] / 1000.0,
            sub["sol_anom"],
            c=color,
            s=46,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.5,
            label=f"{cl} (n={len(sub)})",
        )
    rasgos = list(X.columns)
    umbral_vol = float(arbol.tree_.threshold[0]) / 1000.0
    ejes[0].axvline(umbral_vol, color="#1f2937", ls="--", lw=1.8, label=f"Corte raiz: {umbral_vol:.1f}k trans.")
    nodos_sol = [i for i in range(arbol.tree_.node_count) if arbol.tree_.feature[i] == rasgos.index("sol_anom")]
    if nodos_sol:
        umbral_sol = float(arbol.tree_.threshold[nodos_sol[0]])
        ejes[0].hlines(
            umbral_sol,
            xmin=ejes[0].get_xlim()[0],
            xmax=umbral_vol,
            colors="#8e44ad",
            linestyles=":",
            linewidth=1.8,
            label=f"Corte nivel 2: sol_anom={umbral_sol:.2f}",
        )
    ejes[0].set_xlabel("Transacciones anuales (miles)")
    ejes[0].set_ylabel("Anomalia de horas de sol (sol_anom)")
    ejes[0].set_title("A. Particion espacial (corte raiz y nivel 2)")
    ejes[0].legend(fontsize=7.5)
    ejes[0].grid(alpha=0.25)

    # Panel B: importancia Gini
    imp = pd.Series(arbol.feature_importances_, index=X.columns).sort_values(ascending=True)
    imp_act = imp[imp > 0]
    ejes[1].barh(imp_act.index, imp_act.values, color="#2e86c1", edgecolor="white")
    for i, v in enumerate(imp_act.values):
        ejes[1].text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5)
    ejes[1].set_xlim(0, max(imp_act.max() * 1.18, 0.5))
    ejes[1].set_xlabel("Importancia Gini (feature_importances_)")
    ejes[1].set_title("B. Variables activas del arbol de decision")
    ejes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    _guardar_png(fig, "particion_e_importancia.png")
    return fig


def main() -> None:
    print("=" * 74)
    print("CLASIFICACION I · ARBOL DE DECISION INTERPRETATIVO (CART, max_depth=4)")
    print("=" * 74)

    X, y, df, umbral = preparar_xy()
    print(f"Insumo: {df.shape[0]} filas | umbral mediana = {umbral:.4f} | y = {y.value_counts().to_dict()}")
    print(f"X sin fuga: {X.shape[1]} rasgos ({len(RASGOS_NUMERICOS)} numericos + {X.shape[1]-len(RASGOS_NUMERICOS)} dummies)")

    tabla_esc, tabla_umb, matrices_tr = comparar_escalamiento(X, y)
    print("\n[1] Invarianza ante 4 esquemas de escalamiento:")
    print(tabla_esc.round(3).to_string(index=False))
    assert tabla_esc["train_acc"].nunique() == 1
    assert tabla_esc["test_acc"].nunique() == 1
    assert tabla_esc["nodos"].nunique() == 1
    print("    -> assert OK: arbol CART es invariante ante transformaciones monotonas.")

    print("\n[2] Curva sesgo-varianza por profundidad:")
    modelos, tabla = entrenar_arboles(X, y)
    print(tabla.round(3).to_string(index=False))

    arbol = modelos[PROFUNDIDAD]
    print(f"\n[3] Reglas del arbol final (max_depth={PROFUNDIDAD}):")
    print(export_text(arbol, feature_names=list(X.columns), decimals=2))

    cruce_reg, comp_temp = diagnostico_elegibilidad_predictiva(X, y, df)
    print("\n[4] Diagnostico de elegibilidad predictiva (taxonomia vs. temporal):")
    print(comp_temp.round(3).to_string(index=False))

    # Generar y cerrar figuras en modo terminal
    for fig in (
        fig_diagrama_flujo(),
        fig_eda_esfuerzo_clases(df, umbral),
        fig_escalamiento_comparacion(matrices_tr),
        fig_curva_profundidad_y_cm(tabla, X, y, arbol),
        guardar_figura(arbol, X),
        fig_particion_e_importancia(df, arbol, X),
    ):
        plt.close(fig)
    print("\n[5] Figuras guardadas en md/figuras/ (6 PNG).")


if __name__ == "__main__":
    main()
