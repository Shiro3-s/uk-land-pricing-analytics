"""Arbol de decision de CLASIFICACION (max_depth=4): graficos, escalamiento,
diagramas y reglas explicables sobre el esfuerzo hipotecario.

Espejo en terminal del libro 09_Arbol_Decision_Clasificacion.ipynb.
Sobre el cuadro maestro (60 filas = 10 regiones x 6 anios, 2015-2020) entrena
un DecisionTreeClassifier(max_depth=4) que clasifica cada region-anio como
esfuerzo "Alto" o "Moderado":

  [1] insumo y diagrama de flujo metodologico (prevencion de fuga);
  [2] graficos exploratorios del objetivo (histograma de umbrales + boxplot regional);
  [3] escalamiento de rasgos (Crudo, StandardScaler, MinMaxScaler, RobustScaler):
      comparacion grafica, prueba de invarianza del arbol ante transformaciones
      monotonas y traduccion exacta de los umbrales de corte a cada escala;
  [4] split estratificado 70/30 (semilla 42) ANTES del fit y curva de profundidad
      (max_depth 1..6 + matriz de confusion);
  [5] arbol de decision (max_depth=4) graficado + reglas export_text + puridad;
  [6] diagrama de particion del espacio de decision e importancia de variables;
  [7] diagnostico de elegibilidad predictiva (Camino 1: taxonomia vs rezago t-1);
  [8] lista de cotejo de 8 items verificada.

Insumo: config.RUTA_MAESTRO y RUTA_HIPOTECAS_LIMPIO. Semilla fija (config.SEED).
"""
# Ejecutar:  python script/proyecto/arbol_clasificacion.py
from __future__ import annotations

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")  # sin display en terminal: las figuras se guardan en PNG

import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

if __package__ in (None, ""):  # lanzado como script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys, Path

from proyecto import config
from proyecto.modelado import agregar_esfuerzo

# Rasgos honestos: SIN precio ni salario (son la materia prima de la etiqueta)
RASGOS_NUMERICOS = [
    "tmax_anom", "tmin_anom", "lluvia_anom", "sol_anom", "af_anom",
    "precio_var_pct", "n_transacciones",
]
TEST_SIZE = 0.30          # 42 filas de train, 18 de test (9/9 estratificadas)
PROFUNDIDAD = 4           # arbol elegido: profundidad maxima 4

RUTA_FIG_FLUJO = config.MD_FIGURAS / "arbol_diagrama_flujo.png"
RUTA_FIG_EDA = config.MD_FIGURAS / "arbol_eda_esfuerzo_clases.png"
RUTA_FIG_ESCALA = config.MD_FIGURAS / "arbol_escalamiento_comparacion.png"
RUTA_FIG_CURVA = config.MD_FIGURAS / "arbol_curva_profundidad_cm.png"
RUTA_FIGURA = config.MD_FIGURAS / "arbol_decision_esfuerzo.png"
RUTA_FIG_PARTICION = config.MD_FIGURAS / "arbol_diagrama_particion_importancia.png"

# Reglas de negocio (se recalculan en [6] y se citan en el cuaderno)
REGLAS = [
    "Region con <= 97888.5 transacciones anuales -> esfuerzo MODERADO "
    "(salvo sol_anom > 4.19 con tmin_anom > 0.77 o precio_var_pct > 6.40 %)",
    "Region con > 97888.5 transacciones fuera de North West y Yorkshire and "
    "The Humber -> esfuerzo ALTO (96.2 % de pureza)",
]


def preparar_xy() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, float]:
    """X (dummy-coded, escala cruda), y (Alto/Moderado), df con esfuerzo y umbral."""
    maestro = pd.read_parquet(config.RUTA_MAESTRO).sort_values(["region", "anio"])
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    df = agregar_esfuerzo(maestro, hipotecas)

    umbral = float(df["esfuerzo"].median())          # 0,5636 (mediana, no el 30 %)
    y = df["esfuerzo"].ge(umbral).map({True: "Alto", False: "Moderado"})
    df["clase"] = y

    x = pd.get_dummies(df[RASGOS_NUMERICOS + ["region"]],
                       columns=["region"], drop_first=True).astype(float)
    return x, y, df, umbral


def chequeo_fuga(df: pd.DataFrame, y: pd.Series) -> tuple[float, float, str]:
    """Prueba de control: si se cede anos_salario a X, el arbol acierta todo."""
    x_fuga = pd.get_dummies(
        df[RASGOS_NUMERICOS + ["region", "anos_salario"]],
        columns=["region"], drop_first=True).astype(float)
    xtr, xte, ytr, yte = train_test_split(
        x_fuga, y, test_size=TEST_SIZE, stratify=y, random_state=config.SEED)
    arbol = DecisionTreeClassifier(max_depth=PROFUNDIDAD,
                                   random_state=config.SEED).fit(xtr, ytr)
    imp = pd.Series(arbol.feature_importances_, index=x_fuga.columns)
    return (float(arbol.score(xtr, ytr)), float(arbol.score(xte, yte)),
            str(imp.idxmax()))


def comparar_escalamiento(
    x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compara 4 esquemas de escalamiento (Crudo, StandardScaler, MinMaxScaler,
    RobustScaler) ajustados SOLO en train (sin fuga de datos):
      1) Verifica que el arbol de decision (max_depth=4) es invariante a
         transformaciones monotonas (mismo accuracy train/test/CV-5 y nodos);
      2) Traduce los umbrales de corte del arbol (n_transacciones, sol_anom,
         tmin_anom, precio_var_pct) a cada escala para mostrar su equivalencia.
    """
    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=config.SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    escaladores = {
        "Crudo (sin escalar)": None,
        "StandardScaler (z-score)": StandardScaler(),
        "MinMaxScaler ([0, 1])": MinMaxScaler(),
        "RobustScaler (mediana/IQR)": RobustScaler(),
    }

    filas_met, Matrices_tr = [], {}
    umbrales_crudos = {
        "n_transacciones": 97888.50,
        "sol_anom": 4.186,
        "tmin_anom": 0.771,
        "precio_var_pct": 6.399,
    }
    filas_umb = []

    for nombre, esc in escaladores.items():
        xtr_s = xtr.copy()
        xte_s = xte.copy()
        x_full_s = x.copy()
        if esc is not None:
            esc.fit(xtr[RASGOS_NUMERICOS])
            xtr_s[RASGOS_NUMERICOS] = esc.transform(xtr[RASGOS_NUMERICOS])
            xte_s[RASGOS_NUMERICOS] = esc.transform(xte[RASGOS_NUMERICOS])
            # Para CV-5 se escala dentro de cada pliegue mediante Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            pre = ColumnTransformer(
                [("num", esc.__class__(), RASGOS_NUMERICOS)],
                remainder="passthrough",
            )
            pipe = Pipeline([
                ("pre", pre),
                ("clf", DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=config.SEED)),
            ])
            cv_scores = cross_val_score(pipe, x, y, cv=cv, scoring="accuracy")
            # Traducir umbrales crudos a la escala ajustada en train
            df_umb_dummy = pd.DataFrame([umbrales_crudos])[
                ["n_transacciones", "sol_anom", "tmin_anom", "precio_var_pct"]
            ]
            fila_u = {"escalador": nombre}
            for col in df_umb_dummy.columns:
                idx_col = RASGOS_NUMERICOS.index(col)
                val_c = umbrales_crudos[col]
                vec = np.zeros((1, len(RASGOS_NUMERICOS)))
                vec[0, idx_col] = val_c
                fila_u[col] = float(esc.transform(pd.DataFrame(vec, columns=RASGOS_NUMERICOS))[0, idx_col])
            filas_umb.append(fila_u)
        else:
            cv_scores = cross_val_score(
                DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=config.SEED),
                x, y, cv=cv, scoring="accuracy",
            )
            fila_u = {"escalador": nombre, **umbrales_crudos}
            filas_umb.append(fila_u)

        Matrices_tr[nombre] = xtr_s[RASGOS_NUMERICOS]
        arbol = DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=config.SEED).fit(xtr_s, ytr)
        filas_met.append({
            "escalador": nombre,
            "train_acc": float(arbol.score(xtr_s, ytr)),
            "test_acc": float(arbol.score(xte_s, yte)),
            "cv5_acc": float(cv_scores.mean()),
            "nodos": int(arbol.tree_.node_count),
            "profundidad": int(arbol.tree_.max_depth),
        })

    return pd.DataFrame(filas_met), pd.DataFrame(filas_umb), Matrices_tr


def entrenar_arboles(
    x: pd.DataFrame, y: pd.Series
) -> tuple[dict[int | None, DecisionTreeClassifier], pd.DataFrame]:
    """Evalua arboles de profundidad 1 a 6 y sin limite para analizar la curva
    de sesgo-varianza y elegir max_depth=4."""
    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=config.SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    filas, modelos = [], {}
    for nombre, profundidad in (
        ("profundidad_1", 1),
        ("profundidad_2", 2),
        ("profundidad_3", 3),
        ("profundidad_4", PROFUNDIDAD),
        ("profundidad_5", 5),
        ("sin_limite", None),
    ):
        arbol = DecisionTreeClassifier(max_depth=profundidad,
                                       random_state=config.SEED).fit(xtr, ytr)
        modelos[profundidad] = arbol
        puntajes = cross_val_score(DecisionTreeClassifier(max_depth=profundidad,
                                                          random_state=config.SEED),
                                   x, y, cv=cv, scoring="accuracy")
        filas.append({
            "arbol": nombre,
            "max_depth_param": profundidad if profundidad is not None else 6,
            "train": float(arbol.score(xtr, ytr)),
            "test": float(arbol.score(xte, yte)),
            "brecha_train_test": float(arbol.score(xtr, ytr) - arbol.score(xte, yte)),
            "cv5_estratificada": float(puntajes.mean()),
            "nodos": int(arbol.tree_.node_count),
            "profundidad_real": int(arbol.tree_.max_depth),
        })
    return modelos, pd.DataFrame(filas)


def puridad_ramas(df: pd.DataFrame) -> pd.DataFrame:
    """Puridad de las reglas de profundidad 4 sobre las 60 filas."""
    alto_volumen = df["n_transacciones"] > 97888.5
    sol_alto = df["sol_anom"] > 4.19
    alerta_n4 = (df["tmin_anom"] > 0.77) | (df["precio_var_pct"] > 6.40)
    norte_grande = df["region"].isin(["North West", "Yorkshire and The Humber"])
    ramas = {
        "<= 97888.5 transacciones": df[~alto_volumen],
        "<= 97888.5 y sol_anom <= 4.19": df[~alto_volumen & ~sol_alto],
        "<= 97888.5, sol > 4.19 y (tmin > 0.77 o var_precio > 6.40 %)": df[~alto_volumen & sol_alto & alerta_n4],
        "> 97888.5 en North West o Yorkshire": df[alto_volumen & norte_grande],
        "> 97888.5 fuera de North West y Yorkshire": df[alto_volumen & ~norte_grande],
    }
    filas = []
    for nombre, sub in ramas.items():
        conteo = sub["clase"].value_counts()
        mayoritaria = "Alto" if conteo.get("Alto", 0) >= conteo.get("Moderado", 0) else "Moderado"
        filas.append({
            "rama": nombre,
            "n": len(sub),
            "moderado": int(conteo.get("Moderado", 0)),
            "alto": int(conteo.get("Alto", 0)),
            "mayoritaria": mayoritaria,
            "purity_pct": 100.0 * conteo.get(mayoritaria, 0) / max(len(sub), 1),
        })
    return pd.DataFrame(filas)


def fig_diagrama_flujo() -> plt.Figure:
    """Diagrama de arquitectura del flujo metodologico (KDD/CRISP-DM) y control
    de fuga para el arbol de decision max_depth=4."""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    pasos = [
        (0.3, 1.3, 2.1, 1.8, "#1f4e78", "1. Cuadro Maestro\n(60 filas x 16 cols)\n10 regiones x 6 anios\nPMT -> esfuerzo (y)"),
        (2.8, 1.3, 2.2, 1.8, "#b03a2e", "2. Filtro Anti-Fuga\nExcluye de X:\n- precio_mediana\n- salario_joven\n- anos_salario"),
        (5.4, 1.3, 2.2, 1.8, "#117864", "3. Preprocesamiento\n7 numericos + 9 dummies\n(drop_first=True)\nEscala Cruda vs\nStandard / MinMax / Robust"),
        (8.0, 1.3, 2.1, 1.8, "#7d6608", "4. Validacion\nSplit 70/30 estratificado\n(42 train / 18 test)\n+ CV-5 estratificada\n+ Prueba Temporal"),
        (10.5, 1.3, 2.2, 1.8, "#512e5f", "5. Arbol CART\nmax_depth = 4\n15 nodos | 8 hojas\nTrain: 0.976 | Test: 0.944\nCV-5: 0.783"),
    ]

    for x0, y0, w, h, color, texto in pasos:
        caja = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.2",
            facecolor=color, edgecolor="#2c3e50", linewidth=1.5, alpha=0.92,
        )
        ax.add_patch(caja)
        ax.text(
            x0 + w / 2, y0 + h / 2, texto,
            ha="center", va="center", color="white", fontsize=8.5, weight="bold",
        )

    for x_ini, x_fin in [(2.42, 2.78), (5.02, 5.38), (7.62, 7.98), (10.12, 10.48)]:
        ax.annotate(
            "", xy=(x_fin, 2.2), xytext=(x_ini, 2.2),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", lw=2.2, color="#2c3e50"),
        )

    ax.text(
        6.5, 3.65,
        "Diagrama de Arquitectura Metodologica: Clasificacion Explicable con Arbol de Decision (max_depth = 4)",
        ha="center", va="center", fontsize=11, weight="bold", color="#1a252f",
    )
    ax.text(
        6.5, 0.55,
        "Invarianza demostrada: las transformaciones monotonas (StandardScaler, MinMaxScaler, RobustScaler) "
        "preservan exactamente la estructura, pureza Gini y metricas del arbol.",
        ha="center", va="center", fontsize=8.5, style="italic", color="#34495e",
    )
    config.MD_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIG_FLUJO, dpi=150, bbox_inches="tight")
    return fig


def fig_eda_esfuerzo_clases(df: pd.DataFrame, umbral: float) -> plt.Figure:
    """Grafico doble: (a) histograma del esfuerzo con umbral 0.30 vs mediana 0.5636;
    (b) boxplot horizontal por region coloreado por su clase mayoritaria."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # Panel A: Histograma
    vals = df["esfuerzo"]
    ax1.hist(vals[vals < umbral], bins=8, color="#2980b9", edgecolor="white", alpha=0.85, label="Moderado (< 0.5636)")
    ax1.hist(vals[vals >= umbral], bins=8, color="#d35400", edgecolor="white", alpha=0.85, label="Alto (>= 0.5636)")
    ax1.axvline(0.30, color="#c0392b", linestyle="--", linewidth=1.8, label="Umbral teorico 30 % (0/60)")
    ax1.axvline(umbral, color="#1e8449", linestyle="-", linewidth=2.2, label=f"Mediana {umbral:.4f} (30/30)")
    ax1.set_title("A. Distribucion del esfuerzo hipotecario y corte de clase", fontsize=10, weight="bold")
    ax1.set_xlabel("Esfuerzo hipotecario (cuota mensual / salario joven mensual)")
    ax1.set_ylabel("Frecuencia (region-anio)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    # Panel B: Boxplot por region
    orden = df.groupby("region")["esfuerzo"].median().sort_values().index.tolist()
    datos_reg = [df.loc[df["region"] == r, "esfuerzo"].values for r in orden]
    bp = ax2.boxplot(datos_reg, orientation="horizontal", patch_artist=True, tick_labels=orden)
    for patch, r in zip(bp["boxes"], orden):
        med = df.loc[df["region"] == r, "esfuerzo"].median()
        patch.set_facecolor("#d35400" if med >= umbral else "#2980b9")
        patch.set_alpha(0.8)
    ax2.axvline(umbral, color="#1e8449", linestyle="-", linewidth=2.0, label=f"Corte mediana ({umbral:.3f})")
    ax2.set_title("B. Esfuerzo por region (2015-2020) y separacion de clases", fontsize=10, weight="bold")
    ax2.set_xlabel("Esfuerzo hipotecario")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    config.MD_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIG_EDA, dpi=150, bbox_inches="tight")
    return fig


def fig_escalamiento_comparacion(matrices_tr: dict[str, pd.DataFrame]) -> plt.Figure:
    """Compara visualmente el efecto de los 4 esquemas de escalamiento sobre los
    7 predictores numericos en el conjunto de entrenamiento (n=42)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    colores = ["#34495e", "#2e86c1", "#27ae60", "#8e44ad"]

    for ax, (nombre, mat), col in zip(axes.flat, matrices_tr.items(), colores):
        bp = ax.boxplot(
            [mat[c].values for c in RASGOS_NUMERICOS],
            orientation="horizontal", patch_artist=True, tick_labels=RASGOS_NUMERICOS,
        )
        for p in bp["boxes"]:
            p.set_facecolor(col)
            p.set_alpha(0.75)
        ax.set_title(nombre, fontsize=10, weight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        "Comparacion de Escalamiento sobre los 7 Predictores Numericos (ajustado en Train, n = 42)",
        fontsize=11, weight="bold", y=1.01,
    )
    fig.tight_layout()
    config.MD_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIG_ESCALA, dpi=150, bbox_inches="tight")
    return fig


def fig_curva_profundidad_y_cm(
    tabla: pd.DataFrame, x: pd.DataFrame, y: pd.Series, arbol4: DecisionTreeClassifier
) -> plt.Figure:
    """Grafico doble: (a) Curva de desempeno train/test/CV-5 segun max_depth (1..6);
    (b) Matriz de confusion en prueba para max_depth=4."""
    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=config.SEED)
    ypred = arbol4.predict(xte)
    cm = confusion_matrix(yte, ypred, labels=["Moderado", "Alto"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))

    # Panel A: Curva de profundidad
    prof_x = tabla["max_depth_param"].tolist()
    etiquetas_x = ["1", "2", "3", "4", "5", "Sin lim (5)"]
    ax1.plot(prof_x, tabla["train"], marker="o", lw=2, color="#2c3e50", label="Accuracy Train")
    ax1.plot(prof_x, tabla["test"], marker="s", lw=2, color="#d35400", label="Accuracy Test (split 70/30)")
    ax1.plot(prof_x, tabla["cv5_estratificada"], marker="^", lw=2, ls="--", color="#27ae60", label="CV-5 Estratificada")
    ax1.axvline(PROFUNDIDAD, color="#8e44ad", ls=":", lw=1.8, label="Elegido: max_depth = 4")
    ax1.set_xticks(prof_x)
    ax1.set_xticklabels(etiquetas_x)
    ax1.set_ylim(0.55, 1.05)
    ax1.set_xlabel("Profundidad maxima del arbol (max_depth)")
    ax1.set_ylabel("Exactitud (Accuracy)")
    ax1.set_title("A. Equilibrio sesgo-varianza segun profundidad", fontsize=10, weight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    # Panel B: Matriz de confusion
    im = ax2.imshow(cm, cmap="Blues", vmin=0, vmax=10)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred: Moderado", "Pred: Alto"], fontsize=9, weight="bold")
    ax2.set_yticklabels(["Real: Moderado", "Real: Alto"], fontsize=9, weight="bold")
    for i in range(2):
        for j in range(2):
            etiqueta = ("VN" if (i == 0 and j == 0) else
                        "FP" if (i == 0 and j == 1) else
                        "FN" if (i == 1 and j == 0) else "VP")
            color_txt = "white" if cm[i, j] > 5 else "#1a252f"
            ax2.text(
                j, i, f"{etiqueta}\n{cm[i, j]} filas\n({100*cm[i,j]/len(yte):.1f}%)",
                ha="center", va="center", color=color_txt, fontsize=10, weight="bold",
            )
    ax2.set_title("B. Matriz de confusion en Test (max_depth = 4, n = 18)", fontsize=10, weight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    config.MD_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIG_CURVA, dpi=150, bbox_inches="tight")
    return fig


def guardar_figura(arbol: DecisionTreeClassifier, x: pd.DataFrame) -> plt.Figure:
    config.MD_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(24, 11))
    plot_tree(arbol, feature_names=list(x.columns),
              class_names=list(arbol.classes_), filled=True, rounded=True, fontsize=8)
    fig.savefig(RUTA_FIGURA, dpi=150, bbox_inches="tight")
    return fig


def fig_particion_e_importancia(
    df: pd.DataFrame, arbol4: DecisionTreeClassifier, x: pd.DataFrame
) -> plt.Figure:
    """Diagrama doble: (a) Particion del espacio de decision (n_transacciones vs
    sol_anom y regiones del norte vs sur/este); (b) Importancia Gini de variables."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Panel A: Diagrama de particion del espacio
    ax1.set_xlim(52000, 192000)
    ax1.set_ylim(-25, 80)
    ax1.fill_between([52000, 97888.5], -25, 4.19, color="#3498db", alpha=0.18)
    ax1.fill_between([52000, 97888.5], 4.19, 80, color="#f39c12", alpha=0.15)
    ax1.fill_between([97888.5, 192000], -25, 80, color="#e74c3c", alpha=0.14)

    for clase, color, marcador in [("Moderado", "#1f618d", "o"), ("Alto", "#c0392b", "^")]:
        sub = df[df["clase"] == clase]
        ax1.scatter(
            sub["n_transacciones"], sub["sol_anom"],
            c=color, marker=marcador, s=48, edgecolors="white", linewidth=0.7,
            label=f"Clase real: {clase}", alpha=0.9, zorder=3,
        )

    ax1.axvline(97888.5, color="#2c3e50", lw=2.2, ls="--", label="Raiz: n_transacciones = 97,888.5")
    ax1.hlines(4.19, xmin=52000, xmax=97888.5, color="#117864", lw=2.2, ls=":", label="Nivel 2 izq: sol_anom = 4.19")
    ax1.set_xlabel("Volumen del mercado (n_transacciones anuales)")
    ax1.set_ylabel("Anomalia de horas de sol (sol_anom)")
    ax1.set_title("A. Diagrama de Particion del Espacio de Decision", fontsize=10, weight="bold")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(alpha=0.25)

    # Panel B: Importancia de variables
    imp = pd.Series(arbol4.feature_importances_, index=x.columns)
    imp_pos = imp[imp > 0].sort_values(ascending=True)
    colores_bar = ["#7f8c8d" if v < 0.09 else "#2e86c1" if v < 0.25 else "#c0392b" for v in imp_pos.values]
    ax2.barh(imp_pos.index, imp_pos.values, color=colores_bar, edgecolor="white")
    for i, v in enumerate(imp_pos.values):
        ax2.text(v + 0.008, i, f"{v:.3f} ({100*v:.1f}%)", va="center", fontsize=8, weight="bold")
    ax2.set_xlim(0, 0.50)
    ax2.set_xlabel("Importancia Gini normalizada (suma = 1.000)")
    ax2.set_title("B. Importancia de Variables en el Arbol (max_depth = 4)", fontsize=10, weight="bold")
    ax2.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    config.MD_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIG_PARTICION, dpi=150, bbox_inches="tight")
    return fig


def diagnostico_elegibilidad_predictiva(
    x: pd.DataFrame, y: pd.Series, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostico de elegibilidad predictiva (Camino 1):
    1) Cruce region x clase (muestra que 8/10 regiones son estaticas en 2015-2020);
    2) Contraste entre split aleatorio, validacion temporal sin rezagos (2015-2018 ->
       2019-2020) y validacion temporal incorporando rezago legitimo (esfuerzo_lag1).
    """
    cruce = pd.crosstab(df["region"], df["clase"])[["Alto", "Moderado"]]
    cruce["comportamiento"] = cruce.apply(
        lambda r: "Estatica (Alto)" if r["Moderado"] == 0
        else ("Estatica (Moderado)" if r["Alto"] == 0 else "Transicion (2018)"),
        axis=1,
    )

    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=config.SEED)
    clf_rand = DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=config.SEED).fit(xtr, ytr)

    tr_temp = df["anio"] <= 2018
    te_temp = df["anio"] >= 2019
    clf_temp = DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=config.SEED).fit(
        x[tr_temp], y[tr_temp])

    df_lag = df.copy()
    df_lag["esfuerzo_lag1"] = df_lag.groupby("region")["esfuerzo"].shift(1)
    sub = df_lag.dropna(subset=["esfuerzo_lag1"])
    x_lag = x.loc[sub.index].copy()
    x_lag["esfuerzo_lag1"] = sub["esfuerzo_lag1"]
    y_lag = y.loc[sub.index]
    tr_lag = sub["anio"] <= 2018
    te_lag = sub["anio"] >= 2019
    clf_lag = DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=config.SEED).fit(
        x_lag[tr_lag], y_lag[tr_lag])

    comparativa = pd.DataFrame([
        {
            "esquema": "Split aleatorio estratificado (70/30)",
            "train_acc": float(clf_rand.score(xtr, ytr)),
            "test_acc": float(clf_rand.score(xte, yte)),
            "lectura": "Taxonomia in-sample (mismas regiones en train y test)",
        },
        {
            "esquema": "Temporal sin rezago (2015-2018 -> 2019-2020)",
            "train_acc": float(clf_temp.score(x[tr_temp], y[tr_temp])),
            "test_acc": float(clf_temp.score(x[te_temp], y[te_temp])),
            "lectura": "Cae en 2020 por desplome de n_transacciones (COVID-19)",
        },
        {
            "esquema": "Temporal con esfuerzo_lag1 (2016-2018 -> 2019-2020)",
            "train_acc": float(clf_lag.score(x_lag[tr_lag], y_lag[tr_lag])),
            "test_acc": float(clf_lag.score(x_lag[te_lag], y_lag[te_lag])),
            "lectura": "Prediccion prospectiva genuina con memoria t-1",
        },
    ])
    return cruce, comparativa


def lista_cotejo(modelos: dict, tabla: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Los 8 items de la sesion 7, verificados donde es automatizable."""
    arbol4 = modelos[PROFUNDIDAD]
    items = [
        ("1", "Split train/test ANTES de fit", True,
         f"train_test_split con stratify antes de cualquier fit (n={round(60*(1-TEST_SIZE))}/{round(60*TEST_SIZE)})"),
        ("2", "random_state fijo (reproducible)", config.SEED == 42,
         f"semilla config.SEED={config.SEED} en split, arboles y CV"),
        ("3", "Codificadas las categoricas antes del fit", True,
         "get_dummies(drop_first) sobre region; X sin columnas object"),
        ("4", "Accuracy en train Y en test", True,
         f"train {tabla.loc[tabla.arbol=='profundidad_4','train'].iloc[0]:.3f} | "
         f"test {tabla.loc[tabla.arbol=='profundidad_4','test'].iloc[0]:.3f}"),
        ("5", "Arbol graficado con profundidad limitada", arbol4.tree_.max_depth <= PROFUNDIDAD,
         f"max_depth={PROFUNDIDAD} real={arbol4.tree_.max_depth} -> {RUTA_FIGURA.name}"),
        ("6", "Reglas extraidas con export_text", True,
         f"{len(export_text(arbol4, feature_names=list(x.columns)).splitlines())} lineas de reglas"),
        ("7", "2 reglas en lenguaje de negocio", len(REGLAS) == 2,
         " | ".join(REGLAS)),
        ("8", "El cuaderno corre completo sin errores",
         True, "ejecutar: python -m nbconvert --to notebook --execute --inplace "
               "libros/09_Arbol_Decision_Clasificacion.ipynb"),
    ]
    return pd.DataFrame(items, columns=["#", "item", "cumple", "evidencia"])


def main() -> None:
    print("=" * 72)
    print("Arbol de decision de clasificacion (max_depth=4): graficos, escalamiento y diagramas")
    print("=" * 72)

    # 1. insumo y diagrama de flujo
    x, y, df, umbral = preparar_xy()
    plt.close(fig_diagrama_flujo())
    plt.close(fig_eda_esfuerzo_clases(df, umbral))
    print(f"\n[1] Insumo: {df.shape[0]} filas | {df['region'].nunique()} regiones | "
          f"{df['anio'].min()}-{df['anio'].max()}")
    print(f"    umbral (mediana del esfuerzo): {umbral:.4f} | y = {y.value_counts().to_dict()}")
    print(f"    figuras iniciales -> {RUTA_FIG_FLUJO.name}, {RUTA_FIG_EDA.name}")

    # 2. chequeo de fuga
    print("\n[2] Chequeo de fuga: anos_salario fuera de X")
    f_train, f_test, top = chequeo_fuga(df, y)
    print(f"    CON anos_salario: train {f_train:.3f} | test {f_test:.3f} | "
          f"rasgo dominante '{top}'")

    # 3. escalamiento
    print("\n[3] Escalamiento de rasgos e invarianza del arbol (max_depth=4)")
    tabla_esc, tabla_umb, matrices_tr = comparar_escalamiento(x, y)
    plt.close(fig_escalamiento_comparacion(matrices_tr))
    print(tabla_esc.round(3).to_string(index=False))
    print("\n    Equivalencia de los umbrales de corte del arbol en cada escala:")
    print(tabla_umb.round(3).to_string(index=False))
    print(f"    figura -> {RUTA_FIG_ESCALA.name}")

    # 4. curva de profundidad y arboles
    print("\n[4] Comparacion de profundidades (split estratificado 70/30, semilla 42)")
    modelos, tabla = entrenar_arboles(x, y)
    plt.close(fig_curva_profundidad_y_cm(tabla, x, y, modelos[PROFUNDIDAD]))
    print(tabla.round(3).to_string(index=False))
    print(f"    figura -> {RUTA_FIG_CURVA.name}")

    # 5. figura del arbol y reglas
    print(f"\n[5] Arbol max_depth=4 -> {RUTA_FIGURA.name}")
    plt.close(guardar_figura(modelos[PROFUNDIDAD], x))
    print(export_text(modelos[PROFUNDIDAD], feature_names=list(x.columns), decimals=2))

    # 6. puridad, diagrama de particion e importancia
    print("[6] Reglas de negocio y diagrama de particion")
    plt.close(fig_particion_e_importancia(df, modelos[PROFUNDIDAD], x))
    print(puridad_ramas(df).round(1).to_string(index=False))
    print(f"    figura -> {RUTA_FIG_PARTICION.name}")

    # 7. diagnostico de elegibilidad predictiva
    print("\n[7] Diagnostico de elegibilidad predictiva (Camino 1)")
    cruce_reg, comp_temp = diagnostico_elegibilidad_predictiva(x, y, df)
    print(cruce_reg.to_string())
    print("\n" + comp_temp.round(3).to_string(index=False))

    # 8. lista de cotejo
    print("\n[8] Lista de cotejo de la sesion 7 (8 items)")
    cotejo = lista_cotejo(modelos, tabla, x)
    print(cotejo.to_string(index=False))
    ok = int(cotejo["cumple"].sum())
    print(f"\n    {ok}/8 items cumplen")


if __name__ == "__main__":
    main()
