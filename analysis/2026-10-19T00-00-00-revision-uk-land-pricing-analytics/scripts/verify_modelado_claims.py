# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.9",
# ]
# ///
"""Verifica las afirmaciones cuantitativas de modelado del README/memoria.

Reproduce EXACTAMENTE la configuracion de script/proyecto/modelado.py
(arbol max_depth=4, RF n_estimators=200, KNN tuneado con tuneo interno
barajado) y solo varia el esquema de pliegues del CV externo:

 1. cv=5 sin barajar:  KNN 0.638, arbol 0.900, RF 0.943  (README/libro 08)
 2. CV-5 barajado:     KNN 0.844, arbol 0.963, RF 0.980  (README/libro 08)
 3. Regiones de TEST por pliegue: sin barajar = 2 siempre;
    barajado = "entre 5 y 9" (README).
 4. VIF con 10 dummies = infinito; con 9 (drop_first) max <= 1.8.
 5. Matriz X con drop_first = 60 x 18 (README/diccionario).
Salida: ../plots/verificacion_modelado.json + grafico WebP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "script"))

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from proyecto.modelado import RASGOS_NUMERICOS, cargar_insumo  # noqa: E402
from proyecto.modelado import agregar_esfuerzo, construir_modelos, cv_barajado  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "plots" / "verificacion_modelado.json"
res: dict = {}

maestro, hipotecas = cargar_insumo()
df = agregar_esfuerzo(maestro, hipotecas)
y = df["esfuerzo"].astype(float)

# Los modelos son EXACTOS los de modelado.py: tuneo interno del KNN siempre
# barajado (cv_barajado), solo cambia el CV externo.
modelos = construir_modelos()
kf_contiguo = KFold(n_splits=5, shuffle=False)

# [1] cv externo sin barajar (el "cv=5" por defecto de sklearn)
res["cv_r2_sin_barajar"] = {
    n: float(cross_val_score(m, df[RASGOS_NUMERICOS + ["region"]], y,
                             cv=kf_contiguo, scoring="r2").mean())
    for n, m in modelos.items()
}
# [2] cv externo barajado (la correccion publicada)
res["cv_r2_barajado"] = {
    n: float(cross_val_score(m, df[RASGOS_NUMERICOS + ["region"]], y,
                             cv=cv_barajado(), scoring="r2").mean())
    for n, m in modelos.items()
}

# [3] regiones distintas en cada pliegue de TEST
res["regiones_test_sin_barajar"] = [
    int(df.iloc[test]["region"].nunique()) for _, test in kf_contiguo.split(df)]
res["regiones_test_barajado"] = [
    int(df.iloc[test]["region"].nunique()) for _, test in cv_barajado().split(df)]

# [4] VIF 10 vs 9 dummies (mismo calculo que matriz_xy._vif_columnas)
ohe10 = OneHotEncoder(sparse_output=False, drop=None)
d10 = pd.DataFrame(ohe10.fit_transform(df[["region"]]),
                   columns=[str(c) for c in ohe10.categories_[0]])
ohe9 = OneHotEncoder(sparse_output=False, drop="first")
d9 = pd.DataFrame(ohe9.fit_transform(df[["region"]]),
                  columns=[str(c) for c in ohe9.categories_[0]][1:])


def vif_max(X: pd.DataFrame) -> float:
    mx = 0.0
    for col in X.columns:
        otros = X.drop(columns=[col])
        r2 = LinearRegression().fit(otros, X[col]).score(otros, X[col])
        mx = max(mx, float("inf") if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2))
    return mx


res["vif_max_10_dummies"] = vif_max(d10)
res["vif_max_9_dummies"] = round(vif_max(d9), 3)

# [5] matriz X con drop_first (9 dummies + 9 z_ = 18 columnas)
region_dummies = pd.get_dummies(df["region"], drop_first=True, dtype=float)
X = pd.concat([df[RASGOS_NUMERICOS].reset_index(drop=True),
               region_dummies.reset_index(drop=True)], axis=1)
res["matriz_X_shape"] = list(X.shape)

# comparacion contra lo publicado (README linea 274 y libro 08)
leido = {"knn_tuneado": [0.638, 0.844], "arbol_decision": [0.900, 0.963],
         "random_forest": [0.943, 0.980]}
res["coincide_readme"] = {
    m: bool(abs(res["cv_r2_sin_barajar"][m] - a) < 0.005
            and abs(res["cv_r2_barajado"][m] - b) < 0.005)
    for m, (a, b) in leido.items()}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

# grafico WebP de la sesion (skill: WebP para matplotlib)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

orden = ["regresion_lineal", "random_forest", "arbol_decision", "knn_tuneado"]
etiq = {"regresion_lineal": "RL", "random_forest": "RF",
        "arbol_decision": "Árbol(d=4)", "knn_tuneado": "KNN tuneado"}
x = np.arange(len(orden))
w = 0.36
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(x - w / 2, [res["cv_r2_sin_barajar"][m] for m in orden], w,
       label="cv=5 sin barajar", color="#C44E52")
ax.bar(x + w / 2, [res["cv_r2_barajado"][m] for m in orden], w,
       label="CV-5 barajado (seed 42)", color="#4C72B0")
for i, m in enumerate(orden):
    ax.text(i - w / 2, res["cv_r2_sin_barajar"][m] + 0.012,
            f"{res['cv_r2_sin_barajar'][m]:.3f}", ha="center", fontsize=8)
    ax.text(i + w / 2, res["cv_r2_barajado"][m] + 0.012,
            f"{res['cv_r2_barajado'][m]:.3f}", ha="center", fontsize=8)
ax.set_xticks(x, [etiq[m] for m in orden])
ax.set_ylabel("R² (CV-5)")
ax.set_ylim(0.5, 1.08)
ax.set_title("Verificación: efecto del barajado en el CV (maestro 60 filas)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.savefig(Path(__file__).resolve().parents[1] / "plots"
            / "cv_barajado_vs_no_barajado.webp", format="webp", dpi=140,
            bbox_inches="tight")
plt.close(fig)

print(json.dumps(res, indent=2, ensure_ascii=False))
