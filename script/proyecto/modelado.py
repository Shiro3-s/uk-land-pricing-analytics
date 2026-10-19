"""Modelado de la accesibilidad: predecir el esfuerzo hipotecario (y) desde X.

Espejo en terminal del libro 08_Modelado_Accesibilidad.ipynb.
Sobre la matriz X·y (60 filas, 9 rasgos numericos + one-hot de
region con drop_first: 9 columnas) compara 4 familias de modelos con
validacion cruzada de 5 pliegues barajados (KFold con semilla):
  [1] insumo: maestro + hipotecas -> y (esfuerzo) y X (ColumnTransformer);
  [2] modelos: regresion lineal, KNN (k tuneado), arbol de decision y random
      forest (pipeline con escalador/codificador DENTRO de cada pliegue, sin
      fuga de datos);
  [3] evaluacion: R2, RMSE y MAE con CV-5 (media +/- desviacion);
  [4] importancia de rasgos: coeficientes de la RL e importancia de permutation
      del random forest (que variable explica el esfuerzo);
  [5] limitaciones: la serie temporal (6 anios por region) queda documentada
      como fuera de alcance de este modelado tabular.

Insumo: config.RUTA_MAESTRO y RUTA_HIPOTECAS_LIMPIO. Semilla fija (config.SEED).
"""
# Ejecutar:  python script/proyecto/modelado.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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
    maestro = pd.read_parquet(config.RUTA_MAESTRO).sort_values(["region", "anio"])
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    return maestro, hipotecas


def agregar_esfuerzo(maestro: pd.DataFrame, hipotecas: pd.DataFrame
                     ) -> pd.DataFrame:
    tasa = float(hipotecas["tasa_inicial_pct"].mean())
    comisiones = float(hipotecas["comisiones_total"].mean())
    df = maestro.copy()
    cuota = df["precio_mediana"].apply(
        lambda p: indicadores.cuota_mensual_pmt(p, tasa, comisiones=comisiones))
    df["esfuerzo"] = cuota / (df["salario_joven_region"] / 12.0)
    for col in ("sol_anom", "precio_var_pct"):
        df[col] = df[col].fillna(0.0)
    return df


def cv_barajado() -> KFold:
    """KFold con barajado y semilla fija.

    El cuadro maestro esta ordenado por (region, anio). Sin barajar, los
    pliegues serian contiguos y no representativos: el fold 1 receberia solo
    la primera region y el ultimo solo la ultima. El barajado con semilla
    hace el CV reproducible (mismo split en cada corrida) y representativo.
    """
    return KFold(n_splits=5, shuffle=True, random_state=config.SEED)


def construir_preprocesador() -> ColumnTransformer:
    """StandardScaler (elegido en el libro 06) + one-hot de region; se ajusta
    DENTRO de cada pliegue del CV (el pipeline lo garantiza, sin fuga).

    `drop="first"` elimina una de las 10 columnas dummy: sin el, las 10
    indicadores de region suman siempre 1 y quedan linealmenteDEPENDientes
    (VIF infinito, la "trampa de variables mudas"). Con 9 columnas la
    referencia es implicita y los coeficientes de la regresion se pueden
    interpretar. El libro 06 mide ese contraste: VIF infinito con 10 dummies
    frente a VIF maximo 1,8 con 9.
    """
    return ColumnTransformer(
        transformers=[
            ("escala", StandardScaler(), RASGOS_NUMERICOS),
            ("region", OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                                     drop="first"),
             ["region"]),
        ],
        remainder="drop",
    )


def construir_modelos() -> dict[str, Pipeline]:
    """Pipeline (preprocesador + modelo) por familia. KNN incluye tuneo de k
    con GridSearchCV interno (3, 5, 7, 9, 11 vecinos)."""
    pre = construir_preprocesador()
    knn = Pipeline([("pre", pre), ("modelo", KNeighborsRegressor())])
    return {
        "regresion_lineal": Pipeline([("pre", pre), ("modelo", LinearRegression())]),
        "knn_tuneado": GridSearchCV(
            knn, {"modelo__n_neighbors": [3, 5, 7, 9, 11]},
            cv=cv_barajado(), scoring="r2", n_jobs=None),
        "arbol_decision": Pipeline([
            ("pre", pre),
            ("modelo", DecisionTreeRegressor(max_depth=4, random_state=config.SEED))]),
        "random_forest": Pipeline([
            ("pre", pre),
            ("modelo", RandomForestRegressor(n_estimators=200, random_state=config.SEED))]),
    }


def evaluar(df: pd.DataFrame) -> pd.DataFrame:
    """CV-5 para cada modelo: R2, RMSE y MAE (media +/- desviacion)."""
    x = df[RASGOS_NUMERICOS + ["region"]]
    y = df["esfuerzo"].astype(float)
    filas = []
    for nombre, modelo in construir_modelos().items():
        res = cross_validate(modelo, x, y, cv=cv_barajado(),
                             scoring=["r2", "neg_root_mean_squared_error",
                                      "neg_mean_absolute_error"],
                             return_train_score=False)
        filas.append({
            "modelo": nombre,
            "r2_media": float(res["test_r2"].mean()),
            "r2_desv": float(res["test_r2"].std()),
            "rmse_media": float(-res["test_neg_root_mean_squared_error"].mean()),
            "rmse_desv": float(-res["test_neg_root_mean_squared_error"].std()),
            "mae_media": float(-res["test_neg_mean_absolute_error"].mean()),
            "mae_desv": float(-res["test_neg_mean_absolute_error"].std()),
        })
    return pd.DataFrame(filas).sort_values("r2_media", ascending=False)


def importancia_rasgos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Coeficientes de la RL (valor absoluto, por columna transformada) e
    importancia de permutation del random forest (por columna de entrada: los
    9 rasgos + la region como un solo feature categorical)."""
    x = df[RASGOS_NUMERICOS + ["region"]]
    y = df["esfuerzo"].astype(float)
    pre = construir_preprocesador()

    rl = Pipeline([("pre", pre), ("modelo", LinearRegression())]).fit(x, y)
# con drop="first" la salida tiene 9 dummies pero categories_ conserva las 10:
# hay que quitar la categoria descartada para que los nombres cuadren.
    codificador = rl.named_steps["pre"].named_transformers_["region"]
    categorias = list(codificador.categories_[0])
    if getattr(codificador, "drop_idx_", None) is not None:
        categorias.pop(int(codificador.drop_idx_[0]))
    nombres_transformados = RASGOS_NUMERICOS + categorias
    coef = pd.Series(np.abs(rl.named_steps["modelo"].coef_),
                     index=nombres_transformados, name="coef_abs_RL")

    rf = Pipeline([("pre", pre),
                   ("modelo", RandomForestRegressor(n_estimators=200,
                                                    random_state=config.SEED))]
                  ).fit(x, y)
    imp = permutation_importance(rf, x, y, n_repeats=20,
                                 random_state=config.SEED)
    # permutation_importance opera sobre x (columnas de ENTRADA: 9 rasgos + region)
    nombres_entrada = RASGOS_NUMERICOS + ["region"]
    perm = pd.Series(imp.importances_mean, index=nombres_entrada,
                     name="importancia_permutacion_RF")

    # alineamos ambas visiones en una tabla: entrada (RF) y transformada (RL)
    tabla = (pd.concat([perm, coef], axis=1)
             .sort_values("importancia_permutacion_RF", ascending=False))
    return tabla, coef


def main() -> None:
    print("=" * 70)
    print("Modelado de la accesibilidad")
    print("=" * 70)

    maestro, hipotecas = cargar_insumo()
    df = agregar_esfuerzo(maestro, hipotecas)

    # 1. insumo
    print(f"\n[1] Insumo: {df.shape[0]} filas | {df['region'].nunique()} regiones | "
          f"{df['anio'].min()}-{df['anio'].max()}")
    print(f"    y = esfuerzo: media {df['esfuerzo'].mean():.3f} | "
          f"rango {df['esfuerzo'].min():.3f}-{df['esfuerzo'].max():.3f}")
    print(f"    X: {len(RASGOS_NUMERICOS)} rasgos numericos + one-hot de region")

    # 2-3. modelos y evaluacion
    print("\n[2-3] Evaluacion con CV-5 (sin fuga: escalador dentro del pliegue)")
    tabla = evaluar(df)
    print(tabla.round(3).to_string(index=False))
    mejor = tabla.iloc[0]
    peor = tabla.iloc[-1]
    print(f"\n    mejor: {mejor['modelo']} (R2 {mejor['r2_media']:.3f} +/- "
          f"{mejor['r2_desv']:.3f})")
    print(f"    peor:  {peor['modelo']} (R2 {peor['r2_media']:.3f} +/- "
          f"{peor['r2_desv']:.3f})")

    # 4. importancia
    print("\n[4] Importancia de rasgos (que explica el esfuerzo)")
    tabla_imp, coef = importancia_rasgos(df)
    print("    top 5 por importancia de permutation (random forest):")
    print(tabla_imp.head(5).round(4).to_string())
    print("\n    top 5 por coeficiente absoluto (regresion lineal):")
    print(coef.sort_values(ascending=False).head(5).round(4).to_string())

    # 5. limitaciones
    print("\n[5] Limitaciones")
    print("    - serie temporal: 6 anios por region (2015-2020) no alcanzan para")
    print("      un ARIMA/SARIMA confiable; se documenta como fuera de alcance.")
    print("    - muestra pequena (60 filas): el CV-5 es la validacion mas honesta;")
    print("      no hay set de prueba separado (se reservaria con mas datos).")
    print("    - los coeficientes/ importancias son asociaciones del modelo, no")
    print("      causalidad (el libro 07 dejo la discusion EDA).")


if __name__ == "__main__":
    main()
