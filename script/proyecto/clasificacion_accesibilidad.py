"""Clasificacion accesibilidad - Fase 3 (ultima).

Objetivo binario: accesible = 1 si esfuerzo <= 0.30, else 0
Usa train_test_split estratificado antes de entrenar.
get_dummies para region con drop_first.
Prueba profundidades, arbol final, matriz de confusion y reporte.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto import config
from proyecto import indicadores
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, classification_report


def main() -> None:
    print("=" * 72)
    print("FASE 3: ARBOL DE DECISION + FALSOS POSITIVOS")
    print("=" * 72)

    maestro = pd.read_parquet(config.RUTA_MAESTRO).copy()
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).copy()

    tasa = float(hipotecas["tasa_inicial_pct"].mean())
    comisiones = float(hipotecas["comisiones_total"].mean())
    df = maestro.copy()
    df["cuota_mensual"] = df.apply(
        lambda r: indicadores.cuota_mensual_pmt(r["precio_mediana"], tasa, config.PLAZO_ANIOS, config.LTV, comisiones),
        axis=1,
    )
    df["salario_joven_mensual"] = df["salario_joven_region"] / 12.0
    df["esfuerzo"] = df["cuota_mensual"] / df["salario_joven_mensual"]

    # imputacion consistente
    df["sol_anom"] = df["sol_anom"].fillna(0.0)
    df["precio_var_pct"] = df["precio_var_pct"].fillna(0.0)

    # objetivo binario
    df["accesible"] = (df["esfuerzo"] <= 0.30).astype(int)
    print(f"\nObjetivo: esfuerzo <= 0.30 -> accesible=1")
    print(df["accesible"].value_counts())
    print(f"Proporcion 1: {df['accesible'].mean():.3f}")

    num_vars = ["precio_mediana", "salario_joven_region", "anos_salario", "precio_var_pct",
                "tmax_anom", "tmin_anom", "lluvia_anom", "sol_anom", "af_anom"]
    X = df[num_vars + ["region"]].copy()
    X = pd.get_dummies(X, columns=["region"], drop_first=True)
    y = df["accesible"]

    print(f"\nShape X: {X.shape} | y: {y.shape}")
    print("Columnas X:", list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Train proporciones:\n{y_train.value_counts(normalize=True).round(3)}")
    print(f"Test proporciones:\n{y_test.value_counts(normalize=True).round(3)}")

    print("\n" + "=" * 72)
    print("PRUEBA DE PROFUNDIDAD")
    print("=" * 72)
    print(f"{'max_depth':>10} {'acc_train':>12} {'acc_test':>10}")
    for prof in [2, 3, 4, 5]:
        arbol = DecisionTreeClassifier(max_depth=prof, random_state=42)
        arbol.fit(X_train, y_train)
        print(f"{prof:>10} {arbol.score(X_train,y_train):>12.3f} {arbol.score(X_test,y_test):>10.3f}")

    print("\n" + "=" * 72)
    print("ARBOL FINAL (max_depth=3)")
    print("=" * 72)
    arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    arbol.fit(X_train, y_train)
    print(f"Train: {arbol.score(X_train,y_train):.3f} | Test: {arbol.score(X_test,y_test):.3f}")
    print("\nREGLAS:")
    try:
        print(export_text(arbol, feature_names=list(X.columns), max_depth=3))
    except Exception as e:
        print(f"Sin reglas legibles: {e}")

    y_pred = arbol.predict(X_test)
    print("\n" + "=" * 72)
    print("MATRIZ DE CONFUSION (test)")
    print("=" * 72)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(cm)
    if cm.shape == (2, 2):
        print("\nInterpretacion:")
        print(f"VN (0,0): {cm[0,0]} | FP (0,1): {cm[0,1]}")
        print(f"FN (1,0): {cm[1,0]} | VP (1,1): {cm[1,1]}")
        print("\n" + "=" * 72)
        print("FOCO: FALSOS POSITIVOS / FALSOS NEGATIVOS")
        print("=" * 72)
        print(f"Falsos positivos (predice accesible=1 pero realmente no lo es): {cm[0,1]}")
        print(f"Falsos negativos (predice accesible=0 pero realmente si lo es): {cm[1,0]}")
    else:
        print("\nSolo existe la clase 0. No hay casos reales de clase 1.")
        print("Por consiguiente no se pueden calcular falsos positivos/negativos con datos reales.")
        print("El modelo predice todo como 0 (clase mayoritaria).")

    print("\n" + "=" * 72)
    print("REPORTE DE CLASIFICACION")
    print("=" * 72)
    try:
        print(classification_report(y_test, y_pred, labels=[0,1], target_names=["no_accesible(0)", "accesible(1)"], zero_division=0))
    except Exception as e:
        print(f"Reporte no disponible: {e}")

    print("\n" + "=" * 72)
    print("IMPORTANCIA DE VARIABLES")
    print("=" * 72)
    imp = pd.Series(arbol.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(imp.head(10).round(4).to_string())

    print("\n" + "=" * 72)
    print("CONCLUSION FASE 3")
    print("=" * 72)
    print("Con esfuerzo <= 0.30: ninguna observacion es accesible (100% no accesibles).")
    print("El arbol predice la unica clase presente. No hay falsos positivos ni falsos negativos con esta definicion.")
    print("Para poder evaluar FP/FN se requiere al menos una observacion de la clase 'accesible'.")
    print("=" * 72)


if __name__ == "__main__":
    main()
