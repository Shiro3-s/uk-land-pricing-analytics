"""Fase 4: Sensibilidad de umbral y clasificacion con umbral alternativo.

Prueba distintos umbrales de esfuerzo para ver cuando aparece la clase 'accesible'.
Elige un umbral con ambas clases, entrena arbol, matriz de confusion y falsos positivos/negativos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto import config
from proyecto import indicadores

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, classification_report

CARPETA_FIGURAS = config.MD_FIGURAS


def figura(nombre: str) -> Path:
    return CARPETA_FIGURAS / nombre


def regla_iqr(s: pd.Series):
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {}
    q1 = s_clean.quantile(0.25)
    q3 = s_clean.quantile(0.75)
    iqr = q3 - q1
    lim_sup = q3 + 1.5 * iqr
    lim_inf = q1 - 1.5 * iqr
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lim_inf": float(lim_inf),
        "lim_sup": float(lim_sup),
        "out_sup": int((s_clean > lim_sup).sum()),
        "out_inf": int((s_clean < lim_inf).sum()),
    }


def main() -> None:
    print("=" * 72)
    print("FASE 4: SENSIBILIDAD DE UMBRAL Y CLASIFICACION CON AMBAS CLASES")
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

    df["sol_anom"] = df["sol_anom"].fillna(0.0)
    df["precio_var_pct"] = df["precio_var_pct"].fillna(0.0)

    print("\n[1] DISTRIBUCION DEL ESFUERZO")
    print(df["esfuerzo"].describe())
    iq = regla_iqr(df["esfuerzo"])
    if iq:
        print(f"IQR: Q1={iq['q1']:.3f} Q3={iq['q3']:.3f} lim_sup={iq['lim_sup']:.3f} | out_sup={iq['out_sup']} out_inf={iq['out_inf']}")

    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.6))
    datos = df["esfuerzo"].dropna()
    plt.hist(datos, bins=20, color="#0f766e", edgecolor="white")
    plt.axvline(0.30, color="#dc2626", ls="--", lw=2, label="Umbral 0.30")
    plt.axvline(0.40, color="#f59e0b", ls="--", lw=2, label="Umbral 0.40")
    plt.axvline(0.50, color="#2563eb", ls="--", lw=2, label="Umbral 0.50")
    plt.axvline(datos.median(), color="#22c55e", ls=":", lw=2, label=f"Mediana {datos.median():.3f}")
    plt.xlabel("Esfuerzo (cuota/salario joven mensual)")
    plt.ylabel("Frecuencia")
    plt.title("Sensibilidad de umbral sobre la distribucion del esfuerzo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figura("fase4_hist_esfuerzo_umbral.png"), dpi=110)
    plt.close()
    print("Guardado: fase4_hist_esfuerzo_umbral.png")

    print("\n[2] SENSIBILIDAD POR UMBRAL")
    umbrales = [0.30, 0.35, 0.40, 0.45, 0.50]
    resumen = []
    for u in umbrales:
        tmp = df.copy()
        tmp["accesible_u"] = (tmp["esfuerzo"] <= u).astype(int)
        cnt = tmp["accesible_u"].value_counts().to_dict()
        resumen.append({
            "umbral": u,
            "accesible(1)": int(cnt.get(1, 0)),
            "no_accesible(0)": int(cnt.get(0, 0)),
            "pct_accesible": round(tmp["accesible_u"].mean() * 100, 2)
        })
    res_um = pd.DataFrame(resumen)
    print(res_um.to_string(index=False))

    plt.figure(figsize=(7, 4.2))
    plt.bar([str(x) for x in res_um["umbral"]], res_um["pct_accesible"], color="#1f77b4", edgecolor="white")
    plt.xlabel("Umbral esfuerzo <= x")
    plt.ylabel("% accesible")
    plt.title("Proporcion accesible segun umbral")
    plt.tight_layout()
    plt.savefig(figura("fase4_barras_umbral.png"), dpi=110)
    plt.close()
    print("Guardado: fase4_barras_umbral.png")

    umbral_sel = 0.50
    print("\n[3] CLASIFICACION CON UMBRAL ALTERNATIVO")
    print(f"Umbral seleccionado: esfuerzo <= {umbral_sel}")
    df["accesible_sel"] = (df["esfuerzo"] <= umbral_sel).astype(int)
    print(df["accesible_sel"].value_counts())
    print(f"Proporcion 1: {df['accesible_sel'].mean():.3f}")

    print("\nPor region (umbral 0.50):")
    print(df.groupby("region")["accesible_sel"].sum().sort_values(ascending=False).to_string())

    num_vars = ["precio_mediana", "salario_joven_region", "anos_salario", "precio_var_pct",
                "tmax_anom", "tmin_anom", "lluvia_anom", "sol_anom", "af_anom"]
    X = df[num_vars + ["region"]].copy()
    X = pd.get_dummies(X, columns=["region"], drop_first=True)
    y = df["accesible_sel"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("\nPRUEBA DE PROFUNDIDAD")
    print(f"{'max_depth':>10} {'acc_train':>12} {'acc_test':>10}")
    for prof in [2, 3, 4, 5]:
        arbol = DecisionTreeClassifier(max_depth=prof, random_state=42)
        arbol.fit(X_train, y_train)
        print(f"{prof:>10} {arbol.score(X_train,y_train):>12.3f} {arbol.score(X_test,y_test):>10.3f}")

    arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    arbol.fit(X_train, y_train)
    print("\nARBOL FINAL (max_depth=3)")
    print(f"Train: {arbol.score(X_train,y_train):.3f} | Test: {arbol.score(X_test,y_test):.3f}")
    print("\nREGLAS:")
    try:
        print(export_text(arbol, feature_names=list(X.columns), max_depth=3))
    except Exception as e:
        print(e)

    y_pred = arbol.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\nMATRIZ DE CONFUSION (test) - labels [0,1]")
    print(cm)
    if cm.shape == (2, 2):
        print(f"VN(0,0):{cm[0,0]} FP(0,1):{cm[0,1]} FN(1,0):{cm[1,0]} VP(1,1):{cm[1,1]}")
        print("\nFOCO FALSOS POSITIVOS / FALSOS NEGATIVOS")
        print(f"Falsos positivos: {cm[0,1]}")
        print(f"Falsos negativos: {cm[1,0]}")

    print("\nREPORTE DE CLASIFICACION")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["no_accesible(0)", "accesible(1)"], zero_division=0))

    imp = pd.Series(arbol.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nIMPORTANCIA TOP 10")
    print(imp.head(10).round(4).to_string())

    print("\n" + "=" * 72)
    print("FIN FASE 4")
    print("=" * 72)


if __name__ == "__main__":
    main()
