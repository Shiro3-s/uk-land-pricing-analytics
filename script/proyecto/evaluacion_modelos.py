"""Evaluacion profunda de modelos (Fase 5 CRISP-DM): desbalance de clases,
modelo ingenuo, matriz de confusion con costos asimetricos, precision/recall/F1
verificados manualmente, validacion cruzada k-fold en Pipeline, curvas ROC-AUC /
Precision-Recall, optimizacion del umbral de decision y comparativa global de
que algoritmos funcionan mejor en nuestro proyecto.

Espejo en terminal y biblioteca del cuaderno 13_Evaluacion_Modelos_y_Umbrales.ipynb.

Ejecutar:  python script/proyecto/evaluacion_modelos.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys

from proyecto import config
from proyecto.arbol_clasificacion import PROFUNDIDAD, RASGOS_NUMERICOS, TEST_SIZE, preparar_xy

SEED = config.SEED


def _guardar_png(fig: plt.Figure, nombre: str) -> Path:
    ruta = config.MD_FIGURAS / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    return ruta


def _peso_kb(modelo: object) -> float:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        ruta = Path(tmp.name)
    try:
        joblib.dump(modelo, ruta, compress=3)
        return ruta.stat().st_size / 1024.0
    finally:
        if ruta.exists():
            ruta.unlink()


def construir_modelos_evaluacion() -> dict[str, object]:
    """Catalogo completo de clasificadores comparados sobre la misma particion."""
    return {
        "Ingenuo (mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Arbol CART (d=4)": DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=SEED),
        "Naive Bayes (GaussianNB)": GaussianNB(),
        "KNN escalado (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "KNN escalado (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "Random Forest (ensamble)": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=SEED
        ),
    }


def preparar_escenarios() -> dict[str, dict[str, object]]:
    """Prepara los 3 escenarios reales del proyecto (1 balanceado + 2 desbalanceados)."""
    X_m, y_m_str, df_m, umbral_med = preparar_xy()
    y_m_bal = (y_m_str == "Alto").astype(int)
    # Escenario desbalanceado 1: esfuerzo critico >= 0.80 (11/60 = 18.3 %)
    y_m_crit = (df_m["esfuerzo"] >= 0.80).astype(int)

    # Escenario desbalanceado 2: ofertas hipotecarias de tasa atipica IQR (762/7796 = 9.77 %)
    hip = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).copy()
    X_h = pd.get_dummies(
        hip[["comisiones_total", "meses_inicial", "tipo", "subtitulo"]],
        columns=["tipo", "subtitulo"],
        drop_first=True,
    ).astype(float)
    y_h_atip = hip["es_atipico_tasa"].astype(int)

    return {
        "maestro_balanceado": {
            "nombre": "Cuadro Maestro · Esfuerzo Alto vs. Moderado (50/50)",
            "X": X_m,
            "y": y_m_bal,
            "stratify_ref": y_m_str,
            "test_size": TEST_SIZE,
            "df": df_m,
            "umbral_ref": umbral_med,
        },
        "maestro_critico": {
            "nombre": "Cuadro Maestro · Esfuerzo Critico >= 0,80 (18,3 % minoritario)",
            "X": X_m,
            "y": y_m_crit,
            "stratify_ref": y_m_crit,
            "test_size": TEST_SIZE,
            "df": df_m,
            "umbral_ref": 0.80,
        },
        "hipotecas_atipico": {
            "nombre": "Catalogo Hipotecario F2 · Tasa Atipica IQR (9,77 % minoritario)",
            "X": X_h,
            "y": y_h_atip,
            "stratify_ref": y_h_atip,
            "test_size": 0.25,
            "df": hip,
            "umbral_ref": 8.13,
        },
    }


def verificar_metricas_manuales(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Calcula TN, FP, FN, TP, Precision, Recall, F1 armonica vs aritmetica SIN sklearn y verifica con assert."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec_man = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_man = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_armonica = (2.0 * prec_man * rec_man / (prec_man + rec_man)) if (prec_man + rec_man) > 0 else 0.0
    media_aritmetica = 0.5 * (prec_man + rec_man)
    acc_man = (tp + tn) / max(tp + tn + fp + fn, 1)

    # Asercion exacta contra scikit-learn
    assert np.isclose(prec_man, precision_score(y_true, y_pred, zero_division=0), atol=1e-12)
    assert np.isclose(rec_man, recall_score(y_true, y_pred, zero_division=0), atol=1e-12)
    assert np.isclose(f1_armonica, f1_score(y_true, y_pred, zero_division=0), atol=1e-12)
    assert np.isclose(acc_man, accuracy_score(y_true, y_pred), atol=1e-12)

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy_manual": round(acc_man, 4),
        "precision_manual": round(prec_man, 4),
        "recall_manual": round(rec_man, 4),
        "f1_armonica": round(f1_armonica, 4),
        "media_aritmetica": round(media_aritmetica, 4),
        "penalizacion_armonica": round(media_aritmetica - f1_armonica, 4),
    }


def evaluar_escenario_completo(esc: dict[str, object], clave_esc: str) -> dict[str, object]:
    """Evalua todos los clasificadores (incluyendo Ingenuo y RF) con metricas forenses y CV-5 en Pipeline."""
    X: pd.DataFrame = esc["X"]  # type: ignore[assignment]
    y: pd.Series = esc["y"]  # type: ignore[assignment]
    strat = esc["stratify_ref"]
    test_sz = float(esc["test_size"])

    idx_tr, idx_te = train_test_split(X.index, test_size=test_sz, stratify=strat, random_state=SEED)
    X_tr, X_te = X.loc[idx_tr], X.loc[idx_te]
    y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    filas_res = []
    filas_cv = []
    reportes: dict[str, str] = {}
    cms: dict[str, np.ndarray] = {}
    probs: dict[str, np.ndarray] = {}
    modelos_fit: dict[str, object] = {}

    for nombre, modelo in construir_modelos_evaluacion().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo.fit(X_tr, y_tr)
            _ = modelo.predict(X_te)  # warm-up

            t0 = time.perf_counter()
            for _ in range(5):
                y_pred = modelo.predict(X_te)
            pred_ms = ((time.perf_counter() - t0) / 5.0) * 1000.0

            y_prob = modelo.predict_proba(X_te)[:, 1]
            verif = verificar_metricas_manuales(y_te.to_numpy(), y_pred)

            cv_acc = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
            cv_f1 = cross_val_score(modelo, X, y, cv=cv, scoring="f1")
            cv_rec = cross_val_score(modelo, X, y, cv=cv, scoring="recall")
            cv_auc = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")

        auc_te = float(roc_auc_score(y_te, y_prob))
        ap_te = float(average_precision_score(y_te, y_prob))
        brier = float(brier_score_loss(y_te, y_prob))
        kb = _peso_kb(modelo)

        cms[nombre] = confusion_matrix(y_te, y_pred, labels=[0, 1])
        probs[nombre] = y_prob
        modelos_fit[nombre] = modelo
        reportes[nombre] = classification_report(
            y_te, y_pred, labels=[0, 1], target_names=["No critico (0)", "Positivo (1)"], zero_division=0
        )

        filas_res.append(
            {
                "escenario": clave_esc,
                "modelo": nombre,
                "tasa_positiva_pct": round(100.0 * float(y.mean()), 2),
                "acc_train": round(float(accuracy_score(y_tr, modelo.predict(X_tr))), 4),
                "acc_test": verif["accuracy_manual"],
                "precision_pos": verif["precision_manual"],
                "recall_pos": verif["recall_manual"],
                "f1_pos": verif["f1_armonica"],
                "roc_auc": round(auc_te, 4),
                "pr_auc": round(ap_te, 4),
                "brier": round(brier, 4),
                "cv5_acc_media": round(float(cv_acc.mean()), 4),
                "cv5_acc_desv": round(float(cv_acc.std()), 4),
                "cv5_f1_media": round(float(cv_f1.mean()), 4),
                "cv5_f1_desv": round(float(cv_f1.std()), 4),
                "cv5_rec_media": round(float(cv_rec.mean()), 4),
                "cv5_auc_media": round(float(cv_auc.mean()), 4),
                "TN": verif["TN"],
                "FP": verif["FP"],
                "FN": verif["FN"],
                "TP": verif["TP"],
                "predict_ms": round(pred_ms, 3),
                "peso_kb": round(kb, 2),
            }
        )

        for fold_i in range(len(cv_acc)):
            filas_cv.append(
                {
                    "escenario": clave_esc,
                    "modelo": nombre,
                    "pliegue": fold_i + 1,
                    "accuracy": round(float(cv_acc[fold_i]), 4),
                    "f1": round(float(cv_f1[fold_i]), 4),
                    "recall": round(float(cv_rec[fold_i]), 4),
                    "roc_auc": round(float(cv_auc[fold_i]), 4),
                }
            )

    return {
        "tabla": pd.DataFrame(filas_res),
        "cv_pliegues": pd.DataFrame(filas_cv),
        "cms": cms,
        "probs": probs,
        "reportes": reportes,
        "modelos_fit": modelos_fit,
        "y_test": y_te,
        "X_test": X_te,
    }


def barrido_umbrales_decision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    umbrales: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int | str]]]:
    """Recorre umbrales de probabilidad y extrae los 3 puntos operativos por costo asimetrico."""
    if umbrales is None:
        umbrales = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    filas = []
    for u in umbrales:
        y_p = (y_prob >= u).astype(int)
        m = verificar_metricas_manuales(y_true, y_p)
        fpr = m["FP"] / max(m["FP"] + m["TN"], 1)
        youden_j = float(m["recall_manual"]) - fpr
        filas.append(
            {
                "umbral": u,
                "TP_detectados": m["TP"],
                "FN_perdidos": m["FN"],
                "FP_falsas_alarmas": m["FP"],
                "TN_correctos": m["TN"],
                "precision": m["precision_manual"],
                "recall": m["recall_manual"],
                "f1": m["f1_armonica"],
                "media_aritmetica": m["media_aritmetica"],
                "fpr": round(fpr, 4),
                "youden_J": round(youden_j, 4),
                "total_alarmas": int(m["TP"] + m["FP"]),
            }
        )
    df_u = pd.DataFrame(filas)

    # 1) Equilibrado (maximo F1)
    fila_f1 = df_u.loc[df_u["f1"].idxmax()]
    # 2) Alerta temprana informativa (FP barato: recall alto con precision razonable >= 0.25)
    cand_alerta = df_u[df_u["precision"] >= 0.25]
    fila_alerta = cand_alerta.loc[cand_alerta["recall"].idxmax()] if not cand_alerta.empty else df_u.iloc[0]
    # 3) Alta precision para subsidio limitado (FP caro: maxima precision con al menos 15 % recall)
    cand_prec = df_u[df_u["recall"] >= 0.15]
    fila_prec = cand_prec.loc[cand_prec["precision"].idxmax()] if not cand_prec.empty else fila_f1

    perfiles = {
        "alerta_temprana": {
            "umbral": float(fila_alerta["umbral"]),
            "uso": "Alerta informativa web de bajo costo (prioriza Recall para no dejar pasar ofertas riesgosas).",
            "precision": float(fila_alerta["precision"]),
            "recall": float(fila_alerta["recall"]),
            "f1": float(fila_alerta["f1"]),
            "TP": int(fila_alerta["TP_detectados"]),
            "FP": int(fila_alerta["FP_falsas_alarmas"]),
        },
        "equilibrado_f1": {
            "umbral": float(fila_f1["umbral"]),
            "uso": "Operacion general y tablero ejecutivo (maximiza F1 armonica equilibrando Precision y Recall).",
            "precision": float(fila_f1["precision"]),
            "recall": float(fila_f1["recall"]),
            "f1": float(fila_f1["f1"]),
            "TP": int(fila_f1["TP_detectados"]),
            "FP": int(fila_f1["FP_falsas_alarmas"]),
        },
        "alta_precision": {
            "umbral": float(fila_prec["umbral"]),
            "uso": "Asignacion de subsidios o revisiones costosas (prioriza Precision para minimizar falsas alarmas).",
            "precision": float(fila_prec["precision"]),
            "recall": float(fila_prec["recall"]),
            "f1": float(fila_prec["f1"]),
            "TP": int(fila_prec["TP_detectados"]),
            "FP": int(fila_prec["FP_falsas_alarmas"]),
        },
    }
    return df_u, perfiles


def simular_predicciones_por_region(esc_bal: dict[str, object]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Compara lado a lado que predice CADA algoritmo sobre el perfil real mediano de cada region del UK."""
    X: pd.DataFrame = esc_bal["X"]  # type: ignore[assignment]
    y: pd.Series = esc_bal["y"]  # type: ignore[assignment]
    df: pd.DataFrame = esc_bal["df"]  # type: ignore[assignment]

    modelos = construir_modelos_evaluacion()
    for m in modelos.values():
        m.fit(X, y)

    regiones_orden = (
        df.groupby("region", observed=True)["esfuerzo"].median().sort_values(ascending=False).index.tolist()
    )
    filas = []
    perfiles_ref: dict[str, dict[str, float]] = {}

    for reg in regiones_orden:
        sub = df[df["region"] == reg]
        med_esf = float(sub["esfuerzo"].median())
        real_lbl = "Alto" if med_esf >= float(esc_bal["umbral_ref"]) else "Moderado"

        fila_x = {c: 0.0 for c in X.columns}
        perfil_reg = {}
        for col in RASGOS_NUMERICOS:
            val = float(sub[col].median())
            fila_x[col] = val
            perfil_reg[col] = round(val, 4)
        col_dum = f"region_{reg}"
        if col_dum in fila_x:
            fila_x[col_dum] = 1.0
        perfiles_ref[reg] = perfil_reg

        X_q = pd.DataFrame([fila_x])
        registro: dict[str, object] = {
            "region": reg,
            "esfuerzo_mediano_real": round(med_esf, 3),
            "real": real_lbl,
        }
        for nombre, mod in modelos.items():
            if nombre.startswith("Ingenuo"):
                continue
            p = int(mod.predict(X_q)[0])
            pr = float(mod.predict_proba(X_q)[0, 1])
            et = "Alto" if p == 1 else "Moderado"
            ok = "OK" if et == real_lbl else "ERR"
            corto = (
                nombre.replace("Arbol CART (d=4)", "Arbol_d4")
                .replace("Naive Bayes (GaussianNB)", "NaiveBayes")
                .replace("KNN escalado (k=5)", "KNN_k5")
                .replace("KNN escalado (k=15)", "KNN_k15")
                .replace("Random Forest (ensamble)", "RandomForest")
            )
            registro[corto] = f"{et} ({pr:.2f}) [{ok}]"
        filas.append(registro)

    return pd.DataFrame(filas), perfiles_ref


def tabla_semaforo_global(
    ev_bal: dict[str, object],
    ev_crit: dict[str, object],
    ev_hip: dict[str, object],
) -> pd.DataFrame:
    """Sintesis global de que modelos funcionan mejor, cuales fallan y cual elegir en el proyecto."""
    b = ev_bal["tabla"].set_index("modelo")
    c = ev_crit["tabla"].set_index("modelo")
    h = ev_hip["tabla"].set_index("modelo")

    filas = [
        {
            "Algoritmo / Familia": "Arbol de Decision CART (max_depth=4)",
            "Estado": "GANADOR PRINCIPAL",
            "Maestro 50/50 (F1 / CV-5)": f"{b.loc['Arbol CART (d=4)','f1_pos']:.3f} / {b.loc['Arbol CART (d=4)','cv5_acc_media']:.3f}",
            "Critico 18% (F1 / AUC)": f"{c.loc['Arbol CART (d=4)','f1_pos']:.3f} / {c.loc['Arbol CART (d=4)','roc_auc']:.3f}",
            "Hipotecas Atipico 9,8% (F1 / AUC)": f"{h.loc['Arbol CART (d=4)','f1_pos']:.3f} / {h.loc['Arbol CART (d=4)','roc_auc']:.3f}",
            "Peso / Latencia F2": f"{h.loc['Arbol CART (d=4)','peso_kb']:.1f} KB / {h.loc['Arbol CART (d=4)','predict_ms']:.1f} ms",
            "Veredicto practico": "Mejor balance global: explicable con reglas, liviano (1,4 KB), sin dependencia de escalado y alto AUC.",
        },
        {
            "Algoritmo / Familia": "Random Forest (ensamble 200 arboles)",
            "Estado": "MEJOR ENSAMBLE (CHALLENGER)",
            "Maestro 50/50 (F1 / CV-5)": f"{b.loc['Random Forest (ensamble)','f1_pos']:.3f} / {b.loc['Random Forest (ensamble)','cv5_acc_media']:.3f}",
            "Critico 18% (F1 / AUC)": f"{c.loc['Random Forest (ensamble)','f1_pos']:.3f} / {c.loc['Random Forest (ensamble)','roc_auc']:.3f}",
            "Hipotecas Atipico 9,8% (F1 / AUC)": f"{h.loc['Random Forest (ensamble)','f1_pos']:.3f} / {h.loc['Random Forest (ensamble)','roc_auc']:.3f}",
            "Peso / Latencia F2": f"{h.loc['Random Forest (ensamble)','peso_kb']:.1f} KB / {h.loc['Random Forest (ensamble)','predict_ms']:.1f} ms",
            "Veredicto practico": "Maxima robustez en desbalance y probabilidades muy calibradas, pero pierde reglas simples y pesa mucho mas.",
        },
        {
            "Algoritmo / Familia": "KNN escalado (k=5, Pipeline)",
            "Estado": "CHALLENGER LOCAL",
            "Maestro 50/50 (F1 / CV-5)": f"{b.loc['KNN escalado (k=5)','f1_pos']:.3f} / {b.loc['KNN escalado (k=5)','cv5_acc_media']:.3f}",
            "Critico 18% (F1 / AUC)": f"{c.loc['KNN escalado (k=5)','f1_pos']:.3f} / {c.loc['KNN escalado (k=5)','roc_auc']:.3f}",
            "Hipotecas Atipico 9,8% (F1 / AUC)": f"{h.loc['KNN escalado (k=5)','f1_pos']:.3f} / {h.loc['KNN escalado (k=5)','roc_auc']:.3f}",
            "Peso / Latencia F2": f"{h.loc['KNN escalado (k=5)','peso_kb']:.1f} KB / {h.loc['KNN escalado (k=5)','predict_ms']:.1f} ms",
            "Veredicto practico": "Excelente precision local, pero lento en predict(), almacena datos en RAM y falla si se olvida escalar.",
        },
        {
            "Algoritmo / Familia": "KNN escalado (k=15, Pipeline)",
            "Estado": "ESTABLE / SUBAJUSTE EN MINORITARIA",
            "Maestro 50/50 (F1 / CV-5)": f"{b.loc['KNN escalado (k=15)','f1_pos']:.3f} / {b.loc['KNN escalado (k=15)','cv5_acc_media']:.3f}",
            "Critico 18% (F1 / AUC)": f"{c.loc['KNN escalado (k=15)','f1_pos']:.3f} / {c.loc['KNN escalado (k=15)','roc_auc']:.3f}",
            "Hipotecas Atipico 9,8% (F1 / AUC)": f"{h.loc['KNN escalado (k=15)','f1_pos']:.3f} / {h.loc['KNN escalado (k=15)','roc_auc']:.3f}",
            "Peso / Latencia F2": f"{h.loc['KNN escalado (k=15)','peso_kb']:.1f} KB / {h.loc['KNN escalado (k=15)','predict_ms']:.1f} ms",
            "Veredicto practico": "Suaviza fronteras, pero con 9,8 % de positivos tiende a diluirse frente a k=5.",
        },
        {
            "Algoritmo / Familia": "Naive Bayes (GaussianNB)",
            "Estado": "DESCARTADO (SOLO BASELINE)",
            "Maestro 50/50 (F1 / CV-5)": f"{b.loc['Naive Bayes (GaussianNB)','f1_pos']:.3f} / {b.loc['Naive Bayes (GaussianNB)','cv5_acc_media']:.3f}",
            "Critico 18% (F1 / AUC)": f"{c.loc['Naive Bayes (GaussianNB)','f1_pos']:.3f} / {c.loc['Naive Bayes (GaussianNB)','roc_auc']:.3f}",
            "Hipotecas Atipico 9,8% (F1 / AUC)": f"{h.loc['Naive Bayes (GaussianNB)','f1_pos']:.3f} / {h.loc['Naive Bayes (GaussianNB)','roc_auc']:.3f}",
            "Peso / Latencia F2": f"{h.loc['Naive Bayes (GaussianNB)','peso_kb']:.1f} KB / {h.loc['Naive Bayes (GaussianNB)','predict_ms']:.1f} ms",
            "Veredicto practico": "Muy veloz, pero viola normalidad e independencia en variables financieras (calibracion pobre).",
        },
        {
            "Algoritmo / Familia": "Modelo Ingenuo (clase mayoritaria)",
            "Estado": "CONTROL NEGATIVO (NO USAR)",
            "Maestro 50/50 (F1 / CV-5)": f"{b.loc['Ingenuo (mayoritaria)','f1_pos']:.3f} / {b.loc['Ingenuo (mayoritaria)','cv5_acc_media']:.3f}",
            "Critico 18% (F1 / AUC)": f"{c.loc['Ingenuo (mayoritaria)','f1_pos']:.3f} / {c.loc['Ingenuo (mayoritaria)','roc_auc']:.3f}",
            "Hipotecas Atipico 9,8% (F1 / AUC)": f"{h.loc['Ingenuo (mayoritaria)','f1_pos']:.3f} / {h.loc['Ingenuo (mayoritaria)','roc_auc']:.3f}",
            "Peso / Latencia F2": f"{h.loc['Ingenuo (mayoritaria)','peso_kb']:.1f} KB / {h.loc['Ingenuo (mayoritaria)','predict_ms']:.1f} ms",
            "Veredicto practico": "Exhibe 90,2 % accuracy en F2 detectando 0 % de casos atipicos (prueba de la trampa del accuracy).",
        },
        {
            "Algoritmo / Familia": "Clustering (K-Means + Ward + MiniBatchKMeans 1M)",
            "Estado": "ACTIVO (NO SUPERVISADO)",
            "Maestro 50/50 (F1 / CV-5)": "Enriquece CART CV-5: 0,783 -> 0,800",
            "Critico 18% (F1 / AUC)": "Separa 100 % London/SE (esf. 0,90)",
            "Hipotecas Atipico 9,8% (F1 / AUC)": "F2 silueta 0,464 · 1M F1 en 45,8 MB",
            "Peso / Latencia F2": "Streaming O(1) RAM (local/cloud)",
            "Veredicto practico": "Ideal para segmentar mercados, hipotecas y 1M+ ventas sin etiquetas.",
        },
        {
            "Algoritmo / Familia": "Reglas de Asociacion (Apriori puro)",
            "Estado": "ACTIVO (PATRONES)",
            "Maestro 50/50 (F1 / CV-5)": "195 reglas F1 / 481 reglas F2",
            "Critico 18% (F1 / AUC)": "Detached+Freehold -> Extremo (lift 2,00)",
            "Hipotecas Atipico 9,8% (F1 / AUC)": "Discounted -> Tasa Baja Q1 (lift 3,96)",
            "Peso / Latencia F2": "23,6 MB RAM · auditable",
            "Veredicto practico": "Ideal para alertas de combinaciones de riesgo y orientacion al comprador joven.",
        },
    ]
    return pd.DataFrame(filas)


def enriquecer_contrato_json(
    perfiles_umbral: dict[str, dict[str, object]],
    perfiles_reg: dict[str, dict[str, float]],
    semaforo: pd.DataFrame,
) -> Path:
    """Enriquece data/modelos/contrato_prediccion.json con umbrales operativos y perfiles regionales."""
    ruta = config.RUTA_CONTRATO_PREDICCION
    contrato = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    contrato["umbrales_operativos_atipico_f2"] = perfiles_umbral
    contrato["perfiles_regionales_referencia"] = perfiles_reg
    contrato["resumen_semaforo_modelos"] = semaforo[
        ["Algoritmo / Familia", "Estado", "Veredicto practico"]
    ].to_dict(orient="records")
    ruta.write_text(json.dumps(contrato, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


# ------------------------------------------------------------------ figuras
def fig_ingenuo_vs_modelos(ev_hip: dict[str, object], ev_crit: dict[str, object]) -> plt.Figure:
    """Desenmascara la trampa del accuracy: Accuracy alto del ingenuo vs Recall=0 y F1=0."""
    fig, ejes = plt.subplots(1, 2, figsize=(12.8, 4.6))

    for ax, ev, titulo in (
        (ejes[0], ev_hip, "A. Hipotecas F2 · Tasa Atipica (9,77 % positivos, n=7.796)"),
        (ejes[1], ev_crit, "B. Cuadro Maestro · Esfuerzo Critico >= 0,80 (18,3 % positivos)"),
    ):
        t = ev["tabla"]
        xs = np.arange(len(t))
        w = 0.25
        ax.bar(xs - w, t["acc_test"], w, color="#95a5a6", label="Accuracy (engana)")
        ax.bar(xs, t["recall_pos"], w, color="#e67e22", label="Recall minoritaria")
        ax.bar(xs + w, t["f1_pos"], w, color="#117a65", label="F1 minoritaria")
        for i, r in t.iterrows():
            ax.text(i - w, r["acc_test"] + 0.015, f"{r['acc_test']:.2f}", ha="center", fontsize=6.8)
            ax.text(i, r["recall_pos"] + 0.015, f"{r['recall_pos']:.2f}", ha="center", fontsize=6.8)
            ax.text(i + w, r["f1_pos"] + 0.015, f"{r['f1_pos']:.2f}", ha="center", fontsize=6.8, weight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([m.replace(" (", "\n(") for m in t["modelo"]], fontsize=7.2)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Puntaje (0 a 1)")
        ax.set_title(titulo, fontsize=9.5, weight="bold")
        ax.legend(fontsize=7.6, loc="lower right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Trampa del Accuracy en clases desbalanceadas: por que Accuracy sola miente", fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "eval_ingenuo_vs_modelos.png")
    return fig


def fig_curvas_roc_pr(ev_hip: dict[str, object]) -> plt.Figure:
    """Curvas ROC (AUC) y Precision-Recall sobre el escenario desbalanceado F2 (n_test=1.949)."""
    y_te = ev_hip["y_test"].to_numpy()
    probs = ev_hip["probs"]
    seleccion = [
        ("Arbol CART (d=4)", "#2e86c1"),
        ("Random Forest (ensamble)", "#117a65"),
        ("KNN escalado (k=5)", "#e67e22"),
        ("KNN escalado (k=15)", "#8e44ad"),
        ("Naive Bayes (GaussianNB)", "#c0392b"),
    ]

    fig, ejes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for nombre, col in seleccion:
        p = probs[nombre]
        fpr, tpr, _ = roc_curve(y_te, p)
        auc_v = roc_auc_score(y_te, p)
        ejes[0].plot(fpr, tpr, lw=2, color=col, label=f"{nombre} (AUC={auc_v:.3f})")

        prec, rec, _ = precision_recall_curve(y_te, p)
        ap_v = average_precision_score(y_te, p)
        ejes[1].plot(rec, prec, lw=2, color=col, label=f"{nombre} (AP={ap_v:.3f})")

    ejes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC=0,500)")
    ejes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
    ejes[0].set_ylabel("Tasa de Verdaderos Positivos (Recall / TPR)")
    ejes[0].set_title("A. Curva ROC · Capacidad de ordenamiento")
    ejes[0].legend(fontsize=7.6, loc="lower right")
    ejes[0].grid(alpha=0.25)

    prevalencia = float(y_te.mean())
    ejes[1].axhline(prevalencia, color="black", ls="--", lw=1, label=f"Base prevalencia ({prevalencia:.3f})")
    ejes[1].set_xlabel("Recall (cobertura de tasas atipicas)")
    ejes[1].set_ylabel("Precision (calidad de alarmas)")
    ejes[1].set_title("B. Curva Precision-Recall · Enfoque en clase minoritaria")
    ejes[1].legend(fontsize=7.6, loc="lower left")
    ejes[1].grid(alpha=0.25)

    fig.suptitle("Evaluacion probabilistica en F2 (n_test=1.949, tasa atipica 9,77 %)", fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "eval_curvas_roc_pr.png")
    return fig


def fig_barrido_umbral_costos(df_u: pd.DataFrame, perfiles: dict[str, dict[str, object]]) -> plt.Figure:
    """Panel A: Precision, Recall y F1 segun umbral. Panel B: TP detectados vs FP falsas alarmas."""
    fig, ejes = plt.subplots(1, 2, figsize=(12.4, 4.5))

    ejes[0].plot(df_u["umbral"], df_u["precision"], marker="o", lw=2, color="#2e86c1", label="Precision")
    ejes[0].plot(df_u["umbral"], df_u["recall"], marker="s", lw=2, color="#e67e22", label="Recall")
    ejes[0].plot(df_u["umbral"], df_u["f1"], marker="^", lw=2.3, color="#117a65", label="F1 (armonica)")
    for clave, color_v, etiq in (
        ("alerta_temprana", "#d35400", "Alerta temprana"),
        ("equilibrado_f1", "#1e8449", "Optimo F1"),
        ("alta_precision", "#1b4f72", "Alta precision"),
    ):
        u_v = float(perfiles[clave]["umbral"])
        ejes[0].axvline(u_v, color=color_v, ls="--", lw=1.3, label=f"{etiq} (u={u_v:.2f})")
    ejes[0].set_xlabel("Umbral de decision sobre predict_proba")
    ejes[0].set_ylabel("Valor de metrica (0 a 1)")
    ejes[0].set_title("A. Compromiso Precision / Recall / F1 por umbral (Arbol CART)")
    ejes[0].legend(fontsize=7.2, loc="best")
    ejes[0].grid(alpha=0.25)

    ejes[1].plot(df_u["umbral"], df_u["TP_detectados"], marker="s", lw=2.2, color="#117a65", label="TP: atipicas detectadas")
    ejes[1].plot(df_u["umbral"], df_u["FP_falsas_alarmas"], marker="o", lw=2.2, color="#c0392b", label="FP: falsas alarmas")
    ejes[1].plot(df_u["umbral"], df_u["FN_perdidos"], marker="^", ls="--", lw=1.6, color="#7f8c8d", label="FN: atipicas perdidas")
    ejes[1].set_xlabel("Umbral de decision")
    ejes[1].set_ylabel("Cantidad de productos en test (n=1.949)")
    ejes[1].set_title("B. Costo operativo real: detecciones vs. falsas alarmas")
    ejes[1].legend(fontsize=8)
    ejes[1].grid(alpha=0.25)

    fig.suptitle("Optimizacion del umbral de decision ante costos asimetricos (Arbol CART, F2)", fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "eval_barrido_umbral_costos.png")
    return fig


def fig_cv5_pliegues(cv_hip: pd.DataFrame) -> plt.Figure:
    """Desempeno pliegue a pliegue en CV-5 estratificada (Pipeline sin fuga) con media +/- desv."""
    modelos = ["Arbol CART (d=4)", "KNN escalado (k=5)", "KNN escalado (k=15)", "Random Forest (ensamble)", "Naive Bayes (GaussianNB)"]
    colores = ["#2e86c1", "#e67e22", "#8e44ad", "#117a65", "#c0392b"]

    fig, ejes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    for met, ax, titulo in (
        ("f1", ejes[0], "A. F1 minoritaria por pliegue (CV-5 en Pipeline)"),
        ("roc_auc", ejes[1], "B. ROC-AUC por pliegue (CV-5 en Pipeline)"),
    ):
        for mod, col in zip(modelos, colores):
            sub = cv_hip[cv_hip["modelo"] == mod].sort_values("pliegue")
            mu = sub[met].mean()
            sd = sub[met].std()
            ax.plot(sub["pliegue"], sub[met], marker="o", lw=2, color=col, label=f"{mod} ({mu:.3f} +/- {sd:.3f})")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Pliegue de validacion cruzada")
        ax.set_ylabel(met.upper())
        ax.set_title(titulo)
        ax.legend(fontsize=7.3, loc="lower right")
        ax.grid(alpha=0.25)

    fig.suptitle("Estabilidad entre pliegues CV-5 (Catalogo Hipotecario F2 · tasa atipica)", fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "eval_cv5_pliegues_estabilidad.png")
    return fig


def fig_ranking_global_prediccion(ev_bal: dict[str, object], ev_hip: dict[str, object]) -> plt.Figure:
    """Resumen visual definitivo: F1 promedio (Maestro + F2) vs. velocidad y peso en disco."""
    b = ev_bal["tabla"].set_index("modelo")
    h = ev_hip["tabla"].set_index("modelo")
    modelos = ["Arbol CART (d=4)", "Random Forest (ensamble)", "KNN escalado (k=5)", "KNN escalado (k=15)", "Naive Bayes (GaussianNB)"]
    colores = ["#27ae60", "#117a65", "#e67e22", "#8e44ad", "#c0392b"]

    f1_prom = [0.5 * (float(b.loc[m, "f1_pos"]) + float(h.loc[m, "f1_pos"])) for m in modelos]
    auc_prom = [0.5 * (float(b.loc[m, "roc_auc"]) + float(h.loc[m, "roc_auc"])) for m in modelos]
    pesos = [float(h.loc[m, "peso_kb"]) for m in modelos]
    lat = [float(h.loc[m, "predict_ms"]) for m in modelos]

    fig, ejes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    y_pos = np.arange(len(modelos))
    ejes[0].barh(y_pos - 0.17, f1_prom, 0.32, color="#2e86c1", label="F1 promedio (Maestro + F2)")
    ejes[0].barh(y_pos + 0.17, auc_prom, 0.32, color="#117a65", label="ROC-AUC promedio")
    for i, (f1v, aucv) in enumerate(zip(f1_prom, auc_prom)):
        ejes[0].text(f1v + 0.012, i - 0.17, f"{f1v:.3f}", va="center", fontsize=7.8)
        ejes[0].text(aucv + 0.012, i + 0.17, f"{aucv:.3f}", va="center", fontsize=7.8)
    ejes[0].set_yticks(y_pos)
    ejes[0].set_yticklabels(modelos, fontsize=8.2)
    ejes[0].set_xlim(0, 1.12)
    ejes[0].set_xlabel("Desempeno predictivo promedio (0 a 1)")
    ejes[0].set_title("A. Quien predice mejor (F1 y ROC-AUC)")
    ejes[0].legend(fontsize=7.8, loc="lower right")
    ejes[0].grid(axis="x", alpha=0.25)

    for m, col, kb, ms, f1v in zip(modelos, colores, pesos, lat, f1_prom):
        ejes[1].scatter(ms, f1v, s=max(kb * 1.4, 70), c=col, alpha=0.85, edgecolors="black", label=f"{m} ({kb:.1f} KB)")
    ejes[1].set_xlabel("Latencia predict() en F2 (ms, menor = mejor)")
    ejes[1].set_ylabel("F1 promedio (mayor = mejor)")
    ejes[1].set_title("B. Eficiencia operativa (tamano burbuja = peso KB)")
    ejes[1].legend(fontsize=7.3, loc="lower right")
    ejes[1].grid(alpha=0.25)

    fig.suptitle("Resumen ejecutivo: que algoritmo funciona mejor para nuestro proyecto", fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "eval_ranking_global_prediccion.png")
    return fig


def main() -> None:
    print("=" * 78)
    print("EVALUACION PROFUNDA DE MODELOS · DESBALANCE, CV-5, UMBRALES Y RANKING GLOBAL")
    print("=" * 78)

    escenarios = preparar_escenarios()
    ev_bal = evaluar_escenario_completo(escenarios["maestro_balanceado"], "maestro_balanceado")
    ev_crit = evaluar_escenario_completo(escenarios["maestro_critico"], "maestro_critico")
    ev_hip = evaluar_escenario_completo(escenarios["hipotecas_atipico"], "hipotecas_atipico")

    cols_p = ["modelo", "acc_test", "precision_pos", "recall_pos", "f1_pos", "roc_auc", "cv5_f1_media", "cv5_f1_desv", "TP", "FP", "FN"]
    print("\n[1] Escenario Desbalanceado A · Cuadro Maestro Esfuerzo Critico >= 0,80 (18,3 %):")
    print(ev_crit["tabla"][cols_p].to_string(index=False))

    print("\n[2] Escenario Desbalanceado B · Hipotecas F2 Tasa Atipica IQR (9,77 %, n=7.796):")
    print(ev_hip["tabla"][cols_p].to_string(index=False))

    print("\n[3] Classification Report forense en F2 (Arbol CART vs. KNN k=15):")
    print("--- Arbol CART (d=4) ---")
    print(ev_hip["reportes"]["Arbol CART (d=4)"])
    print("--- KNN escalado (k=15) ---")
    print(ev_hip["reportes"]["KNN escalado (k=15)"])

    # Barrido de umbrales sobre el modelo ganador (Arbol CART) en F2 desbalanceado
    df_umb, perfiles_u = barrido_umbrales_decision(
        ev_hip["y_test"].to_numpy(), ev_hip["probs"]["Arbol CART (d=4)"]
    )
    print("\n[4] Barrido del umbral de decision (Arbol CART sobre F2 atipico, n_test=1.949):")
    print(df_umb.to_string(index=False))
    print("\n    Perfiles de umbral recomendados por costo asimetrico:")
    for clave, info in perfiles_u.items():
        print(f"      - {clave:<16}: u={info['umbral']:.2f} | P={info['precision']:.3f} | "
              f"R={info['recall']:.3f} | F1={info['f1']:.3f} | TP={info['TP']} | FP={info['FP']}")

    # Simulador de prediccion por region real
    sim_reg, perfiles_reg = simular_predicciones_por_region(escenarios["maestro_balanceado"])
    print("\n[5] Simulador lado a lado sobre perfiles reales de las 10 regiones del UK:")
    print(sim_reg.to_string(index=False))

    # Semaforo global de modelos
    semaforo = tabla_semaforo_global(ev_bal, ev_crit, ev_hip)
    print("\n[6] Panel Comparativo Global (cuales funcionan, cuales fallan y cual predice mejor):")
    print(semaforo[["Algoritmo / Familia", "Estado", "Maestro 50/50 (F1 / CV-5)", "Hipotecas Atipico 9,8% (F1 / AUC)"]].to_string(index=False))

    # Exportar CSVs livianos + enriquecer contrato JSON
    config.DATA_INTEGRADO.mkdir(parents=True, exist_ok=True)
    df_umb.to_csv(config.RUTA_EVALUACION_UMBRALES, index=False)
    cv_all = pd.concat([ev_bal["cv_pliegues"], ev_crit["cv_pliegues"], ev_hip["cv_pliegues"]], ignore_index=True)
    cv_all.to_csv(config.RUTA_EVALUACION_CV5, index=False)
    semaforo.to_csv(config.RUTA_COMPARATIVA_GLOBAL, index=False)
    ruta_json = enriquecer_contrato_json(perfiles_u, perfiles_reg, semaforo)

    # Figuras (5 PNG)
    for fig in (
        fig_ingenuo_vs_modelos(ev_hip, ev_crit),
        fig_curvas_roc_pr(ev_hip),
        fig_barrido_umbral_costos(df_umb, perfiles_u),
        fig_cv5_pliegues(ev_hip["cv_pliegues"]),
        fig_ranking_global_prediccion(ev_bal, ev_hip),
    ):
        plt.close(fig)

    print(f"\n[7] Artefactos guardados: {config.RUTA_EVALUACION_UMBRALES.name}, {config.RUTA_EVALUACION_CV5.name}, "
          f"{config.RUTA_COMPARATIVA_GLOBAL.name}, contrato actualizado ({ruta_json.name}) y 5 figuras eval_*.png.")


if __name__ == "__main__":
    main()
