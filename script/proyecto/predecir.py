"""Motor directo de Prediccion Dual (Tipo 1 Regresion + Tipo 2 Clasificacion).

Carga los 4 modelos preentrenados en data/modelos/ (inferencia pura < 5 ms sin
reentrenar) y su contrato oficial (data/modelos/contrato_prediccion.json):
  1. modelo_ganador_clasificacion.joblib    -> Tipo 2 Regional: Alto vs Moderado + Reglas CART
  2. modelo_regresion_esfuerzo.joblib       -> Tipo 1 Regional: esfuerzo salarial exacto (%) +/- MAE
  3. modelo_clasificacion_tasa_f2.joblib    -> Tipo 2 Hipoteca F2: P(tasa atipica) + 3 umbrales
  4. modelo_regresion_tasa_f2.joblib        -> Tipo 1 Hipoteca F2: tasa inicial exacta (%) +/- MAE

Compatible 100 % con macOS, Linux y Windows (UTF-8 y rutas POSIX).

Ejemplos de uso:
  python script/proyecto/predecir.py
  python script/proyecto/predecir.py --region London --precio_var_pct 5.0 --comparar
  python script/proyecto/predecir.py --region "North East" --json
  python script/proyecto/predecir.py --hipoteca --comisiones 999 --meses 24 --tipo fixed --subtitulo "First time buyer"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto import config, sistema_recursos
from proyecto.arbol_clasificacion import preparar_xy

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ETIQUETAS_HUMANAS = {
    "n_transacciones": "Volumen anual de transacciones",
    "tmax_anom": "Anomalia temperatura maxima (C)",
    "tmin_anom": "Anomalia temperatura minima (C)",
    "lluvia_anom": "Anomalia lluvia anual (mm)",
    "sol_anom": "Anomalia horas de sol (h)",
    "af_anom": "Anomalia dias de helada",
    "precio_var_pct": "Variacion anual del precio (%)",
}


def cargar_artefactos_prediccion() -> dict[str, Any]:
    """Carga los 4 modelos .joblib preentrenados y el contrato JSON oficial."""
    requeridos = [
        config.RUTA_MODELO_GANADOR,
        config.RUTA_MODELO_REGRESION_ESFUERZO,
        config.RUTA_MODELO_CLASIFICACION_TASA_F2,
        config.RUTA_MODELO_REGRESION_TASA_F2,
        config.RUTA_CONTRATO_PREDICCION,
    ]
    faltantes = [sistema_recursos.ruta_posix(r, config.PROYECTO) for r in requeridos if not r.exists()]
    if faltantes:
        raise FileNotFoundError(
            "Faltan artefactos en data/modelos/: "
            + ", ".join(faltantes)
            + ". Ejecute: python script/proyecto/prediccion_dual.py"
        )
    return {
        "clf_regional": joblib.load(config.RUTA_MODELO_GANADOR),
        "reg_esfuerzo": joblib.load(config.RUTA_MODELO_REGRESION_ESFUERZO),
        "clf_f2": joblib.load(config.RUTA_MODELO_CLASIFICACION_TASA_F2),
        "reg_f2": joblib.load(config.RUTA_MODELO_REGRESION_TASA_F2),
        "contrato": json.loads(config.RUTA_CONTRATO_PREDICCION.read_text(encoding="utf-8")),
    }


def construir_vector_entrada(
    contrato: dict[str, Any],
    region: str,
    overrides: dict[str, float | None] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], list[str], str]:
    """Construye el vector X exacto usando el perfil real de la region y overrides opcionales."""
    regiones_validas: list[str] = contrato["regiones_validas"]
    mapa_ci = {r.lower(): r for r in regiones_validas}
    clave = region.strip().lower()
    if clave not in mapa_ci:
        raise ValueError(
            f"Region invalida: '{region}'. Regiones validas: {', '.join(regiones_validas)}"
        )
    region_canonica = mapa_ci[clave]

    perfiles = contrato.get("perfiles_regionales_referencia", {})
    base_num: dict[str, float] = {}
    for col, rango in contrato["entradas_numericas"].items():
        if region_canonica in perfiles and col in perfiles[region_canonica]:
            base_num[col] = float(perfiles[region_canonica][col])
        else:
            base_num[col] = float(rango["mediana"])

    advertencias: list[str] = []
    for col, val in (overrides or {}).items():
        if val is None or col not in base_num:
            continue
        fval = float(val)
        base_num[col] = fval
        rango = contrato["entradas_numericas"][col]
        if fval < float(rango["min"]) or fval > float(rango["max"]):
            advertencias.append(
                f"{col}={fval:.2f} queda fuera del rango historico [{rango['min']:.2f}, {rango['max']:.2f}]"
            )

    fila = {c: 0.0 for c in contrato["columnas_modelo_X"]}
    for col, val in base_num.items():
        fila[col] = val
    col_reg = f"region_{region_canonica}"
    if col_reg in fila:
        fila[col_reg] = 1.0

    return pd.DataFrame([fila], columns=contrato["columnas_modelo_X"]), base_num, advertencias, region_canonica


def explicar_ruta_arbol(modelo: DecisionTreeClassifier, X_fila: pd.DataFrame) -> list[str]:
    """Extrae la ruta exacta de reglas del Arbol CART que produjo la prediccion."""
    tree = modelo.tree_
    nombres = list(X_fila.columns)
    valores = X_fila.iloc[0].to_numpy(dtype=float)
    camino = modelo.decision_path(X_fila).indices
    pasos: list[str] = []

    for nodo in camino:
        if tree.children_left[nodo] == tree.children_right[nodo]:
            conteos = tree.value[nodo][0]
            pasos.append(
                f"Hoja #{int(nodo)}: distribucion historica [Moderado={conteos[0]:.2f}, Alto={conteos[1]:.2f}]"
            )
            continue
        feat_idx = int(tree.feature[nodo])
        feat_name = nombres[feat_idx]
        umbral = float(tree.threshold[nodo])
        val = float(valores[feat_idx])
        cumple = val <= umbral
        op = "<=" if cumple else ">"
        if feat_name.startswith("region_"):
            reg_nom = feat_name.replace("region_", "")
            texto = f"Region NO es '{reg_nom}'" if cumple else f"Region ES '{reg_nom}'"
        else:
            etiq = ETIQUETAS_HUMANAS.get(feat_name, feat_name)
            texto = f"{etiq} ({val:,.2f}) {op} {umbral:,.2f}"
        pasos.append(f"Regla #{len(pasos) + 1}: {texto}")
    return pasos


def predecir_escenario_regional(
    region: str,
    n_transacciones: float | None = None,
    tmax_anom: float | None = None,
    tmin_anom: float | None = None,
    lluvia_anom: float | None = None,
    sol_anom: float | None = None,
    af_anom: float | None = None,
    precio_var_pct: float | None = None,
) -> dict[str, Any]:
    """Ejecuta PREDICCION DUAL regional: Tipo 2 Clasificacion (CART) + Tipo 1 Regresion (Ridge)."""
    art = cargar_artefactos_prediccion()
    clf = art["clf_regional"]
    reg = art["reg_esfuerzo"]
    contrato = art["contrato"]

    overrides = {
        "n_transacciones": n_transacciones,
        "tmax_anom": tmax_anom,
        "tmin_anom": tmin_anom,
        "lluvia_anom": lluvia_anom,
        "sol_anom": sol_anom,
        "af_anom": af_anom,
        "precio_var_pct": precio_var_pct,
    }
    X_fila, entradas_usadas, advertencias, region_canon = construir_vector_entrada(contrato, region, overrides)

    t0 = time.perf_counter()
    pred_id = int(clf.predict(X_fila)[0])
    proba = clf.predict_proba(X_fila)[0]
    esfuerzo_est = float(reg.predict(X_fila)[0])
    lat_ms = (time.perf_counter() - t0) * 1000.0

    etiqueta = contrato["objetivo"]["clases"][str(pred_id)]
    reglas = explicar_ruta_arbol(clf, X_fila)
    mae_esf = float(
        contrato.get("tipos_de_prediccion_implementados", {})
        .get("tipo_1_regresion_continua", {})
        .get("esfuerzo_regional_sin_fuga", {})
        .get("mae_cv5", 0.0421)
    )

    return {
        "region": region_canon,
        "modelo": contrato["modelo_ganador"],
        "prediccion": etiqueta,
        "clase_id": pred_id,
        "probabilidad_alto": round(float(proba[1]), 4),
        "probabilidad_moderado": round(float(proba[0]), 4),
        "umbral_esfuerzo_mediana": float(contrato["objetivo"]["umbral_mediana"]),
        "tipo1_regresion_esfuerzo_ratio": round(esfuerzo_est, 4),
        "tipo1_regresion_esfuerzo_pct": round(esfuerzo_est * 100.0, 1),
        "tipo1_mae_pp": round(mae_esf * 100.0, 1),
        "latencia_ms": round(lat_ms, 3),
        "entradas_numericas_usadas": {k: round(v, 4) for k, v in entradas_usadas.items()},
        "advertencias_rango": advertencias,
        "explicacion_reglas": reglas,
    }


def comparar_prediccion_modelos(
    region: str,
    overrides: dict[str, float | None] | None = None,
) -> pd.DataFrame:
    """Compara la prediccion de un mismo escenario entre CART, KNN (k=5), Random Forest y Naive Bayes."""
    art = cargar_artefactos_prediccion()
    contrato = art["contrato"]
    X_fila, _, _, _ = construir_vector_entrada(contrato, region, overrides)
    X_m, y_m_str, _, _ = preparar_xy()
    y_m = (y_m_str == "Alto").astype(int)

    competidores = {
        "Arbol CART (d=4) [GANADOR]": art["clf_regional"],
        "KNN escalado (k=5) [Retador]": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_m, y_m),
        "Random Forest (200) [Ensamble]": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=config.SEED
        ).fit(X_m, y_m),
        "Naive Bayes (GaussianNB) [Baseline]": GaussianNB().fit(X_m, y_m),
    }

    filas = []
    for nombre, clf in competidores.items():
        t0 = time.perf_counter()
        pred = int(clf.predict(X_fila)[0])
        prob = float(clf.predict_proba(X_fila)[0, 1])
        lat = (time.perf_counter() - t0) * 1000.0
        filas.append(
            {
                "modelo": nombre,
                "prediccion": contrato["objetivo"]["clases"][str(pred)],
                "prob_alto": round(prob, 4),
                "latencia_ms": round(lat, 3),
            }
        )
    return pd.DataFrame(filas)


def evaluar_hipoteca_con_umbrales(
    comisiones_total: float,
    meses_inicial: int,
    tipo: str = "fixed",
    subtitulo: str = "First time buyer",
) -> dict[str, Any]:
    """Ejecuta PREDICCION DUAL hipotecaria F2 sin reentrenar (< 5 ms):
    Tipo 1 Regresion (tasa exacta % +/- MAE) + Tipo 2 Clasificacion (P atipica + 3 umbrales)."""
    art = cargar_artefactos_prediccion()
    contrato = art["contrato"]
    reg_f2 = art["reg_f2"]
    clf_f2 = art["clf_f2"]

    meta_f2 = (
        contrato["tipos_de_prediccion_implementados"]["tipo_1_regresion_continua"]["tasa_hipotecaria_f2"]
    )
    cols_f2 = meta_f2["columnas_X"]
    fila = {c: 0.0 for c in cols_f2}
    fila["comisiones_total"] = float(comisiones_total)
    fila["meses_inicial"] = float(meses_inicial)
    if f"tipo_{tipo}" in fila:
        fila[f"tipo_{tipo}"] = 1.0
    if f"subtitulo_{subtitulo}" in fila:
        fila[f"subtitulo_{subtitulo}"] = 1.0

    X_q = pd.DataFrame([fila], columns=cols_f2)
    t0 = time.perf_counter()
    tasa_est = float(reg_f2.predict(X_q)[0])
    prob_atip = float(clf_f2.predict_proba(X_q)[0, 1])
    lat_ms = (time.perf_counter() - t0) * 1000.0

    umbrales = contrato["umbrales_operativos_atipico_f2"]
    decisiones = {}
    for nombre_u, info in umbrales.items():
        u = float(info["umbral"])
        decisiones[nombre_u] = {
            "umbral": u,
            "alerta_atipica": prob_atip >= u,
            "etiqueta": "ALERTA TASA ATIPICA" if prob_atip >= u else "Normal",
            "uso": info["uso"],
        }
    return {
        "entrada": {
            "comisiones_total_gbp": comisiones_total,
            "meses_inicial": meses_inicial,
            "tipo": tipo,
            "subtitulo": subtitulo,
        },
        "tipo1_tasa_estimada_pct": round(tasa_est, 2),
        "tipo1_mae_pp": round(float(meta_f2["mae_test_pp"]), 2),
        "tipo2_probabilidad_tasa_atipica": round(prob_atip, 4),
        "latencia_ms": round(lat_ms, 3),
        "decisiones_por_umbral": decisiones,
    }


def tabla_guia_metodos_y_aciertos() -> pd.DataFrame:
    """Devuelve la guia comparativa completa de metodos CON Etiqueta y SIN Etiqueta,
    cantidad exacta de aciertos, metricas y cuando utilizar cada uno."""
    filas = [
        {
            "paradigma": "CON Etiqueta (Supervisado)",
            "tipo": "Tipo 2A · Clasificacion Binaria Regional",
            "metodo": "Arbol CART (max_depth=4) [GANADOR 4,82/5,00]",
            "aciertos_exactos": "17/18 Test (94,44%) | 47/60 CV-5 (78,33%; 48/60=80,0% con K-Means) | 9/10 regiones (90,0%)",
            "metrica_clave": "F1=0,9412 | AUC=0,783 | Brier=0,208",
            "cuando_usar": "Clasificar si una region tendra Esfuerzo Alto (>=56,4%) o Moderado con reglas SI/ENTONCES auditables.",
        },
        {
            "paradigma": "CON Etiqueta (Supervisado)",
            "tipo": "Tipo 2A · Clasificacion Binaria Regional",
            "metodo": "KNN Escalado (k=5, StandardScaler) [RETADOR]",
            "aciertos_exactos": "17/18 Test (94,44%) | 54/60 CV-5 (90,00%) | 9/10 regiones (90,0%)",
            "metrica_clave": "F1=0,9412 | AUC=0,945 | Brier=0,064",
            "cuando_usar": "Segunda opinion por vecindad territorial (exige StandardScaler; sin escalar cae a 12/18 = 66,67%).",
        },
        {
            "paradigma": "CON Etiqueta (Supervisado)",
            "tipo": "Tipo 2A · Clasificacion Binaria Regional",
            "metodo": "Random Forest (200 arboles, d=6) [ENSAMBLE]",
            "aciertos_exactos": "17/18 Test (94,44%) | 46/60 CV-5 (76,67%)",
            "metrica_clave": "F1=0,9412 | 418,86 KB disco",
            "cuando_usar": "Referencia de ensamble (empata en Test con CART pero pierde trazabilidad por reglas y pesa 310x mas).",
        },
        {
            "paradigma": "CON Etiqueta (Supervisado)",
            "tipo": "Tipo 2A · Clasificacion Binaria Regional",
            "metodo": "Naive Bayes (GaussianNB) [BASELINE]",
            "aciertos_exactos": "8/18 Test (44,44%) | 37/60 CV-5 (61,67%)",
            "metrica_clave": "F1=0,2857 | Brier=0,536",
            "cuando_usar": "Solo linea base teorica (colapsa por colinealidad entre las 5 anomalias climaticas y dummies).",
        },
        {
            "paradigma": "CON Etiqueta (Supervisado)",
            "tipo": "Tipo 1A · Regresion Continua Regional (sin fuga)",
            "metodo": "Regresion Ridge (alpha=1,0 + StandardScaler) [GANADOR]",
            "aciertos_exactos": "R2 CV-5 = 0,9094 +/- 0,0375 | Error medio MAE = +/- 4,21 pp",
            "metrica_clave": "RMSE = 5,10 pp (vs RF 0,761, KNN 0,759, CART 0,706)",
            "cuando_usar": "Estimar el porcentaje exacto de sueldo joven que absorbe la hipoteca en una region (sin fuga de precio/salario).",
        },
        {
            "paradigma": "CON Etiqueta (Supervisado)",
            "tipo": "Tipo 1B · Regresion Continua Hipotecaria F2",
            "metodo": "RandomForestRegressor (200 arboles, d=10) [GANADOR]",
            "aciertos_exactos": "R2 Test = 0,9642 (n_test=1.949) | Error medio MAE = +/- 0,1692 pp",
            "metrica_clave": "RMSE = 0,2562 pp (vs Lineal R2=0,6316)",
            "cuando_usar": "Cotizar la tasa inicial exacta (%) de un producto hipotecario segun comision, plazo, tipo y perfil.",
        },
        {
            "paradigma": "CON Etiqueta (Supervisado)",
            "tipo": "Tipo 2B · Clasificacion Tasa Elevada F2 (>=6,62%)",
            "metodo": "Arbol CART (d=5) / KNN (k=5)",
            "aciertos_exactos": "CART: 1.805/1.949 (92,61%) | KNN: 1.842/1.949 (94,51%) | NB: 945/1.949 (48,49%)",
            "metrica_clave": "CART AUC=0,943 | KNN AUC=0,975",
            "cuando_usar": "Anticipar si una hipoteca caera en el cuartil mas caro del mercado (Q3 >= 6,62%).",
        },
        {
            "paradigma": "CON Etiqueta (Supervisado)",
            "tipo": "Tipo 2C · Clasificacion Tasa Atipica F2 (>8,13%) + 3 Umbrales",
            "metodo": "Arbol CART (d=4) con Umbral Ajustado (0,05 / 0,30 / 0,40)",
            "aciertos_exactos": "u=0,40: 59/190 (Prec=100%, 0 FP) | u=0,30: 188/190 (98,95% recall) | u=0,05: 190/190 (100% recall)",
            "metrica_clave": "ROC-AUC=0,9429 | Youden J=0,859 (vs Ingenuo 0/190 = 0% recall)",
            "cuando_usar": "Detectar hipotecas usurarias (>8,13%): usar u=0,30 para balance F1, u=0,05 para consumidor y u=0,40 para bloqueo sin FP.",
        },
        {
            "paradigma": "SIN Etiqueta (No Supervisado)",
            "tipo": "Tipo 3 · Clustering Territorial, Hipotecario y Masivo 11,35M",
            "metodo": "K-Means (k=3) / MiniBatchKMeans Streaming (float32, 24 GB)",
            "aciertos_exactos": "Separa esfuerzo post-hoc 0,50 -> 0,60 -> 0,90 (100% Alto en London/SE) | 11,35M filas en 49,87 MB RAM",
            "metrica_clave": "Silhouette=0,321 (regiones), 0,479 (F2), 0,333 (11,35M F1)",
            "cuando_usar": "Descubrir segmentos naturales de regiones, hipotecas o millones de viviendas sin usar la etiqueta y.",
        },
        {
            "paradigma": "SIN Etiqueta (No Supervisado)",
            "tipo": "Tipo 4 · Reglas de Asociacion (Market Basket)",
            "metodo": "Apriori Puro Python (tidsets, 23,58 MB RAM)",
            "aciertos_exactos": "10/10 tickets manuales verificados (assert) | 195 reglas F1 + 481 reglas F2",
            "metrica_clave": "F1: Detached+Freehold -> Extremo (conf=68%, lift=2,00) | F2: discounted -> Baja_Q1 (conf=99,1%, lift=3,96)",
            "cuando_usar": "Descubrir co-ocurrencias frecuentes SI antecedente -> ENTONCES consecuente con soporte, confianza y lift.",
        },
    ]
    return pd.DataFrame(filas)


def ejecutar_demo_completa() -> None:
    """Demostracion ejecutable de PREDICCION DUAL (Regresion + Clasificacion) y comparativa."""
    sistema_recursos.configurar_entorno_portable()
    print("=" * 82)
    print("MOTOR DE PREDICCION DUAL (REGRESION CONTINUA + CLASIFICACION) · REINO UNIDO")
    print("=" * 82)

    print("\n[1] Prediccion Dual en vivo sobre 5 regiones reales (Clasificacion CART + Regresion Ridge):")
    for reg in ("London", "South East", "West Midlands", "Wales", "North East"):
        res = predecir_escenario_regional(reg)
        print(
            f"  - {res['region']:<15} | Tipo 2 (CART): {res['prediccion']:<8} (P={res['probabilidad_alto']:.3f}) | "
            f"Tipo 1 (Ridge): {res['tipo1_regresion_esfuerzo_pct']:5.1f}% (+/- {res['tipo1_mae_pp']:.1f} pp) | "
            f"{res['latencia_ms']:.2f} ms"
        )
        print(f"    Reglas CART: {'  =>  '.join(res['explicacion_reglas'])}")

    print("\n[2] Comparativa multi-modelo para un escenario personalizado (London, variacion +5,0 %):")
    comp = comparar_prediccion_modelos("London", {"precio_var_pct": 5.0})
    print(comp.to_string(index=False))

    print("\n[3] Prediccion Dual sobre 2 ofertas hipotecarias F2 (Tasa Exacta RF + 3 Umbrales CART):")
    ejemplos_hip = [
        ("Oferta estandar First Time Buyer (comision £999, 24m, fixed)", 999.0, 24, "fixed", "First time buyer"),
        ("Oferta Remortgage sin comision (comision £0, 60m, fixed)", 0.0, 60, "fixed", "Remortgage"),
    ]
    for desc, com, mes, tip, sub in ejemplos_hip:
        ev = evaluar_hipoteca_con_umbrales(com, mes, tip, sub)
        print(
            f"  * {desc}\n"
            f"    -> Tipo 1 (RF): tasa estimada = {ev['tipo1_tasa_estimada_pct']:.2f}% (+/- {ev['tipo1_mae_pp']:.2f} pp) | "
            f"Tipo 2 (CART): P(atipica) = {ev['tipo2_probabilidad_tasa_atipica']:.3f} | {ev['latencia_ms']:.2f} ms"
        )
        for k, d in ev["decisiones_por_umbral"].items():
            print(f"      [{k:<15} u={d['umbral']:.2f}] -> {d['etiqueta']}")


def main() -> None:
    sistema_recursos.configurar_entorno_portable()
    parser = argparse.ArgumentParser(
        description="Predictor Dual (Regresion + Clasificacion) de vivienda joven en el Reino Unido"
    )
    parser.add_argument("--region", type=str, default=None, help="Region UK (ej. London, Wales, North East)")
    parser.add_argument("--n_transacciones", type=float, default=None)
    parser.add_argument("--tmax_anom", type=float, default=None)
    parser.add_argument("--tmin_anom", type=float, default=None)
    parser.add_argument("--lluvia_anom", type=float, default=None)
    parser.add_argument("--sol_anom", type=float, default=None)
    parser.add_argument("--af_anom", type=float, default=None)
    parser.add_argument("--precio_var_pct", type=float, default=None)
    parser.add_argument("--comparar", action="store_true", help="Mostrar comparativa contra KNN, RF y NB")
    parser.add_argument("--hipoteca", action="store_true", help="Evaluar una oferta hipotecaria F2")
    parser.add_argument("--comisiones", type=float, default=999.0)
    parser.add_argument("--meses", type=int, default=24)
    parser.add_argument("--tipo", type=str, default="fixed")
    parser.add_argument("--subtitulo", type=str, default="First time buyer")
    parser.add_argument("--json", action="store_true", help="Imprimir salida en JSON puro")
    args = parser.parse_args()

    if args.hipoteca:
        ev = evaluar_hipoteca_con_umbrales(args.comisiones, args.meses, args.tipo, args.subtitulo)
        if args.json:
            print(json.dumps(ev, indent=2, ensure_ascii=False))
            return
        print(f"Oferta       : {args.subtitulo} / {args.tipo} | comision £{args.comisiones:.0f} | {args.meses} meses")
        print(f"Tipo 1 (RF)  : tasa inicial estimada = {ev['tipo1_tasa_estimada_pct']:.2f}% (+/- {ev['tipo1_mae_pp']:.2f} pp)")
        print(f"Tipo 2 (CART): P(tasa atipica > 8,13%) = {ev['tipo2_probabilidad_tasa_atipica']:.3f} | {ev['latencia_ms']:.2f} ms")
        for k, d in ev["decisiones_por_umbral"].items():
            print(f"  - [{k:<15} u={d['umbral']:.2f}] -> {d['etiqueta']}")
        return

    if args.region is None:
        ejecutar_demo_completa()
        return

    overrides = {
        "n_transacciones": args.n_transacciones,
        "tmax_anom": args.tmax_anom,
        "tmin_anom": args.tmin_anom,
        "lluvia_anom": args.lluvia_anom,
        "sol_anom": args.sol_anom,
        "af_anom": args.af_anom,
        "precio_var_pct": args.precio_var_pct,
    }
    res = predecir_escenario_regional(args.region, **overrides)
    if args.comparar:
        comp_df = comparar_prediccion_modelos(args.region, overrides)
        res["comparativa_modelos"] = comp_df.to_dict(orient="records")

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    print(f"Region          : {res['region']}")
    print(f"Tipo 2 (CART)   : {res['prediccion']} (P(Alto) = {res['probabilidad_alto']:.3f} | umbral mediana = {res['umbral_esfuerzo_mediana']:.4f})")
    print(f"Tipo 1 (Ridge)  : esfuerzo estimado = {res['tipo1_regresion_esfuerzo_pct']:.1f}% del salario (+/- {res['tipo1_mae_pp']:.1f} pp)")
    print(f"Latencia dual   : {res['latencia_ms']:.2f} ms")
    if res["advertencias_rango"]:
        print("Advertencias    :", "; ".join(res["advertencias_rango"]))
    print("Explicacion CART:")
    for paso in res["explicacion_reglas"]:
        print(f"  - {paso}")
    if args.comparar:
        print("\nComparativa entre algoritmos para este mismo caso:")
        print(pd.DataFrame(res["comparativa_modelos"]).to_string(index=False))


if __name__ == "__main__":
    main()
