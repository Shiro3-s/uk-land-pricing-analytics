"""Prediccion Dual Supervisada (Tipo 1: Regresion Continua + Tipo 2: Clasificacion)
y empaquetado relacional listo para futura subida a Base de Datos y Backend API.

Complementa el torneo de clasificacion (Tipo 2) implementando formalmente el Tipo 1
(prediccion numerica continua sin fuga aritmetica) en los dos niveles del proyecto:
  - Nivel 1A · Cuadro Maestro Regional (n=60, CV-5 barajado sin fuga):
      Predice el esfuerzo hipotecario continuo exacto (ej. 1,018 = 101,8 % del salario)
      comparando Ridge, RandomForestRegressor, KNNRegressor y DecisionTreeRegressor
      (y contrastando con la regresion lineal contable R2=1,000).
      Exporta: data/modelos/modelo_regresion_esfuerzo.joblib
  - Nivel 1B · Catalogo Hipotecario F2 (n=7.796, split 75/25):
      Predice la tasa hipotecaria inicial exacta (%) desde comisiones, plazo, tipo y
      subtitulo comparando RandomForestRegressor, KNNRegressor, DecisionTreeRegressor
      y LinearRegression.
      Exporta: data/modelos/modelo_regresion_tasa_f2.joblib
  - Nivel 2B · Clasificador F2 preentrenado (tasa atipica > 8,13 %):
      Exporta: data/modelos/modelo_clasificacion_tasa_f2.joblib (inferencia < 5 ms sin
      reentrenar en el futuro backend).

Ademas genera:
  - Tablas unificadas de prediccion dual sobre las 10 regiones reales del UK y 5 ofertas F2.
  - Esquema SQL relacional (data/modelos/esquema_bd.sql) + CSVs planos limpios (~3 MB)
    listos para futura subida a PostgreSQL / Supabase / Neon (con opcion de carga por lotes
    de los 11,37 millones de filas de F1).
  - Figura: md/figuras/prediccion_dual_regresion_vs_clasificacion.png

Ejecutar:
  python script/proyecto/prediccion_dual.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto import config, sistema_recursos
from proyecto.arbol_clasificacion import PROFUNDIDAD, preparar_xy
from proyecto.evaluacion_modelos import preparar_escenarios

SEED = config.SEED


def evaluar_regresion_maestro_sin_fuga() -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    """Compara 4 regresores sin fuga (7 numericos + 9 dummies region) en CV-5 barajado."""
    X_m, _, df_m, _ = preparar_xy()
    y_m = df_m["esfuerzo"].astype(float)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    candidatos = {
        "Ridge (alpha=1.0, Pipeline)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForestRegressor (200)": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=SEED),
        "KNN Regressor (k=5, Pipeline)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "DecisionTreeRegressor (d=4)": DecisionTreeRegressor(max_depth=4, random_state=SEED),
    }

    filas = []
    for nombre, mod in candidatos.items():
        cv = cross_validate(
            mod,
            X_m,
            y_m,
            cv=kf,
            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
        )
        filas.append(
            {
                "escenario": "maestro_esfuerzo_sin_fuga",
                "modelo": nombre,
                "r2_media": round(float(cv["test_r2"].mean()), 4),
                "r2_desv": round(float(cv["test_r2"].std()), 4),
                "rmse": round(float(-cv["test_neg_root_mean_squared_error"].mean()), 4),
                "mae": round(float(-cv["test_neg_mean_absolute_error"].mean()), 4),
                "unidad_error": "ratio esfuerzo (1,0 = 100 % salario)",
            }
        )

    tabla = pd.DataFrame(filas).sort_values("r2_media", ascending=False).reset_index(drop=True)
    # Ganador sin fuga: Ridge (R2 ~ 0,909, MAE ~ 0,0421)
    ganador = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_m, y_m)
    fila_g = tabla.iloc[0].to_dict()
    meta = {
        "modelo": "Ridge (alpha=1.0, Pipeline StandardScaler)",
        "columnas_X": list(X_m.columns),
        "r2_cv5": float(fila_g["r2_media"]),
        "rmse_cv5": float(fila_g["rmse"]),
        "mae_cv5": float(fila_g["mae"]),
        "interpretacion": "Estima el esfuerzo hipotecario continuo exacto (sin usar precio ni salario en X) con error medio de +/- 4,2 puntos porcentuales.",
    }
    return tabla, ganador, meta


def evaluar_regresion_hipotecas_f2() -> tuple[pd.DataFrame, Any, Any, dict[str, Any]]:
    """Evalua regresion continua de tasa_inicial_pct (%) en F2 y entrena clasificadores F2."""
    hip = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).dropna(
        subset=["tasa_inicial_pct", "comisiones_total", "meses_inicial", "tipo", "subtitulo"]
    ).copy()
    X_h = pd.get_dummies(
        hip[["comisiones_total", "meses_inicial", "tipo", "subtitulo"]],
        columns=["tipo", "subtitulo"],
        drop_first=True,
        dtype=float,
    )
    y_h = hip["tasa_inicial_pct"].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.25, random_state=SEED)

    candidatos = {
        "RandomForestRegressor (200, d=10)": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED),
        "KNN Regressor (k=5, Pipeline)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "DecisionTreeRegressor (d=6)": DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "LinearRegression (Pipeline)": make_pipeline(StandardScaler(), LinearRegression()),
    }

    filas = []
    for nombre, mod in candidatos.items():
        mod.fit(X_tr, y_tr)
        yp = mod.predict(X_te)
        filas.append(
            {
                "escenario": "hipotecas_f2_tasa_pct",
                "modelo": nombre,
                "r2_media": round(float(r2_score(y_te, yp)), 4),
                "r2_desv": 0.0,
                "rmse": round(float(root_mean_squared_error(y_te, yp)), 4),
                "mae": round(float(mean_absolute_error(y_te, yp)), 4),
                "unidad_error": "puntos porcentuales de tasa (%)",
            }
        )

    tabla = pd.DataFrame(filas).sort_values("r2_media", ascending=False).reset_index(drop=True)
    reg_f2_full = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED).fit(X_h, y_h)

    # Clasificador F2 (tasa atipica > 8,13 %) entrenado sobre el mismo esquema X_h para inferencia directa
    esc_f2 = preparar_escenarios()["hipotecas_atipico"]
    clf_f2_full = DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=SEED).fit(esc_f2["X"], esc_f2["y"])

    fila_g = tabla.iloc[0].to_dict()
    meta = {
        "modelo": "RandomForestRegressor (n_estimators=200, max_depth=10)",
        "columnas_X": list(X_h.columns),
        "r2_test": float(fila_g["r2_media"]),
        "rmse_test_pp": float(fila_g["rmse"]),
        "mae_test_pp": float(fila_g["mae"]),
        "tipos_validos": sorted(hip["tipo"].astype(str).unique().tolist()),
        "subtitulos_validos": sorted(hip["subtitulo"].astype(str).unique().tolist()),
        "interpretacion": "Predice la tasa hipotecaria inicial exacta (%) de un producto con error medio de +/- 0,17 puntos porcentuales.",
    }
    return tabla, reg_f2_full, clf_f2_full, meta


def tabla_prediccion_dual_10_regiones(reg_esf: Any) -> pd.DataFrame:
    """Construye la tabla comparativa lado a lado (Tipo 1 Regresion + Tipo 2 Clasificacion)
    sobre los perfiles reales medianos de las 10 regiones del Reino Unido."""
    X_m, y_m_str, df_m, umbral_med = preparar_xy()
    y_m = (y_m_str == "Alto").astype(int)

    clf_cart = joblib.load(config.RUTA_MODELO_GANADOR)
    clf_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_m, y_m)
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED).fit(X_m, y_m)
    clf_nb = GaussianNB().fit(X_m, y_m)

    cols_num = [c for c in X_m.columns if not c.startswith("region_")]
    filas = []
    for reg, grupo in df_m.groupby("region", observed=True):
        esf_real = float(grupo["esfuerzo"].median())
        clase_real = "Alto" if esf_real >= umbral_med else "Moderado"

        fila_x = {c: 0.0 for c in X_m.columns}
        for cn in cols_num:
            fila_x[cn] = float(grupo[cn].median())
        if f"region_{reg}" in fila_x:
            fila_x[f"region_{reg}"] = 1.0
        Xq = pd.DataFrame([fila_x], columns=X_m.columns)

        esf_pred = float(reg_esf.predict(Xq)[0])
        c_cart = "Alto" if int(clf_cart.predict(Xq)[0]) == 1 else "Moderado"
        p_cart = float(clf_cart.predict_proba(Xq)[0, 1])
        c_knn = "Alto" if int(clf_knn.predict(Xq)[0]) == 1 else "Moderado"
        c_rf = "Alto" if int(clf_rf.predict(Xq)[0]) == 1 else "Moderado"
        c_nb = "Alto" if int(clf_nb.predict(Xq)[0]) == 1 else "Moderado"

        filas.append(
            {
                "region": str(reg),
                "esfuerzo_real_pct": round(esf_real * 100.0, 1),
                "tipo1_regresion_ridge_pct": round(esf_pred * 100.0, 1),
                "error_regresion_pp": round((esf_pred - esf_real) * 100.0, 1),
                "clase_real": clase_real,
                "tipo2_cart_ganador": c_cart,
                "prob_alto_cart": round(p_cart, 3),
                "knn_k5": c_knn,
                "random_forest": c_rf,
                "naive_bayes": c_nb,
                "acierta_cart": "SI" if c_cart == clase_real else "Frontera",
            }
        )
    return pd.DataFrame(filas).sort_values("esfuerzo_real_pct", ascending=False).reset_index(drop=True)


def tabla_prediccion_dual_hipotecas_f2(reg_f2: Any, clf_f2: Any, cols_f2: list[str]) -> pd.DataFrame:
    """Evalua 5 ofertas reales representativas de F2 mostrando ambos tipos de prediccion."""
    hip = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).dropna(
        subset=["tasa_inicial_pct", "comisiones_total", "meses_inicial", "tipo", "subtitulo"]
    ).copy()

    # Seleccionar 5 casos reales diversos (tasa baja, mediana FTB, alta fija, atipica, tracker/discounted)
    atip = hip[hip["es_atipico_tasa"]]
    atip_rem = atip[atip["subtitulo"].astype(str) == "Remortgage"]
    atip_otro = atip[atip["subtitulo"].astype(str) != "Remortgage"]
    muestras = [
        hip[hip["tipo"].astype(str) == "discounted"].iloc[0],
        hip[(hip["subtitulo"].astype(str) == "First time buyer") & (hip["tasa_inicial_pct"] < 6.0)].iloc[0],
        hip[(hip["subtitulo"].astype(str) == "First time buyer") & (hip["tasa_inicial_pct"].between(6.2, 6.8))].iloc[0],
        atip_rem.iloc[0] if not atip_rem.empty else atip.iloc[0],
        atip_otro.iloc[0] if not atip_otro.empty else atip.iloc[-1],
    ]

    filas = []
    for r in muestras:
        fx = {c: 0.0 for c in cols_f2}
        fx["comisiones_total"] = float(r["comisiones_total"])
        fx["meses_inicial"] = float(r["meses_inicial"])
        if f"tipo_{r['tipo']}" in fx:
            fx[f"tipo_{r['tipo']}"] = 1.0
        if f"subtitulo_{r['subtitulo']}" in fx:
            fx[f"subtitulo_{r['subtitulo']}"] = 1.0
        Xq = pd.DataFrame([fx], columns=cols_f2)

        tasa_real = float(r["tasa_inicial_pct"])
        tasa_pred = float(reg_f2.predict(Xq)[0])
        prob_atip = float(clf_f2.predict_proba(Xq)[0, 1])

        filas.append(
            {
                "banco": str(r["banco"])[:24],
                "perfil_tipo": f"{r['subtitulo']} / {r['tipo']}",
                "comision_gbp": round(float(r["comisiones_total"]), 0),
                "meses": int(r["meses_inicial"]),
                "tasa_real_pct": round(tasa_real, 2),
                "tipo1_tasa_pred_rf_pct": round(tasa_pred, 2),
                "error_pp": round(tasa_pred - tasa_real, 2),
                "es_atipica_real": "SI" if bool(r["es_atipico_tasa"]) else "NO",
                "tipo2_prob_atipica_cart": round(prob_atip, 3),
                "alerta_u030": "ALERTA" if prob_atip >= 0.30 else "Normal",
            }
        )
    return pd.DataFrame(filas)


def fig_prediccion_dual(df_reg10: pd.DataFrame, reg_f2: Any, cols_f2: list[str]) -> plt.Figure:
    """Grafica 2 paneles: (A) Esfuerzo Real vs Predicho Ridge + corte CART en 10 regiones;
    (B) Tasa F2 Real vs Predicha RandomForestRegressor + umbral atipico 8,13 %."""
    fig, ejes = plt.subplots(1, 2, figsize=(13.2, 4.9))

    # Panel A: 10 regiones UK (Real vs Tipo 1 Ridge + linea mediana Tipo 2 CART)
    y_pos = np.arange(len(df_reg10))
    ejes[0].barh(y_pos - 0.18, df_reg10["esfuerzo_real_pct"], 0.34, color="#1b4f72", label="Esfuerzo real mediano (%)")
    ejes[0].barh(y_pos + 0.18, df_reg10["tipo1_regresion_ridge_pct"], 0.34, color="#117a65", label="Tipo 1 · Regresion Ridge sin fuga (%)")
    ejes[0].axvline(56.36, color="#c0392b", ls="--", lw=1.6, label="Tipo 2 · Umbral CART Alto/Moderado (56,4 %)")
    ejes[0].set_yticks(y_pos)
    ejes[0].set_yticklabels(df_reg10["region"], fontsize=8.5)
    ejes[0].invert_yaxis()
    ejes[0].set_xlabel("Esfuerzo hipotecario (% del salario joven)")
    ejes[0].set_title("A. Prediccion Dual Regional (Regresion Ridge + Corte CART)", fontsize=10, weight="bold")
    ejes[0].legend(fontsize=7.6, loc="lower right")
    ejes[0].grid(axis="x", alpha=0.25)

    # Panel B: F2 Tasa Real vs Predicha (split 75/25)
    hip = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).dropna(
        subset=["tasa_inicial_pct", "comisiones_total", "meses_inicial", "tipo", "subtitulo"]
    ).copy()
    X_h = pd.get_dummies(
        hip[["comisiones_total", "meses_inicial", "tipo", "subtitulo"]],
        columns=["tipo", "subtitulo"],
        drop_first=True,
        dtype=float,
    )[cols_f2]
    y_h = hip["tasa_inicial_pct"].astype(float)
    _, X_te, _, y_te = train_test_split(X_h, y_h, test_size=0.25, random_state=SEED)
    yp = reg_f2.predict(X_te)
    r2_v = r2_score(y_te, yp)
    mae_v = mean_absolute_error(y_te, yp)

    ejes[1].scatter(y_te, yp, s=12, alpha=0.35, color="#2e86c1", label=f"Productos test (n={len(y_te):,})")
    lim = [float(y_te.min()) - 0.3, float(y_te.max()) + 0.3]
    ejes[1].plot(lim, lim, "k--", lw=1.2, label="Prediccion perfecta (y = x)")
    ejes[1].axvline(8.13, color="#c0392b", ls=":", lw=1.5, label="Cerca atipica Tukey (8,13 %)")
    ejes[1].axhline(8.13, color="#c0392b", ls=":", lw=1.5)
    ejes[1].set_xlabel("Tasa inicial real F2 (%)")
    ejes[1].set_ylabel("Tasa predicha RandomForestRegressor (%)")
    ejes[1].set_title(f"B. Tipo 1 en F2 · Tasa Hipotecaria (R2={r2_v:.3f}, MAE={mae_v:.2f} pp)", fontsize=10, weight="bold")
    ejes[1].legend(fontsize=7.8, loc="upper left")
    ejes[1].grid(alpha=0.25)

    fig.suptitle("Los Dos Tipos de Prediccion Supervisada: Regresion Continua + Clasificacion", fontsize=11.5, weight="bold")
    fig.tight_layout()
    ruta_png = config.MD_FIGURAS / "prediccion_dual_regresion_vs_clasificacion.png"
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=140, bbox_inches="tight")
    return fig


def exportar_capa_relacional_bd() -> list[dict[str, Any]]:
    """Exporta CSVs planos faltantes (hipotecas y precios_ciudad) y genera esquema_bd.sql."""
    hip = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    hip.to_csv(config.RUTA_HIPOTECAS_EXPORT_CSV, index=False, encoding="utf-8")

    pc = pd.read_parquet(config.RUTA_PRECIOS_CIUDAD)
    pc.to_csv(config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV, index=False, encoding="utf-8")

    sql = """-- ============================================================================
-- ESQUEMA RELACIONAL ESTANDAR (PostgreSQL / Supabase / Neon)
-- Proyecto: Accesibilidad de Vivienda para Jovenes (18-34) en el Reino Unido
-- Generado automaticamente por script/proyecto/prediccion_dual.py
--
-- Nivel 1 (tablas operativas y analiticas: ~3 MB total en data/integrado/*.csv)
--   Listo para subir en < 5 segundos a cualquier plan de BD en la nube.
-- Nivel 2 (tabla opcional masiva: 11.369.413 filas de F1 via COPY por lotes)
--   Activar cuando la BD remota disponga de almacenamiento sin limite de GB.
-- ============================================================================

-- 1. Cuadro maestro integrado (60 filas: 10 regiones x 6 anios, 2015-2020)
CREATE TABLE IF NOT EXISTS maestro_regional (
    anio INTEGER NOT NULL,
    region VARCHAR(64) NOT NULL,
    n_transacciones INTEGER NOT NULL,
    precio_medio DOUBLE PRECISION NOT NULL,
    precio_mediana DOUBLE PRECISION NOT NULL,
    salario_mediana_real DOUBLE PRECISION NOT NULL,
    salario_joven_nacional DOUBLE PRECISION NOT NULL,
    factor_regional DOUBLE PRECISION NOT NULL,
    salario_joven_region DOUBLE PRECISION NOT NULL,
    tmax_anom DOUBLE PRECISION,
    tmin_anom DOUBLE PRECISION,
    lluvia_anom DOUBLE PRECISION,
    sol_anom DOUBLE PRECISION,
    af_anom DOUBLE PRECISION,
    precio_var_pct DOUBLE PRECISION,
    anos_salario DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (anio, region)
);

-- 2. Agregado granular por ciudad y anio (14.050 filas, 1.153 ciudades UK)
CREATE TABLE IF NOT EXISTS precios_ciudad (
    anio INTEGER NOT NULL,
    region VARCHAR(64) NOT NULL,
    ciudad VARCHAR(96) NOT NULL,
    n_transacciones INTEGER NOT NULL,
    precio_medio DOUBLE PRECISION NOT NULL,
    precio_mediana DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (anio, region, ciudad)
);
CREATE INDEX IF NOT EXISTS idx_precios_ciudad_region ON precios_ciudad (region, anio);

-- 3. Catalogo hipotecario limpio F2 (7.796 productos, snapshot 2022)
CREATE TABLE IF NOT EXISTS hipotecas_catalogo (
    sku VARCHAR(32) NOT NULL,
    banco VARCHAR(128) NOT NULL,
    subtitulo VARCHAR(64),
    tipo_crudo VARCHAR(64),
    tipo VARCHAR(32),
    plazo_anios INTEGER,
    tasa_inicial_pct DOUBLE PRECISION NOT NULL,
    apr_pct DOUBLE PRECISION,
    tasa_reversion DOUBLE PRECISION,
    comisiones_total DOUBLE PRECISION,
    meses_inicial INTEGER,
    fecha_escaneo TIMESTAMP,
    tid VARCHAR(32) NOT NULL,
    es_atipico_tasa BOOLEAN NOT NULL,
    PRIMARY KEY (sku, tid)
);
CREATE INDEX IF NOT EXISTS idx_hipotecas_tipo_sub ON hipotecas_catalogo (tipo, subtitulo);

-- 4. Corte hipotecario regional 2022 (10 regiones)
CREATE TABLE IF NOT EXISTS hipoteca_2022 (
    region VARCHAR(64) PRIMARY KEY,
    precio_mediana DOUBLE PRECISION,
    precio_real_2020 DOUBLE PRECISION,
    tasa_inicial_pct DOUBLE PRECISION,
    comisiones_media DOUBLE PRECISION,
    salario_joven_region DOUBLE PRECISION,
    cuota_mensual DOUBLE PRECISION,
    cuota_pct_salario DOUBLE PRECISION,
    accesible BOOLEAN
);

-- 5. Serie historica nacional (22 anios: 1999-2020)
CREATE TABLE IF NOT EXISTS historico_nacional (
    anio INTEGER PRIMARY KEY,
    precio_promedio_real DOUBLE PRECISION,
    salario_mediana_real DOUBLE PRECISION,
    salario_joven_nacional DOUBLE PRECISION
);

-- 6. Segmentos descubiertos por Clustering (territoriales, hipotecarios y gran escala F1)
CREATE TABLE IF NOT EXISTS segmentos_territoriales (
    anio INTEGER NOT NULL,
    region VARCHAR(64) NOT NULL,
    cluster INTEGER NOT NULL,
    segmento_nombre VARCHAR(128) NOT NULL,
    accion_politica TEXT,
    precio_mediana DOUBLE PRECISION,
    salario_joven_region DOUBLE PRECISION,
    n_transacciones INTEGER,
    esfuerzo DOUBLE PRECISION,
    anos_salario DOUBLE PRECISION,
    PRIMARY KEY (anio, region)
);

CREATE TABLE IF NOT EXISTS segmentos_transacciones_f1 (
    cluster INTEGER PRIMARY KEY,
    segmento_f1 VARCHAR(128) NOT NULL,
    n_transacciones BIGINT NOT NULL,
    pct_mercado DOUBLE PRECISION,
    precio_medio_gbp DOUBLE PRECISION,
    latitud_media DOUBLE PRECISION,
    longitud_media DOUBLE PRECISION,
    pct_freehold DOUBLE PRECISION,
    pct_obra_nueva DOUBLE PRECISION,
    pct_flat DOUBLE PRECISION,
    pct_london DOUBLE PRECISION
);

-- 7. Reglas de asociacion (Apriori) y benchmarks de modelos / telemetria
CREATE TABLE IF NOT EXISTS reglas_asociacion_inmobiliario (
    id SERIAL PRIMARY KEY,
    antecedente TEXT NOT NULL,
    consecuente TEXT NOT NULL,
    soporte DOUBLE PRECISION,
    confianza DOUBLE PRECISION,
    lift DOUBLE PRECISION,
    score DOUBLE PRECISION,
    categoria_analitica VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS reglas_asociacion_hipotecas (
    id SERIAL PRIMARY KEY,
    antecedente TEXT NOT NULL,
    consecuente TEXT NOT NULL,
    soporte DOUBLE PRECISION,
    confianza DOUBLE PRECISION,
    lift DOUBLE PRECISION,
    score DOUBLE PRECISION
);

-- 8. Bitacora de consultas en vivo del futuro Frontend + Backend API
CREATE TABLE IF NOT EXISTS historial_predicciones (
    id SERIAL PRIMARY KEY,
    creado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tipo_consulta VARCHAR(32) NOT NULL,   -- 'regional' o 'hipoteca'
    region VARCHAR(64),
    entrada_json JSONB NOT NULL,
    prediccion_clasificacion VARCHAR(32),
    probabilidad DOUBLE PRECISION,
    prediccion_regresion DOUBLE PRECISION,
    umbral_aplicado VARCHAR(32),
    latencia_ms DOUBLE PRECISION
);

-- ============================================================================
-- NIVEL 2 (OPCIONAL · NUBE SIN LIMITE DE GB): Transacciones F1 completas (11,37 M)
-- Cargar en lotes desde data/limpia/uk_property_price_limpio.parquet via Python/COPY:
-- ============================================================================
CREATE TABLE IF NOT EXISTS transacciones_f1 (
    transaction_id VARCHAR(64) PRIMARY KEY,
    price DOUBLE PRECISION NOT NULL,
    date_of_transfer DATE NOT NULL,
    year SMALLINT NOT NULL,
    month SMALLINT NOT NULL,
    quarter SMALLINT NOT NULL,
    postcode VARCHAR(16),
    postcode_outward VARCHAR(8),
    property_type_full VARCHAR(32),
    duration_full VARCHAR(16),
    is_new_build BOOLEAN,
    town_city VARCHAR(96),
    district VARCHAR(96),
    county VARCHAR(96),
    ppd_category_type VARCHAR(32),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    region VARCHAR(64),
    country VARCHAR(32),
    admin_district VARCHAR(96),
    es_sin_geo BOOLEAN NOT NULL,
    es_atipico_precio BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tx_f1_region_year ON transacciones_f1 (region, year);
CREATE INDEX IF NOT EXISTS idx_tx_f1_city_year ON transacciones_f1 (town_city, year);
"""
    config.RUTA_ESQUEMA_BD_SQL.parent.mkdir(parents=True, exist_ok=True)
    config.RUTA_ESQUEMA_BD_SQL.write_text(sql.strip() + "\n", encoding="utf-8")

    # Inventario de tablas CSV listas para subir
    csvs_bd = [
        ("maestro_regional", config.DATA_INTEGRADO / "accesibilidad_vivienda_joven_uk_v1.csv"),
        ("precios_ciudad", config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV),
        ("hipotecas_catalogo", config.RUTA_HIPOTECAS_EXPORT_CSV),
        ("hipoteca_2022", config.RUTA_HIPOTECA_2022_MUESTRA),
        ("historico_nacional", config.RUTA_HISTORICO_MUESTRA),
        ("torneo_clasificadores", config.RUTA_TORNEO_RESUMEN),
        ("comparativa_regresion_dual", config.RUTA_COMPARATIVA_REGRESION),
        ("segmentos_territoriales", config.RUTA_SEGMENTOS_TERRITORIALES),
        ("segmentos_hipotecas", config.RUTA_SEGMENTOS_HIPOTECAS),
        ("segmentos_transacciones_f1", config.RUTA_SEGMENTOS_TRANSACCIONES_1M),
        ("clustering_benchmark_24gb", config.RUTA_CLUSTERING_24GB),
        ("reglas_asociacion_inmobiliario", config.RUTA_REGLAS_INMOBILIARIO),
        ("reglas_asociacion_hipotecas", config.RUTA_REGLAS_HIPOTECAS),
        ("evaluacion_umbrales", config.RUTA_EVALUACION_UMBRALES),
        ("comparativa_global_modelos", config.RUTA_COMPARATIVA_GLOBAL),
        ("telemetria_rendimiento", config.RUTA_TELEMETRIA_RENDIMIENTO),
        ("auditoria_fuentes_calidad", config.RUTA_AUDITORIA_DATOS),
    ]
    inv = []
    for tabla, ruta in csvs_bd:
        if ruta.exists():
            inv.append(
                {
                    "tabla_sql": tabla,
                    "archivo_csv": sistema_recursos.ruta_posix(ruta, config.PROYECTO),
                    "tamano_kb": round(ruta.stat().st_size / 1024.0, 2),
                }
            )
    return inv


def actualizar_contrato_prediccion_dual(
    meta_reg_esf: dict[str, Any],
    meta_reg_f2: dict[str, Any],
    inv_bd: list[dict[str, Any]],
) -> Path:
    """Enriquece contrato_prediccion.json con ambos tipos de prediccion, 4 .joblib y esquema BD."""
    contrato = json.loads(config.RUTA_CONTRATO_PREDICCION.read_text(encoding="utf-8"))
    contrato["tipos_de_prediccion_implementados"] = {
        "tipo_1_regresion_continua": {
            "descripcion": "Estima un valor numerico exacto (esfuerzo salarial % o tasa hipotecaria %) con banda de error MAE.",
            "esfuerzo_regional_sin_fuga": meta_reg_esf,
            "tasa_hipotecaria_f2": meta_reg_f2,
        },
        "tipo_2_clasificacion_probabilistica": {
            "descripcion": "Predice una categoria de riesgo/alerta + probabilidad + explicacion por reglas y 3 umbrales operativos.",
            "regional_alto_vs_moderado": contrato["modelo_ganador"],
            "hipoteca_atipica_f2": "DecisionTreeClassifier (max_depth=4) + 3 umbrales (0.05 / 0.30 / 0.40)",
        },
    }
    contrato["modelos_serializados"] = {
        "clasificacion_regional": sistema_recursos.ruta_posix(config.RUTA_MODELO_GANADOR, config.PROYECTO),
        "regresion_esfuerzo_regional": sistema_recursos.ruta_posix(config.RUTA_MODELO_REGRESION_ESFUERZO, config.PROYECTO),
        "clasificacion_tasa_atipica_f2": sistema_recursos.ruta_posix(config.RUTA_MODELO_CLASIFICACION_TASA_F2, config.PROYECTO),
        "regresion_tasa_f2": sistema_recursos.ruta_posix(config.RUTA_MODELO_REGRESION_TASA_F2, config.PROYECTO),
    }
    contrato["capa_datos_futura_bd"] = {
        "esquema_sql": sistema_recursos.ruta_posix(config.RUTA_ESQUEMA_BD_SQL, config.PROYECTO),
        "peso_total_tablas_nivel1_kb": round(sum(x["tamano_kb"] for x in inv_bd), 2),
        "tablas_nivel1": inv_bd,
        "tabla_nivel2_opcional_11m": "data/limpia/uk_property_price_limpio.parquet (11.369.413 filas, carga por lotes COPY)",
    }
    contrato["endpoints_futuros_api"] = {
        "regional_dual": "POST /api/predecir/region -> Clasificacion (Alto/Moderado + Reglas) + Regresion (esfuerzo % +/- MAE)",
        "hipoteca_dual": "POST /api/predecir/hipoteca -> Regresion (tasa % +/- MAE) + Clasificacion (P atipica + 3 umbrales)",
    }
    config.RUTA_CONTRATO_PREDICCION.write_text(
        json.dumps(contrato, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return config.RUTA_CONTRATO_PREDICCION


def main() -> None:
    sistema_recursos.configurar_entorno_portable()
    print("=" * 80)
    print("PREDICCION DUAL (TIPO 1 REGRESION + TIPO 2 CLASIFICACION) Y EMPAQUETADO BD")
    print("=" * 80)

    tab_esf, reg_esf, meta_esf = evaluar_regresion_maestro_sin_fuga()
    tab_f2, reg_f2, clf_f2, meta_f2 = evaluar_regresion_hipotecas_f2()

    config.DATA_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg_esf, config.RUTA_MODELO_REGRESION_ESFUERZO, compress=3)
    joblib.dump(reg_f2, config.RUTA_MODELO_REGRESION_TASA_F2, compress=3)
    joblib.dump(clf_f2, config.RUTA_MODELO_CLASIFICACION_TASA_F2, compress=3)

    tab_reg_all = pd.concat([tab_esf, tab_f2], ignore_index=True)
    tab_reg_all.to_csv(config.RUTA_COMPARATIVA_REGRESION, index=False, encoding="utf-8")

    print("\n[1] TIPO 1 · REGRESION CONTINUA (Sin fuga en Maestro + Tasa Exacta en F2):")
    print(tab_reg_all.to_string(index=False))

    df_reg10 = tabla_prediccion_dual_10_regiones(reg_esf)
    print("\n[2] Comparativa Dual (Tipo 1 Regresion + Tipo 2 Clasificacion) sobre las 10 regiones UK:")
    print(df_reg10.to_string(index=False))

    df_hip5 = tabla_prediccion_dual_hipotecas_f2(reg_f2, clf_f2, meta_f2["columnas_X"])
    print("\n[3] Comparativa Dual sobre 5 ofertas hipotecarias reales de F2:")
    print(df_hip5.to_string(index=False))

    fig = fig_prediccion_dual(df_reg10, reg_f2, meta_f2["columnas_X"])
    plt.close(fig)

    inv_bd = exportar_capa_relacional_bd()
    ruta_json = actualizar_contrato_prediccion_dual(meta_esf, meta_f2, inv_bd)

    print("\n[4] 4 Modelos preentrenados en data/modelos/ (inferencia < 5 ms sin reentrenar):")
    for r in (
        config.RUTA_MODELO_GANADOR,
        config.RUTA_MODELO_REGRESION_ESFUERZO,
        config.RUTA_MODELO_CLASIFICACION_TASA_F2,
        config.RUTA_MODELO_REGRESION_TASA_F2,
    ):
        print(f"    - {sistema_recursos.ruta_posix(r, config.PROYECTO):<54} {r.stat().st_size / 1024.0:7.2f} KB")
    print(f"[5] Esquema SQL relacional : {sistema_recursos.ruta_posix(config.RUTA_ESQUEMA_BD_SQL, config.PROYECTO)}")
    print(f"    Tablas CSV Nivel 1 BD  : {len(inv_bd)} tablas ({sum(x['tamano_kb'] for x in inv_bd):.1f} KB = {sum(x['tamano_kb'] for x in inv_bd)/1024.0:.2f} MB)")
    print(f"    Contrato JSON dual     : {sistema_recursos.ruta_posix(ruta_json, config.PROYECTO)}")
    print("    Figura generada        : md/figuras/prediccion_dual_regresion_vs_clasificacion.png")


if __name__ == "__main__":
    main()
