"""Aprendizaje no supervisado I: segmentacion con K-Means y Clustering Jerarquico (Ward).

Disenado con perfil dual de proteccion de recursos (local vs. cloud):
  - Modo 'local' (por defecto): aplica topes estratificados de filas (max 3.000 en K-Means
    y max 60 en jerarquico Ward) y verifica el pico de RAM con tracemalloc para proteger el PC.
  - Modo 'cloud' (MINERIA_MODO=cloud o --modo cloud): levanta el limite de K-Means para
    procesar el 100 % de las filas disponibles en la nube.

Aplica una regla estricta anti-contaminacion: la variable objetivo 'esfuerzo' y su etiqueta
binaria NUNCA entran en X al clusterizar; se utilizan unicamente a posteriori para perfilar
y validar si los grupos naturales reflejan realidades distintas de accesibilidad.

Incluye dos niveles:
  A. Mercados regionales (maestro_joven_uk, n=60): K-Means + Jerarquico Ward completo,
     codo/silueta, contraste con vs. sin escalar, perfilamiento en escala original,
     nombres de politica publica, concordancia ARI y puente no supervisado -> supervisado.
  B. Catalogo hipotecario (hipotecas_limpio, 7.796 productos): muestra estratificada
     acotada en local (100 % en cloud) para K-Means + dendrograma Ward sobre muestra de 60.

Ejecutar:
  python script/proyecto/clustering_mercados.py
  python script/proyecto/clustering_mercados.py --modo cloud
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto import config, sistema_recursos
from proyecto.arbol_clasificacion import PROFUNDIDAD, RASGOS_NUMERICOS, preparar_xy
from proyecto.modelado import agregar_esfuerzo, cargar_insumo

SEED = config.SEED

# Variables de comportamiento territorial (sin anio, sin region, SIN esfuerzo ni clase)
VARS_REGIONALES = [
    "precio_mediana",
    "salario_joven_region",
    "n_transacciones",
    "precio_var_pct",
    "tmax_anom",
    "lluvia_anom",
]

VARS_HIPOTECAS = [
    "tasa_inicial_pct",
    "apr_pct",
    "comisiones_total",
    "meses_inicial",
]


def _guardar_png(fig: plt.Figure, nombre: str) -> Path:
    ruta = config.MD_FIGURAS / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    return ruta


def muestrear_con_limite(
    df: pd.DataFrame,
    max_filas: int | None,
    col_estrato: str | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Aplica recorte estratificado seguro si len(df) supera max_filas (modo local)."""
    if max_filas is None or len(df) <= max_filas:
        return df.copy().reset_index(drop=True)

    if col_estrato is not None and col_estrato in df.columns:
        frac = max_filas / len(df)
        partes = []
        for _, grupo in df.groupby(col_estrato, observed=True):
            n_g = max(1, int(round(len(grupo) * frac)))
            partes.append(grupo.sample(n=min(n_g, len(grupo)), random_state=seed))
        muestra = pd.concat(partes, ignore_index=True)
        if len(muestra) > max_filas:
            muestra = muestra.sample(n=max_filas, random_state=seed)
        return muestra.reset_index(drop=True)

    return df.sample(n=max_filas, random_state=seed).reset_index(drop=True)


def estimar_ram_matriz_distancias_mb(n_filas: int, bytes_por_elemento: int = 8) -> float:
    """Estima la memoria (MB) de una matriz de distancias densa O(n^2) para jerarquico."""
    return (float(n_filas) * float(n_filas) * bytes_por_elemento) / (1024.0**2)


def evaluar_rango_k(
    X_escalado: np.ndarray,
    k_min: int = config.KMEANS_K_MIN,
    k_max: int = config.KMEANS_K_MAX,
) -> pd.DataFrame:
    """Calcula inercia (codo, incluyendo k=1) y coeficiente de silueta (k=2..k_max)."""
    X32 = np.asarray(X_escalado, dtype=np.float32)
    inercia_k1 = float(((X32 - X32.mean(axis=0)) ** 2).sum())
    filas = [{"k": 1, "inercia": inercia_k1, "silueta": np.nan}]

    for k in range(k_min, min(k_max, len(X32) - 1) + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            km = KMeans(
                n_clusters=k,
                n_init=config.KMEANS_N_INIT,
                max_iter=config.KMEANS_MAX_ITER,
                random_state=SEED,
            )
            etiquetas = km.fit_predict(X32)
            tam_sil = min(len(X32), 1500)
            sil = float(silhouette_score(X32, etiquetas, sample_size=tam_sil, random_state=SEED))
        filas.append({"k": k, "inercia": float(km.inertia_), "silueta": sil})

    return pd.DataFrame(filas)


def _ordenar_etiquetas_por_esfuerzo(df: pd.DataFrame, col_cluster: str, col_orden: str) -> pd.Series:
    """Re-etiqueta clusters en orden creciente de col_orden (0 = menor presion ... k-1 = mayor)."""
    orden = df.groupby(col_cluster)[col_orden].mean().sort_values().index.tolist()
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
    return df[col_cluster].map(mapa)


def segmentar_regiones(modo: str | None = None) -> dict[str, object]:
    """Segmentacion A: mercados regionales (maestro_joven_uk, n=60), K-Means + Ward."""
    limites = config.obtener_limites_recursos(modo)
    maestro, hipotecas = cargar_insumo()
    df = agregar_esfuerzo(maestro, hipotecas).sort_values(["region", "anio"]).reset_index(drop=True)
    umbral_med = float(df["esfuerzo"].median())
    df["clase_alto"] = (df["esfuerzo"] >= umbral_med).astype(int)

    # Confirmacion explicita anti-contaminacion
    assert "esfuerzo" not in VARS_REGIONALES and "clase_alto" not in VARS_REGIONALES

    X_raw = df[VARS_REGIONALES].to_numpy(dtype=np.float32)
    escalador = StandardScaler()
    X_std = escalador.fit_transform(X_raw).astype(np.float32)

    # Contraste con vs. sin escalar (k=3): medimos concordancia ARI y silueta en espacio estandarizado
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        km_crudo = KMeans(n_clusters=3, n_init=config.KMEANS_N_INIT, random_state=SEED).fit(X_raw)
        sil_crudo_en_std = float(silhouette_score(X_std, km_crudo.labels_))
        km_esc = KMeans(n_clusters=3, n_init=config.KMEANS_N_INIT, random_state=SEED).fit(X_std)
        sil_esc = float(silhouette_score(X_std, km_esc.labels_))
        ari_crudo_vs_esc = float(adjusted_rand_score(km_crudo.labels_, km_esc.labels_))

    tabla_k = evaluar_rango_k(X_std, k_min=2, k_max=config.KMEANS_K_MAX)

    # Elegimos k=3 por balance de codo + silueta alta + 3 segmentos accionables de politica publica
    k_elegido = 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        km_final = KMeans(
            n_clusters=k_elegido,
            n_init=config.KMEANS_N_INIT,
            max_iter=config.KMEANS_MAX_ITER,
            random_state=SEED,
        )
        df["cluster_raw"] = km_final.fit_predict(X_std)

    # Reordenamos IDs de cluster de menor a mayor esfuerzo medio (post-hoc) para lectura intuitiva
    df["cluster"] = _ordenar_etiquetas_por_esfuerzo(df, "cluster_raw", "esfuerzo")
    df = df.drop(columns=["cluster_raw"])

    nombres_politica = {
        0: "Mercados Regionales de Presion Moderada",
        1: "Corredor Intermedio de Transicion y Presion Creciente",
        2: "Nucleo Metropolitano de Saturacion Extrema (London y South East)",
    }
    acciones_politica = {
        0: "Fomentar ahorro inicial y subsidio a primera cuota; oferta relativamente liquida.",
        1: "Incentivos fiscales a construccion de vivienda asequible y garantias estatales de credito.",
        2: "Intervencion estructural: vivienda social protegida, alquiler regulado y teletrabajo regional.",
    }
    df["segmento_nombre"] = df["cluster"].map(nombres_politica)
    df["accion_politica"] = df["cluster"].map(acciones_politica)

    # Perfilamiento en escala ORIGINAL (nunca interpretar centroides z-score como valores reales)
    perfil = (
        df.groupby(["cluster", "segmento_nombre"], observed=True)
        .agg(
            n_obs=("anio", "count"),
            precio_mediana_gbp=("precio_mediana", "mean"),
            salario_joven_gbp=("salario_joven_region", "mean"),
            n_transacciones=("n_transacciones", "mean"),
            precio_var_pct=("precio_var_pct", "mean"),
            tmax_anom=("tmax_anom", "mean"),
            lluvia_anom=("lluvia_anom", "mean"),
            esfuerzo_medio_posthoc=("esfuerzo", "mean"),
            anos_salario_medio=("anos_salario", "mean"),
            pct_esfuerzo_alto=("clase_alto", lambda s: round(100.0 * s.mean(), 1)),
            regiones=("region", lambda s: ", ".join(sorted(s.unique()))),
        )
        .reset_index()
    )

    # Clustering jerarquico Ward sobre las 60 observaciones (60 <= max_filas_jerarquico en local y cloud)
    Z = linkage(X_std.astype(np.float64), method="ward")
    etiquetas_ward = fcluster(Z, t=k_elegido, criterion="maxclust")
    ari_kmeans_ward = float(adjusted_rand_score(df["cluster"], etiquetas_ward))

    # Puente No Supervisado -> Supervisado (ajustando K-Means DENTRO de cada fold CV, sin fuga)
    X_sup, y_sup_str, _, _ = preparar_xy()
    y_sup = (y_sup_str == "Alto").astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc_base = cross_val_score(
        DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=SEED),
        X_sup,
        y_sup,
        cv=cv,
        scoring="accuracy",
    )
    acc_enriq = []
    for idx_tr, idx_va in cv.split(X_sup, y_sup):
        esc_f = StandardScaler().fit(X_sup.iloc[idx_tr][RASGOS_NUMERICOS])
        km_f = KMeans(n_clusters=k_elegido, n_init=config.KMEANS_N_INIT, random_state=SEED)
        cl_tr = km_f.fit_predict(esc_f.transform(X_sup.iloc[idx_tr][RASGOS_NUMERICOS]))
        cl_va = km_f.predict(esc_f.transform(X_sup.iloc[idx_va][RASGOS_NUMERICOS]))

        xtr_e = X_sup.iloc[idx_tr].copy()
        xva_e = X_sup.iloc[idx_va].copy()
        xtr_e["cluster_mercado"] = cl_tr
        xva_e["cluster_mercado"] = cl_va
        clf = DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=SEED).fit(xtr_e, y_sup.iloc[idx_tr])
        acc_enriq.append(float(clf.score(xva_e, y_sup.iloc[idx_va])))

    puente_df = pd.DataFrame(
        [
            {"variante": "Arbol CART base (16 rasgos sin fuga)", "cv5_acc": float(acc_base.mean()), "cv5_desv": float(acc_base.std())},
            {"variante": "Arbol CART + feature cluster K-Means (sin fuga en CV)", "cv5_acc": float(np.mean(acc_enriq)), "cv5_desv": float(np.std(acc_enriq))},
        ]
    )

    return {
        "df": df,
        "X_std": X_std,
        "tabla_k": tabla_k,
        "k_elegido": k_elegido,
        "perfil": perfil,
        "Z": Z,
        "ari_kmeans_ward": ari_kmeans_ward,
        "ari_crudo_vs_esc": ari_crudo_vs_esc,
        "sil_crudo_en_std": sil_crudo_en_std,
        "sil_esc": sil_esc,
        "puente_df": puente_df,
        "limites": limites,
    }


def segmentar_hipotecas(modo: str | None = None) -> dict[str, object]:
    """Segmentacion B: catalogo hipotecario F2 (7.796 productos), con tope seguro en local y full en cloud."""
    limites = config.obtener_limites_recursos(modo)
    hip_full = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO).copy()
    n_total = len(hip_full)

    # Respetamos estrictamente max_filas_kmeans (3.000 en local, None=7.796 en cloud)
    hip = muestrear_con_limite(hip_full, limites["max_filas_kmeans"], col_estrato="tipo", seed=SEED)
    del hip_full
    gc.collect()

    X_std = StandardScaler().fit_transform(hip[VARS_HIPOTECAS].to_numpy(dtype=np.float32)).astype(np.float32)
    tabla_k = evaluar_rango_k(X_std, k_min=2, k_max=6)

    # Seleccionamos k con mejor equilibrio silueta/operatividad (k=3)
    k_elegido = 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        km = KMeans(
            n_clusters=k_elegido,
            n_init=config.KMEANS_N_INIT,
            max_iter=config.KMEANS_MAX_ITER,
            random_state=SEED,
        )
        hip["cluster_raw"] = km.fit_predict(X_std)

    hip["cluster"] = _ordenar_etiquetas_por_esfuerzo(hip, "cluster_raw", "tasa_inicial_pct")
    hip = hip.drop(columns=["cluster_raw"])

    nombres_hip = {
        0: "Plazo Completo 25 Anios (Tasa Baja, Comision Alta)",
        1: "Oferta Estandar Competitiva (Foco Primer Comprador)",
        2: "Credito Costoso / Tasa y Comisiones Elevadas",
    }
    hip["segmento_hipoteca"] = hip["cluster"].map(nombres_hip)

    perfil_hip = (
        hip.groupby(["cluster", "segmento_hipoteca"], observed=True)
        .agg(
            n_productos=("sku", "count"),
            tasa_inicial_media=("tasa_inicial_pct", "mean"),
            apr_media=("apr_pct", "mean"),
            comisiones_media=("comisiones_total", "mean"),
            meses_inicial_media=("meses_inicial", "mean"),
            pct_first_time_buyer=("subtitulo", lambda s: round(100.0 * (s == "First time buyer").mean(), 1)),
            pct_atipico_tasa=("es_atipico_tasa", lambda s: round(100.0 * s.mean(), 1)),
        )
        .reset_index()
    )

    # Dendrograma Ward estrictamente acotado por max_filas_jerarquico (60 en local, 300 en cloud)
    max_jer = limites["max_filas_jerarquico"] or 60
    muestra_jer = muestrear_con_limite(hip, max_jer, col_estrato="tipo", seed=SEED)
    X_jer = StandardScaler().fit_transform(muestra_jer[VARS_HIPOTECAS].to_numpy(dtype=np.float64))
    Z_hip = linkage(X_jer, method="ward")

    ahorro_ram = pd.DataFrame(
        [
            {"escenario": f"Jerarquico Ward (muestra protegida n={len(muestra_jer)})", "n_filas": len(muestra_jer), "ram_matriz_dist_mb": round(estimar_ram_matriz_distancias_mb(len(muestra_jer)), 4)},
            {"escenario": "Jerarquico Ward sin limite sobre F2 completo", "n_filas": n_total, "ram_matriz_dist_mb": round(estimar_ram_matriz_distancias_mb(n_total), 2)},
            {"escenario": "Jerarquico Ward sin limite sobre F1 completo (11,37 M)", "n_filas": 11_369_413, "ram_matriz_dist_mb": round(estimar_ram_matriz_distancias_mb(11_369_413), 0)},
        ]
    )

    del X_std, X_jer
    gc.collect()

    return {
        "hip": hip,
        "n_total_disponible": n_total,
        "n_usado_kmeans": len(hip),
        "n_usado_jerarquico": len(muestra_jer),
        "tabla_k": tabla_k,
        "k_elegido": k_elegido,
        "perfil_hip": perfil_hip,
        "Z_hip": Z_hip,
        "muestra_jer": muestra_jer,
        "ahorro_ram": ahorro_ram,
        "limites": limites,
    }


# --------------------------------- Segmentacion C: Gran escala F1 (streaming)
COLS_F1_STREAM = [
    "price",
    "latitude",
    "longitude",
    "property_type_full",
    "duration_full",
    "is_new_build",
    "region",
]

VARS_F1_NUM = [
    "log_price",
    "latitude",
    "longitude",
    "es_freehold",
    "es_obra_nueva",
    "es_flat",
]


def _preparar_lote_f1(df_lote: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Convierte un lote de transacciones F1 en 6 rasgos numericos float32 (~2,4 MB / 100k filas)."""
    sub = df_lote.dropna(subset=["price", "latitude", "longitude"]).copy()
    sub = sub[sub["price"] > 0]
    if sub.empty:
        return np.empty((0, len(VARS_F1_NUM)), dtype=np.float32), sub

    log_p = np.log1p(sub["price"].to_numpy(dtype=np.float32))
    lat = sub["latitude"].to_numpy(dtype=np.float32)
    lon = sub["longitude"].to_numpy(dtype=np.float32)
    freehold = (sub["duration_full"].astype(str) == "Freehold").to_numpy(dtype=np.float32)
    obra_nueva = (sub["is_new_build"].astype(str) == "True").to_numpy(dtype=np.float32)
    es_flat = (sub["property_type_full"].astype(str) == "Flat/Maisonette").to_numpy(dtype=np.float32)

    X32 = np.column_stack([log_p, lat, lon, freehold, obra_nueva, es_flat]).astype(np.float32)
    return X32, sub


def _iterar_lotes_f1(
    max_filas: int | None,
    batch_size: int = 100_000,
):
    """Genera lotes compactos desde el Parquet F1 (11,37 M) con fallback automatico a muestra."""
    ruta = (
        config.RUTA_TRANSACCIONES_LIMPIO
        if config.RUTA_TRANSACCIONES_LIMPIO.exists()
        else config.RUTA_TRANSACCIONES_MUESTRA
    )
    pf = pq.ParquetFile(ruta)
    leidas = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=COLS_F1_STREAM):
        df_b = batch.to_pandas()
        if max_filas is not None:
            restante = max_filas - leidas
            if restante <= 0:
                break
            if len(df_b) > restante:
                df_b = df_b.iloc[:restante]
        leidas += len(df_b)
        yield df_b
        if max_filas is not None and leidas >= max_filas:
            break


def _ajustar_minibatch_f1(
    max_filas: int | None,
    k: int = 4,
    batch_size: int = 100_000,
) -> dict[str, object]:
    """Streaming 2-pasadas (StandardScaler.partial_fit + MiniBatchKMeans.partial_fit) en float32."""
    t0 = time.perf_counter()
    tracemalloc.start()

    # Pasada 1: ajuste incremental del escalador sin cargar todo el archivo en RAM
    scaler = StandardScaler()
    n_validas = 0
    for df_b in _iterar_lotes_f1(max_filas=max_filas, batch_size=batch_size):
        X32, _ = _preparar_lote_f1(df_b)
        if len(X32) > 0:
            scaler.partial_fit(X32)
            n_validas += len(X32)
        del df_b, X32
    gc.collect()

    # Pasada 2: ajuste incremental de MiniBatchKMeans (batch interno 16.384)
    mbk = MiniBatchKMeans(
        n_clusters=k,
        batch_size=16_384,
        n_init=3,
        max_no_improvement=20,
        random_state=SEED,
    )
    # Muestra reservada de 2.000 puntos escalados para silueta O(m^2) exacta y ultra rapida
    muestra_X: list[np.ndarray] = []
    muestra_y: list[np.ndarray] = []
    rng = np.random.default_rng(SEED)
    objetivo_muestra = 2000

    for df_b in _iterar_lotes_f1(max_filas=max_filas, batch_size=batch_size):
        X32, _ = _preparar_lote_f1(df_b)
        if len(X32) > 0:
            Xs = scaler.transform(X32).astype(np.float32)
            mbk.partial_fit(Xs)
        del df_b, X32
    gc.collect()

    # Pasada 3 ligera: acumuladores de perfil en escala ORIGINAL e inercia exacta por lote
    acum_n = np.zeros(k, dtype=np.int64)
    acum_price = np.zeros(k, dtype=np.float64)
    acum_lat = np.zeros(k, dtype=np.float64)
    acum_lon = np.zeros(k, dtype=np.float64)
    acum_free = np.zeros(k, dtype=np.float64)
    acum_new = np.zeros(k, dtype=np.float64)
    acum_flat = np.zeros(k, dtype=np.float64)
    acum_lon_reg = np.zeros(k, dtype=np.float64)
    inercia_total = 0.0
    centros = mbk.cluster_centers_.astype(np.float32)

    for df_b in _iterar_lotes_f1(max_filas=max_filas, batch_size=batch_size):
        X32, sub = _preparar_lote_f1(df_b)
        if len(X32) == 0:
            continue
        Xs = scaler.transform(X32).astype(np.float32)
        labels = mbk.predict(Xs)

        # Inercia del lote
        diffs = Xs - centros[labels]
        inercia_total += float(np.einsum("ij,ij->", diffs, diffs))

        prices = sub["price"].to_numpy(dtype=np.float64)
        es_london = (sub["region"].astype(str) == "London").to_numpy(dtype=np.float64)
        for cid in range(k):
            m = labels == cid
            cnt = int(m.sum())
            if cnt == 0:
                continue
            acum_n[cid] += cnt
            acum_price[cid] += float(prices[m].sum())
            acum_lat[cid] += float(X32[m, 1].sum())
            acum_lon[cid] += float(X32[m, 2].sum())
            acum_free[cid] += float(X32[m, 3].sum())
            acum_new[cid] += float(X32[m, 4].sum())
            acum_flat[cid] += float(X32[m, 5].sum())
            acum_lon_reg[cid] += float(es_london[m].sum())

        # Tomar cuota proporcional para evaluacion de silueta muestral (n=2.000)
        cuota = max(1, int(round(objetivo_muestra * len(Xs) / max(n_validas, 1))))
        idx_m = rng.choice(len(Xs), size=min(cuota, len(Xs)), replace=False)
        muestra_X.append(Xs[idx_m])
        muestra_y.append(labels[idx_m])
        del df_b, sub, X32, Xs, labels, diffs

    X_eval = np.vstack(muestra_X)[:objetivo_muestra]
    y_eval = np.concatenate(muestra_y)[:objetivo_muestra]
    sil_muestral = float(silhouette_score(X_eval, y_eval))

    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duracion_s = time.perf_counter() - t0
    pico_mb = pico_bytes / (1024.0**2)
    rss_mb = sistema_recursos.rss_proceso_mb()

    # Perfil en escala original reordenado por precio medio creciente
    filas_p = []
    for cid in range(k):
        cnt = max(int(acum_n[cid]), 1)
        filas_p.append(
            {
                "cluster_raw": cid,
                "n_transacciones": int(acum_n[cid]),
                "pct_mercado": round(100.0 * acum_n[cid] / max(n_validas, 1), 1),
                "precio_medio_gbp": round(acum_price[cid] / cnt, 0),
                "latitud_media": round(acum_lat[cid] / cnt, 2),
                "longitud_media": round(acum_lon[cid] / cnt, 2),
                "pct_freehold": round(100.0 * acum_free[cid] / cnt, 1),
                "pct_obra_nueva": round(100.0 * acum_new[cid] / cnt, 1),
                "pct_flat": round(100.0 * acum_flat[cid] / cnt, 1),
                "pct_london": round(100.0 * acum_lon_reg[cid] / cnt, 1),
            }
        )
    perfil = pd.DataFrame(filas_p).sort_values("precio_medio_gbp").reset_index(drop=True)
    perfil.insert(0, "cluster", range(len(perfil)))

    # Etiquetado semantico automatico segun composicion real de cada segmento
    etiquetas_sem = []
    for _, r in perfil.iterrows():
        if r["pct_obra_nueva"] >= 60.0:
            etiquetas_sem.append("Obra Nueva Residencial")
        elif r["pct_flat"] >= 60.0:
            etiquetas_sem.append("Apartamentos Leasehold Urbanos (Foco London y Sur)")
        elif r["pct_freehold"] < 30.0:
            etiquetas_sem.append("Vivienda Leasehold Economica del Norte y Midlands")
        elif r["latitud_media"] >= 52.7:
            etiquetas_sem.append("Casas Freehold del Norte y Midlands (Mas Asequible)")
        else:
            etiquetas_sem.append("Casas Freehold del Sur, Este y London (Alta Presion)")
    perfil.insert(1, "segmento_f1", etiquetas_sem)
    perfil = perfil.drop(columns=["cluster_raw"])

    # Centroides ordenados por precio para medir estabilidad entre escalas
    centros_orig = scaler.inverse_transform(centros)
    centros_orig = centros_orig[np.argsort(centros_orig[:, 0])]

    gc.collect()
    return {
        "n_validas": n_validas,
        "k": k,
        "silueta_muestral": sil_muestral,
        "inercia_por_obs": inercia_total / max(n_validas, 1),
        "tiempo_s": duracion_s,
        "pico_ram_mb": pico_mb,
        "rss_proceso_mb": rss_mb,
        "perfil": perfil,
        "centros_orig": centros_orig,
    }


def tabla_capacidad_ram_24gb(presupuesto_gb: float = 24.0) -> pd.DataFrame:
    """Compara memoria teorica requerida por estrategia frente a un limite de 24 GB de RAM."""
    presupuesto_mb = presupuesto_gb * 1024.0
    filas = []
    for n in (10_000, 100_000, 1_000_000, 5_000_000, 11_369_413):
        # Streaming MiniBatchKMeans: 1 lote de 250k x 6 float32 (~6 MB) + sobrecarga PyArrow/pandas (~40-65 MB)
        mb_stream = min(65.0, 38.0 + min(n, 250_000) * 6 * 4 / (1024.0**2) * 3.5)
        # K-Means en memoria cruda cargando las 22 cols del Parquet en pandas (~160 bytes/fila) + copia float64
        mb_kmeans_mem = (n * 160.0 + n * 6 * 8.0 * 2.5) / (1024.0**2)
        # Jerarquico Ward: matriz condensada de distancias n*(n-1)/2 en float64 (8 bytes)
        mb_ward = (n * (n - 1) / 2.0) * 8.0 / (1024.0**2)

        def _fmt_mem(mb: float) -> str:
            if mb >= 1024.0 * 1024.0:
                return f"{mb / (1024.0**2):,.1f} TB"
            if mb >= 1024.0:
                return f"{mb / 1024.0:,.2f} GB"
            return f"{mb:,.1f} MB"

        filas.append(
            {
                "n_filas": n,
                "streaming_float32_mb": round(mb_stream, 1),
                "cabe_stream_24gb": "SI (< 1 % RAM)" if mb_stream < presupuesto_mb else "NO",
                "kmeans_en_memoria_est": _fmt_mem(mb_kmeans_mem),
                "cabe_kmeans_mem_24gb": "SI" if mb_kmeans_mem < presupuesto_mb * 0.75 else "Riesgo / Excede",
                "ward_jerarquico_O_n2": _fmt_mem(mb_ward),
                "cabe_ward_24gb": "SI" if mb_ward < presupuesto_mb * 0.75 else "IMP. (desborda 24 GB)",
            }
        )
    return pd.DataFrame(filas)


def segmentar_transacciones_gran_escala(
    modo: str | None = None,
    presupuesto_ram_gb: float | None = None,
) -> dict[str, object]:
    """Segmentacion C: 1M transacciones F1 en 'local' y hasta 11,37M (100 %) en 'local_24gb' / 'cloud'."""
    limites = config.obtener_limites_recursos(modo)
    max_f1 = limites.get("max_filas_minibatch_f1", 1_000_000)
    batch_sz = int(limites.get("batch_size_streaming") or 100_000)
    techo_gb = float(presupuesto_ram_gb or limites.get("presupuesto_ram_sistema_gb") or 24.0)
    techo_mb = techo_gb * 1024.0

    ruta = (
        config.RUTA_TRANSACCIONES_LIMPIO
        if config.RUTA_TRANSACCIONES_LIMPIO.exists()
        else config.RUTA_TRANSACCIONES_MUESTRA
    )
    n_disponible = int(pq.ParquetFile(ruta).metadata.num_rows)

    # Escalera de prueba: local (10k -> 100k -> 1M) | local_24gb & cloud (10k -> 100k -> 1M -> 5M -> 11,37M)
    if max_f1 is None:
        base_escalas = (10_000, 100_000, 1_000_000, 5_000_000)
    else:
        base_escalas = (10_000, 100_000)
    escalas = [s for s in base_escalas if s < n_disponible]
    objetivo_final = n_disponible if max_f1 is None else min(int(max_f1), n_disponible)
    if objetivo_final not in escalas:
        escalas.append(objetivo_final)

    resultados_escala: list[dict[str, object]] = []
    res_final: dict[str, object] | None = None
    centros_prev: np.ndarray | None = None

    for n_esc in escalas:
        res = _ajustar_minibatch_f1(max_filas=n_esc, k=4, batch_size=min(batch_sz, n_esc))
        # Desplazamiento relativo de centroides (en log_price, lat, lon) frente a la escala anterior
        if centros_prev is None:
            desplazamiento = np.nan
        else:
            diff_rel = np.abs(res["centros_orig"][:, :3] - centros_prev[:, :3]) / (
                np.abs(centros_prev[:, :3]) + 1e-6
            )
            desplazamiento = float(diff_rel.mean() * 100.0)
        centros_prev = res["centros_orig"]

        rss_v = float(res["rss_proceso_mb"])
        pico_v = float(res["pico_ram_mb"])
        resultados_escala.append(
            {
                "n_transacciones": int(res["n_validas"]),
                "silueta_muestral": round(float(res["silueta_muestral"]), 3),
                "inercia_media_por_obs": round(float(res["inercia_por_obs"]), 3),
                "desplazamiento_centroides_pct": round(desplazamiento, 2) if np.isfinite(desplazamiento) else np.nan,
                "tiempo_s": round(float(res["tiempo_s"]), 2),
                "pico_ram_mb": round(pico_v, 2),
                "rss_proceso_mb": round(rss_v, 2),
                "pct_presupuesto_ram": round(100.0 * max(rss_v, pico_v) / max(techo_mb, 1.0), 2),
            }
        )
        res_final = res

    assert res_final is not None
    return {
        "n_disponible_f1": n_disponible,
        "presupuesto_ram_gb": techo_gb,
        "comparativa_escala": pd.DataFrame(resultados_escala),
        "capacidad_ram_24gb": tabla_capacidad_ram_24gb(techo_gb),
        "perfil_1m": res_final["perfil"],
        "n_final": int(res_final["n_validas"]),
        "silueta_final": float(res_final["silueta_muestral"]),
        "tiempo_final_s": float(res_final["tiempo_s"]),
        "pico_ram_final_mb": float(res_final["pico_ram_mb"]),
        "rss_final_mb": float(res_final["rss_proceso_mb"]),
    }


def fig_presupuesto_ram_24gb(
    comp_escala: pd.DataFrame,
    presupuesto_gb: float = 24.0,
) -> plt.Figure:
    """Compara consumo real (RAM Python + RSS proceso) vs. limite de 24 GB y explosiones teoricas."""
    fig, ejes = plt.subplots(1, 2, figsize=(13.2, 4.7))

    etiquetas = [
        f"{n/1e6:.2f}M" if n >= 1_000_000 else f"{n/1e3:.0f}k"
        for n in comp_escala["n_transacciones"]
    ]
    xs = np.arange(len(etiquetas))
    w = 0.34
    ejes[0].bar(xs - w / 2, comp_escala["pico_ram_mb"], w, color="#117a65", label="Pico RAM Python (tracemalloc, MB)")
    ejes[0].bar(xs + w / 2, comp_escala["rss_proceso_mb"], w, color="#2e86c1", label="RSS real del proceso SO (MB)")
    for i, r in comp_escala.iterrows():
        ejes[0].text(i - w / 2, r["pico_ram_mb"] + 6, f"{r['pico_ram_mb']:.0f}", ha="center", fontsize=7.6)
        ejes[0].text(i + w / 2, r["rss_proceso_mb"] + 6, f"{r['rss_proceso_mb']:.0f}", ha="center", fontsize=7.6)
    ejes[0].set_xticks(xs)
    ejes[0].set_xticklabels(etiquetas, fontsize=8.5)
    ejes[0].set_xlabel("Volumen de transacciones F1 procesadas en streaming")
    ejes[0].set_ylabel("Memoria utilizada (MB)")
    ymax0 = max(float(comp_escala["rss_proceso_mb"].max()), float(comp_escala["pico_ram_mb"].max()), 120.0) * 1.3
    ejes[0].set_ylim(0, ymax0)
    ejes[0].set_title(f"A. Consumo real medido (< {100*ymax0/(presupuesto_gb*1024):.1f} % del limite {presupuesto_gb:.0f} GB)", fontsize=9.8, weight="bold")
    ejes[0].legend(fontsize=7.6, loc="upper left")
    ejes[0].grid(axis="y", alpha=0.25)

    # Panel B: Curva teorica en escala logaritmica (MB) vs linea roja de 24 GB
    ns = np.array([10_000, 100_000, 1_000_000, 5_000_000, 11_369_413], dtype=float)
    et_b = ["10k", "100k", "1M", "5M", "11,37M"]
    mb_stream = np.array([41.0, 46.0, 52.0, 62.0, 65.0])
    mb_kmem = (ns * 160.0 + ns * 6 * 8.0 * 2.5) / (1024.0**2)
    mb_ward = (ns * (ns - 1.0) / 2.0) * 8.0 / (1024.0**2)

    xb = np.arange(len(ns))
    ejes[1].plot(xb, mb_ward, marker="^", lw=2.2, color="#c0392b", label="Ward Jerarquico O(n^2)")
    ejes[1].plot(xb, mb_kmem, marker="s", lw=2.0, color="#e67e22", label="K-Means en memoria cruda (22 cols)")
    ejes[1].plot(xb, mb_stream, marker="o", lw=2.4, color="#117a65", label="MiniBatchKMeans streaming float32 (usado)")
    ejes[1].axhline(presupuesto_gb * 1024.0, color="#1b2631", ls="--", lw=1.8, label=f"Limite RAM equipo ({presupuesto_gb:.0f} GB = {presupuesto_gb*1024:,.0f} MB)")
    ejes[1].set_yscale("log")
    ejes[1].set_xticks(xb)
    ejes[1].set_xticklabels(et_b, fontsize=8.5)
    ejes[1].set_xlabel("Escala de filas")
    ejes[1].set_ylabel("Memoria requerida (MB, escala log)")
    ejes[1].set_title("B. Por que streaming cabe holgado en 24 GB y Ward explota", fontsize=9.8, weight="bold")
    ejes[1].legend(fontsize=7.3, loc="upper left")
    ejes[1].grid(alpha=0.25)

    fig.suptitle(f"Analisis de Escalabilidad del Clustering frente al Limite de {presupuesto_gb:.0f} GB de RAM", fontsize=11.3, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "clustering_presupuesto_24gb.png")
    return fig


def fig_escala_1m_comparativa(comp_escala: pd.DataFrame, perfil_1m: pd.DataFrame) -> plt.Figure:
    """Panel A: silueta, tiempo y RAM de 10k vs 100k vs 1M. Panel B: perfiles descubiertos a 1M."""
    fig, ejes = plt.subplots(1, 2, figsize=(12.8, 4.5))

    etiquetas_x = [
        f"{n/1e6:.1f}M" if n >= 1_000_000 else f"{n/1e3:.0f}k"
        for n in comp_escala["n_transacciones"]
    ]
    xs = np.arange(len(etiquetas_x))
    ejes[0].bar(xs - 0.18, comp_escala["silueta_muestral"], 0.34, color="#117a65", label="Silueta muestral (0-1)")
    for i, r in comp_escala.iterrows():
        ejes[0].text(i - 0.18, r["silueta_muestral"] + 0.015, f"{r['silueta_muestral']:.3f}", ha="center", fontsize=8)
    ejes[0].set_ylim(0, max(comp_escala["silueta_muestral"].max() * 1.32, 0.7))
    ejes[0].set_xticks(xs)
    ejes[0].set_xticklabels(etiquetas_x)
    ejes[0].set_xlabel("Volumen de transacciones F1 procesadas (streaming)")
    ejes[0].set_ylabel("Coeficiente de silueta")
    ejes[0].set_title("A. Escalamiento MiniBatchKMeans: calidad vs. RAM")
    ejes[0].grid(axis="y", alpha=0.25)

    ax2 = ejes[0].twinx()
    ax2.plot(xs + 0.18, comp_escala["pico_ram_mb"], marker="o", lw=2, color="#c0392b", label="Pico RAM (MB)")
    for i, r in comp_escala.iterrows():
        ax2.text(i + 0.18, r["pico_ram_mb"] + 3, f"{r['pico_ram_mb']:.0f} MB\n({r['tiempo_s']:.1f}s)", ha="center", fontsize=7.5, color="#922b21")
    ax2.set_ylabel("Pico RAM Python (MB)", color="#c0392b")
    ax2.set_ylim(0, max(comp_escala["pico_ram_mb"].max() * 1.35, 80))

    # Leyenda combinada
    h1, l1 = ejes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ejes[0].legend(h1 + h2, l1 + l2, fontsize=7.8, loc="upper left")

    # Panel B: perfiles de precio medio y cuota de mercado en la escala final
    colores = ["#2e86c1", "#117a65", "#e67e22", "#c0392b"]
    y_pos = np.arange(len(perfil_1m))
    ejes[1].barh(y_pos, perfil_1m["precio_medio_gbp"] / 1000.0, color=colores[: len(perfil_1m)], edgecolor="white")
    ejes[1].set_yticks(y_pos)
    ejes[1].set_yticklabels([f"C{r['cluster']} · {r['segmento_f1'][:34]}" for _, r in perfil_1m.iterrows()], fontsize=7.8)
    for i, r in perfil_1m.iterrows():
        ejes[1].text(
            r["precio_medio_gbp"] / 1000.0 + 6,
            i,
            f"GBP {r['precio_medio_gbp']/1000:.0f}k ({r['pct_mercado']:.1f}%)",
            va="center",
            fontsize=7.8,
        )
    ejes[1].set_xlim(0, perfil_1m["precio_medio_gbp"].max() / 1000.0 * 1.35)
    ejes[1].set_xlabel("Precio medio de transaccion (miles GBP)")
    n_tot = int(perfil_1m["n_transacciones"].sum())
    ejes[1].set_title(f"B. Segmentos reales descubiertos (n={n_tot:,} ventas F1)")
    ejes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Clustering a gran escala por lotes (PyArrow + MiniBatchKMeans float32)", fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "clustering_escala_1m_comparativa.png")
    return fig


# ------------------------------------------------------------------ figuras
def fig_codo_y_silueta(tabla_k: pd.DataFrame, k_elegido: int, titulo: str, archivo: str) -> plt.Figure:
    """Curvas gemelas: metodo del codo (inercia) y coeficiente de silueta."""
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.1))

    ejes[0].plot(tabla_k["k"], tabla_k["inercia"], marker="o", lw=2, color="#2e86c1")
    ejes[0].axvline(k_elegido, color="#c0392b", ls="--", lw=1.6, label=f"k elegido = {k_elegido}")
    ejes[0].set_xlabel("Numero de clusters (k)")
    ejes[0].set_ylabel("Inercia (suma de distancias^2)")
    ejes[0].set_title("A. Metodo del codo (Inercia)")
    ejes[0].legend(fontsize=8.5)
    ejes[0].grid(alpha=0.25)

    sub = tabla_k.dropna(subset=["silueta"])
    ejes[1].plot(sub["k"], sub["silueta"], marker="s", lw=2, color="#117a65")
    ejes[1].axvline(k_elegido, color="#c0392b", ls="--", lw=1.6, label=f"k elegido = {k_elegido}")
    for _, r in sub.iterrows():
        ejes[1].text(r["k"], r["silueta"] + 0.01, f"{r['silueta']:.3f}", ha="center", fontsize=8)
    ejes[1].set_xlabel("Numero de clusters (k)")
    ejes[1].set_ylabel("Coeficiente de silueta promedio")
    ejes[1].set_title("B. Coeficiente de silueta (cohesion vs. separacion)")
    ejes[1].set_ylim(0, max(sub["silueta"].max() * 1.25, 0.5))
    ejes[1].legend(fontsize=8.5)
    ejes[1].grid(alpha=0.25)

    fig.suptitle(titulo, fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, archivo)
    return fig


def fig_pca_segmentos_regiones(df: pd.DataFrame, X_std: np.ndarray) -> plt.Figure:
    """Proyeccion PCA 2D de los segmentos K-Means (sin haber usado esfuerzo en X) + siluetas."""
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X_std)
    var = pca.explained_variance_ratio_ * 100.0
    colores = ["#2e86c1", "#e67e22", "#c0392b"]

    fig, ejes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    for cid, color in enumerate(colores):
        mask = df["cluster"].values == cid
        nombre = df.loc[mask, "segmento_nombre"].iloc[0]
        ejes[0].scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=55,
            alpha=0.85,
            edgecolors="white",
            label=f"C{cid}: {nombre} (n={mask.sum()})",
        )
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ejes[0].scatter(cx, cy, c=color, marker="X", s=180, edgecolors="black", linewidth=1.2)
    # Etiquetar un punto representativo por region
    for reg, g in df.groupby("region", observed=True):
        idx = g.index[-1]
        ejes[0].annotate(reg, (coords[idx, 0], coords[idx, 1]), fontsize=7.2, xytext=(4, 3), textcoords="offset points")
    ejes[0].set_xlabel(f"PC1 ({var[0]:.1f} % varianza)")
    ejes[0].set_ylabel(f"PC2 ({var[1]:.1f} % varianza)")
    ejes[0].set_title("A. Proyeccion PCA 2D de los clusters K-Means (k=3)")
    ejes[0].legend(fontsize=7.3, loc="best")
    ejes[0].grid(alpha=0.25)

    # Panel B: diagrama de siluetas individuales
    sils = silhouette_samples(X_std, df["cluster"].values)
    sil_avg = float(sils.mean())
    y_lower = 2
    for cid, color in enumerate(colores):
        s_c = np.sort(sils[df["cluster"].values == cid])
        y_upper = y_lower + len(s_c)
        ejes[1].fill_betweenx(np.arange(y_lower, y_upper), 0, s_c, facecolor=color, edgecolor=color, alpha=0.8)
        ejes[1].text(-0.08, y_lower + 0.5 * len(s_c), f"C{cid}", fontsize=8.5, weight="bold")
        y_lower = y_upper + 3
    ejes[1].axvline(sil_avg, color="black", ls="--", lw=1.5, label=f"Silueta media = {sil_avg:.3f}")
    ejes[1].set_xlabel("Valor de silueta individual")
    ejes[1].set_ylabel("Observaciones agrupadas por cluster")
    ejes[1].set_yticks([])
    ejes[1].set_title("B. Cohesion interna por observacion")
    ejes[1].legend(fontsize=8.2)
    ejes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    _guardar_png(fig, "clustering_pca_segmentos_regiones.png")
    return fig


def fig_dendrograma(
    Z: np.ndarray,
    etiquetas: list[str] | None,
    k_corte: int,
    titulo: str,
    archivo: str,
) -> plt.Figure:
    """Dendrograma jerarquico aglomerativo (linkage Ward) con linea de corte horizontal."""
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    # Altura de corte a mitad entre las fusiones (n - k) y (n - k + 1)
    h_corte = 0.5 * (Z[-k_corte, 2] + Z[-(k_corte - 1), 2]) if k_corte >= 2 else Z[-1, 2]
    dendrogram(
        Z,
        labels=etiquetas,
        leaf_rotation=90,
        leaf_font_size=6.8 if etiquetas is not None else 7.5,
        color_threshold=h_corte,
        ax=ax,
    )
    ax.axhline(h_corte, color="#c0392b", ls="--", lw=1.6, label=f"Corte horizontal (k={k_corte}, h={h_corte:.2f})")
    ax.set_ylabel("Distancia de fusion (Ward)")
    ax.set_title(titulo, fontsize=10.5, weight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _guardar_png(fig, archivo)
    return fig


def fig_perfil_segmentos(perfil: pd.DataFrame) -> plt.Figure:
    """Compara el esfuerzo hipotecario post-hoc y precio mediano real por segmento descubierto."""
    fig, ejes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    colores = ["#2e86c1", "#e67e22", "#c0392b"]
    etiquetas = [f"C{r['cluster']}\n({r['n_obs']} obs.)" for _, r in perfil.iterrows()]

    ejes[0].bar(etiquetas, perfil["esfuerzo_medio_posthoc"], color=colores, width=0.52)
    ejes[0].axhline(0.30, color="#1e8449", ls="--", lw=1.4, label="Umbral teorico 0,30")
    for i, r in perfil.iterrows():
        ejes[0].text(i, r["esfuerzo_medio_posthoc"] + 0.025, f"{r['esfuerzo_medio_posthoc']:.3f}\n({r['pct_esfuerzo_alto']:.0f}% Alto)", ha="center", fontsize=8)
    ejes[0].set_ylim(0, 1.22)
    ejes[0].set_ylabel("Esfuerzo medio (validacion post-hoc)")
    ejes[0].set_title("A. Validacion externa: esfuerzo real sin haberlo usado en X")
    ejes[0].legend(fontsize=8)
    ejes[0].grid(axis="y", alpha=0.25)

    ejes[1].bar(etiquetas, perfil["precio_mediana_gbp"] / 1000.0, color=colores, width=0.52)
    for i, r in perfil.iterrows():
        ejes[1].text(i, r["precio_mediana_gbp"] / 1000.0 + 12, f"GBP {r['precio_mediana_gbp']/1000:.0f}k\n({r['anos_salario_medio']:.1f} anios)", ha="center", fontsize=8)
    ejes[1].set_ylim(0, perfil["precio_mediana_gbp"].max() / 1000.0 * 1.25)
    ejes[1].set_ylabel("Precio mediano promedio (miles GBP)")
    ejes[1].set_title("B. Precio mediano y anios de salario por segmento")
    ejes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Perfilamiento en escala original de los segmentos territoriales K-Means", fontsize=11, weight="bold")
    fig.tight_layout()
    _guardar_png(fig, "clustering_perfil_segmentos.png")
    return fig


def main(modo: str | None = None, max_ram_gb: float | None = None) -> None:
    sistema_recursos.configurar_entorno_portable()
    modo_activo = (modo or config.MODO_EJECUCION).strip().lower()
    limites = config.obtener_limites_recursos(modo_activo)
    hw = sistema_recursos.info_sistema_ram()
    presupuesto_gb = float(max_ram_gb or limites.get("presupuesto_ram_sistema_gb") or 24.0)

    print("=" * 78)
    print(f"CLUSTERING NO SUPERVISADO (K-MEANS + WARD + STREAMING) · PERFIL: {modo_activo.upper()}")
    print(f"Sistema: {hw['sistema_operativo']} | CPUs: {hw['cpus']} | RAM equipo: {hw['ram_total_gb']} GB "
          f"(disponible: {hw['ram_disponible_gb']} GB) | Presupuesto techo: {presupuesto_gb:.0f} GB")
    print(f"Limites activos -> max_kmeans={limites['max_filas_kmeans']} | "
          f"max_f1={limites['max_filas_minibatch_f1']} | max_jerarquico={limites['max_filas_jerarquico']} | "
          f"alerta_ram={limites['pico_ram_mb_alerta']} MB")
    print("=" * 78)

    tracemalloc.start()
    t0 = time.perf_counter()

    reg = segmentar_regiones(modo_activo)
    print("\n[1] Segmentacion A · Mercados Regionales (n=60, sin esfuerzo en X):")
    print(f"    Contraste escalado (k=3): silueta SIN escalar = {reg['sil_crudo_en_std']:.3f} | "
          f"CON StandardScaler = {reg['sil_esc']:.3f} | ARI crudo vs. escalado = {reg['ari_crudo_vs_esc']:.3f}")
    print("\n    Metodo del codo e indice de silueta:")
    print(reg["tabla_k"].round(3).to_string(index=False))

    print("\n    Perfilamiento en escala ORIGINAL + validacion post-hoc del esfuerzo:")
    cols_p = ["cluster", "segmento_nombre", "n_obs", "precio_mediana_gbp", "salario_joven_gbp",
              "esfuerzo_medio_posthoc", "anos_salario_medio", "pct_esfuerzo_alto"]
    print(reg["perfil"][cols_p].round(2).to_string(index=False))
    print(f"\n    Jerarquico Ward (60 obs.): primeras 5 fusiones = {np.round(reg['Z'][:5, 2], 2)} | "
          f"ultima fusion = {reg['Z'][-1, 2]:.2f}")
    print(f"    Concordancia K-Means vs. Jerarquico Ward (ARI): {reg['ari_kmeans_ward']:.3f}")

    print("\n    Puente No Supervisado -> Supervisado (CV-5 sin fuga):")
    print(reg["puente_df"].round(3).to_string(index=False))

    hip = segmentar_hipotecas(modo_activo)
    print(f"\n[2] Segmentacion B · Catalogo Hipotecario (usado K-Means: {hip['n_usado_kmeans']:,} / "
          f"{hip['n_total_disponible']:,} | muestra Ward: {hip['n_usado_jerarquico']}):")
    print(hip["tabla_k"].round(3).to_string(index=False))
    print("\n    Perfil de segmentos hipotecarios (escala original):")
    print(hip["perfil_hip"].round(2).to_string(index=False))
    print("\n    Impacto en RAM de acotar el clustering jerarquico O(n^2):")
    print(hip["ahorro_ram"].to_string(index=False))

    # Exportar tablas livianas A y B
    config.DATA_INTEGRADO.mkdir(parents=True, exist_ok=True)
    cols_exp_reg = [
        "anio", "region", "cluster", "segmento_nombre", "accion_politica",
        "precio_mediana", "salario_joven_region", "n_transacciones", "esfuerzo", "anos_salario",
    ]
    reg["df"][cols_exp_reg].to_csv(config.RUTA_SEGMENTOS_TERRITORIALES, index=False)
    hip["perfil_hip"].to_csv(config.RUTA_SEGMENTOS_HIPOTECAS, index=False)

    # Cerrar telemetria de A+B antes de iniciar telemetria propia de Segmentacion C (1M+)
    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pico_mb = pico_bytes / (1024.0**2)
    duracion_ab = time.perf_counter() - t0

    # Segmentacion C: Gran escala F1 en streaming (1M en local / 11,37M en local_24gb & cloud)
    f1_res = segmentar_transacciones_gran_escala(modo_activo, presupuesto_ram_gb=presupuesto_gb)
    print(f"\n[3] Segmentacion C · Gran Escala F1 en Streaming "
          f"(procesado: {f1_res['n_final']:,} / {f1_res['n_disponible_f1']:,} transacciones):")
    print("    Experimento comparativo de escala (con RAM Python + RSS real del proceso):")
    print(f1_res["comparativa_escala"].to_string(index=False))
    print(f"\n    Capacidad de algoritmos frente al limite de {presupuesto_gb:.0f} GB de RAM:")
    print(f1_res["capacidad_ram_24gb"].to_string(index=False))
    print("\n    Perfiles de mercado descubiertos en escala final (escala original):")
    print(f1_res["perfil_1m"].to_string(index=False))
    f1_res["perfil_1m"].to_csv(config.RUTA_SEGMENTOS_TRANSACCIONES_1M, index=False, encoding="utf-8")
    if modo_activo in ("local_24gb", "cloud") or not config.RUTA_CLUSTERING_24GB.exists():
        f1_res["comparativa_escala"].to_csv(config.RUTA_CLUSTERING_24GB, index=False, encoding="utf-8")

    # Figuras (8 PNG)
    etiquetas_reg = [f"{r['region'][:12]}-{int(r['anio'])%100:02d}" for _, r in reg["df"].iterrows()]
    for fig in (
        fig_codo_y_silueta(reg["tabla_k"], reg["k_elegido"], "Eleccion de k · Mercados Regionales de Vivienda (n=60)", "clustering_codo_silueta_regiones.png"),
        fig_pca_segmentos_regiones(reg["df"], reg["X_std"]),
        fig_dendrograma(reg["Z"], etiquetas_reg, reg["k_elegido"], "Dendrograma Jerarquico Ward · 60 observaciones Region-Anio (UK)", "clustering_dendrograma_regiones.png"),
        fig_perfil_segmentos(reg["perfil"]),
        fig_codo_y_silueta(hip["tabla_k"], hip["k_elegido"], f"Eleccion de k · Catalogo Hipotecario (n={hip['n_usado_kmeans']:,}, perfil {modo_activo})", "clustering_codo_silueta_hipotecas.png"),
        fig_dendrograma(hip["Z_hip"], None, hip["k_elegido"], f"Dendrograma Ward · Muestra estratificada de {hip['n_usado_jerarquico']} productos hipotecarios", "clustering_dendrograma_hipotecas_muestra60.png"),
        fig_escala_1m_comparativa(f1_res["comparativa_escala"], f1_res["perfil_1m"]),
        fig_presupuesto_ram_24gb(f1_res["comparativa_escala"], presupuesto_gb=presupuesto_gb),
    ):
        plt.close(fig)

    pico_max_mb = max(pico_mb, float(f1_res["pico_ram_final_mb"]))
    rss_final_mb = float(f1_res["rss_final_mb"])
    duracion_total = time.perf_counter() - t0
    print(f"\n[4] Telemetria de recursos ({modo_activo}, techo {presupuesto_gb:.0f} GB):")
    print(f"    - Segmentaciones A+B : {duracion_ab:.2f} s | pico RAM Python = {pico_mb:.2f} MB")
    print(f"    - Segmentacion C F1  : {f1_res['tiempo_final_s']:.2f} s | pico RAM Python = {f1_res['pico_ram_final_mb']:.2f} MB | RSS SO = {rss_final_mb:.2f} MB ({100*rss_final_mb/(presupuesto_gb*1024):.2f} % de {presupuesto_gb:.0f} GB)")
    print(f"    - Total              : {duracion_total:.2f} s | pico maximo Python = {pico_max_mb:.2f} MB")
    alerta_mb = limites["pico_ram_mb_alerta"]
    if alerta_mb is not None:
        assert pico_max_mb < alerta_mb, f"Pico de RAM ({pico_max_mb:.1f} MB) excedio el limite ({alerta_mb} MB)"
        assert rss_final_mb < presupuesto_gb * 1024.0, f"RSS ({rss_final_mb:.1f} MB) excedio el techo ({presupuesto_gb} GB)"
        print(f"    -> assert OK: consumo acotado (Python {pico_max_mb:.2f} MB < {alerta_mb} MB | RSS {rss_final_mb:.1f} MB << {presupuesto_gb:.0f} GB) con {f1_res['n_final']:,} filas.")
    print("[5] Tablas livianas y 8 figuras guardadas en data/integrado/ y md/figuras/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering K-Means + Jerarquico + Streaming con perfil local / local_24gb / cloud")
    parser.add_argument("--modo", choices=["local", "local_24gb", "cloud"], default=None, help="Perfil de limites de recursos")
    parser.add_argument("--max-ram-gb", type=float, default=None, help="Presupuesto maximo de RAM del equipo en GB (ej. 24)")
    args = parser.parse_args()
    main(modo=args.modo, max_ram_gb=args.max_ram_gb)
