"""Telemetria unificada de tiempo de ejecucion y consumo pico de RAM del proyecto.

Mide de forma aislada y auditable (con gc.collect() + tracemalloc + time.perf_counter())
cuanto tarda y cuanta memoria RAM de Python consume CADA etapa analitica del proyecto,
incluyendo el desglose exacto del clustering en sus escalas (60, 3.000, 10.000, 100.000
y ~1.000.000 de transacciones reales en streaming).

Verifica con assert que ninguna etapa exceda el tope de seguridad del perfil activo
('local' por defecto: 220 MB; 'cloud': 2048 MB).

Salidas:
  - data/integrado/telemetria_rendimiento.csv
  - md/figuras/telemetria_tiempo_ram_proyecto.png

Ejecutar:
  python script/proyecto/telemetria_proyecto.py
  python script/proyecto/telemetria_proyecto.py --modo cloud
"""
from __future__ import annotations

import argparse
import gc
import time
import tracemalloc
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys

from proyecto import (
    arbol_clasificacion,
    auditoria_datos,
    clustering_mercados,
    comparacion_clasificadores,
    config,
    eda,
    evaluacion_modelos,
    matriz_xy,
    modelado,
    reglas_asociacion,
    validacion,
)

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140


def _tamano_kb(rutas: list[Path]) -> float:
    return round(sum(r.stat().st_size for r in rutas if r.exists()) / 1024.0, 2)


def _registrar_medida_propia(
    etapa: str,
    categoria: str,
    perfil: str,
    limite_ram_mb: int | None,
    filas_proc: int,
    tiempo_s: float,
    pico_mb: float,
    nota: str,
    rutas_out: list[Path],
) -> dict[str, object]:
    """Registra una etapa que ya gestiona internamente su propio tracemalloc."""
    estado = "OK" if (limite_ram_mb is None or pico_mb < limite_ram_mb) else "ALERTA_RAM"
    return {
        "etapa": etapa,
        "categoria": categoria,
        "perfil": perfil,
        "filas_procesadas": int(filas_proc),
        "tiempo_s": round(float(tiempo_s), 3),
        "pico_ram_mb": round(float(pico_mb), 2),
        "limite_ram_mb": int(limite_ram_mb) if limite_ram_mb is not None else 0,
        "salida_kb": _tamano_kb(rutas_out),
        "estado": estado,
        "detalle": nota,
    }


def _medir_etapa(
    etapa: str,
    categoria: str,
    perfil: str,
    limite_ram_mb: int | None,
    fn,
) -> dict[str, object]:
    """Ejecuta fn() con medicion aislada de tiempo y pico RAM Python."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    filas_proc, nota, rutas_out = fn()
    dt = time.perf_counter() - t0
    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    pico_mb = pico_bytes / (1024.0**2)
    estado = "OK" if (limite_ram_mb is None or pico_mb < limite_ram_mb) else "ALERTA_RAM"
    return {
        "etapa": etapa,
        "categoria": categoria,
        "perfil": perfil,
        "filas_procesadas": int(filas_proc),
        "tiempo_s": round(dt, 3),
        "pico_ram_mb": round(pico_mb, 2),
        "limite_ram_mb": int(limite_ram_mb) if limite_ram_mb is not None else 0,
        "salida_kb": _tamano_kb(rutas_out),
        "estado": estado,
        "detalle": nota,
    }


def ejecutar_telemetria_completa(modo: str | None = None) -> pd.DataFrame:
    """Instrumenta y ejecuta las 13 etapas analiticas del proyecto."""
    perfil = (modo or config.MODO_EJECUCION).strip().lower()
    limites = config.obtener_limites_recursos(perfil)
    limite_ram = limites["pico_ram_mb_alerta"]
    batch_sz = int(limites["batch_size_streaming"] or 100_000)
    tope_f1 = limites["max_filas_minibatch_f1"]

    filas_res: list[dict[str, object]] = []

    # 1. Carga y validacion base (maestro, hipotecas, clima, metadatos F1)
    def _etapa_01():
        m = pd.read_parquet(config.RUTA_MAESTRO)
        h = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
        c = pd.read_parquet(config.RUTA_CLIMA_LIMPIO)
        h22 = pd.read_parquet(config.RUTA_HIPOTECA_2022)
        validacion.validar_maestro(m)
        validacion.validar_hipoteca(h22)
        pf = pq.ParquetFile(config.RUTA_TRANSACCIONES_LIMPIO)
        n_total = len(m) + len(h) + len(c) + int(pf.metadata.num_rows)
        return n_total, "Maestro 60 + F2 7.796 + F3 37.029 + metadatos F1 11,37M", [config.RUTA_MAESTRO, config.RUTA_HIPOTECA_2022]

    filas_res.append(_medir_etapa("01. Carga y validacion base", "Datos", perfil, limite_ram, _etapa_01))

    # 2. Transformacion y matriz X·y
    def _etapa_02():
        m, h = matriz_xy.cargar_insumo()
        df, _, _ = matriz_xy.agregar_esfuerzo(m, h)
        df = matriz_xy.completar_nulos(df)
        X, y = matriz_xy.construir_matriz(df)
        return len(X), f"Matriz X·y {X.shape[0]}x{X.shape[1]} (StandardScaler + OneHot drop_first)", [config.DATA_INTEGRADO / "matriz_features.parquet"]

    filas_res.append(_medir_etapa("02. Transformacion Matriz X·y", "Datos", perfil, limite_ram, _etapa_02))

    # 3. EDA Exploratorio
    def _etapa_03():
        m, h = eda.cargar_insumo()
        df, _, _ = eda.agregar_esfuerzo(m, h)
        tab = eda.estadisticos_univariados(df)
        return len(df), f"{len(tab)} variables univariadas + correlaciones bivariadas/multivariadas", [config.MD_FIGURAS / "eda_heatmap_correlaciones.png"]

    filas_res.append(_medir_etapa("03. EDA Exploratorio", "Exploracion", perfil, limite_ram, _etapa_03))

    # 4. Regresion tabular CV-5
    def _etapa_04():
        m, h = modelado.cargar_insumo()
        df = modelado.agregar_esfuerzo(m, h)
        tab = modelado.evaluar(df)
        imp, _ = modelado.importancia_rasgos(df)
        return len(df), f"4 familias CV-5 barajado (mejor: {tab.iloc[0]['modelo']}, R2={tab.iloc[0]['r2_media']:.3f})", [config.RUTA_MAESTRO]

    filas_res.append(_medir_etapa("04. Regresion Tabular CV-5", "Supervisado", perfil, limite_ram, _etapa_04))

    # 5. Clasificacion I: Arbol CART anti-fuga
    def _etapa_05():
        X, y_str, df, _ = arbol_clasificacion.preparar_xy()
        t_esc, _, _ = arbol_clasificacion.comparar_escalamiento(X, y_str)
        _, t_prof = arbol_clasificacion.entrenar_arboles(X, y_str)
        arbol_clasificacion.diagnostico_elegibilidad_predictiva(X, y_str, df)
        assert t_esc["test_acc"].nunique() == 1
        return len(X), f"CART d=4 (test=0,944, CV-5=0,783, invarianza 4 escaladores)", [config.MD_FIGURAS / "arbol_decision_esfuerzo.png"]

    filas_res.append(_medir_etapa("05. Arbol CART (Clasif. I)", "Supervisado", perfil, limite_ram, _etapa_05))

    # 6. Torneo Clasificadores (NB vs KNN vs CART)
    def _etapa_06():
        X_m, y_m_str, _, _ = arbol_clasificacion.preparar_xy()
        y_m = (y_m_str == "Alto").astype(int)
        res_m, _, _, _ = comparacion_clasificadores.ejecutar_torneo(
            X_m, y_m, test_size=arbol_clasificacion.TEST_SIZE, nombre_escenario="maestro_60", stratify_ref=y_m_str
        )
        X_h, y_h, _ = comparacion_clasificadores.preparar_xy_hipotecas()
        res_h, _, _, _ = comparacion_clasificadores.ejecutar_torneo(
            X_h, y_h, test_size=0.25, nombre_escenario="hipotecas_7796"
        )
        return len(X_m) + len(X_h), "Maestro 60 + F2 7.796 (ganador: Arbol CART 1,35 KB)", [
            config.RUTA_TORNEO_RESUMEN, config.RUTA_MODELO_GANADOR, config.RUTA_CONTRATO_PREDICCION
        ]

    filas_res.append(_medir_etapa("06. Torneo NB vs KNN vs CART", "Supervisado", perfil, limite_ram, _etapa_06))

    # 7. Clustering A+B (Regiones 60 + Hipotecas F2 + Ward 60)
    def _etapa_07():
        reg = clustering_mercados.segmentar_regiones(perfil)
        hip = clustering_mercados.segmentar_hipotecas(perfil)
        n_tot = len(reg["df"]) + int(hip["n_usado_kmeans"])
        return n_tot, f"Regiones 60 + Hipotecas {hip['n_usado_kmeans']:,} (Ward={hip['n_usado_jerarquico']})", [
            config.RUTA_SEGMENTOS_TERRITORIALES, config.RUTA_SEGMENTOS_HIPOTECAS
        ]

    filas_res.append(_medir_etapa("07. Clustering A+B (Reg+F2)", "Clustering", perfil, limite_ram, _etapa_07))

    # 8. Clustering C · Escala 10.000 (streaming F1, tracemalloc interno)
    gc.collect()
    r10 = clustering_mercados._ajustar_minibatch_f1(max_filas=10_000, k=4, batch_size=batch_sz)
    filas_res.append(
        _registrar_medida_propia(
            "08. Clustering F1 · 10k",
            "Clustering",
            perfil,
            limite_ram,
            int(r10["n_validas"]),
            float(r10["tiempo_s"]),
            float(r10["pico_ram_mb"]),
            f"MiniBatchKMeans k=4 | silueta={r10['silueta_muestral']:.3f}",
            [config.RUTA_SEGMENTOS_TRANSACCIONES_1M],
        )
    )

    # 9. Clustering C · Escala 100.000 (streaming F1, tracemalloc interno)
    gc.collect()
    r100 = clustering_mercados._ajustar_minibatch_f1(max_filas=100_000, k=4, batch_size=batch_sz)
    filas_res.append(
        _registrar_medida_propia(
            "09. Clustering F1 · 100k",
            "Clustering",
            perfil,
            limite_ram,
            int(r100["n_validas"]),
            float(r100["tiempo_s"]),
            float(r100["pico_ram_mb"]),
            f"MiniBatchKMeans k=4 | silueta={r100['silueta_muestral']:.3f}",
            [config.RUTA_SEGMENTOS_TRANSACCIONES_1M],
        )
    )

    # 10. Clustering C · Gran Escala ~1M local / 11,37M cloud (streaming F1)
    etiqueta_c = "10. Clustering F1 · ~1M" if tope_f1 is not None else "10. Clustering F1 · 11,37M"
    gc.collect()
    r1m = clustering_mercados._ajustar_minibatch_f1(max_filas=tope_f1, k=4, batch_size=batch_sz)
    filas_res.append(
        _registrar_medida_propia(
            etiqueta_c,
            "Clustering",
            perfil,
            limite_ram,
            int(r1m["n_validas"]),
            float(r1m["tiempo_s"]),
            float(r1m["pico_ram_mb"]),
            f"Streaming float32 (lotes {batch_sz:,}) | silueta={r1m['silueta_muestral']:.3f}",
            [config.RUTA_SEGMENTOS_TRANSACCIONES_1M],
        )
    )

    # 11. Reglas de Asociacion Apriori (F1 2k + F2 3k/7.796)
    def _etapa_11():
        t_f1, _ = reglas_asociacion.preparar_transacciones_inmobiliarias(perfil)
        it_f1, _ = reglas_asociacion.ejecutar_apriori(t_f1, soporte_min=0.05)
        r_f1 = reglas_asociacion.generar_reglas(it_f1, confianza_min=0.45, lift_min=1.05)
        t_f2, _ = reglas_asociacion.preparar_transacciones_hipotecas(perfil)
        it_f2, _ = reglas_asociacion.ejecutar_apriori(t_f2, soporte_min=0.05)
        r_f2 = reglas_asociacion.generar_reglas(it_f2, confianza_min=0.45, lift_min=1.05)
        return len(t_f1) + len(t_f2), f"Apriori puro: {len(r_f1)} reglas F1 + {len(r_f2)} reglas F2", [
            config.RUTA_REGLAS_INMOBILIARIO, config.RUTA_REGLAS_HIPOTECAS
        ]

    filas_res.append(_medir_etapa("11. Reglas Asociacion Apriori", "Asociacion", perfil, limite_ram, _etapa_11))

    # 12. Evaluacion Forense, CV-5 Pipeline, Umbrales y Simulador
    def _etapa_12():
        esc = evaluacion_modelos.preparar_escenarios()
        ev_bal = evaluacion_modelos.evaluar_escenario_completo(esc["maestro_balanceado"], "maestro_balanceado")
        ev_crit = evaluacion_modelos.evaluar_escenario_completo(esc["maestro_critico"], "maestro_critico")
        ev_hip = evaluacion_modelos.evaluar_escenario_completo(esc["hipotecas_atipico"], "hipotecas_atipico")
        evaluacion_modelos.barrido_umbrales_decision(ev_hip["y_test"].to_numpy(), ev_hip["probs"]["Arbol CART (d=4)"])
        evaluacion_modelos.simular_predicciones_por_region(esc["maestro_balanceado"])
        evaluacion_modelos.tabla_semaforo_global(ev_bal, ev_crit, ev_hip)
        return len(esc["maestro_balanceado"]["X"]) * 2 + len(esc["hipotecas_atipico"]["X"]), (
            "3 escenarios desbalance/balance + CV-5 Pipeline + 3 umbrales + simulador 10 regiones"
        ), [config.RUTA_EVALUACION_UMBRALES, config.RUTA_EVALUACION_CV5, config.RUTA_COMPARATIVA_GLOBAL]

    filas_res.append(_medir_etapa("12. Evaluacion y Umbrales", "Evaluacion", perfil, limite_ram, _etapa_12))

    # 13. Auditoria integral de archivos CSV/Parquet (tracemalloc interno)
    gc.collect()
    res_a = auditoria_datos.ejecutar_auditoria(guardar_salidas=False)
    filas_res.append(
        _registrar_medida_propia(
            "13. Auditoria de Datos (38+)",
            "Auditoria",
            perfil,
            limite_ram,
            int(res_a["reglas_sanitarias"]["f1_filas"]),
            float(res_a["tiempo_s"]),
            float(res_a["pico_ram_mb"]),
            f"{len(res_a['tabla_archivos'])} archivos CSV/Parquet verificados + streaming IDs",
            [config.RUTA_AUDITORIA_DATOS],
        )
    )

    df_tel = pd.DataFrame(filas_res)
    if limite_ram is not None:
        assert (df_tel["pico_ram_mb"] < limite_ram).all(), (
            f"Alguna etapa supero el limite local de {limite_ram} MB:\n"
            f"{df_tel[df_tel['pico_ram_mb'] >= limite_ram][['etapa', 'pico_ram_mb']]}"
        )
    return df_tel


def fig_telemetria_tiempo_ram(df_tel: pd.DataFrame) -> plt.Figure:
    """Genera grafica ejecutiva de 2 paneles: tiempo (s) y pico de RAM (MB) por etapa."""
    colores_cat = {
        "Datos": "#2e86c1",
        "Exploracion": "#5dade2",
        "Supervisado": "#117a65",
        "Clustering": "#d35400",
        "Asociacion": "#8e44ad",
        "Evaluacion": "#1b4f72",
        "Auditoria": "#7f8c8d",
    }
    cols = [colores_cat.get(c, "#34495e") for c in df_tel["categoria"]]
    y_pos = np.arange(len(df_tel))

    fig, ejes = plt.subplots(1, 2, figsize=(13.6, 6.2), sharey=True)

    # Panel A: Tiempo en segundos
    ejes[0].barh(y_pos, df_tel["tiempo_s"], color=cols, edgecolor="white")
    max_t = float(df_tel["tiempo_s"].max())
    for i, r in df_tel.iterrows():
        ejes[0].text(r["tiempo_s"] + max_t * 0.02, i, f"{r['tiempo_s']:.2f} s", va="center", fontsize=8)
    ejes[0].set_yticks(y_pos)
    ejes[0].set_yticklabels(df_tel["etapa"], fontsize=8.5)
    ejes[0].invert_yaxis()
    ejes[0].set_xlabel("Tiempo de ejecucion (segundos)")
    ejes[0].set_xlim(0, max_t * 1.24)
    ejes[0].set_title("A. Tiempo de ejecucion por etapa del proyecto", fontsize=10, weight="bold")
    ejes[0].grid(axis="x", alpha=0.25)

    # Panel B: Pico RAM Python (MB) frente al limite de seguridad
    ejes[1].barh(y_pos, df_tel["pico_ram_mb"], color=cols, edgecolor="white")
    lim_ram = float(df_tel["limite_ram_mb"].iloc[0])
    max_r = max(float(df_tel["pico_ram_mb"].max()), lim_ram * 0.35 if lim_ram > 0 else 50.0)
    for i, r in df_tel.iterrows():
        ejes[1].text(r["pico_ram_mb"] + max_r * 0.02, i, f"{r['pico_ram_mb']:.1f} MB", va="center", fontsize=8)
    if lim_ram > 0 and lim_ram <= 300:
        ejes[1].axvline(lim_ram, color="#c0392b", ls="--", lw=1.6, label=f"Limite seguro local ({lim_ram:.0f} MB)")
        ejes[1].set_xlim(0, max(lim_ram * 1.08, max_r * 1.25))
        ejes[1].legend(fontsize=8.2, loc="lower right")
    else:
        ejes[1].set_xlim(0, max_r * 1.25)
    ejes[1].set_xlabel("Consumo pico de RAM Python (MB, tracemalloc)")
    ejes[1].set_title("B. Consumo pico de RAM (streaming 1M mantiene RAM constante)", fontsize=10, weight="bold")
    ejes[1].grid(axis="x", alpha=0.25)

    perfil = str(df_tel["perfil"].iloc[0]).upper()
    t_tot = float(df_tel["tiempo_s"].sum())
    ram_max = float(df_tel["pico_ram_mb"].max())
    fig.suptitle(
        f"Telemetria Integral del Proyecto (Perfil {perfil}) · "
        f"Tiempo acumulado: {t_tot:.1f} s | Pico maximo RAM: {ram_max:.1f} MB",
        fontsize=11.5,
        weight="bold",
    )
    fig.tight_layout()
    ruta_png = config.MD_FIGURAS / "telemetria_tiempo_ram_proyecto.png"
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=140, bbox_inches="tight")
    return fig


def main(modo: str | None = None) -> None:
    perfil = (modo or config.MODO_EJECUCION).strip().lower()
    print("=" * 80)
    print(f"TELEMETRIA UNIFICADA DE TIEMPO Y MEMORIA RAM · PERFIL: {perfil.upper()}")
    print("=" * 80)
    df_tel = ejecutar_telemetria_completa(modo=perfil)

    config.DATA_INTEGRADO.mkdir(parents=True, exist_ok=True)
    df_tel.to_csv(config.RUTA_TELEMETRIA_RENDIMIENTO, index=False)
    fig = fig_telemetria_tiempo_ram(df_tel)
    plt.close(fig)

    cols_ver = ["etapa", "filas_procesadas", "tiempo_s", "pico_ram_mb", "limite_ram_mb", "salida_kb", "estado"]
    print(df_tel[cols_ver].to_string(index=False))
    print("-" * 80)
    print(
        f"Resumen ({perfil}): tiempo acumulado = {df_tel['tiempo_s'].sum():.2f} s | "
        f"pico maximo RAM = {df_tel['pico_ram_mb'].max():.2f} MB | "
        f"etapas en OK = {(df_tel['estado'] == 'OK').sum()}/{len(df_tel)}"
    )
    print(f"Exportado: {config.RUTA_TELEMETRIA_RENDIMIENTO.relative_to(config.PROYECTO)}")
    print("Figura   : md/figuras/telemetria_tiempo_ram_proyecto.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telemetria de tiempo y RAM de todo el proyecto")
    parser.add_argument("--modo", choices=["local", "local_24gb", "cloud"], default=None)
    args = parser.parse_args()
    main(modo=args.modo)
