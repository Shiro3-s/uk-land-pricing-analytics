"""Script Maestro Ejecutable del Proyecto (espejo 1:1 de Cuaderno_Maestro_Proyecto_UK.ipynb).

Orquesta y presenta en consola, paso a paso y con explicacion clara de cada tabla,
todo el ciclo analitico KDD + CRISP-DM (Fases 1 a 11) del proyecto:
  "Accesibilidad de vivienda para jovenes (18-34) en el Reino Unido"

Compatible 100 % con macOS, Linux y Windows (sin dependencias exclusivas de SO).

Bloques ejecutados:
  [1]  Entorno multiplataforma, hardware disponible y perfiles de RAM (local / local_24gb / cloud)
  [2]  Auditoria integral de TODOS los CSV ("Excel") y Parquet (38+ archivos, 0 duplicados)
  [3]  Calidad, limpieza (MCAR/MAR/MNAR, IQR Tukey vs z-score) y cuadro maestro integrado (60x16)
  [4]  Indicadores financieros (PMT 90% LTV, 25 anios), clima vs. precios, matriz X·y y EDA
  [5]  Los DOS Tipos de Prediccion · Tipo 1: Regresion Continua (sin fuga + tasa exacta F2)
  [6]  Los DOS Tipos de Prediccion · Tipo 2: Clasificacion (CART vs NB vs KNN + Ganador)
  [7]  Evaluacion Forense, Desbalance, 3 Umbrales y Tabla Comparativa Dual de Predicciones
  [8]  Clustering No Supervisado (Regiones + F2 + Gran Escala F1 frente al limite de 24 GB RAM)
  [9]  Reglas de Asociacion (Apriori puro F1 + F2 + comparativa FP-Growth)
  [10] Telemetria unificada de tiempo (s) y consumo de memoria RAM (MB)
  [11] Prediccion Dual en vivo (predecir.py) y capa lista para futura subida a BD + Front/Back

Ejecucion multiplataforma:
  Windows (PowerShell):  python main_maestro_completo.py
                         python main_maestro_completo.py --modo local_24gb --region London
  macOS / Linux (bash):  python3 main_maestro_completo.py
                         python3 main_maestro_completo.py --modo local_24gb --region London
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
RUTA_SCRIPT = str(RAIZ / "script")
if RUTA_SCRIPT not in sys.path:
    sys.path.insert(0, RUTA_SCRIPT)

from proyecto import (
    arbol_clasificacion,
    auditoria_datos,
    cargar,
    clustering_mercados,
    comparacion_clasificadores,
    config,
    eda,
    evaluacion_modelos,
    indicadores,
    modelado,
    predecir,
    prediccion_dual,
    reglas_asociacion,
    sistema_recursos,
    validacion,
)


def seccion(numero: int, titulo: str, explicacion: str) -> None:
    print("\n" + "=" * 84)
    print(f"BLOQUE {numero:02d} · {titulo}")
    print("-" * 84)
    print(f"Que hacemos y por que: {explicacion}")
    print("=" * 84)


def run_maestro(modo: str = "local", region_consulta: str = "London", recalcular_24gb: bool = False) -> None:
    ent = sistema_recursos.configurar_entorno_portable()
    hw = sistema_recursos.info_sistema_ram()
    limites = config.obtener_limites_recursos(modo)
    t_inicio = time.perf_counter()

    # ------------------------------------------------------------------ 01
    seccion(
        1,
        "ENTORNO MULTIPLATAFORMA, HARDWARE Y CONTROL DE RECURSOS",
        "Verificamos SO (macOS/Linux/Windows), RAM fisica disponible y perfil de proteccion "
        "('local' conservador, 'local_24gb' hasta 11,37 M transacciones dentro de 24 GB, o 'cloud').",
    )
    print(f"  Sistema operativo : {hw['sistema_operativo']} ({ent['python']}) | CPUs: {hw['cpus']}")
    print(f"  RAM fisica equipo : {hw['ram_total_gb']} GB total | {hw['ram_disponible_gb']} GB disponibles | RSS actual: {sistema_recursos.rss_proceso_mb():.1f} MB")
    print(f"  Backend graficas  : {ent['matplotlib_backend']} (seguro en Linux headless y notebooks)")
    print(f"  Perfil activo     : {modo.upper()} -> {limites}")

    print("\n  [Tabla 0 · Hasta que fase llegamos en KDD + CRISP-DM (Fases 1-11 completadas + Fase 12 lista en Guia Azure)]")
    fases_df = pd.DataFrame(
        [
            ("1-2", "Problema, DIKW, H1..H3 y 5 fuentes (11,37M filas)", "Seleccion", "Negocio + Datos", "Completada"),
            ("3-4", "Limpieza MCAR/MAR/MNAR, Tukey IQR, cruce (60x16) y PMT", "Preprocesamiento", "Preparacion", "Completada"),
            ("5-6", "Escalado, One-Hot (VIF=1,80), PCA y EDA (6 graficas + 3 derivadas)", "Transformacion", "Preparacion + EDA", "Completada"),
            ("7-9", "CON ETIQUETA: Tipo 1 Regresion (Ridge/RF) + Tipo 2 Clasificacion (CART 4,82/5)", "Mineria Supervisada", "Modelado", "Completada"),
            ("10", "SIN ETIQUETA: Tipo 3 Clustering (11,35M en 49,87 MB) + Tipo 4 Apriori", "Mineria No Sup.", "Modelado", "Completada"),
            ("11", "Evaluacion forense, 3 umbrales F2, Presentacion Interactiva + PPTX", "Evaluacion", "Evaluacion", "Completada"),
            ("12", "Subida 17 CSV ('Excel') + SQL a Azure PostgreSQL y enlace compartible", "Despliegue", "Despliegue", "Lista en Guia MD"),
        ],
        columns=["Fases", "Alcance", "Etapa KDD", "Fase CRISP-DM", "Estado"],
    )
    print(fases_df.to_string(index=False))

    # ------------------------------------------------------------------ 02
    seccion(
        2,
        "INVENTARIO Y AUDITORIA DE TODOS LOS ARCHIVOS CSV ('EXCEL') Y PARQUET",
        "Confirmamos que el 100 % de las fuentes reales (4 CSV en data/crudo/ + F1 completo 11,37M) "
        "se utilizan, que los 5 CSV auxiliares en db/ estan contenidos al 100 % en F1 limpio (sin duplicar), "
        "y que todas las tablas limpias e integradas abren sin error y con 0 duplicados.",
    )
    if config.RUTA_AUDITORIA_DATOS.exists():
        df_aud = pd.read_csv(config.RUTA_AUDITORIA_DATOS)
    else:
        df_aud = auditoria_datos.ejecutar_auditoria(guardar_salidas=True)["tabla_archivos"]
    resumen_aud = (
        df_aud.groupby(["carpeta", "estado_limpieza"])
        .agg(archivos=("archivo", "count"), filas_max=("filas", "max"), duplicados=("duplicados", "sum"))
        .reset_index()
    )
    print("  [Tabla 1 · Resumen de sanidad por carpeta del proyecto]")
    print(resumen_aud.to_string(index=False))

    print("\n  [Tabla 2 · Vista de las 5 fuentes CSV ('Excel') originales que alimentan el proyecto]")
    fuentes_info = [
        ("F1 Ventas UK (limpio)", config.RUTA_TRANSACCIONES_LIMPIO, "11.369.413 filas x 22 cols (origen: db/uk_property_price.csv)"),
        ("F2 Hipotecas 2022", config.RUTA_HIPOTECAS, f"{len(cargar.cargar_hipotecas()):,} productos x 13 cols"),
        ("F3 Clima Met Office", config.RUTA_CLIMA, f"{len(cargar.cargar_clima()):,} lecturas mensuales (36 estaciones)"),
        ("F4a Precios/Salario", config.RUTA_PRECIOS_SALARIO, f"{len(cargar.cargar_precios_salario())} anios reales (1975-2020)"),
        ("F4b Ingreso x Edad", config.RUTA_INGRESO_EDAD, f"{len(cargar.cargar_ingreso_edad())} bandas edad x genero"),
    ]
    print(pd.DataFrame(fuentes_info, columns=["Fuente", "Ruta", "Dimension"]).assign(
        Ruta=lambda d: d["Ruta"].apply(lambda r: sistema_recursos.ruta_posix(r, config.PROYECTO))
    ).to_string(index=False))

    # ------------------------------------------------------------------ 03
    seccion(
        3,
        "CUADRO MAESTRO INTEGRADO (60x16), INDICADORES PMT Y CLIMA VS. PRECIOS",
        "Cruzamos F1+F2+F3+F4 en la ventana comun 2015-2020 (10 regiones x 6 anios). Calculamos "
        "cuota hipotecaria PMT (90% LTV, 25 anios, tasa media winsorizada 6,04%) y contrastamos H3 (clima).",
    )
    maestro = pd.read_parquet(config.RUTA_MAESTRO)
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    hip22 = pd.read_parquet(config.RUTA_HIPOTECA_2022)
    validacion.validar_maestro(maestro)
    validacion.validar_hipoteca(hip22)
    print("  [Tabla 3 · Corte hipotecario regional 2022 — ninguna region cumple el umbral <= 30 %]")
    print(hip22[["region", "precio_real_2020", "cuota_mensual", "cuota_pct_salario", "accesible"]].round(3).to_string(index=False))

    print("\n  [Tabla 4 · Correlacion Pearson entre variacion anual del precio y anomalias climaticas (H3)]")
    print(indicadores.correlacion_clima_precio(maestro).round(3).to_string(index=False))

    # ------------------------------------------------------------------ 04
    seccion(
        4,
        "TRANSFORMACION, MATRIZ X·y Y ANALISIS EXPLORATORIO (EDA)",
        "Medimos tendencia central, dispersion, forma y outliers (Tukey IQR). Confirmamos sesgo positivo "
        "(mediana esfuerzo = 0,564 vs media = 0,625) y brecha territorial de 2,3x (North East vs. London).",
    )
    df_esf = modelado.agregar_esfuerzo(maestro, hipotecas)
    print("  [Tabla 5 · Estadisticos univariados del cuadro maestro y variable objetivo esfuerzo]")
    print(eda.estadisticos_univariados(df_esf)[["media", "mediana", "desv", "asimetria", "outliers_iqr"]].round(3).to_string())

    # ------------------------------------------------------------------ 05
    seccion(
        5,
        "LOS DOS TIPOS DE PREDICCION · TIPO 1: REGRESION CONTINUA (CUANTO?)",
        "Implementamos prediccion numerica continua SIN fuga aritmetica: (1A) esfuerzo hipotecario exacto (%) "
        "por region con Ridge CV-5, y (1B) tasa hipotecaria exacta (%) de cada producto F2 con RandomForestRegressor. "
        "Contrastamos honestamente contra la regresion contable (R2=1,000 cuando entra anos_salario).",
    )
    tab_reg_contable = modelado.evaluar(df_esf)
    print("  [Tabla 6A · Referencia contable (con anos_salario en X -> determinismo PMT, CV-5)]")
    print(tab_reg_contable[["modelo", "r2_media", "rmse_media", "mae_media"]].round(4).to_string(index=False))

    tab_esf_sf, reg_esf, _ = prediccion_dual.evaluar_regresion_maestro_sin_fuga()
    tab_f2_reg, reg_f2, clf_f2, meta_f2 = prediccion_dual.evaluar_regresion_hipotecas_f2()
    print("\n  [Tabla 6B · Regresion Continua SIN FUGA (Esfuerzo Regional + Tasa Exacta F2)]")
    print(pd.concat([tab_esf_sf, tab_f2_reg], ignore_index=True).to_string(index=False))

    # ------------------------------------------------------------------ 06
    seccion(
        6,
        "LOS DOS TIPOS DE PREDICCION · TIPO 2: CLASIFICACION (QUE NIVEL / ALERTA?)",
        "Enfrentamos Arbol CART (d=4), Naive Bayes (GaussianNB), KNN escalado (Pipeline k=5) y control "
        "negativo KNN sin escalar en Maestro (n=60) y F2 (n=7.796) para elegir el clasificador principal.",
    )
    X_m, y_m_str, _, _ = arbol_clasificacion.preparar_xy()
    y_m = (y_m_str == "Alto").astype(int)
    res_m, _, _, _ = comparacion_clasificadores.ejecutar_torneo(
        X_m, y_m, test_size=arbol_clasificacion.TEST_SIZE, nombre_escenario="maestro_60", stratify_ref=y_m_str
    )
    X_h, y_h, _ = comparacion_clasificadores.preparar_xy_hipotecas()
    res_h, _, _, _ = comparacion_clasificadores.ejecutar_torneo(
        X_h, y_h, test_size=0.25, nombre_escenario="hipotecas_7796"
    )
    cols_t = ["modelo", "acc_test", "cv5_acc", "f1_test", "roc_auc_test", "brier_score", "predict_ms", "peso_kb"]
    print("  [Tabla 7A · Torneo Clasificacion Nivel 1 (Cuadro Maestro n=60, sin fuga)]")
    print(res_m[cols_t].round(3).to_string(index=False))
    print("\n  [Tabla 7B · Torneo Clasificacion Nivel 2 (Catalogo Hipotecario F2 n=7.796, tasa >= Q3)]")
    print(res_h[cols_t].round(3).to_string(index=False))
    print("\n  [Tabla 7C · Veredicto Multicriterio del Torneo]")
    print(comparacion_clasificadores.matriz_decision_multicriterio(res_m, res_h).to_string(index=False))

    # ------------------------------------------------------------------ 07
    seccion(
        7,
        "EVALUACION FORENSE, DESBALANCE, 3 UMBRALES Y TABLA DUAL DE PREDICCIONES",
        "Demostramos la trampa del accuracy (Ingenuo 90,25% con 0% recall en F2 atipico), calibramos "
        "3 umbrales operativos (0,05 / 0,30 / 0,40) y comparamos lado a lado las predicciones reales "
        "(Tipo 1 Regresion + Tipo 2 Clasificacion) sobre las 10 regiones del Reino Unido y 5 hipotecas F2.",
    )
    if config.RUTA_COMPARATIVA_GLOBAL.exists():
        print("  [Tabla 8A · Semaforo Global — Que algoritmos funcionan bien y cuales fallan]")
        print(pd.read_csv(config.RUTA_COMPARATIVA_GLOBAL).to_string(index=False))

    df_dual_10 = prediccion_dual.tabla_prediccion_dual_10_regiones(reg_esf)
    print("\n  [Tabla 8B · Comparativa Dual (Regresion Ridge + Clasificacion CART/KNN/RF/NB) en 10 regiones reales]")
    print(df_dual_10.to_string(index=False))

    df_dual_f2 = prediccion_dual.tabla_prediccion_dual_hipotecas_f2(reg_f2, clf_f2, meta_f2["columnas_X"])
    print("\n  [Tabla 8C · Comparativa Dual en 5 ofertas hipotecarias reales F2 (Tasa RF + Alerta CART)]")
    print(df_dual_f2.to_string(index=False))

    # ------------------------------------------------------------------ 08
    seccion(
        8,
        "CLUSTERING NO SUPERVISADO CON PRESUPUESTO DE 24 GB DE RAM (A + B + GRAN ESCALA F1)",
        "Segmentamos sin contaminar X con 'esfuerzo': Regiones (n=60), Hipotecas F2 y Transacciones F1 en "
        "streaming PyArrow float32. Mostramos el benchmark hasta 11,37 millones de filas dentro de 24 GB.",
    )
    reg_cl = clustering_mercados.segmentar_regiones(modo)
    print("  [Tabla 9A · Segmentos territoriales descubiertos sin esfuerzo en X (validacion post-hoc)]")
    print(reg_cl["perfil"][["cluster", "segmento_nombre", "n_obs", "precio_mediana_gbp", "esfuerzo_medio_posthoc", "pct_esfuerzo_alto"]].round(2).to_string(index=False))

    if recalcular_24gb or not config.RUTA_CLUSTERING_24GB.exists():
        f1_cl = clustering_mercados.segmentar_transacciones_gran_escala(modo)
        df_bench_24 = f1_cl["comparativa_escala"]
        df_perf_f1 = f1_cl["perfil_1m"]
    else:
        df_bench_24 = pd.read_csv(config.RUTA_CLUSTERING_24GB)
        df_perf_f1 = pd.read_csv(config.RUTA_SEGMENTOS_TRANSACCIONES_1M)

    print("\n  [Tabla 9B · Escalamiento real de Clustering F1 en streaming (RAM Python + RSS real vs. 24 GB)]")
    print(df_bench_24.to_string(index=False))
    print("\n  [Tabla 9C · Capacidad teorica frente a un limite de 24 GB de RAM (Streaming vs In-Memory vs Ward)]")
    print(clustering_mercados.tabla_capacidad_ram_24gb(24.0).to_string(index=False))
    print("\n  [Tabla 9D · Macro-segmentos inmobiliarios reales descubiertos en F1]")
    print(df_perf_f1.to_string(index=False))

    # ------------------------------------------------------------------ 09
    seccion(
        9,
        "REGLAS DE ASOCIACION (APRIORI PURO F1 + F2 Y COMPARATIVA FP-GROWTH)",
        "Descubrimos combinaciones frecuentes de atributos sin etiquetas, filtramos reglas triviales "
        "juridicas (Flat->Leasehold) y priorizamos patrones accionables por score = lift * soporte.",
    )
    r_f1 = pd.read_csv(config.RUTA_REGLAS_INMOBILIARIO)
    r_f2 = pd.read_csv(config.RUTA_REGLAS_HIPOTECAS)
    acc_f1 = r_f1[r_f1["categoria_analitica"] == "Accionable (accesibilidad/territorio)"]
    print("  [Tabla 10A · Top 5 reglas accionables F1 (transacciones inmobiliarias)]")
    print(acc_f1[["antecedente", "consecuente", "soporte", "confianza", "lift", "score"]].head(5).to_string(index=False))
    print("\n  [Tabla 10B · Top 5 reglas F2 (condiciones de oferta hipotecaria)]")
    print(r_f2[["antecedente", "consecuente", "soporte", "confianza", "lift", "score"]].head(5).to_string(index=False))

    # ------------------------------------------------------------------ 10
    seccion(
        10,
        "TELEMETRIA UNIFICADA DE TIEMPO (s) Y CONSUMO DE RAM (MB)",
        "Consolidamos cuanto demoro y cuanta memoria consumio cada etapa del proyecto para verificar "
        "operacion segura en local y escalabilidad directa a servidor/nube.",
    )
    if config.RUTA_TELEMETRIA_RENDIMIENTO.exists():
        df_tel = pd.read_csv(config.RUTA_TELEMETRIA_RENDIMIENTO)
        print("  [Tabla 11 · Telemetria de tiempo y consumo pico de RAM por etapa]")
        print(df_tel[["etapa", "filas_procesadas", "tiempo_s", "pico_ram_mb", "limite_ram_mb", "estado"]].to_string(index=False))

    # ------------------------------------------------------------------ 11
    seccion(
        11,
        "PREDICCION DUAL EN VIVO Y PAQUETE LISTO PARA FUTURA SUBIDA A BD + FRONT/BACK",
        "Probamos ambos tipos de prediccion sobre los 4 modelos .joblib preentrenados (< 5 ms sin "
        "reentrenar) y listamos los artefactos SQL/CSV/JSON preparados para la etapa posterior de despliegue.",
    )
    res_vivo = predecir.predecir_escenario_regional(region_consulta)
    print(f"  Consulta en vivo -> Region: {res_vivo['region']}")
    print(f"    * Tipo 2 (Clasificacion CART): {res_vivo['prediccion']} (P(Alto) = {res_vivo['probabilidad_alto']:.3f})")
    print(f"    * Tipo 1 (Regresion Ridge)   : esfuerzo estimado = {res_vivo['tipo1_regresion_esfuerzo_pct']:.1f}% (+/- {res_vivo['tipo1_mae_pp']:.1f} pp)")
    print(f"    * Latencia combinada         : {res_vivo['latencia_ms']:.2f} ms")
    print("    * Ruta explicativa CART      :")
    for paso in res_vivo["explicacion_reglas"]:
        print(f"        - {paso}")

    hip_vivo = predecir.evaluar_hipoteca_con_umbrales(999.0, 24, "fixed", "First time buyer")
    print(
        f"\n  Consulta hipotecaria F2 (£999, 24m, fixed, First time buyer):\n"
        f"    * Tipo 1 (Regresion RF)      : tasa estimada = {hip_vivo['tipo1_tasa_estimada_pct']:.2f}% (+/- {hip_vivo['tipo1_mae_pp']:.2f} pp)\n"
        f"    * Tipo 2 (Clasificacion CART): P(atipica > 8,13%) = {hip_vivo['tipo2_probabilidad_tasa_atipica']:.3f} ({hip_vivo['latencia_ms']:.2f} ms)\n"
        f"    * Decisiones por umbral      : "
        + " | ".join(f"{k}={v['etiqueta']}" for k, v in hip_vivo["decisiones_por_umbral"].items())
    )

    print("\n  [Tabla 12 · Guia de Seleccion: CON Etiqueta vs SIN Etiqueta y Cantidad Exacta de Aciertos]")
    print(predecir.tabla_guia_metodos_y_aciertos()[["paradigma", "tipo", "metodo", "aciertos_exactos"]].to_string(index=False))

    contrato = json.loads(config.RUTA_CONTRATO_PREDICCION.read_text(encoding="utf-8"))
    bd_info = contrato.get("capa_datos_futura_bd", {})
    print(
        f"\n  Entregables Fase 2 (Presentacion Interactiva + PPTX) y Fase 3 (Guia Azure guardada):\n"
        f"    * Presentacion Web  : Presentacion_Interactiva_Proyecto_UK.html (Simulador Slide 15: opciones y escrita)\n"
        f"    * Presentacion PPTX : Presentacion_Proyecto_UK.pptx | Servidor API: python app_presentacion_interactiva.py\n"
        f"    * 4 modelos .joblib : {list(contrato.get('modelos_serializados', {}).values())}\n"
        f"    * Esquema SQL BD    : {bd_info.get('esquema_sql', 'data/modelos/esquema_bd.sql')}\n"
        f"    * Tablas CSV Nivel 1: {len(bd_info.get('tablas_nivel1', []))} tablas "
        f"({bd_info.get('peso_total_tablas_nivel1_kb', 0.0)/1024.0:.2f} MB total, carga < 5 s)\n"
        f"    * Guia Fase 3 Azure : md/GUIA_FASE3_SUBIDA_AZURE.md (subir_datos_azure.py + predecir_azure.py)"
    )
    print(f"\n>>> EJECUCION MAESTRA COMPLETA EXITOSA EN {time.perf_counter() - t_inicio:.2f} s "
          f"(RSS final SO: {sistema_recursos.rss_proceso_mb():.1f} MB) <<<")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestador Maestro Completo del Proyecto UK Housing (macOS / Linux / Windows)"
    )
    parser.add_argument("--modo", choices=["local", "local_24gb", "cloud"], default="local")
    parser.add_argument("--region", type=str, default="London", help="Region para la consulta dual final")
    parser.add_argument("--recalcular-24gb", action="store_true", help="Re-ejecutar en vivo el benchmark F1 completo")
    args = parser.parse_args()
    run_maestro(modo=args.modo, region_consulta=args.region, recalcular_24gb=args.recalcular_24gb)


if __name__ == "__main__":
    main()
