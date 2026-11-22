"""Lecturas públicas en Azure PostgreSQL (con respaldo CSV) y registro de predicciones."""
import json
import logging
import os
from functools import cache

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from proyecto import config, predecir

COLUMNAS = ("anio", "region", "n_transacciones", "precio_mediana", "salario_joven_region", "anos_salario")
CONSULTA = """
    SELECT anio, region, n_transacciones, precio_mediana, salario_joven_region, anos_salario
    FROM public.maestro_regional
    WHERE (%s::text IS NULL OR region = %s) AND (%s::int IS NULL OR anio = %s)
    ORDER BY anio DESC, region
"""


def conectar(variable):
    return psycopg.connect(
        os.environ[variable],
        connect_timeout=4,
        sslmode="require",
        options="-c statement_timeout=5000",
        row_factory=dict_row,
    )


@cache
def leer_csv(ruta):
    return pd.read_csv(ruta)


def datos_regionales(region=None, anio=None):
    if os.getenv("DATABASE_URL_READONLY"):
        try:
            with conectar("DATABASE_URL_READONLY") as conn:
                filas = conn.execute(CONSULTA, (region, region, anio, anio)).fetchall()
            return filas, "Azure PostgreSQL"
        except psycopg.Error:
            logging.warning("Consulta regional no disponible en Azure")
    df = leer_csv(config.DATA_INTEGRADO / "accesibilidad_vivienda_joven_uk_v1.csv")
    if region:
        df = df[df.region == region]
    if anio is not None:
        df = df[df.anio == anio]
    df = df.sort_values(["anio", "region"], ascending=[False, True])
    return df[list(COLUMNAS)].to_dict("records"), "CSV de respaldo"


def datos_ciudades(region=None, limite=100):
    if os.getenv("DATABASE_URL_READONLY"):
        try:
            with conectar("DATABASE_URL_READONLY") as conn:
                filas = conn.execute(
                    """
                    SELECT DISTINCT ON (ciudad, region)
                           anio, region, ciudad, n_transacciones, precio_mediana
                    FROM public.precios_ciudad
                    WHERE (%s::text IS NULL OR region = %s)
                    ORDER BY ciudad, region, anio DESC, n_transacciones DESC
                    """,
                    (region, region),
                ).fetchall()
            filas = sorted(filas, key=lambda r: (-int(r["anio"]), -int(r["n_transacciones"])))[:limite]
            return filas, "Azure PostgreSQL"
        except psycopg.Error:
            logging.warning("Consulta de ciudades no disponible en Azure")
    df = leer_csv(config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV)
    if region:
        df = df[df.region == region]
    top = (
        df.sort_values(["anio", "n_transacciones"], ascending=[False, False])
        .drop_duplicates(["ciudad", "region"])
        .head(limite)
    )
    return top[["anio", "region", "ciudad", "n_transacciones", "precio_mediana"]].to_dict("records"), "CSV de respaldo"


def datos_hipoteca_2022(region=None):
    if os.getenv("DATABASE_URL_READONLY"):
        try:
            with conectar("DATABASE_URL_READONLY") as conn:
                filas = conn.execute(
                    """
                    SELECT region, precio_mediana, precio_real_2020, salario_joven_region,
                           tasa_inicial_pct, cuota_mensual, cuota_pct_salario, accesible
                    FROM public.hipoteca_2022
                    WHERE (%s::text IS NULL OR region = %s)
                    ORDER BY cuota_pct_salario DESC
                    """,
                    (region, region),
                ).fetchall()
            return filas, "Azure PostgreSQL"
        except psycopg.Error:
            logging.warning("Consulta hipoteca_2022 no disponible en Azure")
    df = leer_csv(config.RUTA_HIPOTECA_2022_MUESTRA)
    if region:
        df = df[df.region == region]
    cols = ["region", "precio_mediana", "precio_real_2020", "salario_joven_region", "tasa_inicial_pct", "cuota_mensual", "cuota_pct_salario", "accesible"]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values("cuota_pct_salario", ascending=False)[cols].to_dict("records"), "CSV de respaldo"


def comparativa_10_regiones():
    """Devuelve referencia ordenada de las 10 regiones UK para graficar frente al caso del usuario."""
    df = leer_csv(config.RUTA_SEGMENTOS_TERRITORIALES)
    ult = df.sort_values("anio", ascending=False).drop_duplicates("region")
    filas = []
    for r in ult.itertuples():
        filas.append(
            {
                "region": str(r.region),
                "esfuerzo_pct": round(float(r.esfuerzo) * 100.0, 1),
                "precio_mediana_gbp": round(float(r.precio_mediana), 0),
                "salario_joven_gbp": round(float(r.salario_joven_region), 0),
                "anos_salario": round(float(r.anos_salario), 2),
                "cluster_id": int(r.cluster),
            }
        )
    filas.sort(key=lambda x: x["esfuerzo_pct"])
    return filas


def consultar_contexto_bd_o_csv(region=None, ciudad=None, url_azure=None):
    origen = "CSV de respaldo"
    region = region or "London"
    ciudad_info = None
    perfil = None
    historial = []
    if os.getenv("DATABASE_URL_READONLY") or url_azure:
        try:
            with psycopg.connect(
                url_azure or os.environ["DATABASE_URL_READONLY"],
                connect_timeout=4,
                sslmode="require",
                row_factory=dict_row,
                options="-c statement_timeout=5000",
            ) as conn:
                if ciudad:
                    ciudad_info = conn.execute(
                        """
                        SELECT ciudad, region, anio, n_transacciones AS n_transacciones_ciudad,
                               precio_mediana AS precio_mediana_ciudad_gbp
                        FROM public.precios_ciudad WHERE lower(ciudad) = lower(%s)
                        ORDER BY anio DESC, n_transacciones DESC, region LIMIT 1
                        """,
                        (ciudad,),
                    ).fetchone()
                if ciudad_info:
                    region = ciudad_info["region"]
                perfil = conn.execute(
                    """
                    SELECT anio, cluster AS cluster_id, segmento_nombre, accion_politica,
                           precio_mediana AS precio_mediana_hist_gbp, salario_joven_region AS salario_joven_region_gbp,
                           esfuerzo * 100 AS esfuerzo_historico_pct, anos_salario AS anos_salario_hist
                    FROM public.segmentos_territoriales WHERE region = %s ORDER BY anio DESC LIMIT 1
                    """,
                    (region,),
                ).fetchone()
                historial = conn.execute(
                    """
                    SELECT id, region, prediccion_clasificacion, prediccion_regresion
                    FROM public.historial_predicciones ORDER BY id DESC LIMIT 5
                    """
                ).fetchall()
                if perfil:
                    origen = "Azure PostgreSQL"
        except psycopg.Error:
            logging.warning("Contexto de Azure temporalmente no disponible")
            perfil = ciudad_info = None
            historial = []

    if origen == "CSV de respaldo":
        if ciudad:
            ciudades = leer_csv(config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV)
            filas = ciudades[ciudades.ciudad.str.casefold() == ciudad.casefold()].sort_values(
                ["anio", "n_transacciones", "region"], ascending=[False, False, True]
            )
            if not filas.empty:
                fila = filas.iloc[0]
                region = fila.region
                ciudad_info = dict(
                    ciudad=fila.ciudad,
                    region=region,
                    anio=int(fila.anio),
                    n_transacciones_ciudad=int(fila.n_transacciones),
                    precio_mediana_ciudad_gbp=float(fila.precio_mediana),
                )
        segmentos = leer_csv(config.RUTA_SEGMENTOS_TERRITORIALES)
        filas = segmentos[segmentos.region == region].sort_values("anio", ascending=False)
        if filas.empty:
            raise ValueError("Región desconocida")
        fila = filas.iloc[0]
        perfil = dict(
            anio=int(fila.anio),
            cluster_id=int(fila.cluster),
            segmento_nombre=fila.segmento_nombre,
            accion_politica=fila.accion_politica,
            precio_mediana_hist_gbp=float(fila.precio_mediana),
            salario_joven_region_gbp=float(fila.salario_joven_region),
            esfuerzo_historico_pct=round(float(fila.esfuerzo) * 100, 1),
            anos_salario_hist=float(fila.anos_salario),
        )

    reglas = leer_csv(config.RUTA_REGLAS_INMOBILIARIO)
    seleccion = reglas[reglas.categoria_analitica == "Accionable (accesibilidad/territorio)"].head(4)
    return dict(
        region_resuelta=region,
        ciudad_info=ciudad_info,
        segmento_sin_etiqueta=perfil,
        conectado_azure=origen == "Azure PostgreSQL",
        fuente_datos=origen,
        historial_azure=historial,
        reglas_apriori_sin_etiqueta=[f"{r.antecedente} → {r.consecuente}" for r in seleccion.itertuples()],
        sql_ejecutado="Consulta parametrizada del perfil regional y de la ciudad seleccionada en Azure PostgreSQL",
    )


def guardar_historial(datos):
    if not os.getenv("DATABASE_URL_APP"):
        return None
    try:
        with conectar("DATABASE_URL_APP") as conn:
            return conn.execute(
                """
                INSERT INTO public.historial_predicciones
                    (tipo_consulta, region, entrada_json, prediccion_clasificacion, probabilidad,
                     prediccion_regresion, umbral_aplicado, latencia_ms)
                VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s) RETURNING id
                """,
                (
                    datos["tipo_consulta"],
                    datos["region"],
                    json.dumps(datos["entrada_json"]),
                    datos["prediccion_clasificacion"],
                    datos["probabilidad"],
                    datos["prediccion_regresion"],
                    datos["umbral_aplicado"],
                    datos["latencia_ms"],
                ),
            ).fetchone()["id"]
    except psycopg.Error:
        logging.warning("No se pudo guardar la predicción en Azure")
        return None


def guia_metodos_export():
    return predecir.tabla_guia_metodos_y_aciertos().to_dict("records")
