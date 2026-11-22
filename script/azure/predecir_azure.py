"""Motor de Prediccion Conectado al Enlace Compartible de Azure PostgreSQL (o CSV local)
con soporte de prediccion POR OPCIONES y A ESCRITA (texto libre en espanol), mostrando
grafica/resumen, tabla comparativa, explicacion matematica paso a paso segun el metodo,
calculo financiero PMT acoplado a ciudad/region y cantidad exacta de aciertos.

Ejemplos de uso:
  1) Por opciones (Region o Ciudad + comparativa de metodos y aciertos):
       python script/azure/predecir_azure.py --region London --metodo todos
       python script/azure/predecir_azure.py --ciudad Manchester --metodo knn

  2) Por opciones (Oferta hipotecaria F2 + 3 umbrales):
       python script/azure/predecir_azure.py --hipoteca --comisiones 999 --meses 24 --tipo fixed --subtitulo "First time buyer"

  3) A escrita (Consulta en texto libre en espanol):
       python script/azure/predecir_azure.py --texto "Quiero comprar vivienda en London como First time buyer con comision 999 a 24 meses fijo"
       python script/azure/predecir_azure.py --texto "Soy joven comprador en Oxford con vivienda de 420000 libras, LTV 85% a 30 anios explicado con KNN"

  4) Usando explicitamente el link compartido de Azure:
       python script/azure/predecir_azure.py --url "postgresql://lector_uk:..." --region London --metodo todos
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ / "script") not in sys.path:
    sys.path.insert(0, str(RAIZ / "script"))

from proyecto import arbol_clasificacion, config, indicadores, predecir, sistema_recursos


from azure.datos_web import consultar_contexto_bd_o_csv, guardar_historial


def calcular_vecinos_knn_en_vivo(region: str, overrides: dict[str, float | None] | None = None, k: int = 5) -> dict[str, Any]:
    """Calcula en vivo los k vecinos historicos reales mas cercanos (Region + Anio + Distancia + Esfuerzo)."""
    art = predecir.cargar_artefactos_prediccion()
    contrato = art["contrato"]
    X_fila, _, _, _ = predecir.construir_vector_entrada(contrato, region, overrides)
    X_m, y_m_str, df_m, _ = arbol_clasificacion.preparar_xy()
    sc = StandardScaler().fit(X_m)
    X_m_std = sc.transform(X_m)
    x_q_std = sc.transform(X_fila)[0]

    dists = np.sqrt(((X_m_std - x_q_std) ** 2).sum(axis=1))
    orden = np.argsort(dists)[:k]
    vecinos = []
    votos_alto = 0
    for rank, idx in enumerate(orden, 1):
        fila_m = df_m.iloc[int(idx)]
        clase_v = str(y_m_str.iloc[int(idx)])
        if clase_v == "Alto":
            votos_alto += 1
        vecinos.append(
            {
                "orden": rank,
                "region": str(fila_m["region"]),
                "anio": int(fila_m["anio"]),
                "distancia_euclidiana_z": round(float(dists[int(idx)]), 3),
                "esfuerzo_real_pct": round(float(fila_m["esfuerzo"]) * 100.0, 1),
                "precio_mediana_gbp": round(float(fila_m["precio_mediana"]), 0),
                "clase": clase_v,
            }
        )
    return {
        "k": k,
        "votos_alto": votos_alto,
        "votos_moderado": k - votos_alto,
        "prob_alto_knn": round(votos_alto / float(k), 4),
        "prediccion_knn": "Alto" if votos_alto >= (k // 2 + 1) else "Moderado",
        "vecinos": vecinos,
    }


def interpretar_consulta_escrita(texto: str) -> dict[str, Any]:
    """Analiza una consulta escrita en lenguaje natural (espanol) y extrae region, ciudad,
    precio personalizado, LTV, plazo en anios, variacion de precio, hipoteca F2 y metodo."""
    t_low = texto.lower()

    # 1. Detectar region
    regiones = [
        "Yorkshire and The Humber",
        "East of England",
        "East Midlands",
        "West Midlands",
        "North East",
        "North West",
        "South East",
        "South West",
        "London",
        "Wales",
    ]
    aliases_reg = {
        "londres": "London",
        "gales": "Wales",
        "noreste": "North East",
        "noroeste": "North West",
        "sureste": "South East",
        "suroeste": "South West",
        "yorkshire": "Yorkshire and The Humber",
    }
    region_detectada = None
    for r in regiones:
        if r.lower() in t_low:
            region_detectada = r
            break
    if not region_detectada:
        for alias, canon in aliases_reg.items():
            if alias in t_low:
                region_detectada = canon
                break

    # 2. Detectar ciudad desde diccionario popular + CSV precios_ciudad
    ciudades_populares = {
        "manchester": ("MANCHESTER", "North West"),
        "birmigham": ("BIRMINGHAM", "West Midlands"),
        "birmingham": ("BIRMINGHAM", "West Midlands"),
        "leeds": ("LEEDS", "Yorkshire and The Humber"),
        "liverpool": ("LIVERPOOL", "North West"),
        "bristol": ("BRISTOL", "South West"),
        "sheffield": ("SHEFFIELD", "Yorkshire and The Humber"),
        "newcastle": ("NEWCASTLE UPON TYNE", "North East"),
        "nottingham": ("NOTTINGHAM", "East Midlands"),
        "cardiff": ("CARDIFF", "Wales"),
        "cambridge": ("CAMBRIDGE", "East of England"),
        "oxford": ("OXFORD", "South East"),
        "bath": ("BATH", "South West"),
        "york": ("YORK", "Yorkshire and The Humber"),
        "brighton": ("BRIGHTON", "South East"),
        "reading": ("READING", "South East"),
        "coventry": ("COVENTRY", "West Midlands"),
        "leicester": ("LEICESTER", "East Midlands"),
        "southampton": ("SOUTHAMPTON", "South East"),
        "portsmouth": ("PORTSMOUTH", "South East"),
        "plymouth": ("PLYMOUTH", "South West"),
        "exeter": ("EXETER", "South West"),
        "norwich": ("NORWICH", "East of England"),
        "swansea": ("SWANSEA", "Wales"),
        "newport": ("NEWPORT", "Wales"),
        "derby": ("DERBY", "East Midlands"),
        "sunderland": ("SUNDERLAND", "North East"),
        "middlesbrough": ("MIDDLESBROUGH", "North East"),
    }
    ciudad_detectada = None
    for c_key, (c_nom, c_reg) in ciudades_populares.items():
        if re.search(rf"\b{re.escape(c_key)}\b", t_low):
            ciudad_detectada = c_nom
            if not region_detectada:
                region_detectada = c_reg
            break

    # 3. Detectar metodo explicito si el usuario lo pide en el texto
    metodo_detectado = "todos"
    if "knn" in t_low or "vecino" in t_low:
        metodo_detectado = "knn"
    elif "ridge" in t_low or "lineal" in t_low or "regresion" in t_low or "regresión" in t_low:
        metodo_detectado = "ridge"
    elif "apriori" in t_low or "asociacion" in t_low or "asociación" in t_low:
        metodo_detectado = "apriori"
    elif "kmeans" in t_low or "k-means" in t_low or "cluster" in t_low or "clúster" in t_low:
        metodo_detectado = "kmeans"
    elif "random forest" in t_low or "bosque" in t_low:
        metodo_detectado = "rf"
    elif "naive bayes" in t_low or "bayes" in t_low:
        metodo_detectado = "nb"
    elif "arbol" in t_low or "árbol" in t_low or "cart" in t_low:
        metodo_detectado = "cart"

    # 4. Detectar precio de vivienda en libras (ej. "vivienda de 380000 libras", "£250000", "280k")
    precio_gbp = None
    m_precio = re.search(r"(?:vivienda|casa|precio|propiedad|valor)\s*(?:de\s*)?£?\s*(\d{5,7})", t_low)
    if not m_precio:
        m_precio = re.search(r"\b(\d{4,8})\s*(?:libras|gbp)\b", t_low)
    if not m_precio:
        m_precio = re.search(r"£\s*(\d{5,7})", t_low)
    if m_precio:
        precio_gbp = float(m_precio.group(1))
    else:
        m_pk = re.search(r"£?\s*(\d{2,3})\s*k\b", t_low)
        if m_pk and int(m_pk.group(1)) >= 80:
            precio_gbp = float(m_pk.group(1)) * 1000.0

    # 5. Detectar LTV (%) o entrada
    ltv_pct = 90.0
    m_ltv = re.search(r"ltv\s*(?:de\s*|del\s*)?(\d{1,3}(?:[.,]\d+)?)\s*%", t_low)
    if m_ltv:
        ltv_pct = float(m_ltv.group(1).replace(",", "."))
    else:
        m_ent = re.search(r"entrada\s*(?:de\s*|del\s*)?(\d{1,2})\s*%", t_low)
        if m_ent:
            ltv_pct = max(50.0, min(95.0, 100.0 - float(m_ent.group(1))))

    # 6. Extraer comision (£ o numero tras comision)
    comisiones = 999.0
    m_com = re.search(r"(?:comisi[oó]n|fee)\s*(?:de\s*)?£?\s*(\d{1,4}(?:\.\d+)?)", t_low)
    if m_com:
        comisiones = float(m_com.group(1))
    elif "sin comision" in t_low or "sin comisión" in t_low or "comision 0" in t_low or "£0" in t_low:
        comisiones = 0.0

    # 7. Extraer meses (24, 36, 60, 120, 300) y plazo total en anios (15..35)
    meses = 24
    plazo_anios = 25
    m_mes = re.search(r"(\d{2,3})\s*meses", t_low)
    if m_mes:
        meses = int(m_mes.group(1))
    for m_an in re.finditer(r"(\d{1,2})\s*a[ñn]os", t_low):
        v_an = int(m_an.group(1))
        if v_an in (2, 3, 5, 10):
            meses = v_an * 12
        elif 15 <= v_an <= 35:
            plazo_anios = v_an

    # 8. Tipo de tasa y Subtitulo
    tipo = "fixed"
    if "discounted" in t_low or "descuento" in t_low:
        tipo = "discounted"
    elif "tracker" in t_low:
        tipo = "tracker"
    elif "variable" in t_low:
        tipo = "variable"

    subtitulo = "First time buyer"
    if "remortgage" in t_low or "rehipoteca" in t_low:
        subtitulo = "Remortgage"
    elif "buy to let" in t_low or "inversion" in t_low or "alquiler" in t_low:
        subtitulo = "Buy to let"
    elif "home purchase" in t_low:
        subtitulo = "Home Purchase"

    # 9. Variacion de precio (%)
    precio_var_pct = None
    m_var = re.search(r"(?:variaci[oó]n|subida|ca[ií]da|cambio)[^0-9\-\+]*([\-\+]?\d+(?:[\.,]\d+)?)\s*%", t_low)
    if not m_var:
        m_var = re.search(r"([\+\-]\d+(?:[\.,]\d+)?)\s*%", t_low)
    if m_var:
        precio_var_pct = float(m_var.group(1).replace(",", "."))

    return {
        "texto_original": texto,
        "region": region_detectada or "London",
        "ciudad": ciudad_detectada,
        "incluye_hipoteca": True,
        "precio_gbp": precio_gbp,
        "precio_var_pct": precio_var_pct,
        "ltv_pct": ltv_pct,
        "plazo_anios": plazo_anios,
        "comisiones": comisiones,
        "meses": meses,
        "tipo": tipo,
        "subtitulo": subtitulo,
        "metodo": metodo_detectado,
    }


def ejecutar_prediccion_integral(
    region: str = "London",
    ciudad: str | None = None,
    precio_gbp: float | None = None,
    precio_var_pct: float | None = None,
    ltv_pct: float = 90.0,
    plazo_anios: int = 25,
    n_transacciones: float | None = None,
    lluvia_anom: float | None = None,
    tmax_anom: float | None = None,
    evaluar_hipoteca: bool = True,
    comisiones: float = 999.0,
    meses: int = 24,
    tipo: str = "fixed",
    subtitulo: str = "First time buyer",
    metodo: str = "todos",
    texto_escrito: str | None = None,
    url_azure: str | None = None,
    registrar: bool = True,
) -> dict[str, Any]:
    """Ejecuta la prediccion acoplada conectada a Azure PostgreSQL / CSV devolviendo:
    prediccion regional, prediccion hipotecaria F2, calculo financiero PMT exacto para la
    ciudad/region, los 5 vecinos KNN en vivo, tabla comparativa y guia de aciertos."""
    load_dotenv(RAIZ / ".env", override=False)
    parsed_txt = None
    if texto_escrito:
        parsed_txt = interpretar_consulta_escrita(texto_escrito)
        region = parsed_txt["region"]
        ciudad = parsed_txt["ciudad"] or ciudad
        if parsed_txt["precio_gbp"] is not None:
            precio_gbp = parsed_txt["precio_gbp"]
        if parsed_txt["precio_var_pct"] is not None:
            precio_var_pct = parsed_txt["precio_var_pct"]
        ltv_pct = parsed_txt["ltv_pct"]
        if not (0 < ltv_pct <= 100):
            raise ValueError("La financiación debe ser mayor que cero y como máximo del 100 %.")
        plazo_anios = parsed_txt["plazo_anios"]
        comisiones = parsed_txt["comisiones"]
        meses = parsed_txt["meses"]
        tipo = parsed_txt["tipo"]
        subtitulo = parsed_txt["subtitulo"]
        if parsed_txt["metodo"] != "todos":
            metodo = parsed_txt["metodo"]

    ctx_prev = consultar_contexto_bd_o_csv(region=region, ciudad=ciudad, url_azure=url_azure)
    reg_canon = ctx_prev["region_resuelta"]

    overrides = {
        "n_transacciones": n_transacciones,
        "precio_var_pct": precio_var_pct,
        "lluvia_anom": lluvia_anom,
        "tmax_anom": tmax_anom,
    }
    res_reg = predecir.predecir_escenario_regional(reg_canon, **overrides)
    comp_modelos = predecir.comparar_prediccion_modelos(reg_canon, overrides)
    knn_vivo = calcular_vecinos_knn_en_vivo(reg_canon, overrides, k=5)
    res_hip = predecir.evaluar_hipoteca_con_umbrales(comisiones, meses, tipo, subtitulo)

    # Calculo financiero PMT acoplado a la ciudad/region y a la tasa hipotecaria predicha
    seg = ctx_prev.get("segmento_sin_etiqueta") or {}
    c_info = ctx_prev.get("ciudad_info")
    precio_base = (
        float(precio_gbp)
        if precio_gbp is not None
        else (
            float(c_info["precio_mediana_ciudad_gbp"])
            if c_info
            else float(seg.get("precio_mediana_hist_gbp", 240000.0))
        )
    )
    var_aplicada = float(precio_var_pct) if precio_var_pct is not None else 0.0
    precio_ajustado = precio_base * (1.0 + (var_aplicada / 100.0 if precio_gbp is None and precio_var_pct is not None else 0.0))
    salario_joven_anual = float(seg.get("salario_joven_region_gbp", 25353.0))
    tasa_pred_f2 = float(res_hip["tipo1_tasa_estimada_pct"])
    cuota_pmt_caso = float(
        indicadores.cuota_mensual_pmt(
            float(precio_ajustado),
            tasa_pred_f2,
            int(plazo_anios),
            float(ltv_pct) / 100.0,
            float(comisiones),
        )
    )
    esfuerzo_pmt_pct = round((cuota_pmt_caso / (salario_joven_anual / 12.0)) * 100.0, 1)
    anos_salario_caso = round(precio_ajustado / salario_joven_anual, 2)

    # Registrar en historial_predicciones de Azure si esta conectado
    ctx = ctx_prev
    registro = {
        "tipo_consulta": "escrita" if texto_escrito else "opciones",
            "region": reg_canon,
            "entrada_json": {
                "ciudad": c_info["ciudad"] if c_info else None,
                "precio_gbp": round(precio_ajustado, 0),
                "ltv_pct": ltv_pct,
                "plazo_anios": plazo_anios,
                "comisiones": comisiones,
                "meses": meses,
                "tipo": tipo,
                "subtitulo": subtitulo,
                "metodo": metodo,
            },
            "prediccion_clasificacion": res_reg["prediccion"],
            "probabilidad": res_reg["probabilidad_alto"],
            "prediccion_regresion": res_reg["tipo1_regresion_esfuerzo_pct"],
            "umbral_aplicado": "0.30",
            "latencia_ms": res_reg["latencia_ms"],
        }
    registro_id = guardar_historial(registro) if registrar and ctx["conectado_azure"] else None
    guia_aciertos = predecir.tabla_guia_metodos_y_aciertos()

    return {
        "registro_id": registro_id,
        "anio_referencia": int(seg["anio"]),
        "anio_salario": int(seg["anio"]),
        "anio_precio": None if precio_gbp is not None else int(c_info["anio"] if c_info else seg["anio"]),
        "fuente_precio": "Introducido por el visitante" if precio_gbp is not None else ("Mediana histórica de la ciudad" if c_info else "Mediana histórica regional"),
        "modo_entrada": "A Escrita (Texto Libre)" if texto_escrito else "Por Opciones (Selectores)",
        "metodo_seleccionado": metodo,
        "interpretacion_texto": parsed_txt,
        "conectado_azure": ctx["conectado_azure"],
        "fuente_datos_conectada": ctx["fuente_datos"],
        "sql_ejecutado": ctx["sql_ejecutado"],
        "historial_azure": ctx["historial_azure"],
        "ciudad_consultada": ctx["ciudad_info"],
        "calculo_financiero_acoplado": {
            "precio_vivienda_gbp": round(precio_ajustado, 0),
            "salario_joven_anual_gbp": round(salario_joven_anual, 0),
            "ltv_pct": round(ltv_pct, 1),
            "plazo_anios": int(plazo_anios),
            "tasa_hipotecaria_usada_pct": tasa_pred_f2,
            "cuota_mensual_pmt_gbp": round(cuota_pmt_caso, 0),
            "esfuerzo_pmt_acoplado_pct": esfuerzo_pmt_pct,
            "anos_salario_requeridos": anos_salario_caso,
            "cumple_umbral_30pct": esfuerzo_pmt_pct <= 30.0,
        },
        "prediccion_regional_dual": res_reg,
        "knn_vecinos_en_vivo": knn_vivo,
        "comparativa_metodos_regional": comp_modelos.to_dict(orient="records"),
        "prediccion_hipoteca_dual": res_hip,
        "segmento_clustering_sin_etiqueta": ctx["segmento_sin_etiqueta"],
        "reglas_apriori_sin_etiqueta": ctx["reglas_apriori_sin_etiqueta"],
        "guia_seleccion_metodos_y_aciertos": guia_aciertos.to_dict(orient="records"),
    }


def main() -> None:
    sistema_recursos.configurar_entorno_portable()
    parser = argparse.ArgumentParser(
        description="Predictor Conectado a Azure PostgreSQL / CSV (Por Opciones o A Escrita)"
    )
    parser.add_argument("--texto", type=str, default=None, help="Consulta escrita en lenguaje natural (espanol)")
    parser.add_argument("--region", type=str, default="London", help="Region del Reino Unido")
    parser.add_argument("--ciudad", type=str, default=None, help="Ciudad del Reino Unido (ej. Manchester, Oxford)")
    parser.add_argument("--precio_gbp", type=float, default=None, help="Precio personalizado en libras GBP")
    parser.add_argument("--precio_var_pct", type=float, default=None)
    parser.add_argument("--ltv_pct", type=float, default=90.0)
    parser.add_argument("--plazo_anios", type=int, default=25)
    parser.add_argument("--n_transacciones", type=float, default=None)
    parser.add_argument("--hipoteca", action="store_true", help="Evaluar oferta hipotecaria F2")
    parser.add_argument("--comisiones", type=float, default=999.0)
    parser.add_argument("--meses", type=int, default=24)
    parser.add_argument("--tipo", type=str, default="fixed")
    parser.add_argument("--subtitulo", type=str, default="First time buyer")
    parser.add_argument("--metodo", choices=["cart", "knn", "rf", "ridge", "kmeans", "apriori", "nb", "todos"], default="todos")
    parser.add_argument("--url", type=str, default=None, help="Enlace compartible DATABASE_URL_READONLY de Azure")
    parser.add_argument("--comparar", action="store_true", help="Imprimir tabla comparativa completa y guia de aciertos")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON")
    args = parser.parse_args()

    out = ejecutar_prediccion_integral(
        region=args.region,
        ciudad=args.ciudad,
        precio_gbp=args.precio_gbp,
        precio_var_pct=args.precio_var_pct,
        ltv_pct=args.ltv_pct,
        plazo_anios=args.plazo_anios,
        n_transacciones=args.n_transacciones,
        evaluar_hipoteca=True,
        comisiones=args.comisiones,
        meses=args.meses,
        tipo=args.tipo,
        subtitulo=args.subtitulo,
        metodo=args.metodo,
        texto_escrito=args.texto,
        url_azure=args.url,
    )

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("=" * 88)
    print(f"PREDICCION INTEGRAL UK HOUSING · MODO: {out['modo_entrada'].upper()}")
    print(f"Fuente de datos conectada: {out['fuente_datos_conectada']}")
    print(f"SQL ejecutado            : {out['sql_ejecutado']}")
    print("=" * 88)

    if out["interpretacion_texto"]:
        it = out["interpretacion_texto"]
        print(f"  Texto interpretado : \"{it['texto_original']}\"")
        print(f"  Parametros extraidos -> Region: {it['region']} | Ciudad: {it['ciudad'] or 'Regional'} | "
              f"Hipoteca: {it['subtitulo']} / {it['tipo']} (£{it['comisiones']:.0f}, {it['meses']}m, LTV {it['ltv_pct']:.0f}%, {it['plazo_anios']}a)")

    if out["ciudad_consultada"]:
        c = out["ciudad_consultada"]
        print(f"  Ciudad en BD/CSV   : {c['ciudad']} (Region: {c['region']}, Anio {c['anio']}: "
              f"precio mediano £{c['precio_mediana_ciudad_gbp']:,.0f}, {c['n_transacciones_ciudad']:,} ventas)")

    r = out["prediccion_regional_dual"]
    fin = out["calculo_financiero_acoplado"]
    seg = out["segmento_clustering_sin_etiqueta"]
    print(f"\n[1] RESULTADO REGIONAL Y FINANCIERO ACOPLADO ({r['region']}):")
    print(f"  * CON ETIQUETA · Tipo 2 (Clasificacion CART Ganador): {r['prediccion']} (Prob. Alto = {r['probabilidad_alto']*100:.1f}%)")
    print(f"  * CON ETIQUETA · Tipo 1 (Regresion Continua Ridge)  : {r['tipo1_regresion_esfuerzo_pct']:.1f}% del salario joven (+/- {r['tipo1_mae_pp']:.1f} pp)")
    print(f"  * CALCULO FINANCIERO PMT DEL CASO (£{fin['precio_vivienda_gbp']:,.0f}, LTV {fin['ltv_pct']}%, {fin['plazo_anios']}a, tasa {fin['tasa_hipotecaria_usada_pct']:.2f}%): "
          f"Cuota = £{fin['cuota_mensual_pmt_gbp']:,.0f}/mes ({fin['esfuerzo_pmt_acoplado_pct']:.1f}% del salario joven, {fin['anos_salario_requeridos']} anios salario)")
    if seg:
        print(f"  * SIN ETIQUETA · Tipo 3 (Clustering K-Means k=3)    : Cluster #{seg['cluster_id']} - {seg['segmento_nombre']} "
              f"(historico BD: {seg['esfuerzo_historico_pct']:.1f}%, {seg['anos_salario_hist']} anios salario)")
        print(f"    Accion recomendada: {seg['accion_politica']}")
    print("  * EXPLICACION PASO A PASO (Reglas del Arbol CART):")
    for p in r["explicacion_reglas"]:
        print(f"      -> {p}")

    h = out["prediccion_hipoteca_dual"]
    print(f"\n[2] RESULTADO HIPOTECARIO F2 ({h['entrada']['subtitulo']} / {h['entrada']['tipo']}, £{h['entrada']['comisiones_total_gbp']:.0f}, {h['entrada']['meses_inicial']}m):")
    print(f"  * CON ETIQUETA · Tipo 1 (Regresion RandomForest)    : Tasa inicial estimada = {h['tipo1_tasa_estimada_pct']:.2f}% (+/- {h['tipo1_mae_pp']:.2f} pp)")
    print(f"  * CON ETIQUETA · Tipo 2 (Clasificacion CART Atipica): P(Tasa > 8,13%) = {h['tipo2_probabilidad_tasa_atipica']*100:.1f}%")
    for k, d in h["decisiones_por_umbral"].items():
        print(f"      - Umbral {k:<15} (u={d['umbral']:.2f}): {d['etiqueta']}")

    print("\n[3] VECINOS MAS CERCANOS EN VIVO (KNN Escalado k=5):")
    print(pd.DataFrame(out["knn_vecinos_en_vivo"]["vecinos"]).to_string(index=False))

    print("\n[4] TABLA COMPARATIVA DE METODOS PARA ESTE CASO Y CANTIDAD DE ACIERTOS:")
    print(pd.DataFrame(out["comparativa_metodos_regional"]).to_string(index=False))

    print("\n[5] GUIA DE SELECCION: QUE PREDICCION UTILIZAR SEGUN EL METODO Y LA CANTIDAD DE ACIERTOS:")
    print(pd.DataFrame(out["guia_seleccion_metodos_y_aciertos"]).to_string(index=False))


if __name__ == "__main__":
    main()
