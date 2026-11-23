"""Motor de Prediccion Conectado a Azure PostgreSQL (con respaldo CSV)
con soporte bilingue (Espanol e Ingles), monedas GBP (£) y COP (Pesos Colombianos),
LTV/TLV con cuota inicial automatica, plazo en anios/meses (maximo 480 meses / 40 anios
y tasa inicial F2 de 12 a 300 meses), 5 vecinos KNN en vivo y comparativa de metodos.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
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

from azure.datos_web import comparativa_10_regiones, consultar_contexto_bd_o_csv, guardar_historial, leer_csv
from proyecto import arbol_clasificacion, config, indicadores, predecir, sistema_recursos

TASA_GBP_COP_DEFAULT = 5100.0
PLAZO_MIN_ANIOS = 5
PLAZO_MAX_ANIOS = 40
PLAZO_MIN_MESES = 60
PLAZO_MAX_MESES = 480
MESES_INICIAL_MIN_F2 = 12
MESES_INICIAL_MAX_F2 = 300


@lru_cache(maxsize=1)
def _catalogo_ciudades_por_nombre() -> list[tuple[str, str, str]]:
    """Indice ordenado de mayor a menor longitud de las 1.153 ciudades UK para busqueda bilingue."""
    ciudades_extra = {
        "manchester": ("MANCHESTER", "North West"),
        "birmingham": ("BIRMINGHAM", "West Midlands"),
        "birmigham": ("BIRMINGHAM", "West Midlands"),
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
    mapa = dict(ciudades_extra)
    if config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV.exists():
        df_c = leer_csv(config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV)
        col_c = "ciudad" if "ciudad" in df_c.columns else "town_city"
        top = (
            df_c.sort_values(["anio", "n_transacciones"], ascending=[False, False])
            .drop_duplicates(subset=[col_c])
        )
        for r in top.itertuples():
            c_nom = str(getattr(r, col_c))
            clave = c_nom.lower().strip()
            if len(clave) >= 4 and clave not in mapa and clave not in ("london", "wales", "york", "bath", "deal", "rye", "march", "street"):
                mapa[clave] = (c_nom, str(r.region))
    pares = [(k, v[0], v[1]) for k, v in mapa.items()]
    pares.sort(key=lambda x: len(x[0]), reverse=True)
    return pares


def _limpiar_numero(texto_num: str) -> float:
    """Convierte formatos numericos en espanol e ingles (250.000, 250,000, 1.800, 1,5) a float."""
    s = texto_num.strip().replace(" ", "")
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        partes = s.split(".")
        if len(partes) > 2 or (len(partes) == 2 and len(partes[1]) == 3):
            s = s.replace(".", "")
    elif "," in s:
        partes = s.split(",")
        if len(partes) > 2 or (len(partes) == 2 and len(partes[1]) == 3):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    return float(s)


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


def interpretar_consulta_escrita(texto: str, tasa_cop: float = TASA_GBP_COP_DEFAULT) -> dict[str, Any]:
    """Interpreta texto libre en ESPANOL o INGLES (o mezcla) y extrae:
    region, ciudad, moneda (GBP/COP), precio en libras/pesos, LTV/TLV o cuota inicial/deposit,
    plazo total en anios/meses (hasta 480 meses / 40 anios), periodo de tasa inicial (12-300m),
    comision, tipo de hipoteca, perfil comprador y metodo."""
    t_low = texto.lower()
    en_hits = sum(1 for w in ("want to buy", "pounds", "deposit", "years", "months", "house in", "home in", "first time buyer", "down payment", "affordable home", "zero fee") if w in t_low)
    es_hits = sum(1 for w in ("quiero", "libras", "pesos", "cuota inicial", "entrada", "años", "anios", "meses", "comisión", "comision", "primer comprador", "vivienda", "casa") if w in t_low)
    idioma = "EN" if en_hits > es_hits else "ES"

    # 1. Detectar region (Espanol + Ingles)
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
        "greater london": "London",
        "gales": "Wales",
        "noreste": "North East",
        "north-east": "North East",
        "noroeste": "North West",
        "north-west": "North West",
        "sureste": "South East",
        "south-east": "South East",
        "suroeste": "South West",
        "south-west": "South West",
        "este de inglaterra": "East of England",
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

    # 2. Detectar ciudad (1.153 ciudades reales UK)
    ciudad_detectada = None
    for c_key, c_nom, c_reg in _catalogo_ciudades_por_nombre():
        if re.search(rf"\b{re.escape(c_key)}\b", t_low):
            ciudad_detectada = c_nom
            if not region_detectada:
                region_detectada = c_reg
            break

    # 3. Detectar metodo explicito (ES + EN)
    metodo_detectado = "todos"
    if any(k in t_low for k in ("knn", "vecino", "neighbor", "neighbour")):
        metodo_detectado = "knn"
    elif any(k in t_low for k in ("ridge", "lineal", "regresion", "regresión", "regression")):
        metodo_detectado = "ridge"
    elif any(k in t_low for k in ("apriori", "asociacion", "asociación", "rules", "reglas")):
        metodo_detectado = "apriori"
    elif any(k in t_low for k in ("kmeans", "k-means", "cluster", "clúster", "segment")):
        metodo_detectado = "kmeans"
    elif any(k in t_low for k in ("random forest", "bosque")):
        metodo_detectado = "rf"
    elif "bayes" in t_low:
        metodo_detectado = "nb"
    elif any(k in t_low for k in ("arbol", "árbol", "cart", "decision tree", "tree")):
        metodo_detectado = "cart"

    # 4. Detectar moneda y precio (Pesos COP / millones o Libras GBP / £ / k)
    moneda_detectada = "GBP"
    precio_gbp = None
    precio_cop = None

    # 4A. Pesos colombianos / millones / mil millones (ej. "1.800 millones de pesos", "950M cop", "1.8 billion cop", "900.000.000 pesos")
    m_bill_cop = re.search(r"(\d[\d\.,]*)\s*(?:mil\s+millones|billones|billions|billion)\s*(?:de\s*)?(?:pesos|cop)?", t_low)
    m_mill_cop = re.search(r"(\d[\d\.,]*)\s*(?:millones|millon|millón|millions|million|m\b)\s*(?:de\s*)?(?:pesos|cop)?", t_low)
    m_pesos_dir = re.search(r"(?:\$|cop\s*)?(\d{1,3}(?:[\.,]\d{3}){2,3}|\d{8,11})\s*(?:pesos|cop)\b", t_low)
    m_cop_pref = re.search(r"\bcop\s*\$?\s*(\d[\d\.,]*)\s*(m|millones|million|millions)?", t_low)

    if m_bill_cop and any(w in t_low for w in ("peso", "cop", "colomb", "mil millones", "billion")):
        val_b = _limpiar_numero(m_bill_cop.group(1))
        if 0.05 <= val_b <= 50:
            precio_cop = val_b * 1_000_000_000.0
            precio_gbp = round(precio_cop / float(tasa_cop), 0)
            moneda_detectada = "COP"
    if precio_gbp is None and m_mill_cop and any(w in t_low for w in ("peso", "cop", "colomb", "millon", "millón", "million")):
        if not re.search(r"(?:pounds|libras|gbp|£)\s*" + re.escape(m_mill_cop.group(1)), t_low):
            val_m = _limpiar_numero(m_mill_cop.group(1))
            if val_m >= 50:  # >= 50 millones de pesos (evita confundir £1.5m)
                precio_cop = val_m * 1_000_000.0
                precio_gbp = round(precio_cop / float(tasa_cop), 0)
                moneda_detectada = "COP"
    if precio_gbp is None and m_pesos_dir:
        val_p = _limpiar_numero(m_pesos_dir.group(1))
        if val_p >= 50_000_000:
            precio_cop = val_p
            precio_gbp = round(precio_cop / float(tasa_cop), 0)
            moneda_detectada = "COP"
    if precio_gbp is None and m_cop_pref:
        val_c = _limpiar_numero(m_cop_pref.group(1))
        if m_cop_pref.group(2):
            val_c *= 1_000_000.0
        if val_c >= 50_000_000:
            precio_cop = val_c
            precio_gbp = round(precio_cop / float(tasa_cop), 0)
            moneda_detectada = "COP"

    # 4B. Libras esterlinas (ej. "£420,000", "420k pounds", "250.000 libras", "165000 gbp")
    if precio_gbp is None:
        m_pk = re.search(r"£?\s*(\d{2,4}(?:[\.,]\d)?)\s*k\b(?:\s*(?:pounds|gbp|libras))?", t_low)
        if m_pk:
            val_k = _limpiar_numero(m_pk.group(1))
            if 40 <= val_k <= 5000:
                precio_gbp = val_k * 1000.0
        if precio_gbp is None:
            m_gbp = re.search(r"(?:£|gbp)\s*(\d[\d\.,]{3,10})", t_low)
            if not m_gbp:
                m_gbp = re.search(r"\b(\d[\d\.,]{3,10})\s*(?:libras|pounds|pound|gbp)\b", t_low)
            if not m_gbp:
                m_gbp = re.search(r"(?:vivienda|casa|precio|propiedad|valor|house|home|property|price|worth|for|por)\s*(?:de\s*)?£?\s*(\d[\d\.,]{4,10})", t_low)
            if m_gbp:
                cand = _limpiar_numero(m_gbp.group(1))
                if cand >= 10_000:
                    precio_gbp = cand

    # 5. Detectar LTV / TLV (%) o Cuota inicial / Entrada / Deposit / Down payment
    ltv_pct = 90.0
    m_ltv = re.search(r"(?:ltv|tlv|financiaci[oó]n|financing)\s*(?:de\s*|del\s*|of\s*)?(\d{1,3}(?:[\.,]\d+)?)\s*%", t_low)
    if not m_ltv:
        m_ltv = re.search(r"(\d{1,3}(?:[\.,]\d+)?)\s*%\s*(?:ltv|tlv|financiaci[oó]n|financing)", t_low)
    if m_ltv:
        ltv_pct = float(m_ltv.group(1).replace(",", "."))
    else:
        m_dep = re.search(r"(?:entrada|cuota inicial|inicial|enganche|deposit|down\s*payment)\s*(?:de\s*|del\s*|of\s*)?(\d{1,2}(?:[\.,]\d+)?)\s*%", t_low)
        if not m_dep:
            m_dep = re.search(r"(\d{1,2}(?:[\.,]\d+)?)\s*%\s*(?:de\s*)?(?:entrada|cuota inicial|deposit|down\s*payment)", t_low)
        if m_dep:
            ltv_pct = max(5.0, min(100.0, 100.0 - float(m_dep.group(1).replace(",", "."))))

    # 6. Extraer comision (£ / fee / sin comision / zero fee)
    comisiones = 999.0
    if any(k in t_low for k in ("sin comision", "sin comisión", "comision 0", "comisión 0", "zero fee", "no fee", "0 fee", "fee 0", "£0")):
        comisiones = 0.0
    else:
        m_com = re.search(r"(?:comisi[oó]n|comisiones|fee|fees)\s*(?:de\s*|of\s*)?£?\s*(\d[\d\.,]{0,6})", t_low)
        if m_com:
            comisiones = _limpiar_numero(m_com.group(1))

    # 7. Plazo total del credito (5-40 anios = 60-480 meses max) y periodo inicial F2 (12-300 meses)
    meses = 24
    plazo_anios = 25
    plazo_meses = None

    # Buscar todos los numeros con "meses" o "months"
    for m_m in re.finditer(r"(\d{2,3})\s*(?:meses|months|month|m\b)", t_low):
        v_m = int(m_m.group(1))
        contexto_cerca = t_low[max(0, m_m.start() - 16):min(len(t_low), m_m.end() + 18)]
        es_tasa_inicial = any(w in contexto_cerca for w in ("fixed", "fijo", "fija", "discounted", "descuento", "tracker", "tasa inicial", "periodo inicial", "período inicial", "initial rate", "initial period", "intro", "promo"))
        if v_m >= 120 and not es_tasa_inicial:
            plazo_meses = min(PLAZO_MAX_MESES, max(PLAZO_MIN_MESES, v_m))
        elif MESES_INICIAL_MIN_F2 <= v_m <= MESES_INICIAL_MAX_F2:
            if es_tasa_inicial or v_m in (12, 24, 36, 48, 60):
                meses = v_m
            elif v_m >= 60 and plazo_meses is None:
                plazo_meses = v_m

    # Buscar anios / years
    for m_an in re.finditer(r"(\d{1,2})\s*(?:a[ñn]os|anios|years|year|yrs|yr)\b", t_low):
        v_an = int(m_an.group(1))
        contexto_cerca = t_low[max(0, m_an.start() - 20):min(len(t_low), m_an.end() + 20)]
        es_inicial = any(w in contexto_cerca for w in ("fixed", "fijo", "fija", "discounted", "tracker", "inicial", "initial"))
        if v_an in (2, 3, 5, 10) and (es_inicial or v_an <= 5):
            meses = v_an * 12
        elif PLAZO_MIN_ANIOS <= v_an <= PLAZO_MAX_ANIOS:
            plazo_anios = v_an
            plazo_meses = v_an * 12

    if plazo_meses is not None:
        plazo_anios = max(PLAZO_MIN_ANIOS, min(PLAZO_MAX_ANIOS, int(round(plazo_meses / 12.0))))
    else:
        plazo_meses = plazo_anios * 12

    # 8. Tipo de tasa y perfil comprador (ES + EN)
    tipo = "fixed"
    if any(k in t_low for k in ("discounted", "discount", "descuento")):
        tipo = "discounted"
    elif any(k in t_low for k in ("tracker", "indexado", "indexada")):
        tipo = "tracker"
    elif any(k in t_low for k in ("variable", "svr")):
        tipo = "variable"

    subtitulo = "First time buyer"
    if any(k in t_low for k in ("remortgage", "rehipoteca", "refinanciacion", "refinanciación")):
        subtitulo = "Remortgage"
    elif any(k in t_low for k in ("buy to let", "btl", "alquiler", "inversion", "inversión", "rental", "investor")):
        subtitulo = "Buy to let"
    elif any(k in t_low for k in ("home purchase", "compra estandar", "compra estándar", "home mover")):
        subtitulo = "Home Purchase"

    # 9. Variacion de precio (%)
    precio_var_pct = None
    m_var = re.search(r"(?:variaci[oó]n|subida|ca[ií]da|cambio|growth|change)[^0-9\-\+]*([\-\+]?\d+(?:[\.,]\d+)?)\s*%", t_low)
    if not m_var:
        m_var = re.search(r"([\+\-]\d+(?:[\.,]\d+)?)\s*%", t_low)
    if m_var:
        precio_var_pct = float(m_var.group(1).replace(",", "."))

    return {
        "texto_original": texto,
        "idioma_detectado": idioma,
        "moneda_detectada": moneda_detectada,
        "region": region_detectada or "London",
        "ciudad": ciudad_detectada,
        "incluye_hipoteca": True,
        "precio_gbp": precio_gbp,
        "precio_cop": precio_cop,
        "precio_var_pct": precio_var_pct,
        "ltv_pct": ltv_pct,
        "plazo_anios": plazo_anios,
        "plazo_meses": plazo_meses,
        "comisiones": comisiones,
        "meses": meses,
        "tipo": tipo,
        "subtitulo": subtitulo,
        "metodo": metodo_detectado,
    }


def _diagnostico_usuario_claro(esfuerzo_pct: float) -> tuple[str, str]:
    """Devuelve (nivel_riesgo, mensaje_claro) comprensible para cualquier persona."""
    if esfuerzo_pct <= 30.0:
        return "saludable", "Accesible: la cuota cabe dentro del límite bancario recomendado (≤ 30 % del sueldo)."
    if esfuerzo_pct < 56.36:
        return "moderado", "Esfuerzo moderado: supera el 30 % ideal, pero queda por debajo de la mediana británica (56,4 %)."
    if esfuerzo_pct < 100.0:
        return "alto", "Esfuerzo alto: consume más de la mitad del salario joven mensual; conviene subir la cuota inicial o ampliar el plazo."
    return "critico", "Crítico (inaccesible individualmente): la cuota iguala o supera el 100 % del salario joven mensual; requiere compra en pareja o subsidio."


def ejecutar_prediccion_integral(
    region: str = "London",
    ciudad: str | None = None,
    precio_gbp: float | None = None,
    precio_cop: float | None = None,
    moneda: str = "GBP",
    tasa_cop: float = TASA_GBP_COP_DEFAULT,
    precio_var_pct: float | None = None,
    ltv_pct: float = 90.0,
    plazo_anios: int = 25,
    plazo_meses: int | None = None,
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
    """Ejecuta prediccion dual acoplada con GBP + COP, LTV/TLV, plazo hasta 480 meses (40 anios),
    5 vecinos KNN, comparativa de 10 regiones y registro en Azure PostgreSQL."""
    load_dotenv(RAIZ / ".env", override=False)
    tasa_cambio = float(tasa_cop) if tasa_cop and float(tasa_cop) > 0 else TASA_GBP_COP_DEFAULT
    moneda_activa = "COP" if str(moneda).upper() == "COP" else "GBP"

    if precio_gbp is None and precio_cop is not None:
        precio_gbp = float(precio_cop) / tasa_cambio
        moneda_activa = "COP"

    if plazo_meses is not None:
        if not (PLAZO_MIN_MESES <= int(plazo_meses) <= PLAZO_MAX_MESES):
            raise ValueError(f"El plazo total en meses debe estar entre {PLAZO_MIN_MESES} y {PLAZO_MAX_MESES} meses (5 a 40 años).")
        plazo_anios = int(round(int(plazo_meses) / 12.0))
        plazo_meses_efectivo = int(plazo_meses)
    else:
        plazo_meses_efectivo = int(plazo_anios) * 12

    parsed_txt = None
    if texto_escrito:
        parsed_txt = interpretar_consulta_escrita(texto_escrito, tasa_cop=tasa_cambio)
        region = parsed_txt["region"]
        ciudad = parsed_txt["ciudad"] or ciudad
        if parsed_txt["precio_gbp"] is not None:
            precio_gbp = parsed_txt["precio_gbp"]
        if parsed_txt["moneda_detectada"] == "COP":
            moneda_activa = "COP"
        if parsed_txt["precio_var_pct"] is not None:
            precio_var_pct = parsed_txt["precio_var_pct"]
        ltv_pct = parsed_txt["ltv_pct"]
        plazo_anios = parsed_txt["plazo_anios"]
        plazo_meses_efectivo = parsed_txt["plazo_meses"]
        comisiones = parsed_txt["comisiones"]
        meses = parsed_txt["meses"]
        tipo = parsed_txt["tipo"]
        subtitulo = parsed_txt["subtitulo"]
        if parsed_txt["metodo"] != "todos":
            metodo = parsed_txt["metodo"]

    if not (0 < float(ltv_pct) <= 100.0):
        raise ValueError("La financiación (LTV/TLV) debe ser mayor que 0 % y como máximo 100 %.")
    if not (PLAZO_MIN_MESES <= plazo_meses_efectivo <= PLAZO_MAX_MESES):
        raise ValueError(f"El plazo máximo permitido es {PLAZO_MAX_MESES} meses ({PLAZO_MAX_ANIOS} años).")
    meses_f2 = max(MESES_INICIAL_MIN_F2, min(MESES_INICIAL_MAX_F2, int(meses)))

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
    res_hip = predecir.evaluar_hipoteca_con_umbrales(comisiones, meses_f2, tipo, subtitulo)

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
    salario_joven_mensual = salario_joven_anual / 12.0
    tasa_pred_f2 = float(res_hip["tipo1_tasa_estimada_pct"])

    # Formula francesa exacta usando plazo_meses_efectivo (hasta 480 meses / 40 anios)
    plazo_anios_exacto = plazo_meses_efectivo / 12.0
    cuota_pmt_caso = float(
        indicadores.cuota_mensual_pmt(
            float(precio_ajustado),
            tasa_pred_f2,
            plazo_anios_exacto,
            float(ltv_pct) / 100.0,
            float(comisiones),
        )
    )
    monto_prestamo = precio_ajustado * (float(ltv_pct) / 100.0)
    cuota_inicial = precio_ajustado - monto_prestamo
    total_pagado = cuota_pmt_caso * plazo_meses_efectivo
    total_intereses = max(0.0, total_pagado - monto_prestamo - float(comisiones))
    pct_intereses_total = round((total_intereses / total_pagado) * 100.0, 1) if total_pagado > 0 else 0.0

    esfuerzo_pmt_pct = round((cuota_pmt_caso / salario_joven_mensual) * 100.0, 1)
    anos_salario_caso = round(precio_ajustado / salario_joven_anual, 2)
    saldo_libre_mensual = salario_joven_mensual - cuota_pmt_caso
    nivel_riesgo, diag_claro = _diagnostico_usuario_claro(esfuerzo_pmt_pct)

    ctx = ctx_prev
    registro = {
        "tipo_consulta": "escrita" if texto_escrito else "opciones",
        "region": reg_canon,
        "entrada_json": {
            "ciudad": c_info["ciudad"] if c_info else None,
            "precio_gbp": round(precio_ajustado, 0),
            "moneda": moneda_activa,
            "ltv_pct": ltv_pct,
            "plazo_meses": plazo_meses_efectivo,
            "comisiones": comisiones,
            "meses_inicial": meses_f2,
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
        "fuente_precio": (
            f"Introducido por el visitante ({'Pesos COP → GBP' if moneda_activa == 'COP' else 'Libras GBP'})"
            if precio_gbp is not None
            else ("Mediana histórica de la ciudad" if c_info else "Mediana histórica regional")
        ),
        "modo_entrada": "A Escrita (Texto Libre Bilingüe)" if texto_escrito else "Por Opciones (Selectores)",
        "moneda_entrada": moneda_activa,
        "metodo_seleccionado": metodo,
        "interpretacion_texto": parsed_txt,
        "conectado_azure": ctx["conectado_azure"],
        "fuente_datos_conectada": ctx["fuente_datos"],
        "sql_ejecutado": ctx["sql_ejecutado"],
        "historial_azure": ctx["historial_azure"],
        "ciudad_consultada": ctx["ciudad_info"],
        "limites_credito_uk": {
            "plazo_min_meses": PLAZO_MIN_MESES,
            "plazo_max_meses": PLAZO_MAX_MESES,
            "plazo_max_anios": PLAZO_MAX_ANIOS,
            "plazo_estandar_f2_meses": 300,
            "meses_inicial_min_f2": MESES_INICIAL_MIN_F2,
            "meses_inicial_max_f2": MESES_INICIAL_MAX_F2,
            "ltv_recomendado_jovenes_pct": "85 % – 95 % (cuota inicial 5 % – 15 %)",
        },
        "calculo_financiero_acoplado": {
            "tasa_cambio_gbp_cop": round(tasa_cambio, 2),
            "precio_vivienda_gbp": round(precio_ajustado, 0),
            "precio_vivienda_cop": round(precio_ajustado * tasa_cambio, 0),
            "ltv_pct": round(float(ltv_pct), 1),
            "cuota_inicial_pct": round(100.0 - float(ltv_pct), 1),
            "cuota_inicial_gbp": round(cuota_inicial, 0),
            "cuota_inicial_cop": round(cuota_inicial * tasa_cambio, 0),
            "monto_prestamo_gbp": round(monto_prestamo, 0),
            "monto_prestamo_cop": round(monto_prestamo * tasa_cambio, 0),
            "plazo_anios": round(plazo_anios_exacto, 1),
            "plazo_meses": plazo_meses_efectivo,
            "meses_inicial_f2": meses_f2,
            "tasa_hipotecaria_usada_pct": tasa_pred_f2,
            "cuota_mensual_pmt_gbp": round(cuota_pmt_caso, 0),
            "cuota_mensual_pmt_cop": round(cuota_pmt_caso * tasa_cambio, 0),
            "salario_joven_anual_gbp": round(salario_joven_anual, 0),
            "salario_joven_anual_cop": round(salario_joven_anual * tasa_cambio, 0),
            "salario_joven_mensual_gbp": round(salario_joven_mensual, 0),
            "salario_joven_mensual_cop": round(salario_joven_mensual * tasa_cambio, 0),
            "saldo_libre_mensual_gbp": round(saldo_libre_mensual, 0),
            "saldo_libre_mensual_cop": round(saldo_libre_mensual * tasa_cambio, 0),
            "total_pagado_gbp": round(total_pagado, 0),
            "total_pagado_cop": round(total_pagado * tasa_cambio, 0),
            "total_intereses_gbp": round(total_intereses, 0),
            "total_intereses_cop": round(total_intereses * tasa_cambio, 0),
            "pct_intereses_del_total": pct_intereses_total,
            "esfuerzo_pmt_acoplado_pct": esfuerzo_pmt_pct,
            "anos_salario_requeridos": anos_salario_caso,
            "cumple_umbral_30pct": esfuerzo_pmt_pct <= 30.0,
            "nivel_riesgo": nivel_riesgo,
            "diagnostico_sencillo": diag_claro,
        },
        "comparativa_10_regiones": comparativa_10_regiones(),
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
    parser = argparse.ArgumentParser(description="Predictor Bilingue UK Housing conectado a Azure PostgreSQL")
    parser.add_argument("--texto", type=str, default=None)
    parser.add_argument("--region", type=str, default="London")
    parser.add_argument("--ciudad", type=str, default=None)
    parser.add_argument("--precio_gbp", type=float, default=None)
    parser.add_argument("--precio_cop", type=float, default=None)
    parser.add_argument("--ltv_pct", type=float, default=90.0)
    parser.add_argument("--plazo_anios", type=int, default=25)
    parser.add_argument("--plazo_meses", type=int, default=None)
    parser.add_argument("--comisiones", type=float, default=999.0)
    parser.add_argument("--meses", type=int, default=24)
    parser.add_argument("--tipo", type=str, default="fixed")
    parser.add_argument("--subtitulo", type=str, default="First time buyer")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out = ejecutar_prediccion_integral(
        region=args.region,
        ciudad=args.ciudad,
        precio_gbp=args.precio_gbp,
        precio_cop=args.precio_cop,
        ltv_pct=args.ltv_pct,
        plazo_anios=args.plazo_anios,
        plazo_meses=args.plazo_meses,
        comisiones=args.comisiones,
        meses=args.meses,
        tipo=args.tipo,
        subtitulo=args.subtitulo,
        texto_escrito=args.texto,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
