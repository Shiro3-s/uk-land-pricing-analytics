"""API pública del simulador (Render + Azure PostgreSQL + Excel).

Inicio local o en servidor:
  python backend/web_api.py
  python web_api.py          (desde la raíz en deploy/backend)
"""
from datetime import datetime, timezone
import io
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_file
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "script") not in sys.path:
    sys.path.insert(0, str(ROOT / "script"))
load_dotenv(ROOT / ".env")

from azure import datos_web, predecir_azure
from proyecto import config, predecir

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16384
CONTRATO = predecir.cargar_artefactos_prediccion()["contrato"]
REGIONES = CONTRATO["regiones_validas"]
LIMITES = {
    "precio_gbp": (1, 100_000_000),
    "precio_cop": (1_000_000, 500_000_000_000),
    "tasa_cop": (1000, 20000),
    "precio_var_pct": (-99, 1000),
    "ltv_pct": (1, 100),
    "tlv_pct": (1, 100),
    "plazo_anios": (5, 40),
    "plazo_meses": (60, 480),
    "comisiones": (0, 1_000_000),
    "meses": (12, 300),
    "n_transacciones": (1, 100_000_000),
    "lluvia_anom": (-10000, 10000),
    "tmax_anom": (-100, 100),
}
TEXTOS = {"ciudad", "texto", "region", "tipo", "subtitulo", "metodo", "moneda"}


@app.errorhandler(HTTPException)
def error_http(error):
    return jsonify(error=error.description), error.code


@app.after_request
def cabeceras(respuesta):
    respuesta.headers["Cache-Control"] = "no-store"
    respuesta.headers["X-Content-Type-Options"] = "nosniff"
    return respuesta


@app.get("/healthz")
def health():
    return {"estado": "listo"}


@app.get("/api/contexto")
def contexto():
    filas, fuente = datos_web.datos_regionales()
    ciudades, _ = datos_web.datos_ciudades(limite=100)
    return {
        "conectado_azure": fuente == "Azure PostgreSQL",
        "fuente_datos": fuente,
        "azure_servidor": "pg-actividad-db-uk26.postgres.database.azure.com",
        "azure_base_datos": "db_actividad_mineria",
        "tasa_cambio_gbp_cop_default": predecir_azure.TASA_GBP_COP_DEFAULT,
        "limites_credito_uk": {
            "plazo_min_anios": predecir_azure.PLAZO_MIN_ANIOS,
            "plazo_max_anios": predecir_azure.PLAZO_MAX_ANIOS,
            "plazo_min_meses": predecir_azure.PLAZO_MIN_MESES,
            "plazo_max_meses": predecir_azure.PLAZO_MAX_MESES,
            "plazo_estandar_f2_meses": 300,
            "meses_inicial_min_f2": predecir_azure.MESES_INICIAL_MIN_F2,
            "meses_inicial_max_f2": predecir_azure.MESES_INICIAL_MAX_F2,
        },
        "regiones_validas": REGIONES,
        "perfiles_regionales": CONTRATO["perfiles_regionales_referencia"],
        "comparativa_10_regiones": datos_web.comparativa_10_regiones(),
        "anios": sorted({r["anio"] for r in filas}),
        "ciudades_destacadas": ciudades,
        "guia_metodos_y_aciertos": predecir.tabla_guia_metodos_y_aciertos().to_dict("records"),
    }


@app.route("/api/predecir", methods=["GET", "POST"])
def prediccion():
    datos = request.get_json() if request.method == "POST" else request.args.to_dict()
    if not isinstance(datos, dict) or set(datos) - (TEXTOS | set(LIMITES)):
        abort(400, "Campos de predicción desconocidos.")
    opciones = {}
    for clave, valor in datos.items():
        if valor is None or valor == "":
            continue
        if clave in TEXTOS:
            if not isinstance(valor, str) or len(valor) > (1200 if clave == "texto" else 100):
                abort(400, f"Valor de {clave} no válido.")
            opciones["texto_escrito" if clave == "texto" else clave] = valor.strip()
        else:
            try:
                numero = float(valor)
            except (ValueError, TypeError, OverflowError):
                abort(400, f"Valor de {clave} no válido.")
            minimo, maximo = LIMITES[clave]
            if isinstance(valor, bool) or not math.isfinite(numero) or not minimo <= numero <= maximo:
                abort(400, f"{clave} debe estar entre {minimo} y {maximo}.")
            if clave in ("meses", "plazo_anios", "plazo_meses"):
                if not numero.is_integer():
                    abort(400, f"{clave} requiere un número entero.")
                numero = int(numero)
            clave_dest = "ltv_pct" if clave == "tlv_pct" else clave
            opciones[clave_dest] = numero

    if opciones.get("region", "London") not in REGIONES:
        abort(400, "Región desconocida.")
    meta = CONTRATO["tipos_de_prediccion_implementados"]["tipo_1_regresion_continua"]["tasa_hipotecaria_f2"]
    for key, allowed in [("tipo", meta["tipos_validos"]), ("subtitulo", meta["subtitulos_validos"])]:
        if key in opciones and opciones[key] not in allowed:
            abort(400, f"Opción desconocida para {key}.")
    try:
        return predecir_azure.ejecutar_prediccion_integral(**opciones, registrar=request.method == "POST")
    except ValueError as error:
        abort(400, str(error))


def _leer_filtros_datos():
    if set(request.args) - {"region", "anio", "tabla"}:
        abort(400, "Filtro desconocido.")
    region = request.args.get("region") or None
    anio = request.args.get("anio") or None
    tabla = request.args.get("tabla") or "regional"
    if region and region not in REGIONES:
        abort(400, "Región desconocida.")
    if anio and (not anio.isascii() or not anio.isdigit() or len(anio) != 4):
        abort(400, "Año no válido.")
    if tabla not in ("regional", "ciudades", "hipoteca_2022", "metodos"):
        abort(400, "Tabla no reconocida.")
    return region, (int(anio) if anio else None), tabla


@app.get("/api/datos")
def datos():
    region, anio, tabla = _leer_filtros_datos()
    if tabla == "ciudades":
        filas, fuente = datos_web.datos_ciudades(region=region, limite=100)
    elif tabla == "hipoteca_2022":
        filas, fuente = datos_web.datos_hipoteca_2022(region=region)
    elif tabla == "metodos":
        filas = datos_web.guia_metodos_export()
        fuente = "Modelos validados (Azure / ML)"
    else:
        filas, fuente = datos_web.datos_regionales(region, anio)
    return {
        "tabla": tabla,
        "fuente": fuente,
        "azure_servidor": "pg-actividad-db-uk26.postgres.database.azure.com",
        "azure_base_datos": "db_actividad_mineria",
        "filas": filas,
        "total": len(filas),
    }


def _estilizar_hoja(hoja, anchos):
    for celda in hoja[1]:
        celda.font = Font(bold=True, color="FFFFFF")
        celda.fill = PatternFill("solid", fgColor="16483F")
    for letra, ancho in anchos.items():
        hoja.column_dimensions[letra].width = ancho
    hoja.freeze_panes = "A2"
    hoja.auto_filter.ref = hoja.dimensions


def generar_libro_excel_bytes(region=None, anio=None) -> bytes:
    """Construye en memoria el libro Excel multi-hoja (.xlsx, 5 hojas) desde Azure PostgreSQL o CSV."""
    filas_reg, fuente = datos_web.datos_regionales(region, anio)
    filas_ciu, _ = datos_web.datos_ciudades(region=region, limite=100)
    filas_h22, _ = datos_web.datos_hipoteca_2022(region=region)
    filas_met = datos_web.guia_metodos_export()

    libro = Workbook()

    # Hoja 1: Datos Regionales (maestro_regional)
    hoja1 = libro.active
    hoja1.title = "Datos Regionales"
    hoja1.append(["Año", "Región", "Transacciones", "Precio mediano (GBP)", "Salario joven anual (GBP)", "Años de salario"])
    for fila in filas_reg:
        hoja1.append([fila[k] for k in datos_web.COLUMNAS])
    _estilizar_hoja(hoja1, {"A": 10, "B": 28, "C": 18, "D": 24, "E": 26, "F": 18})

    # Hoja 2: Ciudades UK Top 100 (precios_ciudad)
    hoja2 = libro.create_sheet("Ciudades UK Top 100")
    hoja2.append(["Año", "Región", "Ciudad", "Transacciones", "Precio mediano (GBP)"])
    for f in filas_ciu:
        hoja2.append([f.get("anio"), f.get("region"), f.get("ciudad"), f.get("n_transacciones"), f.get("precio_mediana")])
    _estilizar_hoja(hoja2, {"A": 10, "B": 26, "C": 26, "D": 18, "E": 24})

    # Hoja 3: Corte Hipotecario 2022 (hipoteca_2022)
    hoja3 = libro.create_sheet("Corte Hipotecario 2022")
    hoja3.append(["Región", "Precio mediano (£)", "Precio real 2020 (£)", "Salario joven anual (£)", "Tasa inicial (%)", "Cuota mensual (£)", "Esfuerzo (% salario)", "Accesible <=30%"])
    for f in filas_h22:
        hoja3.append([
            f.get("region"), f.get("precio_mediana"), f.get("precio_real_2020"),
            f.get("salario_joven_region"), f.get("tasa_inicial_pct"), f.get("cuota_mensual"),
            round(float(f.get("cuota_pct_salario", 0)) * 100.0, 1), "Sí" if f.get("accesible") else "No"
        ])
    _estilizar_hoja(hoja3, {"A": 28, "B": 18, "C": 18, "D": 22, "E": 16, "F": 18, "G": 20, "H": 16})

    # Hoja 4: Metodos y Aciertos (CON vs SIN etiqueta)
    hoja4 = libro.create_sheet("Metodos y Aciertos")
    hoja4.append(["Paradigma", "Tipo de Predicción", "Método / Algoritmo", "Cantidad Exacta de Aciertos", "Métrica Clave", "Cuándo Utilizar"])
    for m in filas_met:
        hoja4.append([m.get("paradigma"), m.get("tipo"), m.get("metodo"), m.get("aciertos_exactos"), m.get("metrica_clave"), m.get("cuando_usar")])
    _estilizar_hoja(hoja4, {"A": 26, "B": 34, "C": 36, "D": 48, "E": 34, "F": 48})

    # Hoja 5: Fuente Azure
    hoja5 = libro.create_sheet("Fuente Azure")
    hoja5.append(["Campo", "Valor"])
    for clave, valor in [
        ("Fuente de datos", fuente),
        ("Servidor Azure PostgreSQL", "pg-actividad-db-uk26.postgres.database.azure.com"),
        ("Base de datos", "db_actividad_mineria"),
        ("Almacenamiento Excel (Azure Blob)", "https://stactividaduk26.blob.core.windows.net/excel-datos/vivienda-uk-azure-datos.xlsx"),
        ("Filtro Región", region or "Todas las regiones"),
        ("Filtro Año", anio or "Todos los años (2015-2020)"),
        ("Registros regionales exportados", len(filas_reg)),
        ("Tasa de cambio referencia (COP/GBP)", predecir_azure.TASA_GBP_COP_DEFAULT),
        ("Generado (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
    ]:
        hoja5.append([clave, valor])
    _estilizar_hoja(hoja5, {"A": 34, "B": 72})

    archivo = io.BytesIO()
    libro.save(archivo)
    return archivo.getvalue()


@app.get("/api/datos/exportar.xlsx")
def exportar():
    region, anio, _ = _leer_filtros_datos()
    contenido = generar_libro_excel_bytes(region=region, anio=anio)
    return send_file(
        io.BytesIO(contenido),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="vivienda-uk-azure-datos.xlsx",
    )


def serve_app():
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")), threads=3)


if __name__ == "__main__":
    serve_app()
