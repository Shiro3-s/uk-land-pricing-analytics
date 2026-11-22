"""API pública del simulador. Inicio local y Render: python web_api.py."""
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "script"))
load_dotenv(ROOT / ".env")

from azure import datos_web, predecir_azure
from proyecto import config, predecir

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 8192
CONTRATO = predecir.cargar_artefactos_prediccion()["contrato"]
REGIONES = CONTRATO["regiones_validas"]
LIMITES = {"precio_gbp": (1, 100_000_000), "precio_var_pct": (-99, 1000), "ltv_pct": (1, 100),
           "plazo_anios": (1, 60), "comisiones": (0, 1_000_000), "meses": (1, 720),
           "n_transacciones": (1, 100_000_000), "lluvia_anom": (-10000, 10000), "tmax_anom": (-100, 100)}
TEXTOS = {"ciudad", "texto", "region", "tipo", "subtitulo", "metodo"}


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
    ciudades = datos_web.leer_csv(config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV)
    top = ciudades.sort_values(["anio", "n_transacciones"], ascending=False).drop_duplicates(["ciudad","region"]).head(100)
    return {"conectado_azure": fuente == "Azure PostgreSQL", "fuente_datos": fuente,
            "regiones_validas": REGIONES, "perfiles_regionales": CONTRATO["perfiles_regionales_referencia"],
            "anios": sorted({r["anio"] for r in filas}), "ciudades_destacadas": top.to_dict("records"),
            "guia_metodos_y_aciertos": predecir.tabla_guia_metodos_y_aciertos().to_dict("records")}


@app.route("/api/predecir", methods=["GET","POST"])
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
            if clave in ("meses","plazo_anios"):
                if not numero.is_integer():
                    abort(400, f"{clave} requiere un número entero.")
                numero = int(numero)
            opciones[clave] = numero
    if opciones.get("region", "London") not in REGIONES:
        abort(400, "Región desconocida.")
    meta = CONTRATO["tipos_de_prediccion_implementados"]["tipo_1_regresion_continua"]["tasa_hipotecaria_f2"]
    for key, allowed in [("tipo",meta["tipos_validos"]), ("subtitulo",meta["subtitulos_validos"])]:
        if key in opciones and opciones[key] not in allowed:
            abort(400, f"Opción desconocida para {key}.")
    try:
        return predecir_azure.ejecutar_prediccion_integral(**opciones, registrar=request.method == "POST")
    except ValueError as error:
        abort(400, str(error))


def datos_filtrados():
    if set(request.args) - {"region", "anio"}:
        abort(400, "Filtro desconocido.")
    region = request.args.get("region") or None
    anio = request.args.get("anio") or None
    if region and region not in REGIONES:
        abort(400, "Región desconocida.")
    if anio and (not anio.isascii() or not anio.isdigit() or len(anio) != 4):
        abort(400, "Año no válido.")
    return datos_web.datos_regionales(region, int(anio) if anio else None)


@app.get("/api/datos")
def datos():
    filas, fuente = datos_filtrados()
    return {"fuente": fuente, "filas": filas, "total": len(filas)}


@app.get("/api/datos/exportar.xlsx")
def exportar():
    filas, fuente = datos_filtrados()
    libro = Workbook()
    hoja = libro.active
    hoja.title = "Datos regionales"
    hoja.append(["Año", "Región", "Transacciones", "Precio mediano (GBP)", "Salario joven anual (GBP)", "Años de salario"])
    for celda in hoja[1]:
        celda.font = Font(bold=True, color="FFFFFF")
        celda.fill = PatternFill("solid", fgColor="174C4B")
    for fila in filas:
        hoja.append([fila[k] for k in datos_web.COLUMNAS])
    for letra, ancho in zip("ABCDEF", [10,28,20,25,28,20]):
        hoja.column_dimensions[letra].width = ancho
    hoja.freeze_panes = "A2"
    hoja.auto_filter.ref = hoja.dimensions
    ficha = libro.create_sheet("Fuente")
    for clave, valor in [("Fuente", fuente), ("Región",request.args.get("region") or "Todas"),
                         ("Año",request.args.get("anio") or "Todos"), ("Registros",len(filas))]:
        ficha.append([clave,valor])
    ficha.column_dimensions["B"].width = 30
    archivo = io.BytesIO()
    libro.save(archivo)
    archivo.seek(0)
    return send_file(archivo, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="datos-regionales.xlsx")


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")), threads=3)
