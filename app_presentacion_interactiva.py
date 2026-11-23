"""Servidor ligero e interactivo para la Presentacion y Simulador de Prediccion en Vivo (Fase 2 y 3).

Conecta Presentacion_Interactiva_Proyecto_UK.html con:
  1. Azure Database for PostgreSQL (pg-actividad-db-uk26) mediante DATABASE_URL_READONLY / DATABASE_URL_APP.
  2. Las 17 tablas CSV limpias ("Excel") en data/integrado/ como respaldo instantaneo.
  3. Los 4 modelos .joblib preentrenados en data/modelos/ (< 5 ms de latencia).
  4. Soporte simultaneo para prediccion POR OPCIONES (selectores/sliders/grafica movible) y A ESCRITA (texto libre).

Uso:
  python app_presentacion_interactiva.py              # Abre servidor en http://localhost:8765
  python app_presentacion_interactiva.py --puerto 8080
  python app_presentacion_interactiva.py --auto-test  # Verifica endpoints /api/contexto y /api/predecir
"""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

RAIZ = Path(__file__).resolve().parent
if str(RAIZ / "script") not in sys.path:
    sys.path.insert(0, str(RAIZ / "script"))

from azure import predecir_azure
from proyecto import config, predecir, sistema_recursos


def obtener_payload_contexto() -> dict:
    """Devuelve resumen de las tablas en Azure PostgreSQL / CSV para alimentar la presentacion."""
    contrato = json.loads(config.RUTA_CONTRATO_PREDICCION.read_text(encoding="utf-8"))
    ctx_az = predecir_azure.consultar_contexto_bd_o_csv(region="London")
    ciudades = []
    if config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV.exists():
        df_c = pd.read_csv(config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV)
        col_c = "ciudad" if "ciudad" in df_c.columns else "town_city"
        top_c = (
            df_c.sort_values(["anio", "n_transacciones"], ascending=[False, False])
            .drop_duplicates(subset=[col_c])
            .head(100)
        )
        for _, r in top_c.iterrows():
            ciudades.append(
                {
                    "ciudad": str(r[col_c]),
                    "region": str(r["region"]),
                    "anio": int(r["anio"]),
                    "n_transacciones": int(r["n_transacciones"]),
                    "precio_mediana": float(r["precio_mediana"]),
                }
            )
    return {
        "conectado_azure": ctx_az["conectado_azure"],
        "fuente_datos": ctx_az["fuente_datos"],
        "historial_azure": ctx_az["historial_azure"],
        "regiones_validas": contrato["regiones_validas"],
        "perfiles_regionales": contrato["perfiles_regionales_referencia"],
        "ciudades_destacadas": ciudades,
        "guia_metodos_y_aciertos": predecir.tabla_guia_metodos_y_aciertos().to_dict(orient="records"),
    }


class PresentacionRequestHandler(SimpleHTTPRequestHandler):
    """Manejador HTTP que sirve la presentacion HTML, figuras PNG y API JSON de prediccion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RAIZ), **kwargs)

    def _enviar_json(self, datos: dict, status: int = 200) -> None:
        cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.path = "/Presentacion_Interactiva_Proyecto_UK.html"
            return super().do_GET()
        if parsed.path == "/api/contexto":
            self._enviar_json(obtener_payload_contexto())
            return
        if parsed.path == "/api/predecir":
            qs = parse_qs(parsed.query)
            texto = qs.get("texto", [None])[0]
            region = qs.get("region", ["London"])[0]
            ciudad = qs.get("ciudad", [None])[0]
            p_gbp = qs.get("precio_gbp", [None])[0]
            p_var = qs.get("precio_var_pct", [None])[0]
            ltv = float(qs.get("ltv_pct", ["90"])[0])
            plazo = int(qs.get("plazo_anios", ["25"])[0])
            n_tx = qs.get("n_transacciones", [None])[0]
            com = float(qs.get("comisiones", ["999"])[0])
            mes = int(qs.get("meses", ["24"])[0])
            tip = qs.get("tipo", ["fixed"])[0]
            sub = qs.get("subtitulo", ["First time buyer"])[0]
            met = qs.get("metodo", ["todos"])[0]
            url_az = qs.get("url", [None])[0]

            res = predecir_azure.ejecutar_prediccion_integral(
                region=region,
                ciudad=ciudad if ciudad else None,
                precio_gbp=float(p_gbp) if p_gbp not in (None, "") else None,
                precio_var_pct=float(p_var) if p_var not in (None, "") else None,
                ltv_pct=ltv,
                plazo_anios=plazo,
                n_transacciones=float(n_tx) if n_tx not in (None, "") else None,
                evaluar_hipoteca=True,
                comisiones=com,
                meses=mes,
                tipo=tip,
                subtitulo=sub,
                metodo=met,
                texto_escrito=texto if texto else None,
                url_azure=url_az if url_az else None,
            )
            self._enviar_json(res)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/predecir":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            body = json.loads(raw) if raw else {}
            res = predecir_azure.ejecutar_prediccion_integral(
                region=body.get("region", "London"),
                ciudad=body.get("ciudad") or None,
                precio_gbp=float(body["precio_gbp"]) if body.get("precio_gbp") not in (None, "") else None,
                precio_var_pct=float(body["precio_var_pct"]) if body.get("precio_var_pct") not in (None, "") else None,
                ltv_pct=float(body.get("ltv_pct", 90.0)),
                plazo_anios=int(body.get("plazo_anios", 25)),
                n_transacciones=float(body["n_transacciones"]) if body.get("n_transacciones") not in (None, "") else None,
                lluvia_anom=float(body["lluvia_anom"]) if body.get("lluvia_anom") not in (None, "") else None,
                tmax_anom=float(body["tmax_anom"]) if body.get("tmax_anom") not in (None, "") else None,
                evaluar_hipoteca=True,
                comisiones=float(body.get("comisiones", 999.0)),
                meses=int(body.get("meses", 24)),
                tipo=str(body.get("tipo", "fixed")),
                subtitulo=str(body.get("subtitulo", "First time buyer")),
                metodo=str(body.get("metodo", "todos")),
                texto_escrito=body.get("texto") or None,
                url_azure=body.get("url_azure") or None,
            )
            self._enviar_json(res)
            return
        self._enviar_json({"error": "Ruta no encontrada"}, status=404)


def ejecutar_auto_test() -> int:
    """Verifica que tanto la prediccion por opciones como a escrita funcionan (con Azure o CSV)."""
    sistema_recursos.configurar_entorno_portable()
    print("=" * 84)
    print("AUTO-TEST DEL MOTOR INTERACTIVO DE PRESENTACION Y PREDICCION (OPCIONES + ESCRITA)")
    print("=" * 84)
    ctx = obtener_payload_contexto()
    print(f"  [OK] Fuente activa   : {ctx['fuente_datos']} (Azure en vivo={ctx['conectado_azure']})")
    print(f"  [OK] Contexto cargado: {len(ctx['regiones_validas'])} regiones, "
          f"{len(ctx['ciudades_destacadas'])} ciudades destacadas, "
          f"{len(ctx['guia_metodos_y_aciertos'])} metodos en guia de aciertos.")

    r_opc = predecir_azure.ejecutar_prediccion_integral(
        region="London", precio_var_pct=5.0, comisiones=999.0, meses=24, tipo="fixed", subtitulo="First time buyer"
    )
    print(f"  [OK] Prediccion Por Opciones (London): CART={r_opc['prediccion_regional_dual']['prediccion']} | "
          f"Ridge={r_opc['prediccion_regional_dual']['tipo1_regresion_esfuerzo_pct']:.1f}% | "
          f"Tasa F2 RF={r_opc['prediccion_hipoteca_dual']['tipo1_tasa_estimada_pct']:.2f}% | "
          f"Cuota PMT=£{r_opc['calculo_financiero_acoplado']['cuota_mensual_pmt_gbp']:,.0f}/mes")

    r_txt = predecir_azure.ejecutar_prediccion_integral(
        texto_escrito="Quiero comprar vivienda en Manchester como First time buyer con comision 0 a 60 meses fixed explicado con KNN"
    )
    print(f"  [OK] Prediccion A Escrita ('Manchester...'): Region={r_txt['prediccion_regional_dual']['region']} | "
          f"CART={r_txt['prediccion_regional_dual']['prediccion']} | "
          f"Ridge={r_txt['prediccion_regional_dual']['tipo1_regresion_esfuerzo_pct']:.1f}% | "
          f"KNN={r_txt['knn_vecinos_en_vivo']['prediccion_knn']} ({r_txt['knn_vecinos_en_vivo']['votos_alto']}/5 votos) | "
          f"Tasa F2 RF={r_txt['prediccion_hipoteca_dual']['tipo1_tasa_estimada_pct']:.2f}%")
    print("=" * 84)
    return 0


def main() -> int:
    sistema_recursos.configurar_entorno_portable()
    parser = argparse.ArgumentParser(description="Servidor de la Presentacion Interactiva UK Housing")
    parser.add_argument("--puerto", type=int, default=8765, help="Puerto HTTP local (por defecto 8765)")
    parser.add_argument("--auto-test", action="store_true", help="Ejecutar prueba rapida sin bloquear consola")
    args = parser.parse_args()

    if args.auto_test:
        return ejecutar_auto_test()

    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto), PresentacionRequestHandler)
    print("=" * 84)
    print("PRESENTACION INTERACTIVA + SIMULADOR DE PREDICCION EN VIVO (AZURE POSTGRESQL / CSV)")
    print(f"  -> Abre en tu navegador: http://localhost:{args.puerto}")
    print("  -> Presiona Ctrl+C para detener el servidor.")
    print("=" * 84)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        servidor.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
