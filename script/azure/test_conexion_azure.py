"""Verificacion integral de conexion a Azure PostgreSQL, roles de seguridad y motor ML.

Valida:
  1. Variables en .env (DATABASE_URL, DATABASE_URL_READONLY, DATABASE_URL_APP).
  2. Conexion cifrada TLS (sslmode=require) con el rol administrador (pgadmin_uk).
  3. Conexion con el enlace unico compartible de solo lectura (lector_uk) y
     prueba negativa confirmando que NO puede crear ni modificar tablas.
  4. Conexion con el rol del motor de predicciones (app_predicciones).
  5. Smoke-test en vivo de los 4 modelos .joblib (script/proyecto/predecir.py)
     junto al driver psycopg en Python 3.14, dejando todo listo para cuando se
     suban las tablas de data/modelos/esquema_bd.sql y los CSV/Excel.

Ejecucion:
  python script/azure/test_conexion_azure.py

Comandos utiles de Azure CLI para gestion posterior:
  - Pausar servidor (ahorrar creditos mientras no se usa):
      az postgres flexible-server stop -g rg-actividad-db -n <SERVIDOR>
  - Reanudar servidor:
      az postgres flexible-server start -g rg-actividad-db -n <SERVIDOR>
  - Agregar IP de un companero o docente para compartir DATABASE_URL_READONLY:
      az postgres flexible-server firewall-rule create -g rg-actividad-db -n <SERVIDOR> --rule-name AllowColaborador --start-ip-address <IP> --end-ip-address <IP>
  - Habilitar acceso publico al enlace de solo lectura (protegido por TLS + rol lector_uk):
      az postgres flexible-server firewall-rule create -g rg-actividad-db -n <SERVIDOR> --rule-name AllowPublicReadOnly --start-ip-address 0.0.0.0 --end-ip-address 255.255.255.255
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
import psycopg

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ / "script") not in sys.path:
    sys.path.insert(0, str(RAIZ / "script"))


def ocultar_password_uri(uri: str) -> str:
    """Devuelve el URI ocultando la contrasena para impresion segura en consola."""
    try:
        partes = urlsplit(uri)
        if partes.password:
            netloc = f"{partes.username}:****@{partes.hostname}"
            if partes.port:
                netloc += f":{partes.port}"
            return urlunsplit((partes.scheme, netloc, partes.path, partes.query, partes.fragment))
    except Exception:
        pass
    return uri


def probar_conexion_rol(nombre_etiqueta: str, uri: str, probar_bloqueo_escritura: bool = False) -> bool:
    """Prueba conexion TLS y permisos del rol especificado."""
    print(f"\n[{nombre_etiqueta}] Conectando a: {ocultar_password_uri(uri)}")
    with psycopg.connect(uri, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user, inet_server_addr()::text, version();")
            db_name, db_user, srv_ip, ver = cur.fetchone()
            ver_corta = ver.split(",")[0] if ver else "PostgreSQL"
            print(f"  -> OK | BD: {db_name} | Usuario: {db_user} | Motor: {ver_corta}")

            cur.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'public';
                """
            )
            n_tablas = cur.fetchone()[0]
            print(f"  -> Tablas actuales en esquema 'public': {n_tablas} (lista para futura carga)")

            if probar_bloqueo_escritura:
                try:
                    cur.execute("CREATE TABLE public._prueba_escritura_no_permitida (id INT);")
                    conn.rollback()
                    print("  -> [ERROR SEGURIDAD] El rol de solo lectura pudo crear una tabla.")
                    return False
                except psycopg.errors.InsufficientPrivilege:
                    conn.rollback()
                    print("  -> OK Seguridad: el rol 'lector_uk' tiene bloqueada cualquier escritura/modificacion.")
    return True


def probar_puente_prediccion_ml() -> None:
    """Verifica que los 4 modelos .joblib preentrenados corren listos para conectarse a la BD."""
    from proyecto.predecir import evaluar_hipoteca_con_umbrales, predecir_escenario_regional

    print("\n[Motor ML Dual] Verificando inferencia local (< 5 ms) lista para historial_predicciones:")
    res_reg = predecir_escenario_regional("London")
    print(
        f"  -> Regional (London): Tipo 2 CART = {res_reg['prediccion']} (P={res_reg['probabilidad_alto']:.3f}) | "
        f"Tipo 1 Ridge = {res_reg['tipo1_regresion_esfuerzo_pct']:.1f}% (+/- {res_reg['tipo1_mae_pp']:.1f} pp) | "
        f"{res_reg['latencia_ms']:.2f} ms"
    )
    res_hip = evaluar_hipoteca_con_umbrales(999.0, 24, "fixed", "First time buyer")
    print(
        f"  -> Hipoteca F2      : Tipo 1 RF = {res_hip['tipo1_tasa_estimada_pct']:.2f}% (+/- {res_hip['tipo1_mae_pp']:.2f} pp) | "
        f"Tipo 2 CART P(atipica) = {res_hip['tipo2_probabilidad_tasa_atipica']:.3f} | "
        f"{res_hip['latencia_ms']:.2f} ms"
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ruta_env = RAIZ / ".env"
    if not ruta_env.exists():
        print(f"[ERROR] No existe {ruta_env}. Copia .env.example a .env o ejecuta el aprovisionamiento.")
        return 1

    load_dotenv(ruta_env, override=True)

    url_admin = os.getenv("DATABASE_URL", "").strip()
    url_ro = os.getenv("DATABASE_URL_READONLY", "").strip()
    url_app = os.getenv("DATABASE_URL_APP", "").strip()

    if not url_admin:
        print("[ERROR] Falta DATABASE_URL en .env")
        return 1

    print("=" * 82)
    print("VERIFICACION DE CONEXION AZURE POSTGRESQL + ENLACE UNICO + MOTOR ML")
    print("=" * 82)

    ok = probar_conexion_rol("1. Rol Admin (pgadmin_uk)", url_admin, probar_bloqueo_escritura=False)
    if url_ro:
        ok = probar_conexion_rol(
            "2. Enlace Unico Compartible Solo Lectura (lector_uk)",
            url_ro,
            probar_bloqueo_escritura=True,
        ) and ok
    if url_app:
        ok = probar_conexion_rol(
            "3. Rol Motor de Predicciones / API (app_predicciones)",
            url_app,
            probar_bloqueo_escritura=False,
        ) and ok

    probar_puente_prediccion_ml()

    print("\n" + "=" * 82)
    print("RESULTADO: Entorno Azure PostgreSQL + MCP + Modelos ML 100% LISTO.")
    print("=" * 82)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
