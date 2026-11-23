"""Fase 3 · Subida idempotente de esquema_bd.sql y las 17 tablas CSV ('Excel') a Azure PostgreSQL.

Ejecuta:
  1. Conexion autenticada TLS (sslmode=require) como administrador (DATABASE_URL / pgadmin_uk).
  2. Creacion/actualizacion de las tablas de data/modelos/esquema_bd.sql (mas tablas auxiliares de resumen).
  3. Carga rapida e idempotente (staging + INSERT de filas faltantes) de las 17 tablas CSV limpias de data/integrado/
     y la muestra de 2.000 transacciones reales en public.transacciones_f1.
  4. Asignacion de permisos SELECT para el enlace unico compartible (lector_uk / DATABASE_URL_READONLY)
     y SELECT + INSERT en historial_predicciones para app_predicciones (DATABASE_URL_APP).

Uso (cuando se active la Fase 3):
  python script/azure/subir_datos_azure.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import psycopg
from psycopg import sql

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ / "script") not in sys.path:
    sys.path.insert(0, str(RAIZ / "script"))

from proyecto import config


TABLAS_CSV_NIVEL1 = [
    ("maestro_regional", config.DATA_INTEGRADO / "accesibilidad_vivienda_joven_uk_v1.csv"),
    ("precios_ciudad", config.RUTA_PRECIOS_CIUDAD_EXPORT_CSV),
    ("hipotecas_catalogo", config.RUTA_HIPOTECAS_EXPORT_CSV),
    ("hipoteca_2022", config.RUTA_HIPOTECA_2022_MUESTRA),
    ("historico_nacional", config.RUTA_HISTORICO_MUESTRA),
    ("torneo_clasificadores", config.RUTA_TORNEO_RESUMEN),
    ("comparativa_regresion_dual", config.RUTA_COMPARATIVA_REGRESION),
    ("segmentos_territoriales", config.RUTA_SEGMENTOS_TERRITORIALES),
    ("segmentos_hipotecas", config.RUTA_SEGMENTOS_HIPOTECAS),
    ("segmentos_transacciones_f1", config.RUTA_SEGMENTOS_TRANSACCIONES_1M),
    ("clustering_benchmark_24gb", config.RUTA_CLUSTERING_24GB),
    ("reglas_asociacion_inmobiliario", config.RUTA_REGLAS_INMOBILIARIO),
    ("reglas_asociacion_hipotecas", config.RUTA_REGLAS_HIPOTECAS),
    ("evaluacion_umbrales", config.RUTA_EVALUACION_UMBRALES),
    ("comparativa_global_modelos", config.RUTA_COMPARATIVA_GLOBAL),
    ("telemetria_rendimiento", config.RUTA_TELEMETRIA_RENDIMIENTO),
    ("auditoria_fuentes_calidad", config.RUTA_AUDITORIA_DATOS),
]


def _cargar_dataframe_con_copy(cur: psycopg.Cursor, nombre_tabla: str, df: pd.DataFrame) -> int:
    """Carga en staging y agrega las filas ausentes sin alterar registros previos."""
    tipo_map = {
        "int64": "BIGINT",
        "int32": "INTEGER",
        "int16": "SMALLINT",
        "float64": "DOUBLE PRECISION",
        "float32": "DOUBLE PRECISION",
        "bool": "BOOLEAN",
    }
    cols_ddl = []
    for col, dt in df.dtypes.items():
        t_sql = tipo_map.get(str(dt), "TEXT")
        cols_ddl.append(sql.SQL("{} {}").format(sql.Identifier(str(col)), sql.SQL(t_sql)))

    tabla = sql.Identifier("public", nombre_tabla)
    columnas = sql.SQL(", ").join(sql.Identifier(str(c)) for c in df.columns)
    cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(tabla, sql.SQL(", ").join(cols_ddl)))
    cur.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(tabla))
    # Heredar tipos incluso cuando el esquema existente es anterior al actual.
    staging = sql.Identifier("carga_" + nombre_tabla)
    cur.execute(sql.SQL("CREATE TEMP TABLE {} ON COMMIT DROP AS SELECT {} FROM {} WITH NO DATA")
                .format(staging, columnas, tabla))
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="", quoting=csv.QUOTE_MINIMAL)
    with cur.copy(sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV)").format(staging, columnas)) as copy:
        copy.write(buf.getvalue())
    faltantes = sql.SQL("SELECT {} FROM {} EXCEPT ALL SELECT {} FROM {}").format(columnas, staging, columnas, tabla)
    cur.execute(sql.SQL("INSERT INTO {} ({}) {} ON CONFLICT DO NOTHING").format(tabla, columnas, faltantes))
    insertadas = cur.rowcount
    cur.execute(sql.SQL("SELECT EXISTS ({})").format(faltantes))
    if cur.fetchone()[0]:
        raise ValueError(f"Conflicto de datos previos en {nombre_tabla}. Se revierte la carga.")
    return insertadas


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv(RAIZ / ".env", override=False)
    url_admin = os.getenv("DATABASE_URL", "").strip()
    if not url_admin:
        print("[ERROR] No se encontro DATABASE_URL en .env")
        return 1

    ro_user = os.getenv("AZURE_PG_READONLY_USER", "lector_uk")
    app_user = os.getenv("AZURE_PG_APP_USER", "app_predicciones")

    print("=" * 84)
    print("FASE 3 · SUBIDA DE ESQUEMA SQL Y 17 TABLAS CSV ('EXCEL') A AZURE POSTGRESQL")
    print("=" * 84)

    with psycopg.connect(url_admin, autocommit=False, connect_timeout=20) as conn:
        with conn.cursor() as cur:
            # 1. Ejecutar esquema_bd.sql
            sql_texto = config.RUTA_ESQUEMA_BD_SQL.read_text(encoding="utf-8")
            cur.execute(sql_texto)
            print("[1] Esquema SQL base (data/modelos/esquema_bd.sql) aplicado en Azure.")

            # 2. Subir las 17 tablas CSV limpias de data/integrado/
            print("\n[2] Subiendo las 17 tablas CSV limpias (data/integrado/*.csv) via COPY:")
            total_filas = 0
            for nombre_tabla, ruta_csv in TABLAS_CSV_NIVEL1:
                if not ruta_csv.exists():
                    raise FileNotFoundError(ruta_csv)
                df = pd.read_csv(ruta_csv)
                n = _cargar_dataframe_con_copy(cur, nombre_tabla, df)
                total_filas += n
                print(f"  -> OK | public.{nombre_tabla:<32} : {n:7,d} filas ({ruta_csv.name})")

            # 3. Subir muestra de 2.000 transacciones en transacciones_f1
            if config.RUTA_TRANSACCIONES_MUESTRA.exists():
                df_f1 = pd.read_parquet(config.RUTA_TRANSACCIONES_MUESTRA)
                n_f1 = _cargar_dataframe_con_copy(cur, "transacciones_f1", df_f1)
                total_filas += n_f1
                print(f"  -> OK | public.{'transacciones_f1':<32} : {n_f1:7,d} filas (muestra georreferenciada F1)")

            # 4. Permisos para lector_uk y app_predicciones
            for rol in (ro_user, app_user):
                cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {};").format(sql.Identifier(rol)))
                cur.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {};").format(sql.Identifier(rol)))
            cur.execute(
                sql.SQL("GRANT INSERT ON TABLE public.historial_predicciones TO {};").format(sql.Identifier(app_user))
            )
            cur.execute(
                sql.SQL("GRANT USAGE, SELECT ON SEQUENCE public.historial_predicciones_id_seq TO {};").format(
                    sql.Identifier(app_user)
                )
            )
            print(f"\n[3] Carga completada: {total_filas:,} registros subidos y permisos asignados a '{ro_user}' y '{app_user}'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
