"""Inicializa la base de datos db_actividad_mineria y los roles de minimo privilegio en Azure PostgreSQL.

1. Se conecta a la BD por defecto 'postgres' para crear 'db_actividad_mineria' si aun no existe.
2. Crea o actualiza de forma idempotente:
   - lector_uk        -> Solo lectura (SELECT) en tablas actuales y futuras (DATABASE_URL_READONLY).
   - app_predicciones -> Lectura (SELECT) en tablas actuales y futuras + listo para INSERT en
                         historial_predicciones cuando se cargue esquema_bd.sql (DATABASE_URL_APP).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
import psycopg
from psycopg import sql

RAIZ = Path(__file__).resolve().parents[2]


def reemplazar_bd_en_uri(uri: str, nueva_bd: str) -> str:
    partes = urlsplit(uri)
    return urlunsplit((partes.scheme, partes.netloc, f"/{nueva_bd}", partes.query, partes.fragment))


def asegurar_base_de_datos(url_admin: str, db_name: str) -> None:
    url_postgres = reemplazar_bd_en_uri(url_admin, "postgres")
    print(f"Verificando existencia de la base de datos '{db_name}'...")
    with psycopg.connect(url_postgres, autocommit=True, connect_timeout=20) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE DATABASE {} ENCODING 'UTF8';").format(
                        sql.Identifier(db_name)
                    )
                )
                print(f"  -> Base de datos creada: {db_name}")
            else:
                print(f"  -> Base de datos ya existente: {db_name}")


def asegurar_rol(cur: psycopg.Cursor, rol: str, password: str) -> None:
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s;", (rol,))
    existe = cur.fetchone() is not None
    if not existe:
        cur.execute(
            sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {};").format(
                sql.Identifier(rol),
                sql.Literal(password),
            )
        )
        print(f"  -> Rol creado: {rol}")
    else:
        cur.execute(
            sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {};").format(
                sql.Identifier(rol),
                sql.Literal(password),
            )
        )
        print(f"  -> Rol actualizado: {rol}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv(RAIZ / ".env", override=True)
    url_admin = os.environ["DATABASE_URL"]
    db_name = os.getenv("AZURE_PG_DATABASE", "db_actividad_mineria")
    admin_user = os.getenv("AZURE_PG_ADMIN_USER", "pgadmin_uk")
    ro_user = os.getenv("AZURE_PG_READONLY_USER", "lector_uk")
    ro_pass = os.environ["AZURE_PG_READONLY_PASSWORD"]
    app_user = os.getenv("AZURE_PG_APP_USER", "app_predicciones")
    app_pass = os.environ["AZURE_PG_APP_PASSWORD"]

    asegurar_base_de_datos(url_admin, db_name)

    print(f"Conectando como administrador ({admin_user}) a {db_name}...")
    with psycopg.connect(url_admin, autocommit=True, connect_timeout=20) as conn:
        with conn.cursor() as cur:
            asegurar_rol(cur, ro_user, ro_pass)
            asegurar_rol(cur, app_user, app_pass)

            for rol in (ro_user, app_user):
                cur.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO {};").format(
                        sql.Identifier(db_name),
                        sql.Identifier(rol),
                    )
                )
                cur.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA public TO {};").format(
                        sql.Identifier(rol),
                    )
                )
                cur.execute(
                    sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {};").format(
                        sql.Identifier(rol),
                    )
                )
                cur.execute(
                    sql.SQL(
                        "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public GRANT SELECT ON TABLES TO {};"
                    ).format(
                        sql.Identifier(admin_user),
                        sql.Identifier(rol),
                    )
                )

            cur.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC;")
            cur.execute(
                sql.SQL("GRANT ALL ON SCHEMA public TO {};").format(
                    sql.Identifier(admin_user),
                )
            )
            print("  -> Privilegios por defecto (SELECT futuro) y bloqueo de escritura aplicados con exito.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
