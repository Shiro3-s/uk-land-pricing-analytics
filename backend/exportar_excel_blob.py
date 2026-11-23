"""Genera el libro Excel multi-hoja (5 hojas) desde Azure PostgreSQL/CSV y opcionalmente lo sube a Azure Blob Storage.

Uso:
  python backend/exportar_excel_blob.py                  # Solo genera data/integrado/vivienda-uk-azure-datos.xlsx
  python backend/exportar_excel_blob.py --subir-azure    # Genera y sube a stactividaduk26 / excel-datos via Azure CLI
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.web_api import generar_libro_excel_bytes


def _buscar_az_cli() -> str:
    encontrado = shutil.which("az") or shutil.which("az.cmd")
    if encontrado:
        return encontrado
    win_default = Path(r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd")
    if win_default.exists():
        return str(win_default)
    raise FileNotFoundError("No se encontró Azure CLI ('az'). Instálalo o agrégalo al PATH.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportar Excel multi-hoja desde Azure PostgreSQL y subir a Azure Blob Storage")
    parser.add_argument("--salida", type=Path, default=ROOT / "data" / "integrado" / "vivienda-uk-azure-datos.xlsx")
    parser.add_argument("--subir-azure", action="store_true", help="Sube el archivo generado al contenedor excel-datos en stactividaduk26")
    parser.add_argument("--cuenta", default="stactividaduk26")
    parser.add_argument("--grupo", default="rg-actividad-db")
    parser.add_argument("--contenedor", default="excel-datos")
    args = parser.parse_args()

    contenido = generar_libro_excel_bytes()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_bytes(contenido)
    print(f"Excel generado: {args.salida} ({len(contenido):,} bytes)")

    if args.subir_azure:
        az = _buscar_az_cli()
        key = subprocess.check_output(
            [az, "storage", "account", "keys", "list", "-g", args.grupo, "-n", args.cuenta, "--query", "[0].value", "-o", "tsv"],
            text=True,
        ).strip()
        subprocess.run(
            [
                az, "storage", "blob", "upload",
                "--account-name", args.cuenta,
                "--account-key", key,
                "--container-name", args.contenedor,
                "--name", args.salida.name,
                "--file", str(args.salida),
                "--content-type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "--overwrite",
                "-o", "none",
            ],
            check=True,
        )
        print(f"Publicado en Azure Blob: https://{args.cuenta}.blob.core.windows.net/{args.contenedor}/{args.salida.name}")


if __name__ == "__main__":
    main()
