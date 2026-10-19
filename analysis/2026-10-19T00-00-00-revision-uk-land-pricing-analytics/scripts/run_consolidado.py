# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "matplotlib>=3.9",
#   "scipy>=1.11",
#   "scikit-learn>=1.4",
#   "jupyter>=1.0",
#   "nbclient>=0.9",
#   "nbformat>=5.9",
#   "ipykernel>=6.29",
# ]
# ///
"""Ejecuta el notebook consolidado de avances (REPRODUCIR_PIPELINE=True) y
reporta si todos los asserts pasan. No reescribe el notebook original:
se ejecuta en memoria y se guarda una copia en la carpeta de la sesion.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[3]
SESION = Path(__file__).resolve().parents[1]
NB = RAIZ / "Avances" / "Cuaderno_Avance_1_Completo.ipynb"
COPIA = SESION / "plots" / "Cuaderno_Avance_1_ejecutado.ipynb"

nb = nbformat.read(NB, as_version=4)

# forzar REPRODUCIR_PIPELINE=True en la celda de la bandera
for cell in nb.cells:
    if cell.cell_type == "code" and "REPRODUCIR_PIPELINE = False" in cell.source:
        cell.source = cell.source.replace(
            "REPRODUCIR_PIPELINE = False", "REPRODUCIR_PIPELINE = True")
        print("[flag] REPRODUCIR_PIPELINE -> True")
        break

t0 = time.time()
client = NotebookClient(nb, timeout=900, kernel_name="python3",
                        resources={"metadata": {"path": str(RAIZ)}})
try:
    client.execute()
    ok = True
    err = None
except Exception as e:  # noqa: BLE001
    ok = False
    err = str(e)
elapsed = time.time() - t0

nbformat.write(nb, COPIA)

# resumen: contar asserts y errores en la salida
n_assert = 0
n_ok_checks = 0
errores = []
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    src = cell.source
    n_assert += src.count("assert ")
    out = "".join(
        o.get("text", "") for o in cell.get("outputs", [])
        if o.get("output_type") == "stream")
    if "[OK" in out or "[OK ]" in out:
        n_ok_checks += 1
    if "Error" in out or "error" in out.lower():
        errores.append(out[:300])
    for o in cell.get("outputs", []):
        if o.get("output_type") == "error":
            errores.append(f"{o.get('ename')}: {o.get('evalue')}")

print(f"\n[RESULTADO] ok={ok} tiempo={elapsed:.1f}s")
print(f"[ASSERTS] celdas con assert: {n_assert} | celdas con [OK]: {n_ok_checks}")
print(f"[COPIA] {COPIA}")
if err:
    print(f"[EXCEPCION]\n{err[:2000]}")
for e in errores[:10]:
    print(f"[ERROR-EN-SALIDA] {e[:400]}")
sys.exit(0 if ok else 1)
