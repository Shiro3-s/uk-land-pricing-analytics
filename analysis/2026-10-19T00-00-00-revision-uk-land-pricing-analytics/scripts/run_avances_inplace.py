# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14",
#   "matplotlib>=3.9",
#   "scipy>=1.11",
#   "scikit-learn>=1.4",
#   "jupyter>=1.0",
#   "nbclient>=0.9",
#   "nbformat>=5.9",
#   "ipykernel>=6.29",
# ]
# ///
"""Re-ejecuta el cuaderno consolidado de Avances in situ (con
REPRODUCIR_PIPELINE=True) tras filtrar los warnings benignos y verifica:
(a) que pasan todos los asserts, (b) que no quedan rutas de maquina en las
salidas. Ademas guarda una copia en la carpeta de la sesion.
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[3]
SESION = Path(__file__).resolve().parents[1]
NB = RAIZ / "Avances" / "Cuaderno_Avance_1_Completo.ipynb"
COPIA = SESION / "plots" / "Cuaderno_Avance_1_ejecutado.ipynb"

nb = nbformat.read(NB, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code" and "REPRODUCIR_PIPELINE = False" in cell.source:
        cell.source = cell.source.replace(
            "REPRODUCIR_PIPELINE = False", "REPRODUCIR_PIPELINE = True")
        print("[flag] REPRODUCIR_PIPELINE -> True")
        break

t0 = time.time()
client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                        resources={"metadata": {"path": str(RAIZ)}})
try:
    client.execute()
    ok = True
    err = None
except Exception as e:  # noqa: BLE001
    ok = False
    err = str(e)
elapsed = time.time() - t0

if ok:
    # restaurar la bandera del entregable: los artefactos del pipeline son
    # identicos con True o False (verificado), el lector debe tener el run rapido.
    for cell in nb.cells:
        if cell.cell_type == "code" and "REPRODUCIR_PIPELINE = True" in cell.source:
            cell.source = cell.source.replace(
                "REPRODUCIR_PIPELINE = True", "REPRODUCIR_PIPELINE = False")
            break
    nbformat.write(nb, NB)
    nbformat.write(nb, COPIA)
print(f"[RESULTADO] ok={ok} tiempo={elapsed:.1f}s")
if err:
    print(err[:2000])
    raise SystemExit(1)

# --- verificaciones sobre las salidas ----------------------------------------
def salida_completa(nb) -> str:
    partes = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                partes.append(o.get("text", ""))
            elif o.get("output_type") in ("execute_result", "display_data"):
                partes.append(o.get("data", {}).get("text/plain", ""))
    if any(isinstance(p, list) for p in partes):
        partes = ["".join(p) if isinstance(p, list) else p for p in partes]
    return "".join(partes)


salida = salida_completa(nb)
assert "C:\\Users" not in salida, "queda una ruta Windows en las salidas"
assert "site-packages" not in salida, "queda una ruta site-packages en las salidas"
print("[OK] sin rutas de maquina en las salidas")

n_errores = sum(
    1 for c in nb.cells if c.cell_type == "code"
    for o in c.get("outputs", []) if o.get("output_type") == "error")
assert n_errores == 0, f"{n_errores} celdas con error"
print("[OK] todos los asserts del cuaderno pasaron (0 celdas con error)")
print(f"[COPIA] {COPIA}")
