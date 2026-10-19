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
"""Re-ejecuta el libro 08 (modelado) tras filtrar los warnings benignos,
re-escribe el cuaderno in situ y verifica: (a) que no quedan rutas de maquina
en las salidas, (b) que las metricas publicadas se reproducen exactas
(RL 1,000 | RF 0,980 | Arbol 0,963 | KNN 0,844).
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[3]
NB = RAIZ / "libros" / "08_Modelado_Accesibilidad.ipynb"

nb = nbformat.read(NB, as_version=4)

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

if ok:
    nbformat.write(nb, NB)
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

esperadas = {
    "regresion_lineal": "1.000",
    "random_forest": "0.980",
    "arbol_decision": "0.963",
    "knn_tuneado": "0.844",
}
for modelo, r2 in esperadas.items():
    assert modelo in salida, f"falta la fila de {modelo} en la salida"
    lineas = [l for l in salida.splitlines() if modelo in l and r2 in l]
    assert lineas, f"no encontre {modelo} con R2 {r2} en ninguna linea"
    print(f"[OK] {modelo}: {lineas[-1].strip()}")
print("[OK] metricas publicadas reproducidas exactas")
