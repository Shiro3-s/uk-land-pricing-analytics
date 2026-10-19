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
"""Re-ejecuta el libro 06 (transformacion y matriz) tras alinear el one-hot a
drop="first", guarda el cuaderno re-ejecutado in situ y verifica la forma de
matriz_features.parquet (debe ser 60 x 21 = 2 ids + 18 X + y, 9 dummies).
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[3]
NB = RAIZ / "libros" / "06_Transformacion_y_Matriz.ipynb"

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
    nbformat.write(nb, NB)  # solo reescribe si termino limpio
print(f"[RESULTADO] ok={ok} tiempo={elapsed:.1f}s")
if err:
    print(err[:2000])
    raise SystemExit(1)

# --- verificacion del artefacto regenerado -----------------------------------
m = pd.read_parquet(RAIZ / "data" / "integrado" / "matriz_features.parquet")
dummies = [c for c in m.columns if c.startswith("region_")]
zs = [c for c in m.columns if c.startswith("z_")]
print(f"[matriz_features] shape={m.shape} | z_={len(zs)} | dummies={len(dummies)}")
print("dummies:", dummies)
assert m.shape == (60, 21), f"esperaba (60, 21), llego {m.shape}"
assert len(dummies) == 9, f"esperaba 9 dummies, llego {len(dummies)}"
assert m.isna().sum().sum() == 0, "la matriz regenerada tiene nulos"
print("[OK] matriz_features 60x21 con 9 dummies (drop_first) y sin nulos")
