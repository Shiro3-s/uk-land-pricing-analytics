# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "matplotlib>=3.9",
#   "scikit-learn>=1.4",
# ]
# ///
"""Ejecuta el modelado del proyecto (script/proyecto/modelado.py) con uv.

Espejo en terminal del libro 08_Modelado_Accesibilidad.ipynb: RL, KNN tuneado,
arbol y random forest con CV-5 barajado. Salida en ../plots/modelado_run.log.
"""
from __future__ import annotations

import io
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "script"))

LOG = Path(__file__).resolve().parents[1] / "plots" / "modelado_run.log"

t0 = time.time()
buf = io.StringIO()
ok = True
err = ""
try:
    from proyecto import modelado

    with redirect_stdout(buf):
        modelado.main()
except Exception:
    ok = False
    err = traceback.format_exc()
finally:
    elapsed = time.time() - t0
    texto = buf.getvalue()
    if err:
        texto += "\n\n[ERROR]\n" + err
    texto += f"\n[ESTADO] ok={ok} tiempo={elapsed:.1f}s"
    LOG.write_text(texto, encoding="utf-8")

print(f"ok={ok} tiempo={elapsed:.1f}s log={LOG}")
if not ok:
    sys.exit(1)
