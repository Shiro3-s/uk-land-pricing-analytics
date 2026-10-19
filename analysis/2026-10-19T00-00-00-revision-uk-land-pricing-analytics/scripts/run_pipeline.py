# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "matplotlib>=3.9",
# ]
# ///
"""Re-ejecuta el PIPELINE completo del proyecto (main_proyecto.py) con uv.

Equivalente a `python main_proyecto.py` desde la raiz del proyecto:
cruza las 4 fuentes -> maestro_joven_uk (anio x region) en 10 partes.
Salida consola capturada en ../plots/pipeline_run.log (relativo a esta sesion).
"""
from __future__ import annotations

import io
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

# raiz del proyecto = 3 niveces arriba de scripts/ (sesion/scripts/run_pipeline.py)
RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "script"))

LOG = Path(__file__).resolve().parents[1] / "plots" / "pipeline_run.log"

t0 = time.time()
buf = io.StringIO()
ok = True
err = ""
try:
    from proyecto.pipeline import PipelineProyecto

    with redirect_stdout(buf):
        maestro = PipelineProyecto().run()
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
