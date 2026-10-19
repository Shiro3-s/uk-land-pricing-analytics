# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "scipy>=1.11",
# ]
# ///
"""Chequeos finos: correlacion clima (dropna nativo, como el pipeline),
tasas IQR (una sola cola, como comparar_iqr_zscore), y el warning de one-hot.

Nota: agregar_esfuerzo imputa sol_anom y precio_var_pct a 0, lo que cambia la
correlacion clima respecto al pipeline (que usa el maestro crudo con dropna).
Aqui se usa el maestro CRUDO para replicar la parte 8 del pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "script"))

from scipy import stats  # noqa: E402

maestro = pd.read_parquet(RAIZ / "data" / "integrado" / "maestro_joven_uk.parquet")

# --- clima vs precio_var_pct con dropna nativo ( réplica parte 8) ---
print("=== clima vs precio_var_pct (dropna por columna, maestro CRUDO) ===")
for col in ["lluvia_anom", "tmax_anom", "tmin_anom", "af_anom", "sol_anom"]:
    sub = maestro[["precio_var_pct", col]].dropna()
    r = sub[col].corr(sub["precio_var_pct"])
    print(f"  {col:<12} r={r:+.3f}  n={len(sub)}")

# --- tasas: IQR una sola cola (límite superior) como comparar_iqr_zscore ---
print("\n=== tasas: IQR superior (una cola) ===")
crudo = pd.read_csv(RAIZ / "data" / "crudo" / "UK_Mortgage_Rate.csv",
                    low_memory=False)
tasa = crudo["MTG_INITIAL_RATE_PCT"].astype("float64").dropna()
q1, q3 = tasa.quantile(0.25), tasa.quantile(0.75)
iqr = q3 - q1
sup = q3 + 1.5 * iqr
inf = q1 - 1.5 * iqr
fuera_sup = int((tasa > sup).sum())
fuera_inf = int((tasa < inf).sum())
print(f"  n={len(tasa)} | lim_sup={sup:.2f} | fuera(sup)={fuera_sup} "
      f"({100*fuera_sup/len(tasa):.2f}%) | fuera(inf)={fuera_inf}")
wins = tasa.clip(upper=sup)
print(f"  media {tasa.mean():.2f} -> winsor {wins.mean():.2f}")
