# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
# ]
# ///
"""Compara los artefactos re-ejecutados del pipeline con el respaldo previo."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SESION = Path(__file__).resolve().parents[1]
RAIZ = Path(__file__).resolve().parents[3]
ANTES = SESION / "scripts" / "backup_antes"
OUT = SESION / "plots" / "reproducibilidad.json"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


rep: dict = {"comparaciones": {}}

# 1) maestro parquet: hash antes/despes + igualdad de contenidos
antes = ANTES / "maestro_joven_uk.parquet"
despues = RAIZ / "data" / "integrado" / "maestro_joven_uk.parquet"
df_a = pd.read_parquet(antes).sort_values(["anio", "region"]).reset_index(drop=True)
df_d = pd.read_parquet(despues).sort_values(["anio", "region"]).reset_index(drop=True)
iguales = df_a.equals(df_d)
max_diff = None
if not iguales and list(df_a.columns) == list(df_d.columns):
    num = df_a.select_dtypes("number").columns
    max_diff = float((df_a[num] - df_d[num]).abs().max().max())
rep["comparaciones"]["maestro"] = {
    "hash_antes": md5(antes),
    "hash_despues": md5(despues),
    "contenido_identico": bool(iguales),
    "max_diff_numerica": max_diff,
    "shape": list(df_d.shape),
    "nulos": int(df_d.isna().sum().sum()),
}

# 2) bitacora
antes_b = ANTES / "BITACORA_DECISIONES.md"
despues_b = RAIZ / "md" / "BITACORA_DECISIONES.md"
rep["comparaciones"]["bitacora"] = {
    "hash_antes": md5(antes_b),
    "hash_despues": md5(despues_b),
    "identica": md5(antes_b) == md5(despues_b),
}

# 3) csv versionado accesibilidad (no debe tocarse por el pipeline)
antes_c = ANTES / "accesibilidad_vivienda_joven_uk_v1.csv"
despues_c = RAIZ / "data" / "integrado" / "accesibilidad_vivienda_joven_uk_v1.csv"
rep["comparaciones"]["csv_accesibilidad_v1"] = {
    "hash_antes": md5(antes_c),
    "hash_despues": md5(despues_c),
    "identico": md5(antes_c) == md5(despues_c),
}

OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(rep, indent=2, ensure_ascii=False))
