# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "pyarrow",
# ]
# ///

from pathlib import Path
import pandas as pd
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]  # proyecto raíz
OUT = Path(__file__).resolve().parents[1] / "plots" / "shape_report.json"

OUT.parent.mkdir(exist_ok=True)

report = {}
report["ts"] = datetime.now().isoformat()

# maestro parquet
p = ROOT / "data" / "integrado" / "maestro_joven_uk.parquet"
if p.exists():
    df = pd.read_parquet(p)
    report["maestro_parquet"] = {
        "path": str(p.relative_to(ROOT)),
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c,t in df.dtypes.items()},
        "head": df.head().to_dict(orient="split"),
        "nans": df.isna().sum().to_dict(),
    }
else:
    report["maestro_parquet"] = {"exists": False}

# maestro csv versionado
for name in ["maestro_joven_uk.csv", "accesibilidad_vivienda_joven_uk_v1.csv"]:
    p = ROOT / "data" / "integrado" / name
    if p.exists():
        df = pd.read_csv(p)
        report[f"maestro_csv_{name}"] = {
            "path": str(p.relative_to(ROOT)),
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "nans": df.isna().sum().to_dict(),
        }

# hipotecas
p = ROOT / "data" / "integrado" / "hipoteca_2022.parquet"
if p.exists():
    df = pd.read_parquet(p)
    report["hipoteca_2022_parquet"] = {"path": str(p.relative_to(ROOT)), "shape": list(df.shape)}

# clima limpio
p = ROOT / "data" / "limpia" / "metoffice_clima_limpio.parquet"
if p.exists():
    df = pd.read_parquet(p)
    report["metoffice_clima_limpio_parquet"] = {"path": str(p.relative_to(ROOT)), "shape": list(df.shape)}

# f1 muestra
p = ROOT / "db" / "uk_property_price_muestra.csv"
if p.exists():
    df = pd.read_csv(p, nrows=2000)
    report["f1_muestra"] = {"path": str(p.relative_to(ROOT)), "shape": list(df.shape)}

OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print("wrote", OUT)
