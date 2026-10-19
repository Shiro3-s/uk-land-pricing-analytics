# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "scipy>=1.11",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.9",
# ]
# ///
"""Verifica las cifras publicadas del EDA, la bitacora y el diccionario.

Comprobaciones (README, MEMORIA_TECNICA, DICCIONARIO, BITACORA, libro 07):
 1. anos_salario: 6,2 (North East) -> 14,2 (London) por REGION-PROMEDIO.
 2. esfuerzo 2022: 43,0 % (NE) -> 100,0 % (London); ninguno <= 30 %.
 3. esfuerzo global: media 0,625 | rango 0,416-1,065 | 4/60 > 1,0 (6,7 %).
 4. Tukey sobre esfuerzo: IQR 0,284; cercas 0,053 y 1,189; 0 atipicos
    globales; 1 en London con IQR regional (n=6) (libro 07).
 5. correlaciones Pearson vs esfuerzo: precio +0,983 | salario +0,913.
 6. tasas IQR: 762 (9,77 %) | z-score 0 | media 6,11 -> 6,04 winsorizada.
 7. salario joven nacional 25.353 | factor_regional 0,805 (2022).
 8. clima vs variacion de precios (pearson): lluvia, tmax, tmin, af, sol.
Salida: ../plots/verificacion_eda.json + grafico WebP Tukey.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "script"))

from scipy import stats  # noqa: E402

from proyecto import config  # noqa: E402
from proyecto.modelado import RASGOS_NUMERICOS, agregar_esfuerzo, cargar_insumo  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "plots" / "verificacion_eda.json"
res: dict = {}

maestro, hipotecas = cargar_insumo()
df = agregar_esfuerzo(maestro, hipotecas)

# [1] anos de salario por region (promedio 2015-2020)
por_region = df.groupby("region").agg(
    anios=("anos_salario", "mean"), esfuerzo=("esfuerzo", "mean"),
    precio=("precio_mediana", "mean"))
res["anios_salario_min"] = [por_region["anios"].idxmin(),
                            round(float(por_region["anios"].min()), 1)]
res["anios_salario_max"] = [por_region["anios"].idxmax(),
                            round(float(por_region["anios"].max()), 1)]

# [2] esfuerzo 2022 por region (artefacto hipoteca_2022: F4 precios 2022 deflactados + F2)
hip22 = pd.read_parquet(RAIZ / "data" / "integrado" / "hipoteca_2022.parquet")
res["esfuerzo2022_min"] = [hip22.loc[hip22["cuota_pct_salario"].idxmin(), "region"],
                           round(float(hip22["cuota_pct_salario"].min() * 100), 1)]
res["esfuerzo2022_max"] = [hip22.loc[hip22["cuota_pct_salario"].idxmax(), "region"],
                           round(float(hip22["cuota_pct_salario"].max() * 100), 1)]
res["esfuerzo2022_n_menores_30pct"] = int((hip22["cuota_pct_salario"] <= 0.30).sum())

# [3] esfuerzo global
e = df["esfuerzo"]
res["esfuerzo_global"] = {
    "media": round(float(e.mean()), 3),
    "rango": [round(float(e.min()), 3), round(float(e.max()), 3)],
    "n_sobre_1": int((e > 1.0).sum()),
    "pct_sobre_1": round(float((e > 1.0).mean() * 100), 1),
}

# [4] Tukey global y por region (n=6) sobre esfuerzo
q1, q3 = e.quantile(0.25), e.quantile(0.75)
iqr = q3 - q1
baja, alta = q1 - 1.5 * iqr, q3 + 1.5 * iqr
res["tukey_global"] = {
    "iqr": round(float(iqr), 3), "cercas": [round(float(baja), 3),
                                            round(float(alta), 3)],
    "n_atipicos": int(((e < baja) | (e > alta)).sum())}
por_reg = df.groupby("region")["esfuerzo"]
n_reg_atip = 0
detalle_reg = []
for reg, s in por_reg:
    r1, r3 = s.quantile(0.25), s.quantile(0.75)
    ri = r3 - r1
    at = int(((s < r1 - 1.5 * ri) | (s > r3 + 1.5 * ri)).sum())
    n_reg_atip += at
    if at:
        detalle_reg.append([reg, at])
res["tukey_regional_n6"] = {"n_atipicos": n_reg_atip, "detalle": detalle_reg}

# [5] correlaciones Pearson (con y) de precio_mediana y salario
res["pearson_esfuerzo"] = {
    "precio_mediana": round(float(stats.pearsonr(df["precio_mediana"], e)[0]), 3),
    "salario_joven_region": round(float(
        stats.pearsonr(df["salario_joven_region"], e)[0]), 3),
    "anos_salario": round(float(stats.pearsonr(df["anos_salario"], e)[0]), 3),
}

# [6] tasas: IQR vs z-score y winsorizacion (libro 02 / pipeline parte 6)
# Se usa el CSV CRUDO (F2) con el mismo rename que cargar.py
crudo = pd.read_csv(RAIZ / "data" / "crudo" / "UK_Mortgage_Rate.csv",
                    low_memory=False)
col_map = {"MTG_INITIAL_RATE_PCT": "tasa_inicial_pct"}
crudo = crudo.rename(columns=col_map)
tasa = crudo["tasa_inicial_pct"].astype("float64").dropna()
z = np.abs(stats.zscore(tasa))
q1t, q3t = tasa.quantile(0.25), tasa.quantile(0.75)
iqrt = q3t - q1t
at_iqr = int(((tasa < q1t - 1.5 * iqrt) | (tasa > q3t + 1.5 * iqrt)).sum())
wins = tasa.clip(q1t - 1.5 * iqrt, q3t + 1.5 * iqrt)
res["tasas"] = {
    "n": int(len(tasa)), "n_iqr": at_iqr,
    "pct_iqr": round(float(at_iqr / len(tasa) * 100), 2),
    "n_zscore_3": int((z > 3).sum()),
    "media_antes": round(float(tasa.mean()), 2),
    "media_winsor": round(float(wins.mean()), 2),
    "limite_sup_iqr": round(float(q3t + 1.5 * iqrt), 2),
}

# [7] salario joven y factor regional (pipeline parte 7)
sal = pd.read_csv(RAIZ / "data" / "crudo" / "Income_by_age_and_gender.csv")
res["salario_joven_nacional_2022"] = [
    f"GBP {int(maestro['salario_joven_nacional'].iloc[-1]):,}".replace(",", ".")]
res["factor_regional_2022_minmax"] = [
    round(float(maestro["factor_regional"].min()), 3),
    round(float(maestro["factor_regional"].max()), 3)]

# [8] clima vs variacion de precios (pearson, como pipeline parte 9)
pv = df["precio_var_pct"]
res["pearson_clima_vs_precio_var_pct"] = {
    c: round(float(stats.pearsonr(df[c], pv)[0]), 3)
    for c in ["lluvia_anom", "tmax_anom", "tmin_anom", "af_anom", "sol_anom"]}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

# grafico WebP: distribucion del esfuerzo con cercas de Tukey
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.hist(e, bins=12, color="#4C72B0", alpha=0.85, edgecolor="white")
ax.axvline(0.30, color="#55A868", ls="--", lw=1.5,
           label="umbral de asequibilidad 30 %")
ax.axvline(baja, color="#C44E52", ls=":", lw=1.5, label="cercas Tukey")
ax.axvline(alta, color="#C44E52", ls=":", lw=1.5)
ax.axvline(1.0, color="#8172B2", ls="-.", lw=1.5, label="esfuerzo = 1,0")
ax.set_xlabel("esfuerzo (cuota mensual / salario mensual joven)")
ax.set_ylabel("n° de observaciones región-año")
ax.set_title(f"Verificación: esfuerzo (n=60); Tukey IQR={iqr:.3f}, "
             f"cercas {baja:.3f}/{alta:.3f}, 0 atípicos globales")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.savefig(Path(__file__).resolve().parents[1] / "plots"
            / "esfuerzo_tukey_verificacion.webp", format="webp", dpi=140,
            bbox_inches="tight")
plt.close(fig)

print(json.dumps(res, indent=2, ensure_ascii=False))
