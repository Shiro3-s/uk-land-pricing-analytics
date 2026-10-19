# Revisión · UK Land Pricing Analytics

**Fecha:** 2026-10-19 · **Sesión:** `analysis/2026-10-19T00-00-00-revision-uk-land-pricing-analytics/`
**Método:** skill `scientific-eda` (context-first, journal append-only). Todos los Python vía `uv run` + PEP723.
**Alcance (Opción A + re-ejecución del pipeline):** leer artefactos, re-ejecutar pipeline/EDA/modelado/notebook consolidado y verificar cifras de los "avances" contra los datos.

## Veredicto

**Los análisis están bien ejecutados y los "avances" dicen información correcta.** Re-ejecuté el pipeline completo (28 s, ok), `eda.py`, `modelado.py` y el notebook consolidado con `REPRODUCIR_PIPELINE=True` (47 s, **todos los asserts pasan**). El cuadro maestro se reproduce idéntico (DataFrame.equals = True). No hay fuga de datos: escalador y one-hot se ajustan dentro de cada pliegue, y el KNN tunea k con GridSearchCV interno.

Encontré **2 discrepancias menores** (F1, F2) que no invalidan ningún resultado, y **1 nota informativa** (F3).

## Cifras verificadas (todas exactas)

| Afirmación (avances) | Verificado |
|---|---|
| Maestro 60×16, 28 nulos (18 sol + 10 precio_var) | ✔ |
| Años de salario 6,2 (NE) → 14,2 (London) | ✔ |
| Esfuerzo 2022: 43,0 % (NE) → 100,0 % (London); 0 ≤ 30 % | ✔ |
| Esfuerzo global: media 0,625 · rango 0,416–1,065 · 4/60 > 1,0 | ✔ |
| Tukey esfuerzo: IQR 0,284 · cercas 0,053/1,189 · 0 atípicos globales · 1 en London (n=6) | ✔ |
| Pearson: precio +0,983 · salario +0,913 | ✔ |
| Tasas IQR 762 (9,77 %) vs z-score 0 · 6,11→6,04 % (winsor superior) | ✔ |
| Clima vs Δprecio: lluvia −0,241 · tmax −0,388 · tmin −0,399 · af −0,145 · sol +0,238 | ✔ |
| **Modelado CV-5 barajado: RL 1,000 · RF 0,980 · Árbol 0,963 · KNN 0,844** | ✔ |
| Barajado vs sin barajar: 2 regiones/fold → 5–9; VIF ∞ vs 1,8; matriz 60×18 | ✔ |

## Discrepancias detectadas

**F1 · `matriz_features.parquet` (60×22) contradice al código y a los docs.** El artefacto versionado tiene **10 dummies** de región (todas), pero el README/diccionario y el encabezado del libro 06 dicen **9** con `drop="first"` (60×18). La celda del libro 06 que guarda el archivo usa `OneHotEncoder(...)` sin `drop`, aunque su propio checklist (y el script espejo `matriz_xy.py`, que asserta 18) sí aplican `drop="first"`. *Impacto:* bajo — el modelado no lee ese parquet (construye su preprocesador con `drop="first"` dentro del pipeline). *Corrección:* alinear el libro 06 a `drop="first"` y regenerar (quedará 60×21 = 18 X + y + 2 id).

**F2 · El "0,638 / 0,900 / 0,943 (sin barajar)" citado no tiene celda reproducible en el libro 08.** Reproducido con la configuración real de `modelado.py` (tuneo interno barajado): 0,630 / 0,909 / 0,945 (±0,009). La conclusión cualitativa es idéntica (barajado sube KNN ≈ +0,21). *Corrección:* citar la configuración exacta del tuneo al publicar esa columna.

**F3 (informativo) · Warning de sklearn en el CV:** `Found unknown categories ... during transform` — en algún pliegue de TEST barajado aparece una región ausente del train (60 filas, 10 regiones). `handle_unknown="ignore"` la codifica como la referencia. No invalida resultados; se puede silenciar o documentar.

## Artefactos de la sesión

- `journal.md` — bitácora completa (contexto, shape, reproducibilidad, verificaciones, findings)
- `plots/` — `pipeline_run.log`, `eda_run.log`, `modelado_run.log`, `verificacion_{eda,modelado}.json`, `reproducibilidad.json`, `cv_barajado_vs_no_barajado.webp`, `esfuerzo_tukey_verificacion.webp`, `Cuaderno_Avance_1_ejecutado.ipynb`
- `scripts/` — wrappers PEP723 (`run_pipeline`, `run_eda`, `run_modelado`, `run_consolidado`), verificaciones (`verify_*`, `check_*`) y `backup_antes/`

## Correcciones aplicadas (2026-10-19, tras la verificación)

| # | Qué | Estado |
|---|---|---|
| A1 · F1 | Libro 06 alineado a `OneHotEncoder(drop="first")`; `matriz_features.parquet` regenerado a **60×21** (9 dummies, 0 nulos); textos del libro actualizados | ✔ Resuelto |
| A2 | Errata del libro 06 «`Ese_ratio_`» → «`Ese salto`» (el dato 11.369.413 → 60 **es correcto** y se conserva) | ✔ |
| A3 · F3 | Filtro dirigido de los warnings benignos en libro 08 y cuaderno consolidado → **0 rutas de máquina** en las salidas; métricas del libro 08 reproducidas exactas; consolidado con asserts OK | ✔ Resuelto |
| A4 | Roadmap ampliado a **10 fases** (KDD + CRISP-DM); objetivo final = modelo predictivo; **fase 7 en curso** (predictores + validación temporal 2015-2018 / 2019-2020) | ✔ |

F2 se mantiene como nota informativa (cifra reproducida ±0,009; conclusión cualitativa idéntica).
Detalle completo en `journal.md` (secciones `[ACTION]` y `[RESULT]`).
