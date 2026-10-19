# Journal: Revisión del proyecto UK Land Pricing Analytics

**Sesión:** 2026-10-19T00-00-00-revision-uk-land-pricing-analytics  
**Skill:** scientific-eda (context-first, human-guided, append-only)  
**Enfoque:** Opción A + Re-ejecución del PIPELINE con datos existentes (maestro y cruces entre CSVs). Usar scripts PEP723 + uv run.

## [CONTEXT] 2026-10-19T00:00:00
- **Objetivo:** Revisar si los "avances" dicen información correcta y si los análisis (EDA/modelado) están bien ejecutados.
- **Alcance:** README, MEMORIA_TÉCNICA, DICCIONARIO_DATOS, BITÁCORA_DECISIONES, INFORME_AVANCE1, libros 05/07/08, scripts `eda.py`, `modelado.py`, `pipeline.py`, `matriz_xy.py`, `indicadores.py`, `config.py`.
- **Verificaciones:** consistencia numérica con artefactos/parquet; clasificación MCAR/MAR/MNAR, IQR vs z-score, winsorización (6.11%→6.04%); CV barajado, drop_first en one-hot, pipeline sin fuga; imputaciones justificadas (sol_anom, precio_var_pct); H3 honesta (|r| 0.15–0.40); reproducibilidad.
- **Acción:** Ejecutar PIPELINE (usar datos existentes; F1 completo puede no estar). Validar cruces entre CSVs y artefactos.
- **Decisión usuario:** Opción A + re-ejecutar PIPELINE. Usar maestro y cruces entre archivos .csv.

[Siguiente: [SHAPE] Inspeccionar maestro (parquet/CSV), hipotecas, clima, precio-salario, etc. para confirmar ventana 2015-2020 y 60×16.]

## [SHAPE] 2026-10-19T00:04:09
- Maestro (parquet): shape (60,16) — 10 regiones × 6 años (2015–2020). OK.
- Columnas coinciden con DICCIONARIO_DATOS.md. Tipos correctos (int64/float64/str).
- Nulos documentados: sol_anom = 18, precio_var_pct = 10 (primer año por región). af_anom 0. OK (coincide con docs).
- CSV versionado: accesibilidad_vivienda_joven_uk_v1.csv (60,16) presente. OK.
- Hipoteca 2022 (parquet) presente. Clima limpio presente.

[Verificación 1/7: Shape y consistencia con diccionario — CORRECTO.]
[Siguiente: [CHECK] Re-ejecutar PIPELINE completo para validar cruces entre CSVs y reproducibilidad.]

## [CHECK] 2026-10-19T00:12 — Re-ejecución del PIPELINE (uv run, 28.4s, ok=True)
- Log: plots/pipeline_run.log. Las 10 partes completas + parte 11 (bitácora).
- Cifras clave del log == documentación:
  - F1 11.369.413 filas | F2 7.796 | F3 37.049 (36 est.) | F4 46 | F4b 12 ✔
  - IQR 762 (9,77%) vs z-score 0; lim sup 8,13; tasa 6,11→6,04% ✔
  - salario joven GBP 25.353 (factor 0,805) ✔
  - años de salario: 6,2 (North East) → 14,2 (London) ✔
  - esfuerzo 2022: 43,0% (NE) → 100,0% (London); ninguna región ≤30% ✔
  - maestro 60 filas, 28 nulos (18 sol_anom + 10 precio_var_pct) ✔
  - correlaciones clima: lluvia -0,241 | tmax -0,388 | tmin -0,399 | af -0,145 | sol +0,238
- Reproducibilidad (scripts/check_reproducibilidad.py → plots/reproducibilidad.json):
  - maestro: DataFrame.equals = True (contenido idéntico) ✔
  - bitácora: contenido idéntico (solo CRLF→LF) ✔
  - csv accesibilidad v1: byte-idéntico ✔

[Siguiente: [CHECK] Ejecutar eda.py y modelado.py con uv para verificar cifras del libro 07/08.]

## [CHECK] 2026-10-19T00:35 — Verificación de cifras EDA/modelado vs documentación

### scripts/verify_eda_claims.py + verify_clima_tasas.py (EDA, bitácora, diccionario)
TODAS CONFIRMADAS exactas (tras replicar el método correcto del pipeline):
- años_salario: 6,2 (North East) → 14,2 (London) ✔
- esfuerzo 2022: 43,0 % (NE) → 100,0 % (London); 0 regiones ≤ 30 % ✔
- esfuerzo global: media 0,625 | rango 0,416–1,065 | 4/60 > 1,0 (6,7 %) ✔
- Tukey esfuerzo: IQR 0,284 | cercas 0,053 y 1,189 | 0 atípicos globales;
  1 en London con IQR regional n=6 ✔ (libro 07)
- Pearson esfuerzo: precio +0,983 | salario +0,913 ✔
- tasas IQR (una sola cola superior, como comparar_iqr_zscore):
  762 (9,77 %) | z-score 0 | lim_sup 8,13 | media 6,11 → 6,04 ✔
  (nota: 933 valores por debajo del límite inferior NO se tocan; la
  winsorización es solo superior, coherente con la cola real de tasas altas)
- clima vs precio_var_pct (dropna nativo, maestro CRUDO, n=50/35):
  lluvia −0,241 | tmax −0,388 | tmin −0,399 | af −0,145 | sol +0,238 ✔
- salario joven nacional GBP 25.353 | factor 0,805 ✔
- matriz X (drop_first): 60×18 ✔ (README/diccionario)

### scripts/verify_modelado_claims.py (modelado, tuneo exacto de modelado.py)
- CV-5 BARAJADO (reproducido exacto): RL 1,000 | RF 0,980 | Árbol 0,963 |
  KNN 0,844 ✔ (README/libro 08/avances)
- Regiones por fold de TEST: sin barajar = [2,2,2,2,2]; barajado = [7,9,5,7,8]
  → README "sin barajar cada pliegue ve solo 2 regiones; barajado entre 5 y 9" ✔
- VIF: 10 dummies = infinito | 9 dummies (drop_first) = 1,8 ✔ (libro 06)
- CV-5 SIN BARAJAR (con tuneo interno del KNN también barajado, como
  modelado.py): KNN 0,630 | Árbol 0,909 | RF 0,945.
  → Coinciden dentro de ±0,009 con los 0,638/0,900/0,943 publicados.
  El 0,638/0,900/0,943 del libro 08 viene de una celda de "demostración"
  (no existe celda con shuffle=False en el libro; la cifra se cita). La
  diferencia es por el esquema de tuneo interno del KNN. CONCLUSIÓN
  CUALITATIVA ÍNTEGRA: el barajado sube KNN ≈ +0,21, árbol ≈ +0,05,
  RF ≈ +0,035. Correcto.

## [FINDING] 2026-10-19T00:35 — Discrepancias detectadas (2 menores, no invalidan resultados)

### F1 · matriz_features.parquet versionada (60×22) contradice al código y a los docs
- El artefacto `data/integrado/matriz_features.parquet` tiene 22 columnas:
  9 z_ + **10 dummies** de región (todas las regiones, sin drop_first) + esfuerzo + region + anio.
- El libro 06 (celda que lo guarda) usa `OneHotEncoder(sparse_output=False)` SIN `drop="first"`,
  y su propia salida dice "(60, 22)" y "60 × 19 (9 z_ + 10 dummies)" — coherente con el archivo,
  pero INCOHERENTE con su propio encabezado ("Matriz X·y con 60 filas × 18 columnas ... 9 dummies
  con drop_first") y con la conclusión del diccionario/README (60 × 18).
- El script espejo `script/proyecto/matriz_xy.py` SÍ usa `drop="first"` y asserta 18 columnas.
- IMPACTO: bajo. El modelado NO lee matriz_features.parquet (construye su preprocesador con
  drop="first" dentro del pipeline). Solo afecta al artefacto exportado y a la coherencia doc/código.
- CORRECCIÓN SUGERIDA: alinear el libro 06 a `drop="first"` (9 dummies) y regenerar el parquet
  (quedará 60×21 = 18 X + y + 2 identificadores), o corregir README/diccionario/encabezado del libro 06.

### F2 · README/memoria citan KNN 0,638 / árbol 0,900 / RF 0,943 (sin barajar) sin celda reproducible
- Reproducido con la configuración real de modelado.py: 0,630 / 0,909 / 0,945 (±0,009).
- La conclusión cualitativa es idéntica. Sugiero citar la configuración exacta del tuneo interno
  al publicar la columna "sin barajar".

### F3 (informativo) · warning de sklearn en el CV
- `UserWarning: Found unknown categories in columns [0] during transform` proviene de
  `OneHotEncoder(handle_unknown="ignore", drop="first")`: en algún fold de TEST barajado aparece
  una región que no estuvo en el train (60 filas / 10 regiones × 6 años; pliegue de 48 train puede
  perder alguna región). `handle_unknown="ignore"` la codifica como todo ceros (la referencia).
- No invalida los resultados (es el precio de la seguridad frente a fugas), pero es ruidoso.
  Se puede silenciar con `warnings.filterwarnings` o documentándolo.

## [SUMMARY] 2026-10-19 — Revisión completa
- Pipeline re-ejecutado (28 s, maestro idéntico); eda.py y modelado.py reproducen cifras; notebook consolidado con REPRODUCIR_PIPELINE=True ejecuta limpio (47 s, asserts OK).
- Veredicto: análisis bien ejecutados, avances correctos. 2 discrepancias menores (F1 matriz_features 10 dummies vs 9 documentados; F2 cifra "sin barajar" sin celda reproducible) + F3 warning sklearn. Ninguna invalida resultados.
- Resumen en RESUMEN_REVISION.md.

## [ACTION] 2026-10-19 — Correcciones aplicadas (F1, F3) + ampliación de fases

### A1 · Libro 06 alineado a `drop="first"` (resuelve F1)
- `libros/06_Transformacion_y_Matriz.ipynb`: el `ColumnTransformer` pasa a
  `OneHotEncoder(sparse_output=False, drop="first")` y el nombrado de columnas usa
  `get_feature_names_out(["region"])` (con `drop` `categories_[0]` da 10 nombres para 9 columnas).
- Textos del propio libro corregidos: tabla de tipos («9 dummies»), print «9 z_ + 9 region_»,
  cierre «60 × 18 (9 z_ + 9 dummies)».
- Re-ejecutado con `scripts/run_libro06.py` (18,2 s, ok). Artefacto regenerado:
  `data/integrado/matriz_features.parquet` = **60 × 21** (2 ids + 18 X + y) con **9 dummies**,
  sin nulos. Ahora coincide con README, diccionario, memoria (§4.6 ya decía 60 × 21) y con el
  espejo `matriz_xy.py`.
- Aclaración: no hay impacto aguas abajo (el modelado y el cuaderno consolidado construyen su
  propio preprocesador con `drop="first"` y no leen este parquet).

### A2 · Errata corregida en libro 06
- «`Ese_ratio_`» → «`Ese salto`» en la celda del recorrido 11.369.413 → 60. **El dato sigue**
  (el log del pipeline confirma 11.369.413 filas de F1).

### A3 · Rutas de máquina fuera de las salidas (resuelve F3 de forma limpia)
- `libros/08_Modelado_Accesibilidad.ipynb` y `Avances/Cuaderno_Avance_1_Completo.ipynb`:
  `warnings.filterwarnings("ignore", message="Found unknown categories.*")` (one-hot con
  `handle_unknown="ignore"` en el CV barajado) y, en el consolidado, también para
  `FigureCanvasAgg is non-interactive` (plt.show con backend no interactivo).
- Re-ejecutados in situ (`run_libro08.py` 23,3 s; `run_avances_inplace.py` 54,9 s con
  REPRODUCIR_PIPELINE=True). Verificado: **0 rutas `C:\Users` ni `site-packages`** en las salidas;
  métricas del libro 08 reproducidas exactas (RL 1,000 | RF 0,980 | Árbol 0,963 | KNN 0,844);
  el consolidado pasa todos los asserts con 0 celdas de error. La bandera del entregable se
  restauró a False (los artefactos son idénticos con True/False).
- El bloque `sys.path.insert(..., PROYECTO / "script")` NO se toca: es necesario para importar
  el paquete `proyecto`.

### A4 · Fase actual y roadmap ampliados (KDD + CRISP-DM)
- Objetivo final declarado: **modelo predictivo** del esfuerzo hipotecario por región y año.
- Tabla de fases ampliada de 6 a 10 en `README.md` y `md/PROPUESTA_METODOLOGICA.md` (esta última
  estaba con 4 fases y la 3 «En curso»). **Fase 7 (predictores y validación temporal) = En curso**;
  8 (modelo final), 9 (evaluación/interpretación) y 10 (despliegue) pendientes.
- Nota de alcance y limitación «serie temporal» actualizadas en `md/MEMORIA_TECNICA.md`;
  «Alcance actual» y «Cómo vamos» del README, y la salida `matriz_features` (60 × 21).
- Decisión clave documentada: la fase 6 usó CV-5 **barajado**; predecir exige **validación
  temporal** (entrenar 2015-2018, validar 2019-2020) porque los años de una región están
  correlacionados y un fold barajado filtra información del futuro.
- Entradas añadidas a `md/BITACORA_DECISIONES.md`.

## [RESULT] 2026-10-19 — Estado tras las correcciones
- F1 resuelto (artefacto y libro 06 alineados a 9 dummies / 60 × 21); F3 silenciado sin ruido.
- F2 queda como nota informativa (no se toca: la cifra sin barajar se reproduce con ±0,009 y la
  conclusión cualitativa es idéntica).
- Roadmap reencuadrado hacia el modelo predictivo (fases 7-10).
