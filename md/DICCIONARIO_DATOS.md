# Diccionario de datos del cuadro maestro `maestro_joven_uk`

**Proyecto:** Accesibilidad de vivienda para jóvenes (18-34) en el Reino Unido
**Fuente consolidada:** `data/integrado/maestro_joven_uk.parquet` (60 filas = 10 regiones × 6 años, 2015-2020)
**Ventana de cruce:** 2015-2020 (único periodo donde conviven transacciones, clima y salario)
**Semilla del proyecto:** 42 (`config.SEED`)

Formato: **nombre · tipo · descripción · rango/valores válidos · unidades · tratamiento aplicado**.

## 1. Columnas del cuadro maestro (16)

| # | Columna | Tipo | Descripción (lenguaje claro) | Rango / valores válidos | Unidades | Tratamiento |
|---|---|---|---|---|---|---|
| 1 | `anio` | int64 | Año de la observación (clave temporal del cruce) | 2015–2020 | año | sin tratamiento (identificador) |
| 2 | `region` | str | Región del Reino Unido (ONS), 10 niveles nominales | `East Midlands`, `East of England`, `London`, `North East`, `North West`, `South East`, `South West`, `Wales`, `West Midlands`, `Yorkshire and The Humber` | texto | sin tratamiento (se recodifica one-hot al entrar a X) |
| 3 | `n_transacciones` | int64 | Número de transacciones de vivienda limpias (F1) en la región-año | ≥ 0 | conteo | agregación desde F1 (libros 02-03) |
| 4 | `precio_medio` | float64 | Precio medio de venta de vivienda | ≥ 0 | GBP | agregación desde F1; escala auxiliar (no entra a X: redundante con la mediana) |
| 5 | `precio_mediana` | float64 | Precio mediano de venta de vivienda (más robusto que la media ante outliers) | ≥ 0 | GBP | agregación desde F1; **rasgo candidato a X** |
| 6 | `salario_mediana_real` | float64 | Mediana nacional del salario (deflactada a GBP 2020) | ≥ 0 | GBP/año | base del cálculo del salario joven (libro 03); escala auxiliar |
| 7 | `salario_joven_nacional` | float64 | Salario joven (18-34) nacional estimado (mediana × factor de edad) | ≥ 0 | GBP/año | libro 03; escala auxiliar |
| 8 | `factor_regional` | float64 | Factor de regionalización del salario (referencia ASHE 2021) | 0.85–1.25 | ratio | libro 03; escala auxiliar |
| 9 | `salario_joven_region` | float64 | Salario joven estimado por región (nacional × factor regional) | ≥ 0 | GBP/año | libro 03; **rasgo candidato a X** (denominador de la asequibilidad) |
| 10 | `tmax_anom` | float64 | Anomalía de temperatura máxima frente a la normal 1931-2020 (Met Office) | −0.72 a +2.09 | desviación | anomalías por estación agregadas a región (libro 05); **rasgo candidato a X** |
| 11 | `tmin_anom` | float64 | Anomalía de temperatura mínima frente a la normal 1931-2020 | −1.23 a +1.76 | desviación | ídem; **rasgo candidato a X** |
| 12 | `lluvia_anom` | float64 | Anomalía de lluvia anual frente a la normal 1931-2020 | −14.5 a +22.6 | mm | ídem; **rasgo candidato a X** |
| 13 | `sol_anom` | float64 | Anomalía de horas de sol frente a la normal 1931-2020 | −20.5 a +72.6 | horas | ídem; **18 nulos conservados en el maestro** (se imputan a 0 al construir X/EDA/modelado); **rasgo candidato a X** |
| 14 | `af_anom` | float64 | Anomalía de días de helada (air frost) frente a la normal 1931-2020 | −43 a +8.7 | días | ídem; **rasgo candidato a X** |
| 15 | `precio_var_pct` | float64 | Variación porcentual anual del precio mediana (todo positivo 2015-2020) | 0 a +9.4 | % | calculada desde `precio_mediana`; **10 nulos conservados en el maestro** (primer año de cada región, sin año previo; se imputan a 0 al construir X/EDA/modelado); **rasgo candidato a X** |
| 16 | `anos_salario` | float64 | Años de salario joven completos que cuesta la vivienda mediana (precio ÷ salario joven) | 5.9–15.2 | años | libro 04; **rasgo candidato a X** (medida directa de asequibilidad) |

## 2. Columnas derivadas en la matriz X·y (libro 06)

La matriz `data/integrado/matriz_features.parquet` añade el objetivo y las columnas transformadas:

| Columna | Tipo | Descripción | Rango | Unidades | Tratamiento |
|---|---|---|---|---|---|
| `esfuerzo` | float64 | **Objetivo (y).** Cuota mensual PMT (90 % LTV, 25 años, tasa winsorizada 6,04 % + comisiones GBP 1.116) ÷ salario joven mensual regional | 0.416–1.065 | ratio (1.0 = todo el salario) | calculado en libros 04 y 06; umbral de accesibilidad 0.30 |
| `z_precio_mediana` … `z_anos_salario` | float64 | Los 9 rasgos estandarizados (media 0, desv 1) con `StandardScaler` | ~ −2 a +3 | desviaciones | escalado con `ColumnTransformer` dentro de cada pliegue del CV (sin fuga); **elegido en la comparación empírica del libro 06** |
| `region_*` (9 columnas) | float64 | One-hot de la región con `OneHotEncoder` y `drop="first"` | 0 o 1 | binario | codificación elegida en el libro 06 (R² 0,891 vs −0,051 del ordinal). Con `drop="first"` hay **9** columnas para **10** regiones: la región ausente es la de referencia implícita y los coeficientes de la regresión se pueden interpretar. Sin él, las 10 dummy sumarían siempre 1 (trampa de variables mudas, VIF infinito) |

### 2-bis. Variables propuestas en el análisis avanzado (cuaderno consolidado §9-bis)

No forman parte de la matriz X·y: se **proponen y validan** en la sección 9-bis del
cuaderno consolidado (nivel *Excepcional* de la rúbrica). Se documentan aquí para que el
diccionario siga siendo completo, que es lo que exige el ítem 6 del checklist.

| Columna | Tipo | Descripción | Rango | Unidades | Tratamiento y validación |
|---|---|---|---|---|---|
| `indice_tension` | float64 | Precio mediano de la región dividido por la **mediana nacional de su mismo año**. Separa el efecto nivel (inflación) del efecto relativo (esa región es más cara que el país) | 0.68–2.60 | ratio (1.0 = precio típico del año) | propuesta propia; validada con Pearson **r = +0,946** contra `esfuerzo` y **r = +0,537** contra `n_transacciones` (mercado caro y poco líquido) |
| `segmento_accesibilidad` | category | Segmento ordinal por tramos del esfuerzo: moderado (0,30-0,50), alto (0,50-0,80), crítico (0,80-1,00), extremo (>1,00). Los cortes coinciden con el umbral de accesibilidad (0,30) y con el punto de «la cuota se come el sueldo entero» (1,0) | 4 niveles | categoría | segmentación propuesta con `pd.cut`; validada porque el esfuerzo medio crece de forma monótona de un segmento al siguiente |
| `cv_esfuerzo` | float64 | **Coeficiente de variación** del esfuerzo dentro de cada región: `s / x̄`. Mide cuánto **oscila** el esfuerzo año a año, no cuánto es | 0.035–0.094 | adimensional | propuesta propia; validada con Pearson **r = +0,684** contra la volatilidad del precio de la región |

**Nota sobre las unidades de `esfuerzo` (dos versiones del mismo indicador).** El
`esfuerzo` del objetivo `y` y el de `hipoteca_2022` se calculan con la misma fórmula
PMT pero **sobre precios con distinta base monetaria**:

| Indicador | Precio usado | Base |
|---|---|---|
| `esfuerzo` (objetivo `y`, 60 filas) | `precio_mediana` de cada año | **nominal** GBP del año (2015-2020) |
| `hipoteca_2022` (10 filas) | `precio_mediana` de 2022 | **deflatado** a GBP 2020 (÷ 1,137) |

No es un error, pero sí una distinción que hay que tener presente al comparar ambos
cuadros: el primero mide el esfuerzo en el contexto de precios de cada año; el segundo
compara regiones en una base monetaria constante. Cuando se mezclen en un mismo
análisis, hay que declarar cuál de los dos se está usando.

## 3. Columnas excluidas de X y por qué

| Columna | Rol | Razón de la exclusión |
|---|---|---|
| `anio` | identificador | clave temporal, no rasgo; se conserva como metadato de fila |
| `region` | identificador (recodificado) | entra a X solo como one-hot, no como texto |
| `n_transacciones` | agrupador | volumen de mercado, no determinante directo del esfuerzo del joven |
| `precio_medio` | escala auxiliar | duplica la señal de `precio_mediana` (la mediana es más robusta) |
| `salario_mediana_real` | escala auxiliar | insumo intermedio de `salario_joven_region` (meterla duplicaría la señal) |
| `salario_joven_nacional` | escala auxiliar | ídem (nacional, no regional) |
| `factor_regional` | escala auxiliar | ídem (factor multiplicativo ya aplicado en `salario_joven_region`) |

**Nota de fuga (leakage):** la tasa hipotecaria y la cuota PMT viven **dentro de `esfuerzo` (y)**, nunca como rasgo de X: meterlas a X sería predecir el objetivo con el objetivo.

## 4. Trazabilidad de cada columna → libro de origen

| Insumo | Libro de origen | Ver en |
|---|---|---|
| ventana 2015-2020, 4 fuentes | 01 · Carga | `01_Carga_4_Fuentes.ipynb` |
| `precio_mediana`, `precio_medio`, `n_transacciones` | 02 · Limpieza → 03 · Cruces | `02_Limpieza_4_Datasets.ipynb`, `03_Cruces_Affordability_Joven.ipynb` |
| `salario_joven_region` (+ escalas) | 03 · Cruces | `03_Cruces_Affordability_Joven.ipynb` |
| `esfuerzo`, `anos_salario`, tasa winsorizada | 04 · Hipoteca | `04_Hipoteca_Joven.ipynb` |
| anomalías climáticas `*_anom` | 05 · Clima y precios | `05_Clima_y_Precios.ipynb` |
| matriz X·y (z_ + dummies) y comparación de métodos | 06 · Transformación | `06_Transformacion_y_Matriz.ipynb` |
| EDA (univariado/bivariado, 6 figuras, 3 hallazgos) | 07 · EDA | `07_EDA_Accesibilidad.ipynb` |
| modelado (RL/KNN/árbol/forest, CV-5, importancia) | 08 · Modelado | `08_Modelado_Accesibilidad.ipynb` |

---

*Documento vivo: se regenera/verifica al ejecutar los libros y `script/proyecto/matriz_xy.py`. Última verificación: octubre 2026.*
