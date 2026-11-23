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

### 2-bis. Variables propuestas en el análisis avanzado (cuaderno consolidado)

No forman parte de la matriz X·y base: se **proponen y validan** en la etapa exploratoria
avanzada y se documentan aquí para mantener la completitud del diccionario de datos.

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
| modelado de regresión (RL/KNN/árbol/forest, CV-5, importancia) | 08 · Regresión | `08_Modelado_Accesibilidad.ipynb` |
| clasificación I sin fuga (`clase`: Alto/Moderado, CART `d=4`) | 09 · Clasificación I | `09_Arbol_Decision_Clasificacion.ipynb` |
| torneo comparativo (Árbol vs. Naive Bayes vs. KNN) y contrato ganador | 10 · Torneo | `10_Torneo_Clasificadores.ipynb`, `data/modelos/contrato_prediccion.json` |
| segmentación no supervisada (`cluster`, `segmento_nombre`, `segmento_f1`) | 11 · Clustering | `11_Clustering_Segmentacion.ipynb`, `data/integrado/segmentos_*.csv` |
| reglas de asociación (`antecedente`, `consecuente`, `soporte`, `confianza`, `lift`, `score`) | 12 · Asociación | `12_Reglas_Asociacion_Apriori.ipynb`, `data/integrado/reglas_asociacion_*.csv` |
| evaluación forense (umbrales, CV-5 por pliegue y comparativa global) | 13 · Evaluación | `13_Evaluacion_Modelos_y_Umbrales.ipynb`, `data/integrado/evaluacion_*.csv` |

## 5. Artefactos derivados adicionales (Clasificación, Clustering y Asociación)

| Artefacto | Grano / filas | Variables clave | Uso |
|---|---|---|---|
| `data/modelos/modelo_ganador_clasificacion.joblib` | Modelo entrenado (`1,35 KB`) | 7 rasgos numéricos sin fuga + 9 dummies `region` | Tipo 2 · Clasificador regional principal (`DecisionTreeClassifier`, `max_depth=4`) |
| `data/modelos/modelo_regresion_esfuerzo.joblib` | Modelo entrenado (`~1,2 KB`) | 7 rasgos numéricos sin fuga + 9 dummies `region` | Tipo 1A · Regresión continua de esfuerzo salarial (`Ridge`, `R² = 0,909`, `MAE = 4,2 pp`) |
| `data/modelos/modelo_clasificacion_tasa_f2.joblib` | Modelo entrenado (`~1,8 KB`) | `comisiones_total`, `meses_inicial`, `tipo`, `subtitulo` | Tipo 2B · Clasificador F2 de tasa atípica (> 8,13 %) con 3 umbrales operativos |
| `data/modelos/modelo_regresion_tasa_f2.joblib` | Modelo entrenado | `comisiones_total`, `meses_inicial`, `tipo`, `subtitulo` | Tipo 1B · Regresión continua de tasa hipotecaria F2 (`RandomForestRegressor`, `R² = 0,964`, `MAE = 0,17 pp`) |
| `data/modelos/contrato_prediccion.json` | Metadatos JSON | `tipos_de_prediccion_implementados`, `modelos_serializados`, `umbrales_operativos`, `capa_datos_futura_bd` | Contrato técnico integral para futura integración con API y frontend web |
| `data/modelos/esquema_bd.sql` | DDL SQL estándar | Nivel 1 (17 tablas operativas `1,83 MB` + `historial_predicciones`) y Nivel 2 (`transacciones_f1`) | Esquema relacional listo para **Azure Database for PostgreSQL Flexible Server** (`md/GUIA_FASE3_SUBIDA_AZURE.md`) |
| `data/integrado/comparativa_regresion_dual.csv` | 8 filas (2 escenarios × 4 regresores) | `escenario`, `modelo`, `r2_media`, `r2_desv`, `rmse`, `mae` | Benchmark comparativo del Tipo 1 (Regresión Continua sin fuga) |
| `data/integrado/clustering_benchmark_24gb.csv` | Escalas `10k` a `11,37 M` | `n_transacciones`, `silueta_muestral`, `tiempo_s`, `pico_ram_mb`, `rss_proceso_mb`, `pct_presupuesto_ram` | Benchmark real de escalabilidad del clustering frente al límite de 24 GB de RAM |
| `data/integrado/hipotecas_catalogo_limpio.csv` | 7.796 filas × 14 cols | Catálogo hipotecario limpio F2 | Exportación plana lista para carga directa en base de datos |
| `data/integrado/precios_ciudad_limpio.csv` | 14.050 filas × 6 cols | Agregado anual por 1.153 ciudades UK | Exportación plana lista para consultas granulares por ciudad en el futuro frontend |
| `data/integrado/torneo_clasificadores_resumen.csv` | 8 filas (2 escenarios × 4 modelos) | `acc_train`, `acc_test`, `cv5_acc`, `f1_test`, `roc_auc_test`, `brier_score`, `predict_ms`, `peso_kb` | Evidencia empírica del torneo comparativo |
| `data/integrado/segmentos_territoriales.csv` | 60 filas (región × año) | `cluster`, `segmento_nombre`, `accion_politica`, `esfuerzo` (post-hoc) | Segmentación territorial K-Means ($k=3$) |
| `data/integrado/segmentos_hipotecas_muestra.csv` | 3 perfiles agregados | `segmento_hipoteca`, `tasa_inicial_media`, `comisiones_media`, `pct_first_time_buyer` | Taxonomía de oferta hipotecaria F2 |
| `data/integrado/segmentos_transacciones_1m.csv` | 4 perfiles agregados (~1 M ventas) | `segmento_f1`, `n_transacciones`, `precio_medio_gbp`, `pct_freehold`, `pct_flat`, `pct_london` | Clustering en *streaming* (`MiniBatchKMeans`, perfil `local` 1M / `cloud` 11,37M) |
| `data/integrado/reglas_asociacion_inmobiliario.csv` | 195 reglas ($n=2.000$) | `antecedente`, `consecuente`, `soporte`, `confianza`, `lift`, `score`, `categoria_analitica` | Reglas Apriori sobre compraventas reales F1 |
| `data/integrado/reglas_asociacion_hipotecas.csv` | 481 reglas ($n=2.999$) | `antecedente`, `consecuente`, `soporte`, `confianza`, `lift`, `score` | Reglas Apriori sobre condiciones crediticias F2 |
| `data/integrado/evaluacion_umbrales_resumen.csv` | 11 umbrales (0,05–0,80) | `umbral`, `TP_detectados`, `FP_falsas_alarmas`, `precision`, `recall`, `f1`, `youden_J` | Optimización de umbral ante costos asimétricos |
| `data/integrado/evaluacion_cv5_pliegues.csv` | 90 registros (3 esc. × 6 mod. × 5 folds) | `escenario`, `modelo`, `pliegue`, `accuracy`, `f1`, `recall`, `roc_auc` | Estabilidad pliegue por pliegue (CV-5 en Pipeline) |
| `data/integrado/comparativa_global_modelos.csv` | 8 familias de modelos | `Algoritmo / Familia`, `Estado`, `Maestro 50/50`, `Hipotecas Atipico 9,8%`, `Veredicto practico` | Semáforo global de viabilidad y desempeño |
| `data/integrado/telemetria_rendimiento.csv` | 13 etapas analíticas | `etapa`, `categoria`, `perfil`, `filas_procesadas`, `tiempo_s`, `pico_ram_mb`, `limite_ram_mb`, `salida_kb`, `estado` | Telemetría unificada de tiempo y consumo de RAM (incluido clustering 10k/100k/~1M) |
| `data/integrado/auditoria_fuentes_calidad.csv` | 38 archivos CSV / Parquet | `carpeta`, `archivo`, `formato`, `separador`, `tamano_mb`, `filas`, `columnas`, `nulos_detectados`, `duplicados`, `apertura_ok`, `estado_limpieza` | Auditoría integral de cobertura de fuentes y estado sanitario de limpieza |

---

*Documento vivo: se regenera y verifica al ejecutar los cuadernos y módulos de `script/proyecto/`.*
