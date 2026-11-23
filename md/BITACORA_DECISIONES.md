| fuente | problema | cantidad | metodo | razon | impacto |
|---|---|---|---|---|---|
| F2 hipotecas | duplicados por sku+tid | 0 | keep='first' | cada producto unico por escaneo (clave natural sku+tid) | - |
| F2 hipotecas | productos sin tasa inicial (MNAR) | 0 | eliminar filas | sin tasa no hay cuota que estimar; son casos puntuales | - |
| F2 hipotecas | tipos y plazos textuales | 0 | categoria + int | contrato de tipos para el cruce por ano de escaneo | - |
| F3 clima | filas sin fecha (corruptas/footer) | 20 | eliminar fila | no son mediciones: no tienen anio ni variables (MNAR de registro) | - |
| F3 clima | filas temporales duplicadas | 0 | keep='first' | una medicion mensual por estacion | - |
| F3 clima | valores NA/*/# en tmax/tmin/af/lluvia/sol | 0 | NaN numerico | los simbolos Met Office (---, *, #) se trataron como faltantes MCAR | - |
| F4 precios/salario | salario 'null' 1975-1998 | 24 | dejar NaN (no imputar) | el salario solo esta publicado desde 1999; no se inventa historia | - |
| F4 ingreso por edad | duplicados y tipos | 0 | clave grupo+genero | una mediana salarial por banda y genero | - |
| F2 hipotecas | tasa_reversion (faltante) | 328 | conservar NaN (la cuota usa tasa_inicial; reversion es informativa) | MAR: la ausencia depende del tipo (observado): discounted=6.1%; fixed=0.8%; tracker=26.0%; variable=39.5% | - |
| F3 clima | anio/month (faltante) | 20 | eliminar fila (no son mediciones) | MNAR: filas sin fecha (registro corrupto/footer) | - |
| F3 clima | tmax (faltante) | 1599 | NaN (los agregados anuales y la normal toleran meses faltantes) | MCAR: gaps instrumentales de estacion: 1467/1599 con tmax nulo tambien falta tmin (falla conjunta de reporte); no depende del valor | - |
| F3 clima | tmin (faltante) | 1522 | NaN (los agregados anuales y la normal toleran meses faltantes) | MCAR: gaps instrumentales de estacion: 1467/1599 con tmax nulo tambien falta tmin (falla conjunta de reporte); no depende del valor | - |
| F3 clima | af (faltante) | 2949 | NaN (los agregados anuales y la normal toleran meses faltantes) | MCAR: gaps instrumentales de estacion: 1467/1599 con tmax nulo tambien falta tmin (falla conjunta de reporte); no depende del valor | - |
| F3 clima | lluvia (faltante) | 1478 | NaN (los agregados anuales y la normal toleran meses faltantes) | MCAR: gaps instrumentales de estacion: 1467/1599 con tmax nulo tambien falta tmin (falla conjunta de reporte); no depende del valor | - |
| F3 clima | sol (faltante) | 9483 | NaN (la anomalia de sol se usa aparte, solo donde existe) | MNAR estructural: horas de sol no medidas/publicadas antes de 1931 (4138 de 9483 nulos; decadas 1850-1930 casi vacias), no es azar | - |
| F4 precios/salario | salario_mediana_real (faltante) | 24 | NaN (no se imputa historia inventada) | MNAR estructural: mediana salarial no publicada hasta 1999 (serie oficial), ausencia por diseno, no aleatoria | - |
| F2 hipotecas | outliers de tasa_inicial (metodo) | 762 | winsorizar clip al limite IQR 8.13 | IQR vs z-score: IQR detecta 762 y z-score 0 (skew 0.04); IQR no exige normalidad | tasa media 6.11 -> 6.04 % |
| F4 salario joven | bandas de edad sin corte exacto en 34 | 3 | media de medianas 18-21/22-29/30-39 | la fuente publica bandas; 30-39 cubre 30-34 (aprox.) | salario joven = GBP 25353 |
| F2 hipotecas | precios 2022 nominales vs salario GBP 2020 | 10 | deflactar por CPI 2020->2022 (1.137) | homogeneizar moneda entre F1 (nominal) y F4 (real 2020) | cuota en GBP 2020 |
| Transformacion X | tirania de unidades en distancias (n_transacciones vs clima) | 16 | StandardScaler dentro de make_pipeline() | evita fuga de datos (fit solo en X_train) y evita caida de KNN de -27,7 pp | VIF = 1,80 con drop_first=True |
| CON Etiqueta · Tipo 1 | fuga aritmetica por anos_salario (r = +1,000 con esfuerzo) | 60 | excluir precio, salario y anos_salario de X; usar Ridge (alpha=1.0) | Ridge logra R2 CV-5 = 0,9094 (+/- 4,21 pp) sin fuga; en F2 gana RandomForestRegressor (R2 = 0,9642, +/- 0,17 pp) | 2 regresores .joblib listos (< 5 ms) |
| CON Etiqueta · Tipo 2 | seleccion de clasificador principal y desbalance en F2 (9,77%) | 7796 | Arbol CART (max_depth=4) + 3 umbrales (0,05 / 0,30 / 0,40) | gana torneo multicriterio (4,82/5,00): 17/18 Test (94,4%), 9/10 regiones, 188/190 atipicas F2 en u=0,30 (99,0%) con reglas SI/ENTONCES | 2 clasificadores .joblib (< 5 ms) |
| SIN Etiqueta · Tipo 3 | escalar clustering a 11,35 millones de ventas F1 dentro de 24 GB RAM | 11346500 | PyArrow iter_batches + MiniBatchKMeans en float32 (sin esfuerzo en X) | Ward O(n^2) exige ~517 TB; streaming mantiene pico Python en 49,87 MB (0,20% de 24 GB) y silueta 0,333 | 4 macro-segmentos reales descubiertos |
| SIN Etiqueta · Tipo 4 | reglas triviales juridicas en vivienda (Flat -> Leasehold) | 676 | Apriori puro (tidsets, assert 10/10 tickets) + filtro de negocio + score = lift * soporte | separa trivialidades legales y prioriza reglas accionables (195 F1 + 481 F2) | Detached+Freehold -> Extremo (lift 2,00) |
| Fase 2 y Fase 3 | presentacion interactiva con predictor en vivo y subida a Azure | 17 tablas | Presentacion HTML + PPTX + app_presentacion_interactiva.py + GUIA_FASE3_SUBIDA_AZURE.md | permite predecir por opciones y a escrita conectado a CSV/Azure PostgreSQL (DATABASE_URL_READONLY) | 17 CSV (1,83 MB) + esquema SQL listos |
