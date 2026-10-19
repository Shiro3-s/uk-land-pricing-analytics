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
