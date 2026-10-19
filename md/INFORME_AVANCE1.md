# Informe de avance 1 · Limpieza y cruzamiento (Fase 3)

**Accesibilidad de vivienda para jóvenes (18-34) en el Reino Unido**
Proyecto de Minería de Datos · UNIMINUTO Ibagué 2026-2 · Docente: Esteban Ernesto Morales Castro
Fecha: septiembre de 2026

---

## 1. Fase y alcance

- **Fase 3 · Limpieza y cruzamiento** — EN CURSO
- KDD: etapa 2 · Preprocesamiento | CRISP-DM: Preparación de datos
- Entregable de esta fase: cuadro maestro `maestro_joven_uk` (año × región) a partir de
  las 4 fuentes, con calidad verificada, indicadores de accesibilidad y bitácora.

### Por qué KDD + CRISP-DM (y no SEMMA)

El proyecto considera tres marcos: KDD, CRISP-DM y SEMMA. El equipo integra **dos**:
el ciclo **KDD** por su claridad conceptual del descubrimiento de conocimiento y
**CRISP-DM** porque arranca en el problema de negocio, es iterativo y es la metodología
del proyecto (negocio · datos · preparación). **SEMMA** se descarta por ser parcial
(se queda en la parte analítica) y por estar ligado a la herramienta SAS. Cada fase del
proyecto se reporta con su equivalencia en ambos marcos (tabla 1 del documento APA), lo
que hace rastreable cada decisión.

**Lo que llevamos:** fases 1-2 completas (pregunta, objetivos, hipótesis y fuentes
tipadas) y fase 3 en curso con los entregables de este informe; queda pendiente la fase 4
(modelado de las series regionales).

## 2. Fuentes y estado

| # | Fuente | Filas | Periodo | Ubicación |
|---|--------|------:|---------|-----------|
| F1 | Transacciones (Land Registry + geo) | 11.369.413 | 2015-2026 | `db/*` (solo `_muestra` en repo) |
| F2 | Hipotecas (Kaggle, snapshot 2022) | 7.796 | 2022 | `data/crudo/UK_Mortgage_Rate.csv` |
| F3 | Clima Met Office (36 estaciones) | 37.049 | 1931-2020 | `data/crudo/MET_Office_Weather_Data.csv` |
| F4 | Precio medio + salario mediana (GBP 2020) | 46 | 1975-2020 | `data/crudo/Average_UK_houseprices_and_salary.csv` |
| F4b | Salario por edad y género | 12 | 2021 | `data/crudo/Income_by_age_and_gender.csv` |

**Ventana de cruce completo**: 2015-2020 (las 3 series que se cruzan — precio, clima y
salario — coexisten ahí). Las hipotecas son un **corte 2022** y se analizan aparte.

**Enlaces de descarga (Kaggle):** F1
[uk-property-sale-prices-20152026-geo-enriched](https://www.kaggle.com/datasets/mansiaggarwal88/uk-property-sale-prices-20152026-geo-enriched?select=uk_property_price_paid_geo.csv) ·
F2 [uk-mortgage-rates-thousands-of-mortgage-products](https://www.kaggle.com/datasets/thedevastator/uk-mortgage-rates-thousands-of-mortgage-products) ·
F3 [uk-met-office-weather-data](https://www.kaggle.com/datasets/josephw20/uk-met-office-weather-data) ·
F4 [uk-median-house-prices-and-salary-19752020](https://www.kaggle.com/datasets/samuelcortinhas/uk-median-house-prices-and-salary-19752020).

## 3. Limpieza por fuente (resumen)

Orden: duplicados → categorías → nulos (MCAR/MAR/MNAR) → outliers (IQR).

| Fuente | Problema | Cantidad | Clasificación | Tratamiento |
|--------|----------|----------|---------------|-------------|
| F2 | duplicados por `sku+tid` | 0 | - | verificar (clave natural) |
| F2 | productos sin `tasa_inicial_pct` | 0 | MNAR puntual | eliminar fila |
| F3 | filas mensuales duplicadas | 0 | duplicado | conservar primera |
| F3 | filas sin fecha (corruptas/footer) | 20 | MNAR de registro | eliminar fila |
| F3 | símbolos Met Office (---, *, #) | — | MCAR | NaN numérico (no imputar) |
| F4 | salario `null` 1975-1998 | 24 | MNAR estructural (no publicado) | NaN (serie real, no se inventa) |
| F4b | bandas de edad sin corte en 34 | — | aproximación | media de medianas 18-21 / 22-29 / 30-39 |

F1 ya fue limpiada en la fase anterior (IQR de `price` → winsorizar + bandera
`es_atipico_precio`; nulos geo MCAR → bandera `es_sin_geo`). Todas las decisiones quedan
en `md/BITACORA_DECISIONES.md`.

### 3.1 Faltantes: clasificación MCAR / MAR / MNAR por fuente

Diagnóstico con evidencia por variable (`script/proyecto/diagnostico.py`):

| Fuente | Variable | Nulos | Clase | Evidencia y tratamiento |
|--------|----------|------:|-------|-------------------------|
| F2 | `tasa_reversion` | 328 (4,2 %) | **MAR** | la ausencia depende del tipo observado (fixed 0,8 % → variable 39,5 %); se conserva NaN (la cuota usa `tasa_inicial`) |
| F3 | `anio`/`month` | 20 (0,05 %) | **MNAR** | filas sin fecha (registro corrupto); se eliminan |
| F3 | `tmax` / `tmin` | 1.599 / 1.522 | **MCAR** | gaps instrumentales: 1467 de 1599 con `tmax` nulo también falta `tmin` (falla conjunta de reporte); NaN, los agregados toleran |
| F3 | `af` / `lluvia` | 2.949 / 1.478 | **MCAR** | idem, falla conjunta de reporte |
| F3 | `sol` | 9.483 (25,6 %) | **MNAR estructural** | horas de sol no publicadas antes de 1931 (4138 de 9483); NaN, se usa aparte |
| F4 | `salario_mediana_real` | 24 (52,2 %) | **MNAR estructural** | mediana no publicada hasta 1999 (serie oficial); no se imputa historia |

### 3.2 Outliers: IQR (Tukey) vs z-score — decisión

Sobre `tasa_inicial_pct` de F2 (alimenta la cuota hipotecaria): Q1 = 5,62 · mediana = 6,10 ·
Q3 = 6,62 · **IQR = 1,00** → límite superior Tukey = **8,13**.

| Método | Criterio | Detectados | Observación |
|--------|----------|-----------:|-------------|
| **IQR (Tukey)** | fuera de [Q1−1,5·IQR, Q3+1,5·IQR] | **762 (9,77 %)** | no supone normalidad; detecta la cola real (hasta 9,94 %) |
| z-score | \|z\| > 3 | **0 (0 %)** | asume normalidad; con skew 0,04 no detecta la cola |

**Decisión: IQR + winsorizar** (clip al límite superior, sin eliminar filas) y bandera
`es_atipico_tasa`. Impacto: tasa inicial media 6,11 % → **6,04 %**, que es la empleada en
el esfuerzo hipotecario 2022.

### 3.1 Salario joven 18-34

Brecha salarial 2021 (mediana por banda y género):

| Banda | Mediana (GBP) anio | Fuente |
|---|---|---|
| 18 a 21 | 18.392 | Income by age and gender |
| 22 a 29 | 26.856 | ídem |
| 30 a 39 | 34.210 | ídem |

- **Salario joven 18-34 ≈ GBP 25.353/año** (media de las medianas de bandas; la banda
  30-39 cubre 30-34).
- **Factor joven** = 0,805 respecto a la mediana nacional 2020 → se aplica a la serie
  anual F4 para obtener el salario joven de cada año.
- **Regionalización**: se multiplica por `FACTOR_REGIONAL` (referencia ASHE 2021,
  aproximada): London 1,25 · South East 1,06 · East of England 1,01 · South West 0,95 ·
  East Midlands 0,92 · West Midlands 0,91 · North West 0,91 · Yorkshire & Humber 0,88 ·
  North East 0,85 · Wales 0,90.

## 4. Cruce `maestro_joven_uk` (año × región, 2015-2020)

10 regiones × 6 años = **60 filas**, sin nulos en `precio_mediana` ni `salario_joven_region`,
clave única `(anio, region)` (validado con asserts). Las anomalías climáticas por región
provienen del promedio de las estaciones Met Office de Inglaterra y Gales (24 estaciones
→ 10 regiones).

| anio | region | n_transacciones | precio_mediana | salario_joven_region | anos_salario | lluvia_anom | tmax_anom |
|------|--------|----------------:|---------------:|---------------------:|-------------:|------------:|----------:|
| 2015 | East Midlands | 86.170 | 157.500 | 23.192 | 6,8 | -6,2 | +1,0 |
| 2016 | East Midlands | 92.306 | 165.000 | 23.275 | 7,1 | +3,9 | +0,7 |
| 2017 | East Midlands | 92.767 | 178.000 | 22.919 | 7,8 | -1,8 | +0,9 |
| 2018 | East Midlands | 91.765 | 185.500 | 22.795 | 8,1 | -1,0 | +1,3 |
| 2019 | East Midlands | 89.086 | 190.000 | 22.841 | 8,3 | +16,3 | +1,1 |
| 2020 | East Midlands | 78.078 | 202.500 | 23.325 | 8,7 | -11,9 | +0,7 |

> Ver `data/integrado/maestro_joven_uk_muestra.csv` (10 filas) y
> `libros/03_Cruces_Affordability_Joven.ipynb`.

## 5. Indicadores de accesibilidad

### 5.1 Años de salario joven para comprar (media 2015-2020)

| region | años de salario |
|--------|----------------:|
| North East | 6,2 |
| North West | 6,7 |
| Wales | 6,8 |
| Yorkshire and The Humber | 7,0 |
| East Midlands | 7,8 |
| West Midlands | 8,0 |
| South West | 10,2 |
| East of England | 10,7 |
| South East | 11,6 |
| London | 14,2 |

Poniendo los números sobre la mesa, **London exige casi el doble de años que el North
East** (14,2 contra 6,2). Ninguna región está siquiera cerca de la regla prudente de
banca de ≤ 4 años de salario: norte y Gales resultan más accesibles, pero la compra sigue
estando lejos de lo sano (H2 se cumple).

### 5.2 Esfuerzo hipotecario 2022 (corte F2)

Cuota mensual sobre la vivienda mediana 2022, deflactada a GBP-2020 (CPI Banco de
Inglaterra, dic 2020→dic 2022 ≈ 13,7 %), con la tasa inicial media de la oferta
**winsorizada 6,04 %** (IQR), 90 % LTV y 25 años:

| region | cuota mensual | % del salario joven | acorde al 30 % |
|--------|--------------:|--------------------:|:--------------:|
| North East | GBP 772 | 43,0 % | no |
| North West | GBP 1.002 | 52,1 % | no |
| Yorkshire and The Humber | GBP 977 | 52,5 % | no |
| Wales | GBP 1.028 | 54,0 % | no |
| West Midlands | GBP 1.207 | 62,8 % | no |
| East Midlands | GBP 1.233 | 63,4 % | no |
| South West | GBP 1.589 | 79,1 % | no |
| East of England | GBP 1.745 | 81,8 % | no |
| South East | GBP 1.937 | 86,5 % | no |
| London | GBP 2.641 | 100,0 % | no |

El resultado es contundente: en **las 10 regiones la cuota supera el 30 % del salario** de
un joven, y en London el acreedor se llevaría el sueldo mensual por completo. La (H1) no
deja lugar a dudas: el crédito es el eslabón que bloquea el ingreso de los 18-34 a la
compra.

## 6. Clima vs precios (F3 × F1)

Anomalía anual = desviación frente a la normal histórica de cada estación (1931-2020),
promediada por región. Correlación de Pearson con la variación anual del precio
(n = 50 observaciones región-año, 2015-2020):

| variable climática | Pearson |
|--------------------|--------:|
| lluvia (anomalía) | -0,241 |
| temperatura máxima | -0,388 |
| temperatura mínima | -0,399 |
| heladas (af) | -0,145 |
| sol | +0,238 |

Los coeficientes salen **débiles y negativos** (más anomalía → menos variación del
precio, |r| < 0,4). Sencillamente, el clima **no explica** los niveles de precio por
región (H3 queda con evidencia débil); en el corto plazo domina el eje
salario-tasa-oferta que se ve en los cuadernos 04-05.

## 7. Validación

- Cuadro maestro: clave `(anio, region)` única, 60 filas, sin nulos en precio/salario,
  precios y salarios positivos.
- Hipoteca 2022: 10 regiones, cuotas y % positivos.
- Notebooks ejecutados sin errores; figuras embebidas y PNG en `md/figuras/`.

## 8. Limitaciones

- Salario mediano **nacional** (F4): la regionalización usa factores aproximados
  (ASHE 2021), no estadística por ciudad.
- F1 sin edad del comprador → análisis **ecológico** (región-año), no individual.
- Hipotecas **snapshot 2022**: las tasas anteriores a 2022 se aproximan con el corte.
- El clima cruza solo estaciones de Inglaterra y Gales; Escocia e Irlanda del Norte solo
  aportan a la anomalía nacional.

## 9. Figuras

`md/figuras/`: `fases_proyecto.png` (tabla de fases hasta fase 3 EN CURSO),
`faltantes_por_fuente.png` (MCAR/MAR/MNAR), `outliers_tasa_iqr.png` (IQR vs winsorizada),
`historico_precio_salario.png`, `precios_por_region.png`, `anos_salario_por_region.png`,
`anomalia_lluvia.png`, `clima_precio.png`, `hipoteca_2022.png` (+ `boxplot_price.png` y
`cuartiles_price.png` de la fase anterior).

## 10. Pendiente (fase 4)

Modelado y análisis de pronóstico (series y regresión), nuevos notebook(s) y el PDF de
propuesta generado por los autores desde `md/PROPUESTA_METODOLOGICA.md`.

## 11. Buenas prácticas de ingeniería y documentación

Criterios aplicados al entregable (buenas prácticas de código):

- **Funcionalidad**: el pipeline corre de principio a fin (`python main_proyecto.py`) y
  valida filas, nulos y rangos con asserts antes de exportar (`validacion.py`).
- **Justificación técnica**: cada decisión se apoya en evidencia (MCAR/MAR/MNAR, IQR vs
  z-score y contexto de negocio de cada fuente), no en gustos personales.
- **Documentación**: docstrings y comentarios (qué y por qué) en cada módulo, celdas
  markdown en los cuadernos, notas bajo las tablas del APA y tres informes para
  audiencias distintas (propuesta, avance 1 y documento APA).
- **Reproducibilidad**: rutas centralizadas en `config.py`, orígenes intocables,
  `requirements.txt`, cuadernos ejecutados sin errores y git con `.gitignore` (solo se
  versionan muestras `_muestra`).

Además se aplican modularidad (un módulo por responsabilidad), bitácora de decisiones
(`md/BITACORA_DECISIONES.md`) y control de versiones con un commit por corte.