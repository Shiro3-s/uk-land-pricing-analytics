# Accesibilidad de vivienda para jóvenes (18-34) en el Reino Unido

**Proyecto de Minería de Datos · UNIMINUTO Ibagué 2026-2 · Avance 1**

**Propósito.** Estimar **cuántos años de salario necesita un joven (18-34) para comprar
una vivienda** en cada región del Reino Unido, **qué esfuerzo hipotecario** le exige cada
una (porción del salario que se va en la cuota) y **qué papel juega el clima** en la
evolución de los precios.

**Qué contiene este repositorio.** 5 fuentes de datos del Reino Unido · 8 cuadernos por
fase · 1 cuaderno consolidado · 11 módulos de Python · 17 figuras · diccionario de datos ·
memoria técnica · bitácora de decisiones.

**Alcance actual.** Las 6 primeras fases del ciclo están completadas: el modelado
**tabular** comparativo (RL, KNN, árbol, random forest) quedó cerrado con validación
cruzada. El **objetivo final del proyecto es un modelo predictivo** de la asequibilidad
(esfuerzo hipotecario) por región y año, y para llegar a él el ciclo se amplía con cuatro
fases más (7-10), hoy en la **fase 7: ingeniería de predictores y validación temporal**.

## Avance por sesión

| Sesión | Tema | Estado | Evidencia en el repositorio |
|---|---|---|---|
| 1 | Introducción y marcos (KDD, CRISP-DM) | ✔ Completada | `README.md`, `md/PROPUESTA_METODOLOGICA.md` |
| 2 | Fuentes y tipos de datos | ✔ Completada | `libros/01_Carga_4_Fuentes.ipynb` |
| 3 | Limpieza de datos | ✔ Completada | `libros/02`, `libros/03`, `libros/04`, `libros/05`, `Diagnostico_Limpieza` |
| 4 | Transformación y reducción | ✔ Completada | `libros/06_Transformacion_y_Matriz.ipynb` |
| 5 | Análisis exploratorio (EDA) | ✔ Completada | `libros/07_EDA_Accesibilidad.ipynb` |
| 6 | Cierre de unidad · informe de laboratorio | ✔ Completada | `md/README_LABORATORIO.md`, `Avances/Cuaderno_Avance_1_Completo.ipynb` |
| 7 | Modelado (anticipo) | ✔ Completada | `libros/08_Modelado_Accesibilidad.ipynb` |

> El **informe de laboratorio** exigido al cerrar la unidad está en
> [`md/README_LABORATORIO.md`](md/README_LABORATORIO.md). Este README, en cambio, es la
> presentación del proyecto y se actualiza al cerrar cada sesión.

## Los datos que manejamos (nuestro «Excel»)

Este proyecto no usa hojas de cálculo: usa **CSV y Parquet**, que son el equivalente
moderno de un Excel pero con tipos de dato explícitos y compresión. La equivalencia para
entender qué es cada cosa:

| En un Excel | En este proyecto | Qué es |
|---|---|---|
| Libro / cuadrante | `.parquet` | `maestro_joven_uk.parquet`: **60 filas × 16 columnas**, una fila por región-año |
| Hoja de cálculo | `.csv` | `accesibilidad_vivienda_joven_uk_v1.csv` (dataset limpio versionado) |
| Filas / registros | `.parquet` de F1 | `uk_property_price_limpio.parquet`: **11.369.413 transacciones** |
| Columnas / campos | esquema tipado | 16 columnas del maestro (`float64`, `int64`, `str`), documentadas en `md/DICCIONARIO_DATOS.md` |
| Power Query / importador | `script/proyecto/cargar.py` | Carga **defensiva**: cuenta filas y columnas e imprime los tipos inferidos |
| Fórmulas / macros | `script/proyecto/indicadores.py` | PMT, años de salario, esfuerzo, correlaciones |
| Filtros y limpieza | `script/proyecto/limpiar.py` + `diagnostico.py` | Duplicados, nulos, atípicos, con bitácora de cada decisión |

### Los cinco archivos que usamos

| # | Archivo | Formato | Filas | Peso | ¿En el repo? |
|---|---|---|---|---|---|
| F1 | `db/uk_property_price.csv` | CSV | 11.369.413 | **2.220 MB** | ❌ local (solo muestra de 2.000 filas) |
| F1 | `data/limpia/uk_property_price_limpio.parquet` | Parquet | 11.369.413 | 434 MB | ❌ local |
| F2 | `data/crudo/UK_Mortgage_Rate.csv` | CSV | 7.796 | 0,94 MB | ✔ completo |
| F3 | `data/crudo/MET_Office_Weather_Data.csv` | CSV | 37.049 | 1,47 MB | ✔ completo |
| F4 | `data/crudo/Average_UK_houseprices_and_salary.csv` | CSV | 46 | <0,01 MB | ✔ completo |
| F4b | `data/crudo/Income_by_age_and_gender.csv` | CSV | 12 | <0,01 MB | ✔ completo |

**Por qué F1 no se sube.** 2,2 GB de transacciones es inviable para un repositorio; y para
el cruce solo necesitamos su **agregado** por región-año (60 filas). El repositorio
versiona **muestras** (2.000 filas por archivo en `db/`, y **100.000 filas** de la versión
limpia en `data/limpia/`) para que se pueda reproducir parcialmente, y los
datos completos se descargan de los enlaces de Kaggle (ver *Descarga de los datos*).

## Fases del proyecto (ciclo KDD + CRISP-DM)

| Fase | KDD | CRISP-DM | Estado |
|------|-----|----------|--------|
| 1. Introducción y objetivos | Selección | Comprensión del negocio | Completada |
| 2. Fuentes y tipos de datos | Selección y preprocesamiento I | Comprensión de datos | Completada |
| 3. Limpieza y cruzamiento | Preprocesamiento II | Preparación de datos | Completada |
| 4. Transformación y matriz X·y | Transformación | Preparación de datos | Completada |
| 5. EDA | Interpretación | Exploración | Completada |
| 6. Modelado comparativo (anticipo) | Minería | Modelado | Completada (tabular, CV-5 barajado) |
| 7. Ingeniería de predictores y validación temporal | Transformación + Minería | Preparación + Modelado | **En curso** |
| 8. Modelo predictivo final | Minería | Modelado | Pendiente |
| 9. Evaluación e interpretación | Interpretación | Evaluación | Pendiente |
| 10. Despliegue e informe final | — | Despliegue | Pendiente |

**De la fase 6 a la 10 (rumbo al modelo predictivo).** El objetivo final es **predecir** el
esfuerzo hipotecario (y, por extensión, la asequibilidad) por región y año, no solo
comparar modelos. La fase 6 cerró el *benchmark* tabular con **validación cruzada barajada**;
para predecir hace falta un salto metodológico: los 6 años por región están correlacionados
en el tiempo, así que un fold barajado filtra información del futuro. Las fases 7-10
siguen el mismo marco KDD + CRISP-DM:

- **F7 (en curso) — predictores y validación temporal:** nuevas variables predictoras
  (índice de tensión precio/salario nacional, lags del precio y del esfuerzo, tendencia
  temporal; candidatas ya probadas en la sección 9-bis del cuaderno consolidado) y
  reemplazo del CV barajado por **validación temporal** (entrenar 2015-2018, validar
  2019-2020), que es la única que respeta el orden de los años.
- **F8 — modelo predictivo final:** tuning (GridSearchCV *dentro* del pipeline, sin fuga)
  de los cuatro candidatos de la fase 6 más Gradient Boosting, con la validación temporal
  como veredicto; baseline ingenuo (persistencia del año anterior) como suelo.
- **F9 — evaluación e interpretación:** métricas fuera de muestra (R², RMSE, MAE) por
  región y por año, importancia de predictores (permutación + coeficientes de la RL),
  y estabilidad del modelo entre ventanas temporales.
- **F10 — despliegue e informe final:** modelo congelado como módulo `script/proyecto/`,
  función de predicción reproducible, informe final con la lectura para las dos
  audiencias (técnica y de políticas públicas).

## Metodología (KDD + CRISP-DM)

**Qué metodologías utilizamos.** El proyecto se apoya en **dos marcos de forma
combinada** (KDD + CRISP-DM), tal como se justifica en `md/PROPUESTA_METODOLOGICA.md`
y en el `Avances.pdf`:

- **KDD** (Fayyad et al., 1996): el ciclo analítico del descubrimiento — *selección,
  preprocesamiento, transformación, minería e interpretación*.
- **CRISP-DM** (Wirth & Hipp, 2000): el ciclo de negocio — *entendimiento del negocio,
  entendimiento de los datos, preparación, modelado, evaluación y despliegue*.
- **SEMMA** se descartó: está ligada a la herramienta SAS y cubre solo la parte
  analítica, sin el ciclo de negocio.

**Por qué estos dos y no uno solo.**

1. **Cobertura del ciclo completo:** KDD aporta el detalle conceptual de las etapas
   analíticas y CRISP-DM lo enmarca en el negocio; usar uno solo deja un hueco (KDD
   sin despliegue, CRISP-DM sin el detalle de la teoría).
2. **Enfoque de negocio:** CRISP-DM enmarca el análisis en el problema de negocio
   (comprensión del negocio, de los datos, preparación, modelado y evaluación),
   que es exactamente lo que cubre este repositorio.
3. **Trazabilidad:** cada decisión se mapea a su etapa KDD y su fase CRISP-DM
   (columnas de la tabla de fases), una buena práctica de documentación.

**Cómo vamos (estado actual).**

- **Fases 1-3 completas:** pregunta y objetivos, fuentes, tipos de datos, limpieza
  (MCAR/MAR/MNAR, IQR vs z-score, winsorización) y cruce del cuadro maestro.
- **Fase 4 (transformación) completa:** matriz X·y (60 × 18) con `StandardScaler` +
  one-hot (`drop_first`, 9 dummies), y comparación empírica de métodos (escaladores,
  codificación, discretización, PCA, diagnóstico supervisado CV-5) en el libro 06.
- **Fase 5 (EDA) completa:** libro 07 con la estructura de un informe EDA
  (univariado con 6 figuras, bivariado/multivariado, 3 hallazgos evidencia→acción),
  con las fórmulas nombradas de cada medida y la cuantificación de las cercas de Tukey.
- **Fase 6 (modelado comparativo) completa en alcance tabular:** libro 08 compara regresión
  lineal, KNN tuneado, árbol y random forest con **CV-5 barajado** y `drop_first` en
  el one-hot, tras pasar el checklist de 6 ítems de preparación.
- **Fase 7 (en curso) — ingeniería de predictores y validación temporal:** ya se iniciaron los predictores temporales (índice de tensión precio/salario nacional, lags del precio y del esfuerzo, tendencia temporal) y se sustituye el CV barajado por **validación temporal** (entrenar 2015-2018, validar 2019-2020), la única que respeta el orden de los años y evita filtrar información del futuro.
- **Fase 8 (pendiente) — modelo predictivo final:** tuning (GridSearchCV dentro del pipeline, sin fuga) de los cuatro candidatos de la fase 6 más Gradient Boosting, con validación temporal como veredicto; baseline ingenuo (persistencia del año anterior) como suelo.
- **Fase 9 (pendiente) — evaluación e interpretación:** métricas fuera de muestra (R², RMSE, MAE) por región y año, importancia de predictores (permutación + coeficientes de la RL) y estabilidad del modelo entre ventanas temporales.
- **Fase 10 (pendiente) — despliegue e informe final:** modelo congelado como módulo `script/proyecto/`, función de predicción reproducible e informe final para las dos audiencias (técnica y de políticas públicas).

**Resultados del avance (fase 3).**

- Cuadro maestro `maestro_joven_uk`: **60 filas** (10 regiones × 6 años, 2015-2020)
  sin huecos en precio y salario.
- Faltantes: la tasa de hipotecas es **MAR** (depende del tipo de producto); el clima
  es **MCAR** instrumental (falla conjunta de reporte); el sol y el salario histórico
  son **MNAR** estructurales. No se imputa en esta fase.
- Outliers de la tasa inicial: **IQR (Tukey) detecta 762 productos (9,77 %)**; z-score
  con \|z\|>3 detecta 0. Se elige IQR y la tasa se winsoriza (**media 6,11 % → 6,04 %**).
- Salario joven: **GBP 25.353/año** (factor 0,805 sobre la mediana nacional),
  regionalizado con ASHE 2021 (ONS).
- Años de salario para comprar: de **6,2** (North East) a **14,2** (London).
- Esfuerzo hipotecario 2022: de **43,0 %** (North East) a **100,0 %** (London) del
  salario; **ninguna región cumple el umbral del 30 %.**
- Clima: anomalías anuales de lluvia/temperatura/heladas frente a la normal 1931-2020,
  correlacionadas con la variación de precios por región (`md/figuras/*.png`).

Detalle de decisiones: `md/BITACORA_DECISIONES.md` e `md/INFORME_AVANCE1.md`.

## Problema y objetivos

Los jóvenes compran cada vez menos vivienda. Objetivo: **¿cuánta vivienda puede pagar un
joven (18-34) en cada ciudad/región con su salario, con hipoteca o préstamo, y el clima
cómo se asocia con la evolución de precios?**

Indicadores del cruce `maestro_joven_uk` (año × región, 2015-2020):

- `anos_salario` — años completos de salario joven para comprar la vivienda mediana.
- Esfuerzo hipotecario 2022 — cuota mensual (PMT, 90 % LTV, 25 años, tasa inicial media
  **winsorizada con IQR**) como % del salario joven mensual (umbral 30 %).
- Anomalías climáticas — desviación anual de lluvia/temperatura/heladas frente a la
  normal histórica 1931-2020 y su correlación con la variación de precios.

**Calidad de datos tratada en la fase 3:** clasificación de faltantes por fuente
(**MCAR/MAR/MNAR** con su evidencia) y decisión de método de outliers (**IQR (Tukey) vs
z-score**: IQR no exige normalidad y detecta la cola real; z-score con |z|>3 no detecta
nada en la tasa inicial). Ver `script/proyecto/diagnostico.py`.

## Fuentes de datos

| # | Fuente | Periodo | Archivo | Enlace (Kaggle) |
|---|--------|---------|---------|-----------------|
| F1 | Propiedad (Land Registry, geocodificada) | 2015-2026 | `db/*` (en repo solo `_muestra`) | [uk-property-sale-prices-20152026-geo-enriched](https://www.kaggle.com/datasets/mansiaggarwal88/uk-property-sale-prices-20152026-geo-enriched?select=uk_property_price_paid_geo.csv) |
| F2 | Hipotecas (snapshot 2022) | 2022 | `data/crudo/UK_Mortgage_Rate.csv` | [uk-mortgage-rates-thousands-of-mortgage-products](https://www.kaggle.com/datasets/thedevastator/uk-mortgage-rates-thousands-of-mortgage-products) |
| F3 | Met Office (36 estaciones) | 1931-2020 | `data/crudo/MET_Office_Weather_Data.csv` | [uk-met-office-weather-data](https://www.kaggle.com/datasets/josephw20/uk-met-office-weather-data) |
| F4 | Precio medio + salario mediana (GBP 2020) | 1975-2020 | `data/crudo/Average_UK_houseprices_and_salary.csv` | [uk-median-house-prices-and-salary-19752020](https://www.kaggle.com/datasets/samuelcortinhas/uk-median-house-prices-and-salary-19752020) |
| F4b | Salario por edad y género (brecha 2021) | 2021 | `data/crudo/Income_by_age_and_gender.csv` | (misma fuente F4) |

Referencias: Land Registry Price Paid Data (OGL v3.0), Met Office historic station data,
Kaggle (transacciones F1, hipotecas F2, clima F3, precios-salario F4), Banco de Inglaterra.

## Estructura del proyecto

```
mineria/
├── main_proyecto.py            # ejecuta el analisis de accesibilidad (11 partes)
├── main_limpieza.py            # ejecuta la limpieza de F1 (7 fases)
├── Avances.pdf                 # documento del avance en normas APA 7 (PDF)
├── Avances/
│   ├── Avances.pdf             # copia del documento del avance
│   └── Cuaderno_Avance_1_Completo.ipynb   # cuaderno consolidado de las 6 sesiones
├── db/                         # F1 origin + muestras _muestra (locales y repo)
├── data/
│   ├── crudo/                  # F2, F3, F4 y F4b (pequenos, si se suben)
│   ├── limpia/                 # F1 limpia + limpias por fuente (parquet locales)
│   └── integrado/              # maestro_joven_uk + hipoteca_2022 + historico
├── script/
│   ├── limpieza/               # limpieza de F1 (cuartiles/IQR/MCAR/bitacora)
│   └── proyecto/               # 4 fuentes -> cruce maestro_joven_uk
│       ├── config.py           # rutas, reglas, factores regionales, estaciones
│       ├── cargar.py           # carga defensiva por fuente
│       ├── limpiar.py          # duplicados -> tipos -> nulos -> outliers
│       ├── diagnostico.py      # clasificacion MCAR/MAR/MNAR + IQR vs z-score
│       ├── transformaciones.py # agregados, anomalias, salario joven
│       ├── cruces.py           # maestro_joven_uk (anio x region)
│       ├── indicadores.py      # anos_salario, PMT, correlacion clima
│       ├── graficos.py         # 9 figuras PNG + tabla de fases
│       ├── validacion.py       # asserts post-cruce
│       ├── bitacora.py         # md/BITACORA_DECISIONES.md
│       ├── pipeline.py         # orquestador de 11 partes
│       ├── matriz_xy.py        # matriz X·y + comparacion empirica de metodos
│       ├── eda.py              # EDA terminal + 6 figuras eda_*.png
│       └── modelado.py         # RL/KNN/arbol/forest CV-5
└── libros/                     # notebooks reproducibles (01-08 + 2 previos)
```

## Instalación

Requisitos: Python 3.10+ (desarrollado con Python 3.14).

```powershell
pip install -r script/requirements/requirements.txt
```

Extras de Jupyter para ejecutar/exportar los cuadernos:

```powershell
pip install -r script/requirements/requirements-notebooks.txt
```

## Ejecución

Todos los comandos desde la raíz del proyecto.

| Tarea | Comando |
|---|---|
| Analisis de accesibilidad (4 fuentes, 11 partes) | `python main_proyecto.py` |
| Limpieza de F1 (7 fases) | `python main_limpieza.py` |
| Matriz X·y + comparacion de metodos | `python script/proyecto/matriz_xy.py` |
| EDA terminal + figuras | `python script/proyecto/eda.py` |
| Modelado CV-5 | `python script/proyecto/modelado.py` |
| Ejecutar un cuaderno (in-place) | `python -m jupyter nbconvert --to notebook --execute --inplace "libros/03_Cruces_Affordability_Joven.ipynb" --ExecutePreprocessor.timeout=900` |
| Exportar un cuaderno a HTML | `python -m jupyter nbconvert --to html --execute "libros/05_Clima_y_Precios.ipynb" --ExecutePreprocessor.timeout=900` |

### Libros

- `01_Carga_4_Fuentes.ipynb` — contratos, esquemas, ventana de cruce y tabla de fases.
- `02_Limpieza_4_Datasets.ipynb` — MCAR/MAR/MNAR por fuente, IQR vs z-score y bitácora.
- `03_Cruces_Affordability_Joven.ipynb` — `maestro_joven_uk` y ciudades.
- `04_Hipoteca_Joven.ipynb` — esfuerzo hipotecario 2022 por región (tasa winsorizada).
- `05_Clima_y_Precios.ipynb` — **clima vs precios**: anomalías climáticas frente a la
  normal 1931-2020 por estación agregadas a región, y correlación de **Pearson** con
  la variación anual del precio. Fija la hipótesis H3: el clima es contexto, no motor.
- `06_Transformacion_y_Matriz.ipynb` — matriz X·y + tabla de tipos de dato y temas +
  comparación empírica de métodos (escaladores, codificación, discretización, PCA y
  diagnóstico supervisado con CV-5).
- `07_EDA_Accesibilidad.ipynb` — **EDA**: informe con 6 secciones (contexto, calidad,
  univariado con 6 figuras matplotlib, bivariado/multivariado, 3 hallazgos
  evidencia→acción, anexos) + cuantificación de las **cercas de Tukey** (punto 3.7).
- `08_Modelado_Accesibilidad.ipynb` — **modelado**: regresión lineal, KNN
  tuneado, árbol y random forest con CV-5 (R²/RMSE/MAE) e importancia de rasgos,
  precedido del **checklist de 6 ítems de preparación** verificado en ejecución.
- `Lectura.ipynb` / `Diagnostico_Limpieza.ipynb` — lectura y limpieza de F1 (previos).

### Cuaderno consolidado

`Avances/Cuaderno_Avance_1_Completo.ipynb` (70 celdas) reúne **todo el proyecto en un
solo documento ejecutable**: las 5 fuentes, la limpieza, el cruce, los indicadores, el
clima, la transformación, las 14 comparaciones de métodos, el EDA, el modelado, el
checklist y las limitaciones. **Invoca los módulos** en vez de reescribir el código, de
modo que ninguna cifra está escrita a mano.

Trae una bandera `REPRODUCIR_PIPELINE`: en `True` re-ejecuta el pipeline completo y
**verifica con `assert`** que reproduce exactamente el cuadro maestro versionado; en
`False` (por defecto) usa los artefactos y funciona sin el F1 de 434 MB. Tiempos medidos:
**≈ 42 s** en modo artefacto, **≈ 31 s** re-ejecutando el pipeline completo.

### Cambios recientes

- **Cuaderno consolidado del Avance 1** (`Avances/Cuaderno_Avance_1_Completo.ipynb`):
  un cuaderno de 70 celdas que ejecuta las 6 sesiones de punta a punta. Invoca los módulos
  del proyecto en lugar de duplicar el código, embebe las 17 figuras, y trae la bandera
  `REPRODUCIR_PIPELINE` que **verifica con `assert`** que la re-ejecución del pipeline
  reproduce el cuadro maestro versionado. Incluye la comprobación medida de por qué el
  barajado importa: sin él cada pliegue ve solo **2 regiones**; barajado, entre **5 y 9**.
- **Corrección metodológica del modelado** (dos cambios con efecto medido):
  - **CV-5 barajado.** El cuadro maestro está ordenado por región-año, así que
    `cv=5` sin barajar evaluaba cada modelo sobre una región distinta. Con
    `KFold(5, shuffle=True, random_state=42)` el KNN sube de **0,638 a 0,844**, el
    árbol de **0,900 a 0,963** y el bosque de **0,943 a 0,980**.
  - **`drop="first"` en el one-hot.** Las 10 columnas dummy de región sumaban
    siempre 1 (trampa de variables mudas, VIF infinito). Con 9 columnas la región
    de referencia es implícita y **los coeficientes de la regresión se vuelven
    interpretables** (London +0,0002 frente a todos en 0,0000-0,0001).
    La matriz X·y pasa de 60 × 19 a **60 × 18**.
  - Se documenta que las comisiones se suman **fuera** de la fórmula PMT
    (`comisiones/n` lineal): es una aproximación, no un recargo compuesto.
- **Rigor estadístico en el libro 07 (EDA)**: se añadió la especificación exacta de los
  datos (el maestro **60 × 16**, las hipotecas **7.796 × 14**, unidades y periodo), las
  fórmulas con nombre de cada medida (media, mediana, desviación **muestral** de Bessel,
  asimetría de **Fisher-Pearson**, curtosis **excedente**, IQR, **Pearson** y recta de
  mínimos cuadrados), las justificaciones de cada elección (por qué mediana y no media,
  por qué **Tukey** y no z-score, por qué Pearson y no Spearman) y el punto **3.7**, que
  cuantifica las cercas de Tukey sobre el esfuerzo (IQR 0,284; cercas 0,053 y 1,189;
  **0 atípicos globales**, 1 en London por el IQR regional con n = 6).
- **Libro 05 restaurado** (antes `Prueba_05_Clima_y_Precios.ipynb`), lo que completa la
  secuencia 01-08: ahora es un capítulo de pleno derecho del cruce climático. Incorpora
  la especificación de datos (F3: 37.049 registros, 36 estaciones, 1931-2020), la
  fórmula de la **anomalía simple** (valor − normal 1931-2020) con la justificación de
  por qué **no** es un z-score, el criterio de agregado (promedio, no suma), la fórmula
  del **Pearson** con su guía de umbrales y una lectura honesta de la H3: |r| de 0,15 a
  0,40 equivale a explicar del 2 % al 16 % de la varianza del precio anual.
- **Checklist de 6 ítems en el libro 08**: verificación ejecutada y reproducible de
  tipos, nulos justificados, duplicados, categorías consistentes, escala/codificación
  (con la razón por qué el escalado es imprescindible en KNN y neutro en árboles) y
  diccionario de datos. Resultado: **6 de 6 ítems se cumplen**.
- **Libro 6 completado** con la tabla de tipos de dato y temas (16 columnas: dtype +
  tema + rol, verificada en vivo) en la sección 2, cerrando la fase 4 (transformación).
- **Libro 7 nuevo (`07_EDA_Accesibilidad.ipynb`)**: EDA formal con la estructura de un
  informe; 6 figuras (histograma, boxplot, barras, dispersión+recta, heatmap,
  matriz de dispersión) y 3 hallazgos. Espejo en terminal: `script/proyecto/eda.py`
  (exporta las figuras a `md/figuras/eda_*.png`).
- **Libro 8 (`08_Modelado_Accesibilidad.ipynb`)**: compara RL (R² 1,000), random
  forest (0,980), árbol (0,963) y KNN tuneado (0,844) con **CV-5 barajado** y semilla
  42; importancia de rasgos (permutación RF: `anos_salario` 0,580, `precio_mediana`
  0,314). Espejo: `script/proyecto/modelado.py`. Este resultado es el **punto de partida**
  del modelo predictivo (fases 7-10): la validación temporal, que respeta el orden de los
  años, es la que decidirá el modelo final.
- **Diccionario de datos** (`md/DICCIONARIO_DATOS.md`): las 16 columnas del maestro y
  las derivadas en formato tabla (nombre, tipo, descripción, rango, unidades,
  tratamiento) con trazabilidad libro por libro.
- `script/proyecto/matriz_xy.py` con la sección `[7] Comparacion empirica
  de metodos` (espejo en terminal del notebook 06).

## Salidas

- `data/integrado/maestro_joven_uk.parquet` — cuadro año × región (2015-2020) y tablas
  planas CSV.
- `data/integrado/matriz_features.parquet` — matriz X·y (**60 × 21**: 2 identificadores
  `region`/`anio` + 18 X con `drop_first` + la columna objetivo `esfuerzo`).
- `data/integrado/accesibilidad_vivienda_joven_uk_v1.csv` — **dataset limpio versionado**
  del cuadro maestro (60 × 16). Conserva los 28 nulos documentados a propósito.
- `Avances.pdf` — **documento del avance en normas APA 7** (portada, resumen, método,
  resultados, referencias).
- `md/INFORME_AVANCE1.md` — informe de la fase 3 (limpieza + cruzamiento).
- `md/PROPUESTA_METODOLOGICA.md` — propuesta editable (los autores generan el PDF).
- `md/DICCIONARIO_DATOS.md` — diccionario de datos del cuadro maestro (formato tabla).
- `md/MEMORIA_TECNICA.md` — **memoria técnica**: qué datos usamos y por qué, la cadena de
  transformación, los fundamentos estadísticos, matemáticos y de ingeniería, las 14
  comparaciones de alternativas y las limitaciones declaradas.
- `md/BITACORA_DECISIONES.md` — decisiones de limpieza y cruce.
- `md/figuras/*.png` — 9 figuras del analisis + 2 de la limpieza F1 + 6 del EDA
  (`eda_*.png`) (+ tabla de fases).

**Sobre los datos (muestras):** el repositorio **no sube los archivos grandes completos**
(F1: el CSV de 2,22 GB y el parquet limpio de 434 MB son locales). En `db/` y
`data/limpia/` solo se sube **una muestra con sufijo `_muestra`** de cada dataset, para
que el repositorio sea liviano y el análisis sea reproducible en forma parcial: las
muestras de `db/` traen **2.000 filas** y la muestra limpia
**`uk_property_price_limpio_muestra.csv` trae 100.000 filas × 22 columnas (20 MB)**
(su parquet versionado, 2.000 filas). Las fuentes F2-F4 son pequeñas (archivos Kaggle
< 1,5 MB) y se incluyen completas en `data/crudo/`. **Los datasets completos se descargan
de los enlaces** de la sección de fuentes: F1, F2, F3 y F4 (ver más abajo), y
`main_limpieza.py` regenera la versión limpia de F1 a partir del CSV original.

### Descarga de los datos completos

| Dataset | Muestra en el repo | Descarga completa |
|---|---|---|
| F1 transacciones (11,4 M filas, CSV 2,22 GB) | `db/uk_property_price_muestra.csv` | [Kaggle: uk-property-sale-prices-20152026-geo-enriched](https://www.kaggle.com/datasets/mansiaggarwal88/uk-property-sale-prices-20152026-geo-enriched?select=uk_property_price_paid_geo.csv) |
| F1 pre-limpieza | `db/uk_property_semiorganizado_muestra.csv` | (se genera con `main_limpieza.py` desde el CSV de Kaggle) |
| F2 hipotecas 2022 | completa en `data/crudo/` | [Kaggle: uk-mortgage-rates](https://www.kaggle.com/datasets/thedevastator/uk-mortgage-rates-thousands-of-mortgage-products) |
| F3 Met Office 1931-2020 | completa en `data/crudo/` | [Kaggle: uk-met-office-weather-data](https://www.kaggle.com/datasets/josephw20/uk-met-office-weather-data) |
| F4 precios + salario 1975-2020 | completa en `data/crudo/` | [Kaggle: uk-median-house-prices-and-salary-19752020](https://www.kaggle.com/datasets/samuelcortinhas/uk-median-house-prices-and-salary-19752020) |

> Todos los archivos con extensión `_muestra` son **muestras**, no los datos completos:
> 2.000 filas las de `db/` y el parquet limpio, y **100.000 filas** la muestra limpia en
> CSV (`data/limpia/uk_property_price_limpio_muestra.csv`, 20 MB).

## Referencias

- F1 transacciones (Kaggle, 11,4 M): [uk-property-sale-prices-20152026-geo-enriched](https://www.kaggle.com/datasets/mansiaggarwal88/uk-property-sale-prices-20152026-geo-enriched?select=uk_property_price_paid_geo.csv)
- F2 hipotecas (Kaggle): [uk-mortgage-rates-thousands-of-mortgage-products](https://www.kaggle.com/datasets/thedevastator/uk-mortgage-rates-thousands-of-mortgage-products)
- F3 clima (Kaggle): [uk-met-office-weather-data](https://www.kaggle.com/datasets/josephw20/uk-met-office-weather-data)
- F4 precios/salario (Kaggle): [uk-median-house-prices-and-salary-19752020](https://www.kaggle.com/datasets/samuelcortinhas/uk-median-house-prices-and-salary-19752020)
- Met Office: [Historic station data](https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data)
- Land Registry: [Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)