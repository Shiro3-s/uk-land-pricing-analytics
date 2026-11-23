# Propuesta metodológica

**Accesibilidad de vivienda para jóvenes (18-34) en el Reino Unido: salario, hipoteca y clima**

**Proyecto de Minería de Datos · Marco KDD + CRISP-DM**

- **Autores:** Juan Pablo Angel Quitian · Joel Sebastian Bueno Medina
- **Enfoque:** Minería de datos descriptiva, predictiva, segmentación y patrones frecuentes
- **Fecha:** 2026

---

## 1. Problema

En el Reino Unido los jóvenes (18-34 años) compran cada vez menos vivienda. La compra
exige juntar un ahorro inicial y sostener una hipoteca, mientras los precios crecen por
encima de los salarios y el mercado de crédito encarece la cuota mensual. El problema de
negocio es: **¿cuánta vivienda puede pagar un joven (18-34) en cada ciudad/región del Reino
Unido, con hipoteca o préstamo, y de qué manera el clima se asocia con la evolución de los
precios?**

## 2. Pregunta de investigación

¿Qué relación existe entre el salario joven, el esfuerzo hipotecario y la evolución de los
precios por región en el Reino Unido (2015-2020), y qué papel juega el clima en esa
evolución?

## 3. Objetivo general

Analizar la accesibilidad de la vivienda para jóvenes (18-34) por ciudad/región del Reino
Unido mediante el cruce de las transacciones de compra-venta, la oferta hipotecaria, las
series de clima Met Office y los precios/salarios nacionales, aplicando el ciclo KDD y
CRISP-DM con entregables documentados por fase.

## 4. Objetivos específicos

1. Consolidar en un cuadro maestro **año × región** las cuatro fuentes (transacciones,
   hipotecas, clima y precios/salario) con calidad verificable (duplicados, nulos,
   atípicos y contratos de tipos).
2. Regionalizar el salario joven (18-34) a partir de la mediana nacional y de la brecha
   salarial por edad de 2021, con factores regionales de referencia (ASHE 2021).
3. Estimar el esfuerzo hipotecario de un joven por región en el corte 2022 (cuota mensual
   sobre la vivienda mediana: 90 % LTV, 25 años, tasa inicial media de la oferta) y
   compararlo con el umbral del 30 % del salario.
4. Medir la asociación entre la variación anual de precios y las anomalías climáticas
   (lluvia, temperatura, heladas) respecto de la normal histórica 1931-2020.
5. Comunicar los resultados con notebooks reproducibles, figuras y un informe por fase.

## 5. Hipótesis

- **H1.** En ninguna región el esfuerzo hipotecario de un joven (18-34) sobre la vivienda
  mediana de 2022 cumple el umbral del 30 % del salario.
- **H2.** Las regiones del norte y Gales requieren menos años de salario joven que London
  y el sur de Inglaterra.
- **H3.** Las anomalías climáticas (más lluvia o temperaturas extremas) se asocian de
  forma débil y negativa con la variación anual del precio de vivienda.

## 6. Marco metodológico (KDD + CRISP-DM)

El panorama metodológico presenta tres marcos: KDD, CRISP-DM y SEMMA. El equipo integra
**dos**: el ciclo KDD (claridad conceptual del descubrimiento) y **CRISP-DM** (arranca en el
negocio, es iterativo y es el estándar industrial). **SEMMA** se descarta por
parcial y ligado a la herramienta SAS. Cada fase se reporta con su equivalente en ambos marcos.

| Fase | KDD | CRISP-DM | Estado |
|------|-----|----------|--------|
| 1. Introducción, objetivos, hipótesis y Escalera DIKW | Selección | Entendimiento del negocio | ✔ Completada |
| 2. Fuentes (11,37 M filas), formatos y auditoría | Selección y preprocesamiento I | Entendimiento de los datos | ✔ Completada |
| 3. Limpieza (MCAR/MAR/MNAR, Tukey IQR, winsorización) y cruce | Preprocesamiento II | Preparación de los datos | ✔ Completada |
| 4. Transformación (3 Escaladores, One-Hot, VIF, PCA) y EDA (6 figuras) | Transformación y Exploración | Preparación y Exploración | ✔ Completada |
| 5. Modelado CON Etiqueta (Tipo 1 Regresión y Tipo 2 Clasificación) | Minería Supervisada | Modelado | ✔ Completada |
| 6. Modelado SIN Etiqueta (Tipo 3 Clustering 11,35M y Tipo 4 Apriori) | Minería No Supervisada | Modelado de Patrones | ✔ Completada |
| 7. Evaluación Forense (CRISP-DM 5), 3 Umbrales y Presentación Interactiva | Interpretación y Evaluación | Evaluación | ✔ Completada |
| 8. Subida de BD/CSV a Azure Database for PostgreSQL (Fase 3) | Operacionalización | Despliegue en Azure | ⏭ Lista en `md/GUIA_FASE3_SUBIDA_AZURE.md` |

El proyecto aplica el flujo estándar a las 4 fuentes del
Reino Unido: `deduplicar → categorías → nulos (MCAR/MAR/MNAR) → outliers (IQR) → cruzar`.

Cada faltante se **clasifica con evidencia** (MAR si depende de otra variable observada,
MCAR si es instrumental, MNAR si es estructural/de registro) y para los atípicos se
**compara IQR (Tukey) vs z-score**; se usa IQR + winsorización porque no exige normalidad
y detecta la cola real (p. ej., en la tasa inicial 2022: IQR detecta 762, z-score con
\|z\|>3 detecta 0). La clasificación es reproducible desde `script/proyecto/diagnostico.py`.

## 7. Fuentes de datos

| # | Fuente | Descripción | Periodo | Formato | Enlace (Kaggle) |
|---|--------|-------------|---------|---------|-----------------|
| F1 | UK Property Price Paid (Land Registry, vía Kaggle, geocodificado) | ~11,4 M de transacciones con precio, fecha y coordenadas (centroides de postcode-area) | 2015-2026 | `db/*` (en repo solo muestra) | [uk-property-sale-prices-20152026-geo-enriched](https://www.kaggle.com/datasets/mansiaggarwal88/uk-property-sale-prices-20152026-geo-enriched?select=uk_property_price_paid_geo.csv) |
| F2 | UK Mortgage Rates (Kaggle) | 7.796 productos hipotecarios (tasa inicial/APR, comisiones, plazo, tipo) | snapshot 2022 | `data/crudo/UK_Mortgage_Rate.csv` | [uk-mortgage-rates-thousands-of-mortgage-products](https://www.kaggle.com/datasets/thedevastator/uk-mortgage-rates-thousands-of-mortgage-products) |
| F3 | Met Office Historic Station Data | 36 estaciones mensuales de tmax, tmin, heladas (af), lluvia y sol | 1931-2020 | `data/crudo/MET_Office_Weather_Data.csv` | [uk-met-office-weather-data](https://www.kaggle.com/datasets/josephw20/uk-met-office-weather-data) |
| F4 | UK Average House Prices and Salary | Precio medio y salario mediana nacionales ajustados por inflación a 2020 | 1975-2020 (salario 1999+) | `data/crudo/Average_UK_houseprices_and_salary.csv` | [uk-median-house-prices-and-salary-19752020](https://www.kaggle.com/datasets/samuelcortinhas/uk-median-house-prices-and-salary-19752020) |
| F4b | Income by age and gender | Salario mediano por banda de edad y género (brecha salarial) | 2021 | `data/crudo/Income_by_age_and_gender.csv` | (fuente F4) |

Referencias: Land Registry Price Paid Data (OGL v3.0), Met Office historic station data,
Kaggle (4 conjuntos: F1-F4), Banco de Inglaterra (CPI).

## 8. Población, variables e indicadores

- **Población:** transacciones de vivienda del Reino Unido 2015-2020 + oferta hipotecaria 2022.
- **Variables de análisis:** precio de venta (mediana), salario mediano, salario joven
  regionalizado, tasa inicial de hipoteca, anomalía de lluvia/temperatura/heladas.
- **Indicadores calculados:**
  - `anos_salario` = precio mediana ÷ salario joven regional (años completos de salario).
  - `cuota_mensual` = PMT sobre 90 % LTV, 25 años y tasa inicial media (esfuerzo = cuota ÷
    salario mensual; umbral 30 %).
  - `anomalía` = valor anual − normal histórica de la estación (1931-2020).

## 9. Diseño del cruce y análisis

1. Agregar F1 por **año × región** y por **año × ciudad**.
2. Regionalizar el salario joven: mediana nacional del año × factor joven (brecha 2021)
   × factor regional (ASHE 2021).
3. Cruzar en `maestro_joven_uk` (anio × region, 2015-2020): precio + salario joven +
   anomalías climáticas.
4. Corte **hipoteca 2022**: esfuerzo hipotecario por región (precios 2022 deflactados a
   GBP-2020, CPI Banco de Inglaterra).
5. Correlación (Pearson) y regresión lineal entre variación anual del precio y anomalías.

## 10. Limitaciones

- El salario mediano es **nacional**; la regionalización usa factores de referencia
  aproximados (ASHE 2021), no estadística por ciudad.
- Las transacciones F1 **no tienen edad del comprador**: el análisis es ecológico
  (región-año), no individual.
- Las hipotecas son un **snapshot 2022**; las tasas anteriores se aproximan con ese corte.
- Se analizan series en sus rangos reales: el cruce completo abarca **2015-2020**.

## 11. Entregables Consolidados del Proyecto

- `Cuaderno_Maestro_Proyecto_UK.ipynb` y `main_maestro_completo.py` — **Libro Mayor Integral** con todas las fases (1 a 12), EDA completo y explicación individual de cada tabla y gráfica.
- `Presentacion_Interactiva_Proyecto_UK.html` (`app_presentacion_interactiva.py`) y `Presentacion_Proyecto_UK.pptx` — **Presentación Ejecutiva e Interactiva (Fase 2)** con Simulador de Predicción en Vivo (por opciones y a escrita) conectado a los CSV/BD.
- `libros/00` a `13` — 14 cuadernos modulares reproducibles numerados en orden.
- `script/proyecto/` — paquete modular con toda la lógica analítica, auditoría, telemetría y predicción dual.
- `data/modelos/` y `data/integrado/` — 4 modelos `.joblib` preentrenados, `contrato_prediccion.json`, `esquema_bd.sql` y 17 tablas CSV limpias (`1,83 MB`).
- `md/MEMORIA_TECNICA.md`, `md/DICCIONARIO_DATOS.md`, `md/BITACORA_DECISIONES.md` y `md/GUIA_FASE3_SUBIDA_AZURE.md` — documentación técnica oficial y guía guardada para la subida a **Azure Database for PostgreSQL**.