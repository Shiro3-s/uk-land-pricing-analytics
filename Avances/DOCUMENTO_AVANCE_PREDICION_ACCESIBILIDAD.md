# Avance del Proyecto: Predicción de Accesibilidad Vivienda Joven

## 1. Introducción

Este proyecto tiene como objetivo desarrollar un modelo de predicción para evaluar la accesibilidad de la vivienda para personas jóvenes (18-34 años) en el Reino Unido. El análisis parte de la integración de cuatro fuentes de datos: transacciones inmobiliarias, productos hipotecarios, datos climáticos y series de precios y salarios.

El objetivo final es predecir si, dado un conjunto de características económicas y climáticas, una región y año resultan accesibles para la compra de vivienda mediana por un joven bajo condiciones de hipoteca estándar.

El esfuerzo hipotecario se define como la relación entre la cuota mensual de la hipoteca y el salario joven mensual. Se establece un umbral de referencia del 30 %: si el esfuerzo es menor o igual a ese valor, la vivienda se considera accesible.

## 2. Metodología aplicada

El proyecto sigue una organización por etapas con responsabilidades diferenciadas por componente:

- Cada módulo del código tiene una función única y bien delimitada (carga, limpieza, diagnóstico, transformación, cruce, indicadores, validación, gráficos, bitácora, configuración, orquestación).
- Un pipeline maestro coordina la ejecución secuencial de las etapas, garantizando trazabilidad y orden determinista.
- Toda decisión relevante queda registrada en una bitácora de decisiones con justificación técnica.
- Los parámetros y reglas de negocio están centralizados en un módulo de configuración.
- Se utiliza semilla fija para particiones aleatorias y modelos, permitiendo que los resultados sean idénticos en cada ejecución.
- Se ejecutan validaciones post-cruce para verificar unicidad de claves, ausencia de nulos en columnas críticas y cumplimiento de límites de outliers tratados.

## 3. Fuentes de datos

Se integran cuatro fuentes principales:

1. Transacciones inmobiliarias: precios de venta georreferenciados con período 2015-2026. Se utiliza como base para precios mediano por región y año.
2. Productos hipotecarios: snapshot de oferta hipotecaria del año 2022. Incluye tasa inicial, comisiones, plazo y tipo de producto.
3. Datos climáticos: series de estaciones meteorológicas con variables de temperatura, lluvia, horas de sol y días de heladas. Se calculan anomalías anuales respecto a una normal histórica.
4. Precios y salarios: mediana de precios de vivienda y mediana de salarios por edad y género, con ajuste a precios reales de 2020.

El cruce de las fuentes genera el cuadro maestro a nivel año × región para el período 2015-2020.

## 4. Cálculo del esfuerzo hipotecario

### 4.1 Cuota mensual

Se calcula la cuota mensual de una hipoteca francesa (amortización constante) sobre el 90 % del precio mediano de la vivienda, con plazo de 25 años y tasa inicial media de la oferta hipotecaria limpia. Se agregan comisiones promedio prorrateadas.

La fórmula general es:

$$C = P \cdot \frac{r}{1 - (1 + r)^{-n}} + \frac{comisiones}{n}$$

donde:
- $P$ = capital prestado (precio × LTV)
- $r$ = tasa mensual (tasa anual por ciento / 100 / 12)
- $n$ = número de cuotas (plazo en años × 12)

El esfuerzo se obtiene dividiendo la cuota mensual entre el salario joven mensual (salario anual joven dividido entre 12).

### 4.2 Variable objetivo binaria

$$accesible = \begin{cases} 1 & \text{si } esfuerzo \le 0.30 \\ 0 & \text{en caso contrario} \end{cases}$$

## 5. Fase 1 – Análisis exploratorio de datos

### 5.1 Descripción de la muestra

El cuadro maestro contiene 60 observaciones correspondientes a 10 regiones durante 6 años (2015-2020). La clave compuesta (año, región) es única y no presenta duplicados.

### 5.2 Distribución del esfuerzo

| Estadístico | Valor |
|-------------|-------|
| Mínimo | 0.416 |
| Q1 | 0.479 |
| Mediana | 0.564 |
| Media | 0.625 |
| Q3 | 0.763 |
| Máximo | 1.065 |
| Desviación estándar | 0.182 |
| Asimetría | 0.84 |
| Curtosis | -0.37 |

### 5.3 Detección de valores atípicos

Se aplica la regla de Tukey (1.5 × IQR) para detectar valores atípicos:

- Esfuerzo: no se detectan valores atípicos (ni superiores ni inferiores).
- Precio mediano: se detectan 4 valores superiores al límite superior.

La distribución del esfuerzo muestra asimetría positiva moderada.

### 5.4 Análisis por región

El esfuerzo medio varía significativamente entre regiones:

- Mayor esfuerzo: Londres, Sureste de Inglaterra, Este de Inglaterra.
- Menor esfuerzo: Noreste de Inglaterra, Noroeste de Inglaterra, Gales.

Todas las regiones presentan esfuerzo medio superior al umbral de referencia del 30 %.

### 5.5 Correlaciones

Se calcula la matriz de correlación de Pearson entre las variables numéricas clave. Se observa una relación positiva entre precio mediano y esfuerzo, y relación inversa entre años de salario y esfuerzo, lo cual es coherente con la definición del indicador.

## 6. Fase 2 – Validación y depuración

### 6.1 Consistencia estructural

- 60 filas, 16 columnas en el cuadro maestro.
- 10 regiones con 6 observaciones cada una (2015-2020).
- Sin duplicados por clave (año, región).
- Sin valores nulos en columnas críticas del cruce.

### 6.2 Tratamiento de faltantes

Se identifican 28 valores faltantes totales, concentrados en dos variables:

- Horas de sol anomalía: 18 faltantes.
- Variación porcentual de precio: 10 faltantes.

Se opta por imputar estos faltantes a cero, justificado por la naturaleza de las variables.

### 6.3 Conjunto de predictores

Se define el conjunto de predictores para modelado:

Variables numéricas:
- Precio mediano de vivienda
- Salario joven regional
- Años de salario para comprar
- Variación porcentual de precio
- Anomalías climáticas: temperatura máxima, mínima, lluvia, horas de sol, días de heladas

Variable categórica:
- Región (codificada mediante variables ficticias con eliminación de la categoría de referencia)

Excluidas:
- Año (identificador temporal)
- Identificadores de fila y claves de cruce

## 7. Fase 3 – Modelo de clasificación y falsos positivos

### 7.1 Diseño experimental

- Variable objetivo: accesible (1) si esfuerzo ≤ 0.30, no accesible (0) en caso contrario.
- Partición estratificada en entrenamiento y prueba, con semilla fija.
- Variables categóricas convertidas a variables ficticias con eliminación de la categoría de referencia.

### 7.2 Búsqueda de hiperparámetros

Se evalúa el profundidad máxima del árbol en los valores 2, 3, 4 y 5 para controlar el equilibrio entre sesgo y varianza y evitar sobreajuste.

### 7.3 Resultados con umbral 0.30

Con el umbral de referencia del 30 %, ninguna de las 60 observaciones es clasificada como accesible. Toda la muestra pertenece a la clase no accesible.

Matriz de confusión en prueba (solo clase 0 presente):

| | Predicho 0 | Predicho 1 |
|---|---|---|
| Real 0 | 15 | 0 |
| Real 1 | 0 | 0 |

Al no existir observaciones de la clase positiva en los datos, no es posible calcular falsos positivos ni falsos negativos con esta definición.

## 8. Fase 4 – Sensibilidad de umbral

### 8.1 Análisis de sensibilidad

Se evalúa la sensibilidad de la variable objetivo ante distintos umbrales de esfuerzo:

| Umbral de esfuerzo | % observaciones accesibles |
|---|---|
| 0.30 | 0.00 % |
| 0.35 | 0.00 % |
| 0.40 | 1.67 % |
| 0.45 | 6.67 % |
| 0.50 | 26.67 % |

### 8.2 Modelo con umbral alternativo (0.50)

Seleccionando el umbral de 0.50, se obtienen ambas clases (16 observaciones accesibles, 44 no accesibles), lo que permite entrenar y evaluar un modelo de clasificación con información real.

Se entrena un árbol de decisión con profundidad máxima 3 sobre la partición estratificada. Los resultados en prueba muestran precisión global del 100 %, con 1 falso positivo y 0 falsos negativos.

La variable más importante para el modelo es años de salario para comprar, seguida por otras características económicas y climáticas con menor peso relativo.

## 9. Análisis de reducción de dimensionalidad

Se realiza un análisis de componentes principales como diagnóstico para evaluar la redundancia entre los predictores numéricos. Los resultados indican que la compresión de información mediante PCA conlleva una pérdida de desempeño en el modelo predictivo, por lo que se prefiere mantener el conjunto completo de predictores para preservar la interpretabilidad regional.
