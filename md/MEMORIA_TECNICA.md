# Memoria técnica del proyecto

**Accesibilidad de vivienda para jóvenes (18-34) en el Reino Unido**
Análisis de cuántos años de salario cuesta comprar, qué esfuerzo hipotecario exige cada
región y qué papel juega el clima en la evolución de los precios.

Este documento reúne, con su justificación, **qué datos usamos, cómo los transformamos y
por qué cada decisión estadística y de ingeniería se tomó así**. Está organizado según los
seis temas del temario para que se pueda seguir en clase tema por tema, pero todo el
contenido es del proyecto: nuestras fuentes, nuestras cifras y nuestras decisiones.

---

## Mapa: los seis temas y dónde vive cada uno en el repositorio

| Tema | Qué se trabaja | Dónde está en el repositorio |
|---|---|---|
| **1. Introducción y marcos** | KDD, CRISP-DM y por qué descartamos SEMMA; tipo de tarea de minería | `README.md`, `md/PROPUESTA_METODOLOGICA.md` |
| **2. Fuentes y tipos de datos** | Las 5 fuentes, formatos, tipos inferidos, trampas de CSV | `libros/01_Carga_4_Fuentes.ipynb`, `script/proyecto/cargar.py` |
| **3. Limpieza de datos** | Nulos MCAR/MAR/MNAR, duplicados, atípicos por IQR, winsorización | `libros/02_Limpieza_4_Datasets.ipynb`, `Diagnostico_Limpieza.ipynb`, `script/proyecto/diagnostico.py`, `script/limpieza/` |
| **4. Transformación y reducción** | Escalado, one-hot, VIF, discretización, PCA y la matriz X·y | `libros/06_Transformacion_y_Matriz.ipynb`, `script/proyecto/matriz_xy.py` |
| **5. Análisis exploratorio** | Univariado, bivariado, multivariado, figuras y hallazgos | `libros/07_EDA_Accesibilidad.ipynb`, `script/proyecto/eda.py` |
| **6. Cierre y entrega** | Checklist de 6 ítems, diccionario de datos, dataset versionado | `libros/08_Modelado_Accesibilidad.ipynb`, `md/DICCIONARIO_DATOS.md`, este documento |

> **Nota de alcance.** Los libros 03, 04 y 05 (cruces, hipoteca y clima) son material
> propio del proyecto que no corresponde a un tema suelto del temario: son el *puente*
> entre la limpieza y la transformación. El libro 08 (modelado) es el **anticipo** del
> bloque de árboles de decisión que se anuncia después del cierre de la unidad, es decir
> vamos una sesión por delante de la clase.
>
> **Rumbo.** El objetivo final del proyecto es un **modelo predictivo** del esfuerzo
> hipotecario por región y año. Las fases 1-6 (documentadas aquí) dejan los datos y el
> *benchmark* comparativo; las fases 7-10 lo convierten en predictor con **validación
> temporal** (entrenar 2015-2018, validar 2019-2020) y despliegue. El estado y el detalle
> de cada fase están en la tabla de fases del `README.md`.

---

# 1. Introducción y marcos

## 1.1 Qué problema resolvemos y de qué tipo es

**Pregunta de negocio.** ¿Cuánta vivienda puede pagar un joven (18-34) en cada región del
Reino Unido con su salario, con hipoteca o préstamo, y el clima cómo se asocia con la
evolución de los precios?

Por el tipo de salida que produce, nuestro problema es de **regresión** (predecir una
magnitud continua: el esfuerzo hipotecario) con una fuerte base **descriptiva** (el EDA
tiene tanto peso analítico como el modelo). No es clasificación, ni clustering, ni
descubrimiento de reglas de asociación.

## 1.2 Por qué KDD + CRISP-DM y por qué no SEMMA

Elegimos **combinar** KDD (Fayyad et al., 1996) y CRISP-DM (Wirth & Hipp, 2000):

| Marco | Qué aporta al proyecto |
|---|---|
| **KDD** | El detalle conceptual de las etapas analíticas: selección, preprocesamiento, transformación, minería e interpretación |
| **CRISP-DM** | El marco de negocio: entendimiento del negocio, de los datos, preparación, modelado, evaluación y despliegue |
| **SEMMA** | **Descartada**: cubre solo la parte analítica, sin el ciclo de negocio, y está atada a la herramienta SAS |

La combinación cubre el ciclo completo: usar KDD solo dejaría fuera el despliegue;
usar CRISP-DM solo perdería el detalle conceptual de las etapas teóricas. El mapeo de
cada fase del proyecto a su etapa KDD y su fase CRISP-DM está en la tabla de fases del
`README.md`, y la justificación completa en `md/PROPUESTA_METODOLOGICA.md`.

## 1.3 Los cuatro errores que este diseño evita

| Error frecuente | Cómo lo evitamos |
|---|---|
| Empezar por el modelo | El EDA (libro 07) y el checklist de preparación van **antes** de cualquier entrenamiento |
| Confundir reporte con patrón | Las correlaciones se reportan como asociación, nunca como causa (ver § 5.4) |
| Creer que más datos = mejores resultados | Agregamos 11,4 M de transacciones a 60 filas región-año **a propósito**: el grano correcto es la pregunta correcta |
| No versionar nada | Cada decisión de limpieza queda en `md/BITACORA_DECISIONES.md` con su cantidad, método y razón |

## 1.4 El herramental

Python 3.14 · pandas 3.0 · numpy 2.5 · scipy · scikit-learn · matplotlib · Jupyter.
Los requisitos están en `script/requirements/`. No usamos Weka ni Power BI: el análisis
es programático y reproducible.

---

# 2. Fuentes y tipos de datos

## 2.1 Las cinco fuentes y por qué entra cada una

| # | Fuente | Archivo | Filas | Columnas | Tamaño | Periodo | Por qué entra |
|---|---|---|---|---|---|---|---|
| **F1** | Transacciones de vivienda (Land Registry, geocodificada) | `db/uk_property_price.csv` | **11.369.413** | 20 | **2.220 MB** | 2015-2026 | Da el **precio real de venta** por locality: es el numerador de la asequibilidad |
| **F2** | Catálogo de hipotecas (snapshot) | `data/crudo/UK_Mortgage_Rate.csv` | **7.796** | 14 | 0,94 MB | 2022 | Da la **tasa y las comisiones** con las que se amortiza el préstamo |
| **F3** | Meteorología (Met Office) | `data/crudo/MET_Office_Weather_Data.csv` | **37.049** | 8 | 1,47 MB | 1853-2020 | Aporta el **contexto climático** y su posible asociación con los precios |
| **F4** | Precios y salario deflactados | `data/crudo/Average_UK_houseprices_and_salary.csv` | **46** | 4 | <0,01 MB | 1975-2020 | Da la **serie histórica** en moneda constante y la mediana salarial |
| **F4b** | Salario por edad y género | `data/crudo/Income_by_age_and_gender.csv` | **12** | 3 | <0,01 MB | 2021 | Permite **ajustar el salario a la franja joven** (18-34) |

**Por qué 5 fuentes y no una.** La pregunta de accesibilidad es una **razón entre tres
magnitudes de naturaleza distinta**: precio, salario y costo del crédito. Ninguna fuente
del Reino Unido publica las tres juntas. Además el clima entra como variable
explicativa de segundo orden, que es lo que permite contrastar si el problema es
salarial, financiero o geométrico.

**Por qué F3 entra siendo "marginal".** No hay evidencia de que el clima explique el
precio de la vivienda (ver § 5.5). Entra precisamente para **poder descartarlo con
números** en lugar de suponerlo.

## 2.2 Dónde viven los datos y qué formato eligió cada uno

| Fuente | Formato en el repo | Formato de trabajo | Por qué |
|---|---|---|---|
| F1 | CSV de 2,2 GB (local) | **parquet** de 434 MB | El CSV de 2,2 GB no entra en memoria cómodamente; parquet es columnar y comprimido (5× más chico) |
| F2, F3, F4, F4b | CSV (< 1,5 MB) | CSV → parquet intermedio | Son pequeños: se versionan completos |
| Resultado del cruce | `data/integrado/*.parquet` + CSV v1 | parquet + CSV | El parquet conserva tipos; el CSV es la salida legible y versionada |

**Muestreo en el repositorio.** El CSV de F1 y el parquet limpio **no** se suben (son
locales: 2,2 GB y 434 MB). En su lugar se versiona una **muestra con sufijo `_muestra`**
(primeras 2.000 filas de cada uno), más los CSV completos de las fuentes pequeñas. Los
datos completos se descargan de los enlaces de Kaggle del `README.md`.

## 2.3 Las trampas del CSV que encontramos

| Trampa | Dónde apareció | Qué hicimos |
|---|---|---|
| **Símbolos de "sin dato"** | F3 (Met Office) usa `---`, `*`, `#` para los faltantes | `na_values` al leer + `to_numeric(errors="coerce")`: los convertimos en `NaN`, nunca en número |
| **Fechas en texto** | F3 (`anio`, `month`) | Se parsean a tipo fecha y se verifica que no haya fechas futuras ni incoherencia con la fecha de transferencia en F1 |
| **Columnas sin nombre** | F4 trae una columna `Unnamed: 3` residual | Se descarta en la carga |
| **Normalización de texto** | F1 (`town_city`, `county`) están 100 % en mayúsculas | Se normaliza y se comprueba **colisiones**: 0 colisiones entre valores originales y normalizados |

**El tipo que pandas infiere sin decirlo.** Al cargar un CSV, pandas decide cada tipo
por sí solo y avisa poco: una columna de precios puede caer como texto si el archivo trae
un símbolo raro en la primera fila. Por eso el `cargar.py` del proyecto hace una **lectura
defensiva**: cuenta filas y columnas, y después imprime el tipo de cada columna para que
la decisión sea visible y no implícita.

## 2.4 El grano: por qué el cruce es región-año

Cada fuente tiene su grano: F1 es transacción, F3 es estación-mes, F4 es año-nacional.
El **cuadro maestro** las lleva todas a un grano común de **región × año**, que es el
nivel al que la pregunta tiene respuesta:

- Un joven no compra "en Aberdeen"; compra en una región, con un salario regional y una
  tasa nacional.
- El clima de una estación se parece al de su región (microclimas contiguos), así que
  promediar por región es lo correcto.

**Resultado: 60 filas = 10 regiones × 6 años (2015-2020).**

## 2.5 La ventana 2015-2020 y por qué no más

Las cuatro fuentes tienen periodos distintos. La **única ventana donde las cuatro
existen a la vez** es 2015-2020:

| Fuente | Empieza | Termina | ¿Cubre 2015-2020? |
|---|---|---|---|
| F1 | 2015 | 2026 | Sí |
| F3 | 1853 | 2020 | Sí |
| F4 | 1975 | 2020 | Sí |
| F4b | 2021 | 2021 | No (fuera: se usa como factor único) |

Acortar la ventana es una decisión consciente: alargar obligaría a abandonar F1
(que empieza en 2015) y perderíamos el precio real de venta, que es la variable central
del proyecto.

---

# 3. Limpieza de datos

## 3.1 Nulos: no todos son iguales (MCAR / MAR / MNAR)

No existe una única decisión para los faltantes: depende de **por qué** falta el dato.
Clasificamos cada columna por su mecanismo:

| Fuente / variable | Mecanismo | Evidencia | Decisión |
|---|---|---|---|
| F2 · `tasa_reversion` | **MAR** | La tasa de faltantes cambia con el tipo de producto: *discounted* 6,1 %, *fixed* 0,8 %, *tracker* 26,0 %, *variable* 39,5 % | Se conserva el nulo: el tipo de producto explica el faltante |
| F3 · `tmax`/`tmin`/`af`/`lluvia` | **MCAR** instrumental | 1.467 de 1.599 filas con `tmax` nulo tienen también `tmin` nulo: fallan juntas por el reporte, no por el clima | Se conservan; los agregados anuales toleran meses faltantes |
| F3 · `sol` | **MNAR** estructural | 9.483 nulos, de los cuales 4.138 anteriores a 1931: el Met Office no publicó esa variable | Se conserva: son años sin registro, no sin sol |
| F4 · `salario_mediana_real` | **MNAR** estructural | 24 nulos; la serie oficial solo existe desde 1999 | Se conserva el histórico |
| F3 · `anio`/`month` | **MNAR** de registro | 20 filas sin fecha: no son mediciones | Se **eliminan**: sin fecha no hay observación |

**Las tres categorías, en una frase.**

- **MCAR** (al azar): el faltante no depende de nada medible → se puede tolerar o imputar.
- **MAR** (aleatorio condicional): depende de **otra variable observada** → se puede
  modelar o imputar de forma informada.
- **MNAR** (no aleatorio): depende del **propio valor ausente** → imputar inventa
  información; lo correcto es declarar el límite.

## 3.2 Duplicados: el identificador natural manda

| Fuente | Clave natural | Duplicados |
|---|---|---|
| F2 | (`sku`, `tid`) | 0 |
| F3 | (`anio`, `mes`, `estación`) | 10 |
| F4b | (`grupo_edad`, `género`) | 0 |
| F1 | `transaction_id` | 0 |

Usamos la **clave natural** y no `duplicated()` global: dos filas con la misma clave
natural son la misma observación aunque difieran en un campo secondary. En F1 la clave
`transaction_id` es tan fuerte que da 0 duplicados sobre 11,4 M de filas.

## 3.3 Atípicos: IQR contra z-score, con la evidencia de por qué

Este es uno de los puntos donde **no nos quedamos con la teoría: medimos las dos opciones
sobre nuestros datos.**

| Variable | Q1 | Mediana | Q3 | IQR | Cerca superior (Q3+1,5·IQR) | **IQR detecta** | **z-score (\|z\|>3) detecta** |
|---|---|---|---|---|---|---|---|
| F2 · `tasa_inicial_pct` | 5,62 | 6,10 | 6,62 | 1,00 | **8,13** | **762 (9,77 %)** | **0 (0 %)** |
| F1 · `price` | 158.000 | 250.000 | 385.000 | 227.000 | **725.500** | **718.461 (6,32 %)** | 0 (el máximo de 900 M está a z = 591) |

**Por qué gana la regla de Tukey.** El z-score mide cuánto se aparta un valor **de la
media, en unidades de desviación estándar**, y eso solo significa "atípico" si la
distribución es normal. Los precios y las tasas son asimétricos, así que bajo ese supuesto
el z-score **no ve la cola real**: detecta 0 atípicos en la tasa, donde el IQR detecta el
9,77 %. La regla de Tukey (cercas en `Q1 − 1,5·IQR` y `Q3 + 1,5·IQR`) no asume nada
sobre la forma de la distribución: solo mira los cuartiles, que son robustos.

**La asimetría manda en la decisión.** En F1 el `price` tiene asimetría fuerte y su
media (364.248) está contaminada por el máximo; por eso el proyecto usa la **mediana**
como referencia central y no la media.

## 3.4 Winsorización: conservar la fila, marcar el valor

Decidimos **recortar, no eliminar**: en vez de borrar las 718.461 filas de precio
atípico, las conservamos y les ponemos un tope, guardando además una bandera:

| | Sin tratar | Con winsorización |
|---|---|---|
| `price` máximo | 900.000.000 | **725.500** |
| Filas eliminadas | — | **0** |
| Tasa de anguish media | 6,113 % | **6,043 %** |

**Por qué recortar y no borrar.** El precio extremo en una venta es información ("una
vivienda de lujo en Londres"), no un error de dedo; y el indicador que construimos
(años de salario) necesita esas regiones. Recortar deja la fila y su valor alto, que es
lo que el negocio quiere ver; borrar la perdería.

**Y por qué tampoco dejarlo sin tocar.** El `price` entra en un modelo predictivo: un
máximo de 900 millones de libras domina la media de la región y desbalancea el resto.

## 3.5 La bitácora: limpiar sin registro es limpiar a ciegas

Cada decisión de limpieza se escribe en `md/BITACORA_DECISIONES.md` con cinco campos:
fuente, problema, **cantidad**, método y razón. Es lo que permite que otro lector (o
nosotros en diciembre) entienda qué se hizo y por qué, sin tener que releer el código.

---

# 4. Transformación y reducción

## 4.1 Escalado: cuatro opciones comparadas con un experimento

Elegimos **`StandardScaler`** (estandarización z), pero no por moda: lo comparamos con
los otros tres midiendo **cuánto se distorsionan ante un valor extremo**.

**Experimento de estrés.** Inyectamos un precio multiplicado por 100 en una fila y medimos
cuánto se comprime el rango de las 59 filas normales:

| Escalador | Compresión del rango normal | Veredicto |
|---|---|---|
| `MinMaxScaler` | **97,7 %** | El extremo aplasta todo lo demás |
| `StandardScaler` | 95,3 % | Comprime casi todo (el desvío también se infla) |
| **`RobustScaler`** (mediana/IQR) | **4,1 %** | El más resistente |
| **`StandardScaler`** (elegido) | — | El más simple y el mejor en predicción |

Elegimos `StandardScaler` por el resultado en el modelo (tabla siguiente), dejando
`RobustScaler` documentado como plan B si aparecieran colas extremas. `MinMax` se
descartó por ser el más frágil ante un extremo.

**Escalar, ¿para qué?** Porque si `precio_mediana` (~300.000 GBP) comparte matriz con
`tmax_anom` (±2 grados), el precio domina la distancia y KNN acaba midiendo "distancia
al precio" en lugar de "distancia al perfil". `StandardScaler` deja ambas en la misma
escala (media 0, desv 1).

## 4.2 Codificación categórica: one-hot contra ordinal

La región es **nominal**: no hay orden entre "London" y "Wales". Encodificarla como
número (0,1,2...) le inventa un orden que el algoritmo usaría.

**Experimento 1: el mismo dato, signo opuesto.** Codificamos alfabéticamente y
obtuvimos correlación **r = −0,315** con el esfuerzo. Invertimos el orden y el mismo
conjunto de datos dio **r = +0,315**. Mismo dato, correlación espuria de signo opuesto:
prueba de que la correlación ordinal no mide nada real aquí.

**Experimento 2: R² en validación cruzada.** Con la región como único rasgo:

| Codificación | R² (CV-5) | RMSE | Lectura |
|---|---|---|---|
| **One-hot** | **0,891** | 0,056 | Aprende la estructura regional |
| Ordinal | **−0,051** | 0,176 | **Peor que predecir la media** |

El ordinal es 3,1× peor en error y su R² negativo significa que no generaliza. One-hot
gana de forma rotunda.

## 4.3 La trampa de variables mudas y el factor VIF

Con one-hot completo salen **10** columnas dummy (una por región), pero **suman siempre
1** en cada fila: están linealmente dependientes. Eso se llama *trampa de variables
mudas* (*dummy trap*).

El **VIF** (factor de inflación de la varianza, `VIF = 1/(1−R²)`) lo detecta:

| Configuración | VIF |
|---|---|
| 10 dummies | **∞** (dependencia lineal perfecta) |
| 9 dummies (`drop="first"`) | **1,8** (multicolinealidad baja) |

**Por qué usamos 9.** Con `drop="first"` la región de referencia (East Midlands) queda
implícita y las demás se comparan contra ella. Además de evitar la redundancia,
**vuelve interpretables los coeficientes**: antes todos los coeficientes de región salían
en 0,0000-0,0001 (el solucionador reparte el peso de forma arbitraria entre categorías
equivalentes); ahora London es **+0,0002** y significa "cuánto esfuerzo añade London
respecto a la referencia".

## 4.4 Discretización: por qué no

Probamos `KBinsDiscretizer` con tres estrategias (uniform, quantile, kmeans) y medimos
cuánta información conserva cada una:

| Métrica | Mejor estrategia | Resultado |
|---|---|---|
| Orden conservado (ρ² de Spearman) | quantile | 93,8 % |
| **Información retenida** (información mutua) | quantile | **64,3 %** (precio) / 46,5 % (años de salario) |

Retiene como máximo el 64 % de la información del rasgo continuo. **No discretizamos**:
mutilaríamos justo el rasgo continuo (precio) que el esfuerzo necesita. El objetivo es
regresión sobre magnitudes, no clasificación sobre tramos.

## 4.5 PCA: útil como diagnóstico, no como compresión

| Condición | PC1 (varianza 1ª componente) | Componentes para 80 % |
|---|---|---|
| Sin escalar | 100 % | 1 |
| MinMax | 46,5 % | 3 |
| Standard | 35,7 % | 4 |
| Robust | 42,8 % | 3 |

Sin escalar, una sola componente explica el 100 %: eso es la **"tiranía de las
unidades"** (el precio, de mayor magnitud, se lleva toda la varianza). Estandarizar
"libera" las otras direcciones (PC1 baja a 35,7 %).

Pero al usarlo en el modelo, **comprimir cuesta**: Standard+PCA(80 %) rinde R² 0,937
contra 1,000 del pipeline completo. **No usamos PCA en la matriz final**; lo dejamos
como diagnóstico, que es donde su lectura es útil.

## 4.6 La matriz X·y final

| Parte | Columnas | Detalle |
|---|---|---|
| Identificadores | 2 | `region`, `anio` (no entran como rasgos) |
| Rasgos numéricos | 9 | `precio_mediana`, `salario_joven_region`, 5 anomalías climáticas, `precio_var_pct`, `anos_salario` — estandarizados (9 columnas `z_`) |
| Región (one-hot `drop_first`) | 9 | 10 regiones → 9 columnas |
| Objetivo `y` | 1 | `esfuerzo` (cuota ÷ salario) |
| **Total** | **21 (60 × 21)** | 18 columnas de X + 2 identificadores + 1 objetivo |

**Imputación: solo dos columnas, y con constante.** `sol_anom` (18 nulos) y
`precio_var_pct` (10 nulos) se completan con **0**, no con la media: un 0 en "anomalía"
significa "no hubo anomalía", que es un hecho del dato, mientras que imputar la media
inventaría un valor climático que nunca se midió. Es la única imputación del pipeline.

---

# 5. Análisis exploratorio

## 5.1 Qué se mide y con qué nombre

El EDA sigue tres familias, cada una con su fórmula y su pregunta:

| Familia | Medida | Fórmula | Pregunta |
|---|---|---|---|
| Tendencia central | Media | x̄ = (1/n)·Σxᵢ | ¿Dónde está el centro? |
| | Mediana | percentil 50 | ¿Centro por la mitad de los casos? |
| Dispersión | Desviación **muestral** | s = √(Σ(xᵢ−x̄)²/(n−1)) | ¿Cuánto se dispersa? |
| | **IQR** | Q3 − Q1 | ¿Amplitud del 50 % central? |
| Forma | Asimetría **Fisher-Pearson** | g₁ = (n/((n−1)(n−2)))·Σ(((x−x̄)/s)³) | ¿Sesgo? |
| | **Curtosis excedente** | g₂ = (1/n)·Σ(((x−x̄)/s)⁴) − 3 | ¿Colas pesadas? |

La desviación es **muestral** (n−1) porque trabajamos con una muestra y queremos
generalizar, no describir solo estos 60 puntos.

## 5.2 Por qué la mediana y no solo la media

Cuando la distribución está sesgada, la media la arrastra la cola. El **cuarteto de
Anscombe** (F. Anscombe, 1973) es el caso canónico: cuatro conjuntos con medias idénticas
y estructuras opuestas, que solo el gráfico delata. En el maestro, `precio_mediana` tiene
asimetría **+1,27** y `salario_joven_region` **+1,53**: por eso el **ranking regional se
ordena por medianas**, para que un solo año excepcional no reordene el ranking.

## 5.3 Outliers del esfuerzo: 0 global, 1 por el matiz del n

Aplicamos las cercas de Tukey al esfuerzo (`y`):

- **IQR = 0,284**; cerca inferior **0,053**, cerca superior **1,189**.
- **Globalmente: 0 atípicos de 60 (0,0 %).**

Interpretación: la dispersión del esfuerzo **no está anomalizada**; el problema es de
**nivel**, no de forma. El mínimo (0,416) ya supera el umbral de accesibilidad (0,30):
ninguna región es un atípico, sencillamente **nadie puede comprar**.

**El matiz que enseña el método.** Con solo **6 observaciones por región**, el IQR
regional es mucho más estrecho que el global (London: IQR 0,053), y su cerca marca **1
atípico en London**. Eso no es un error: es que **el umbral 1,5·IQR es sensible al
tamaño muestral**. Por eso reportamos la cuantificación **global y por región**, no una
sola cifra.

## 5.4 Correlación no es causalidad

El coeficiente de **Pearson** `r = Σ(xᵢ−x̄)(yᵢ−ȳ) / ((n−1)·sₓ·sᵧ)` mide **asociación
lineal** entre variables observadas. En el maestro, precio y esfuerzo tienen **r = +0,983**:
asociación casi perfecta en estos 60 puntos. Pero eso **no** significa que "el precio
produce el esfuerzo": ambos comparten denominador (el salario) y la cuota es una función
del precio. Para sostener causalidad haría falta un diseño que aislara el efecto.

Por eso our hipótesis se escribe con cuidado: el clima "**se asocia débilmente**" con los
precios, no "**causa**" una menor subida.

## 5.5 La hipótesis H3 (clima): el clima es contexto, no motor

Con la **variación porcentual** del precio (no su nivel: el nivel daría correlaciones
artificialmente altas por latitud y altitud) contra las cinco anomalías climáticas:

| Anomalía | Pearson r | Lectura |
|---|---|---|
| `lluvia_anom` | **−0,241** | Débil negativa |
| `tmax_anom` | **−0,388** | Moderada negativa |
| `tmin_anom` | **−0,399** | Moderada negativa |
| `af_anom` | **−0,145** | Débil negativa |
| `sol_anom` | +0,238 | Débil positiva |

**Qué significa en varianza.** Un r de 0,399 explica r² = 0,159, o sea el **16 %** de la
variación del precio; un r de 0,145 explica el **2 %**. Es decir, **entre el 84 % y el
98 %** de lo que mueve el precio año a año **no** está en las variables climáticas.

**Conclusión de H3:** el clima **no es** el motor de la accesibilidad. La explicación la
da el binomio **salario joven – tasa – precio regional**. El clima es **contexto**, y en
el modelo su importancia de permutación queda en **≈0**.

## 5.6 El hallazgo de negocio

El **esfuerzo** mediano va de **0,437 (North East)** a **1,018 (London)**: una brecha de
**2,3×**. Y **ninguna** de las 10 regiones cumple el umbral del 30 % del salario. Con
4 de 60 filas (6,7 %) por encima de 1,0, donde la cuota se come el sueldo entero.

---

# 6. Cierre, verificación y entrega

## 6.1 El checklist de preparación para modelar (6 ítems, ejecutado)

Antes de entrenar, el proyecto se somete a una verificación de seis ítems, ejecutada en
el libro 08. Los seis pasan:

| # | Ítem | Verificación | Resultado |
|---|---|---|---|
| 1 | Tipos correctos | dtype de las 17 columnas | 14 float64 + 2 int64 + 1 texto, 0 `object` |
| 2 | Nulos justificados | Conteo tras imputar | 0 (imputados solo los 28 documentados) |
| 3 | Sin duplicados | `duplicated()` y clave (región, año) | 0 y 0 |
| 4 | Categorías consistentes | Niveles de `region` | 10 regiones, sin variantes |
| 5 | Escala y codificación | Media≈0, desv≈1, one-hot con `drop_first` | \|media\|max 1,1e-15; 9 dummies, 1 activa por fila |
| 6 | Diccionario de datos | Columnas documentadas | 17/17 |

## 6.2 Validación del cruce (aserciones)

Además del checklist, `script/proyecto/validacion.py` verifica por `assert` que: hay 60
filas, la clave (año, región) es única, no hay nulos en precio ni salario, ambos son
positivos, y la tasa winsorizada respeta su cerca. Si alguna falla, **el pipeline se
detiene** con un mensaje claro, en vez de producir un resultado silenciosamente erróneo.

## 6.3 Qué entregamos

| Entregable | Dónde |
|---|---|
| Cuadernos ejecutables (01-08) | `libros/` |
| Dataset limpio versionado | `data/integrado/accesibilidad_vivienda_joven_uk_v1.csv` (60 × 16) |
| Diccionario de datos | `md/DICCIONARIO_DATOS.md` |
| Bitácora de decisiones | `md/BITACORA_DECISIONES.md` |
| Memoria técnica (este documento) | `md/MEMORIA_TECNICA.md` |

> El CSV versionado conserva 28 nulos documentados (18 `sol_anom` + 10
> `precio_var_pct`). **Es intencional**: el maestro es fiel al dato (la imputación a 0
> ocurre solo al construir X), y por eso el diccionario explica cada nulo. Un dataset
> "limpio" que rellena esos 28 valores escondería una decisión de negocio.

## 6.4 El modelo: qué ganamos al corregir el protocolo

| Modelo | R² (CV-5 barajado) | Qué aporta |
|---|---|---|
| Regresión lineal | **1,000** | Línea base interpretable |
| Random forest | 0,980 | No lineal robusto |
| Árbol de decisión | 0,963 | Reglas interpretables |
| KNN tuneado | 0,844 | No paramétrico |

**El R² de la regresión es 1,000 y eso no es una fuga de datos**: es determinismo
contable. El esfuerzo es `PMT / salario`, y la PMT es una función (casi) lineal del
precio con tasa y plazo fijos; además `anos_salario = precio / salario` comparte
cociente con el objetivo. El modelo recupera la fórmula.

**La corrección que más cambió las cifras** no fue de modelo, sino de **protocolo**: el
CV ahora **baraja** los pliegues (los datos están ordenados por región-año; sin barajar
cada pliegue evaluaba sobre una región distinta). Solo eso subió el KNN de **0,638 a
0,844**.

**Qué variable explica el esfuerzo (importancia de permutación del bosque):**

| Rasgo | Importancia | Coeficiente \|β\| de la RL |
|---|---|---|
| `anos_salario` | **0,580** | **0,181** |
| `precio_mediana` | 0,314 | 0,000 |
| `salario_joven_region` | 0,006 | 0,000 |
| `region` | 0,002 | (implícita) |
| Las 5 anomalías climáticas | ≈ 0 | ≈ 0 |

Dos lecturas: (1) manda el cociente precio/salario, no un factor aislado — coherente
con el determinismo contable; (2) **el clima queda en ≈0 una vez controlados precio y
salario**, que es la confirmación estadística de H3 del § 5.5.

---

# Anexo A · Las 14 comparaciones de alternativas

Cada decisión del proyecto se tomó **midiendo**, no suponiendo. Este es el resumen:

| # | Decisión | Alternativas comparadas | Evidencia | Ganador |
|---|---|---|---|---|
| 1 | Detección de atípicos | IQR (1,5) vs z-score (\|z\|>3) | 762 (9,77 %) vs 0 | **IQR** (no exige normalidad) |
| 2 | Referencia central | Media vs mediana | media de `price` contaminada (z=591 del máx); Anscombe | **Mediana** |
| 3 | Escalado | MinMax / Standard / Robust / sin escalar | compresión 97,7/95,3/4,1 %; R² KNN 0,904/0,861/0,764/0,976 | **Standard** |
| 4 | Codificación | One-hot vs ordinal | R² 0,891 vs −0,051; r espuria ±0,315 | **One-hot** |
| 5 | Multicolinealidad | 10 dummies vs 9 (`drop_first`) | VIF ∞ vs 1,8 | **9 dummies** |
| 6 | Discretización | uniform / quantile / kmeans / ninguna | info retenida máx 64,3 % | **Ninguna** |
| 7 | Reducción | PCA(80 %) vs 9 rasgos | R² 0,937 vs 1,000 | **Sin PCA** |
| 8 | Medida de correlación | Pearson vs Spearman | se busca linealidad; r² interpretable | **Pearson** |
| 9 | Anomalía climática | Simple (diferencia) vs z-score | el z no es comparable entre mm y °C | **Simple** |
| 10 | Respuesta del clima | Variación % vs nivel del precio | el nivel es artefacto de latitud | **Variación** |
| 11 | Modelo | RL / bosque / árbol / KNN tuneado | 1,000 / 0,980 / 0,963 / 0,844 | **RL** (parsimonia) |
| 12 | Imputación | Media vs constante 0 | imputar la media inventa dato climático | **Constante 0** (solo 2 col.) |
| 13 | Formato del objetivo | Continuo vs 5 tramos | el modelo usa el valor continuo | **Continuo** |
| 14 | Validación | `cv=5` vs KFold barajado | KNN 0,638 vs 0,844 | **Barajado** |

---

# Anexo B · Fórmulas del proyecto

Todas las medidas y criterios matemáticos que usa el pipeline, con su fórmula.

| Fórmula | Dónde se usa |
|---|---|
| **Cuota PMT** (anualidad): `C = P·r / (1 − (1+r)^(−n))` | Esfuerzo y amortización |
| **Cercas de Tukey**: `LI = Q1 − 1,5·IQR`, `LS = Q3 + 1,5·IQR` | Atípicos (F1, F2, esfuerzo) |
| **Winsorización**: `x' = min(x, LS)` | `price` y `tasa_inicial_pct` |
| **Z-score** (escala): `z = (x − μ) / σ` | `StandardScaler` |
| **Min-max**: `(x − mín) / (máx − mín)` | Escalador comparado |
| **Escalado robusto**: `(x − mediana) / IQR` | Escalador comparado |
| **Fisher-Pearson (asimetría)**: `g₁ = (n/((n−1)(n−2)))·Σ(((x−x̄)/s)³)` | EDA |
| **Curtosis excedente**: `g₂ = (1/n)·Σ(((x−x̄)/s)⁴) − 3` | EDA |
| **Pearson**: `r = Σ(xᵢ−x̄)(yᵢ−ȳ) / ((n−1)·sₓ·sᵧ)` | Correlaciones |
| **Mínimos cuadrados**: `ŷ = β₀ + β₁·x`, `β₁ = r·sᵧ/sₓ` | Recta del EDA, RL |
| **VIF**: `1 / (1 − R²ⱼ)` | Multicolinealidad |
| **Variación porcentual**: `(x_t − x_{t−1}) / x_{t−1}·100` | `precio_var_pct` |
| **Deflactor**: `nominal(año) / 1,137` | Precios a GBP 2020 |
| **Anomalía climática**: `valor(año) − normal(1931-2020,estación)` | `*_anom` |
| **Años de salario**: `precio_mediana / salario_joven_region` | `anos_salario` |
| **Esfuerzo**: `PMT / (salario_joven/12)` | Objetivo `y` |
| **Sturges (bins)**: `k = 1 + log₂(n)` | Histogramas (n=60 → 6,9) |
| **R²** (validación): media de los 5 pliegues | Modelado |

---

# Anexo C · Limitaciones declaradas

Con lo hecho hasta aquí, estas son las limitaciones **conocidas y documentadas**:

1. **Fuga conceptual en el objetivo.** El esfuerzo comparte fórmula con
   `anos_salario` y `precio_mediana`, así que X contiene transformaciones algebraicas de
   y. Por eso el R² de la RL es 1,000: el problema es casi determinista. Es una
   limitación del diseño del indicador, no del modelo.
2. **Muestra pequeña.** 60 filas (10 regiones × 6 años). Todo es descriptivo o
   comparativo empírico; **no** hay estadística inferencial (ni intervalos de confianza,
   ni pruebas de hipótesis de negocio más allá del Shapiro-Wilk diagnóstico). Con
   n=50 pares, un |r|=0,40 tiene margen de error amplio.
3. **Serie temporal corta (en vía de resolverse).** 6 años por región no alcanzan para
   un ARIMA/SARIMA confiable, por eso la fase 6 modeló en **tabular** con CV barajado.
   La fase 7 (en curso) sustituye ese CV por **validación temporal** y añade predictores
   de rezago y de tendencia; la fase 8 decide el modelo final con métricas fuera de
   muestra (2015-2018 entrena, 2019-2020 valida).
4. **Comisiones aproximadas.** Se suman fuera de la fórmula PMT (prorrateo lineal), no
   como recargo compuesto. Diferencia pequeña (GBP 3,72/mes) pero es una aproximación.
5. **Cobertura geográfica.** Solo las 10 regiones de Inglaterra y Gales: F1 no cubre
   Escocia ni Irlanda del Norte. El total nacional sí incluye las 36 estaciones.
6. **Tasa única.** El modelo usa la tasa media winsorizada de toda la oferta 2022; el
   desglose por tipo de producto existe en el libro 04 como diagnóstico, no como
   rasgo.
7. **Salario joven estimado.** El factor 0,805 viene de bandas 18-21/22-29/30-39
   (la banda 30-39 cubre 30-34 por decimales), y los factores regionales son de
   referencia ASHE 2021. Son una **aproximación**, no una estadística por ciudad.

---

*Documento generado como parte del proyecto. Las cifras provienen de la ejecución real
del pipeline (`python main_proyecto.py`, `matriz_xy.py`, `eda.py`, `modelado.py`) con
semilla 42. Nada de este proyecto se publica en GitHub: es un avance académico local.*