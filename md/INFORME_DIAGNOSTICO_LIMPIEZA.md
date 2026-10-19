# Informe de Diagnóstico y Limpieza · UK Property Price Paid + geo

Foco: `price`, `latitude`, `longitude`, `town_city`, `county`, `country`

## 1. Origen

| Archivo | Rol |
|---|---|
| `db/uk_property_price.csv` (2,22 GB) | Fuente única, intocable · **solo local** (no se sube al repo) |
| `libros/Lectura.ipynb` | Lectura de la base |
| `libros/Diagnostico_Limpieza.ipynb` | Diagnóstico ejecutable (recalcula todo en vivo) |
| `main_limpieza.py` | **Archivo principal**: un solo `.py` que ejecuta todo y muestra las 8 partes (rutas, reglas, IQR, MCAR, validación, bitácora). Uso: `python main_limpieza.py` |
| `script/limpieza/` (paquete) | Pipeline canónico (mismas reglas), equivalente a `python -m script.limpieza` |
| `db/` | Datos de entrada (muestra de 2.000 filas de cada CSV: `uk_property_price_muestra.csv`, `uk_property_muestra.csv`, `uk_property_semiorganizado_muestra.csv`); los completos quedan locales |
| `data/limpia/` | Salida limpia: `uk_property_price_limpio_muestra.csv` (**100.000 filas**) y `uk_property_price_limpio_muestra.parquet` (2.000 filas) |

> El dataset original completo (11,4 M de filas, 2,22 GB) se descarga de Kaggle:
> https://www.kaggle.com/datasets/mansiaggarwal88/uk-property-sale-prices-20152026-geo-enriched?select=uk_property_price_paid_geo.csv
> La carpeta `db/` es **solo local** y no se sube al repositorio (límite de tamaño del
> repositorio): en las carpetas `db/` y `data/limpia/` el repo trae solo una muestra
> (sufijo `_muestra`): 2.000 filas las de `db/`, y **100.000 filas** la salida limpia en
> `.csv` (su `.parquet`, 2.000). La salida completa se regenera
> con `python main_limpieza.py`.
| `data/limpia/uk_property_price_limpio.parquet` (434 MB) | Dataset limpio generado |

## 1.1 Ejecución

Todo corre desde la raíz del proyecto con un solo comando:

```
python main_limpieza.py
```

(equivalente a `python -m script.limpieza`). Cuadernos: abrirlos con `jupyter notebook`
o ejecutarlos desde consola con
`python -m jupyter nbconvert --to notebook --execute --inplace "libros/Diagnostico_Limpieza.ipynb" --ExecutePreprocessor.timeout=900`.
La consola muestra la corrida en **8 partes**:

1. Carga del origen (archivo, tamaño, tiempo).
2. Duplicados: exactos y por `transaction_id`.
3. Categóricas: cardinalidad por columna y países.
4. Faltantes: simultaneidad, medianas con/sin geo y conclusión MCAR.
5. Outliers: regla IQR paso a paso y por qué se descarta el z-score.
6. Validación: 8 criterios real vs esperado.
7. Exportación del parquet limpio.
8. Bitácora de decisiones.

Los cuadernos ejecutan el **mismo paquete** (`script/limpieza`) y añaden las gráficas
de exposición: 8 figuras en `Diagnostico_Limpieza.ipynb` (tasa de nulos geo por año,
top de `town_city`/`county`, país, cuartiles Q1–Q4 en barras, histograma con los
cuartiles Q1/Q2/Q3 y el límite IQR, boxplot original vs winsorizado, volumen y mediana
de `price` por año) y 1 en `Lectura.ipynb` (volumen transaccional por año). La parte 5
del pipeline además **guarda** dos gráficas en `md/figuras/`: `cuartiles_price.png`
(barras Q1–Q4) y `boxplot_price.png` (original vs winsorizado).

## 2. Dimensiones y kpi

- Filas: **11.369.413** × 20 columnas.
- Carga con dtypes optimizados: ~85–96 s.
- PK `transaction_id` (GUID inalterado).

## 3. Problemas encontrados y tratamiento

| # | Columna | Problema | Cantidad | % | Tratamiento |
|---|---|---|---|---|---|
| 1 | `price` | Atípicos (IQR): Q1=158.000 · Q2=250.000 · Q3=385.000 · IQR=227.000. Límite inferior −182.500, superior **725.500**. Fuera: **718.461** (6,32 %), todos por arriba (máx 900.000.000) | 718.461 | 6,32 % | **Winsorizar (clip)** a 725.500 + bandera `es_atipico_precio` |
| 2 | `latitude`/`longitude`/`region`/`country`/`admin_district` | Nulos geo **simultáneos** (MCAR de geocodificación). 0 incoherencias lat↔lon | 22.913 | 0,20 % | Bandera `es_sin_geo`; NO se imputan coordenadas |
| 3 | `transaction_id` | Duplicados exactos y por llave | 0 | 0 % | Solo verificación |
| 4 | `town_city` (1.153) · `county` (117) · `country` (2) | Colisiones al normalizar | 0 | 0 % | Conservar (ya normalizadas) |
| 5 | `year`/`month`/`quarter` | Inconsistencia vs `date_of_transfer` | 0 | 0 % | Conservar |
| 6 | `date_of_transfer` | Fechas 2015-01-01 → 2026-06-30; futuras | 0 | 0 % | Conservar |

## 4. Dataset limpio (salida)

- 22 columnas = 20 originales + `es_sin_geo` + `es_atipico_precio`.
- **Ninguna fila se elimina**: los problemas se resuelven con banderas (integridad estadística).
- `price` queda preparado como **variable objetivo** con cola recortada al límite IQR.
- Verificación post-limpieza (asserts en vivo): duplicados 0, nulos en críticas 0, fechas futuras 0, `es_sin_geo`=22.913, `es_atipico_precio`=718.461, `price` ∈ [100, 725.500]. **TODO en verde.**

## 5. Clasificación de faltantes: MCAR / MAR / MNAR

Solo la geografía tiene nulos (lat/lon/region/country/admin_district, simultáneos).

| Tipo | Definición | En nuestro dataset | Acción |
|---|---|---|---|
| **MCAR** | La ausencia no depende de nada medible | Faltante geo: falló la geocodificación; mediana de `price` casi idéntica con geo (250.000) y sin geo (278.000), y la tasa no crece por año | Bandera `es_sin_geo` (método “constante/bandera”) |
| MAR | Depende de otra variable observada | No aplica (no hay otro nulo) | — |
| MNAR | Depende del propio valor ausente | No aplica | — |

**Por qué no `fillna` ni `dropna`:** inventar coordenadas falsearía un clustering espacial
futuro; y aunque el 0,20 % permite `dropna` (regla < 5 % + MCAR), el `price` de esas filas
sigue siendo válido, así que se conservan marcadas.

## 6. Outliers: 4 cuartiles y regla IQR (Tukey) paso a paso

Dividir `price` en 4 partes iguales da los cuartiles **Q1, Q2, Q3 y Q4**; cada uno agrupa
~25 % de las transacciones. La masa cae en los **cuartiles bajos**: **Q1 + Q2 (50 % de la
base) viven por debajo de la mediana (250.000 £)** — la vivienda típica — por lo que la
referencia central es la mediana (Q2), no la media (contaminada por el máximo de 900M a
z = 591). El IQR toma **Q1 y Q3** como anclas y recorta la cola extrema del Q4:

```
PASO 1  los 4 cuartiles de price (columna objetivo): Q1, Q2, Q3, Q4 → ~25 % c/u
        Q1 = 100 .. 158.000 · Q2 = 158.001 .. 250.000 · Q3 = 250.001 .. 385.000
        Q4 = 385.001 .. 900.000.000   (Q1 + Q2 = mercado típico, por debajo de la mediana)
PASO 2  cuartiles de corte para el IQR: Q1 = 158.000 · Q2 mediana = 250.000 · Q3 = 385.000
PASO 3  IQR = 227.000
PASO 4  límite inferior = −182.500 · límite superior = 725.500
PASO 5  fuera de límites = 718.461 (6,32 %) — todos por arriba
PASO 6  decisión: WINSORIZAR al límite superior
```

Se usa **IQR** en lugar de z-score porque la media (364.248) queda contaminada por el máximo
de 900M (z = 591 desviaciones); el IQR resiste colas largas, típicas en datos comerciales.
Decisión **tratar**: el lujo real es “señal” y normalmente se conservaría, pero `price` entra
a un modelo predictivo → se recorta con `clip()`, conservando la fila.

## 7. Bitácora de decisiones

| Columna | Problema | Método | Razón |
|---|---|---|---|
| geo (5 cols) | nulos (MCAR) | bandera `es_sin_geo` | MCAR; no inventar coordenadas |
| `price` | atípicos (IQR) | clip + bandera | 6,32 % cola derecha; price va a modelo |

*Metodología: diagnóstico con `isna()`/`duplicated()`/`describe()`, clasificación de faltantes y base `propuesta` del PDF.*

## 8. Pendiente (siguientes pasos)

- Análisis temporal: *pandemic shock*, volatilidad mensual, nuevo Código Postal (2025).
- *New-build premium* con `is_new_build`.
- Clustering geoespacial sobre latitudes/longitudes limpias, excluyendo `es_sin_geo`.