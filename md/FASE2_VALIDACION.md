# Fase 2: Validación y depuración de datos

## Objetivo
Comprobar coherencia temporal, nulos, esfuerzo, cruces con lo ya tratado y definir el set de variables X antes de pasar a clasificación.

## Insumos
- `data/integrado/maestro_joven_uk.parquet`
- `data/limpia/hipotecas_limpio.parquet`
- `data/integrado/hipoteca_2022.parquet`

## 1) Consistencia temporal
- Maestro: (60, 16)
- 10 regiones x 6 años (2015-2020). Todas con 6 registros
- Rango: 2015 - 2020
- Sin duplicados por (anio, region)
- Clave única (anio, region): OK

## 2) Nulos y duplicados
- Nulos totales: 28
- Duplicados (anio,region): 0

Detalle de nulos:
- `sol_anom` -> 18 nulos
- `precio_var_pct` -> 10 nulos

Estos se imputan a 0, tal como ya se hace en `script/proyecto/modelado.py`.

## 3) Coherencia del esfuerzo
- Esfuerzo: min 0.416 | Q1 0.479 | mediana 0.564 | Q3 0.763 | max 1.065 | media 0.625
- Filas con esfuerzo > 0.30: 60 / 60
- Todas las regiones por encima del umbral 0.30

Esfuerzo medio por región:
| region | esfuerzo_medio |
|---|---|
| London | 0.996 |
| South East | 0.813 |
| East of England | 0.752 |
| South West | 0.712 |
| West Midlands | 0.561 |
| East Midlands | 0.547 |
| Yorkshire and The Humber | 0.488 |
| Wales | 0.478 |
| North West | 0.472 |
| North East | 0.434 |

## 4) Cruce con lo tratado
- Tasa media hipotecas limpia: 6.043%
- Comisiones media: 1115.78
- `hipoteca_2022`: ninguna región accesible con umbral 0.30

## 5) Regla IQR en esfuerzo
- Q1: 0.4792 | Q3: 0.7632 | IQR: 0.2840
- Límite inferior: 0.0532 | Límite superior: 1.1892
- Atípicos superiores: 0 | Atípicos inferiores: 0

## 6) Set de variables propuesto para X
**Numéricas:**
- `precio_mediana`
- `salario_joven_region`
- `anos_salario`
- `precio_var_pct`
- `tmax_anom`
- `tmin_anom`
- `lluvia_anom`
- `sol_anom`
- `af_anom`

**Categórica:**
- `region` (convertir con `get_dummies(drop_first=True)`)

**Excluir:**
- `anio` (identificador)

**Imputación a aplicar:**
```python
df["sol_anom"] = df["sol_anom"].fillna(0.0)
df["precio_var_pct"] = df["precio_var_pct"].fillna(0.0)
```

Con esta imputación los nulos quedan en 0 y el set queda completo.

## Resultado
**DATOS VALIDADOS OK**

## Siguiente paso
Pasar a **Fase 3: Árbol de decisión + falsos positivos** (última fase).
