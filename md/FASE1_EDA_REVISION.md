# Fase 1: EDA para revisión de datos

Fecha: Generado por revisión local (sin subir a repositorio)

## Objetivo
Validar los datos con un EDA reproducible antes de pasar a clasificación. Se calculó el esfuerzo hipotecario con la lógica del proyecto.

## Insumos
- `data/integrado/maestro_joven_uk.parquet`
- `data/limpia/hipotecas_limpio.parquet`
- `script/proyecto/indicadores.cuota_mensual_pmt()`
- Tasa media hipotecas limpia: 6.043% | Comisiones media: 1115.78

## Resultados generales
- Filas: 60 (10 regiones x 6 años 2015-2020)
- Clave única (anio, region): OK
- Esfuerzo: min 0.416 | mediana 0.564 | max 1.065
- Todas las filas con esfuerzo > 0.30 (60/60)

## Resumen numérico
| Variable | n | media | mediana | min | max | desv | asimetría | curtosis |
|---|---|---|---|---|---|---|---|---|
| esfuerzo | 60 | 0.63 | 0.56 | 0.42 | 1.07 | 0.18 | 0.84 | -0.37 |
| precio_mediana | 60 | 222264.42 | 182747.50 | 127000.00 | 482525.00 | 94948.21 | 1.31 | 0.91 |
| salario_joven_region | 60 | 24160.72 | 23022.25 | 21061.00 | 31691.25 | 2837.53 | 1.57 | 1.72 |
| anos_salario | 60 | 8.92 | 8.04 | 5.93 | 15.23 | 2.61 | 0.84 | -0.37 |

## Regla IQR
| Variable | Q1 | Q3 | IQR | lim_inf | lim_sup | out_sup | out_inf | tot |
|---|---|---|---|---|---|---|---|---|
| esfuerzo | 0.4792 | 0.7632 | 0.2840 | 0.0532 | 1.1892 | 0 | 0 | 0 |
| precio_mediana | 153750.00 | 271250.00 | 117500.00 | -22500.00 | 447500.00 | 4 | 0 | 4 |

## Por región (promedio 2015-2020)
| region | esfuerzo_medio | esfuerzo_mediana | anos_salario_medio | precio_mediana_medio | n |
|---|---|---|---|---|---|
| London | 0.996 | 1.018 | 14.236 | 445920.833 | 6 |
| South East | 0.813 | 0.835 | 11.608 | 308333.333 | 6 |
| East of England | 0.752 | 0.774 | 10.735 | 271666.667 | 6 |
| South West | 0.712 | 0.726 | 10.169 | 242083.333 | 6 |
| West Midlands | 0.561 | 0.569 | 8.000 | 182415.833 | 6 |
| East Midlands | 0.547 | 0.558 | 7.797 | 179750.000 | 6 |
| Yorkshire and The Humber | 0.488 | 0.491 | 6.961 | 153500.000 | 6 |
| Wales | 0.478 | 0.479 | 6.820 | 153824.167 | 6 |
| North West | 0.472 | 0.473 | 6.723 | 153325.000 | 6 |
| North East | 0.434 | 0.437 | 6.189 | 131825.000 | 6 |

## Figuras generadas (prefijo rev_)
- `md/figuras/rev_histograma_esfuerzo.png`
- `md/figuras/rev_histograma_precio_mediana.png`
- `md/figuras/rev_boxplot_esfuerzo.png`
- `md/figuras/rev_dispersion_precio_esfuerzo.png`
- `md/figuras/rev_barras_esfuerzo_anos_salario_region.png`
- `md/figuras/rev_heatmap_correlaciones.png`
- `md/figuras/rev_matriz_dispersion.png`

## Observaciones
- Esfuerzo muy por encima del umbral 0.30 en todas las regiones. London es la más afectada.
- `precio_mediana` presenta 4 valores por encima del límite superior por IQR.
- El esfuerzo muestra asimetría positiva moderada. La relación con `precio_mediana` es clara (pendiente positiva).
- `anos_salario` guarda relación inversa con el esfuerzo, coherente con el objetivo.

## Siguiente paso
Pasar a **Fase 2: Validación y depuración de datos** con base en estas figuras y resultados.
