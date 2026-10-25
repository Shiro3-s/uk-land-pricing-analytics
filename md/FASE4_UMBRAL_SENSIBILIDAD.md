# Fase 4: Sensibilidad de umbral

## Objetivo
Evaluar efecto del umbral. Con 0.30 sin clase accesible. Buscar umbral con ambas clases para analizar falsos positivos/negativos.

## Sensibilidad
| umbral | accesible(1) | no_accesible(0) | % |
|---|---|---|---|
| 0.30 | 0 | 60 | 0.00 |
| 0.35 | 0 | 60 | 0.00 |
| 0.40 | 1 | 59 | 1.67 |
| 0.45 | 4 | 56 | 6.67 |
| 0.50 | 16 | 44 | 26.67 |

## Umbral seleccionado: 0.50
Con esfuerzo <= 0.50 hay ambas clases (16/44). Por region: North East (6), North West (6), Wales (2), Yorkshire and The Humber (2). Otras con 0.

## Árbol (max_depth=3), umbral 0.50
Split estratificado. Matriz de confusión test:
```
[[10  1]
 [ 0  4]]
```
VN(0,0)=10, FP(0,1)=1, FN(1,0)=0, VP(1,1)=4  
Falsos positivos: 1 | Falsos negativos: 0  
Accuracy: 1.00

## Figuras
- md/figuras/fase4_hist_esfuerzo_umbral.png
- md/figuras/fase4_barras_umbral.png

## Conclusión
El umbral 0.30 es muy estricto. Con 0.50 sí aparecen ambas clases y permite medir FP/FN. Con max_depth=3 el árbol clasifica bien, con 1 falso positivo.
