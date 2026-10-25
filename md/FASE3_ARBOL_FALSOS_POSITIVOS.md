# Fase 3: Árbol de decisión + falsos positivos (última)

## Objetivo
Clasificación binaria con árbol de decisión. Enfocar en falsos positivos/falsos negativos.

## Definición
`accesible = 1` si `esfuerzo <= 0.30`, else `0`

Cálculo de `esfuerzo` con `indicadores.cuota_mensual_pmt()` usando:
- Tasa media hipotecas limpia: 6.043%
- Comisiones media: 1115.78
- LTV 0.90, plazo 25 años
- Imputación: `sol_anom` y `precio_var_pct` a 0

## Preparación
- X: numéricas + `region` con `pd.get_dummies(drop_first=True)`
- Split: `train_test_split(test_size=0.25, random_state=42, stratify=y)` antes de entrenar

## Resultado
- Clase 1 (accesible): 0 de 60 (0.000)
- Train: 45 (todos clase 0) | Test: 15 (todos clase 0)
- Árbol con `max_depth` 2-5: accuracy 1.000 (predice todo clase 0)
- Reglas: solo clase 0

## Matriz de confusión (test)
[[15  0]
 [ 0  0]]

Interpretación:
- VN (0,0): 15
- FP (0,1): 0
- FN (1,0): 0
- VP (1,1): 0

## Reporte
- Clase no_accesible(0): precision/recall/f1 1.00
- Clase accesible(1): sin soporte (0). Métricas con zero_division=0

## Importancias
Todas en 0 con este árbol (única clase).

## Conclusión sobre falsos positivos
**No hay falsos positivos ni falsos negativos.**

Con el umbral `esfuerzo <= 0.30` ninguna región/año es accesible en los datos. El modelo clasifica correctamente la única clase presente. Para poder medir falsos positivos/negativos se necesitaría al menos algún caso con `accesible=1`.

## Archivos
- Script: `script/proyecto/clasificacion_accesibilidad.py`
- Notebook: `md/FASE3_ARBOL_FALSOS_POSITIVOS.ipynb`
