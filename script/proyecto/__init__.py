"""Modulo proyecto: analisis de accesibilidad de vivienda para jovenes (18-34).

Proyecto integral de mineria de datos (KDD + CRISP-DM). Este paquete orquesta las
4 fuentes (transacciones, hipotecas, clima, precios/salario) y las cruza en el
cuadro maestro_joven_uk (anio x region). Orden del pipeline:
carga -> limpieza -> transformaciones -> cruce -> indicadores -> analisis clima
-> validacion -> figuras + bitacora.
"""
from __future__ import annotations