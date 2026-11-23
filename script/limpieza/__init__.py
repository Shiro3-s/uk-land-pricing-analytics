"""Paquete de limpieza del integrado UK Property (precio + geo).

Modulos por responsabilidad (arquitectura modular de ingenieria):

    config          rutas, dtypes, columnas y constantes de negocio verificado
    cargar          lectura con tipos optimizados del origen (intocable)
    diagnostico     perfil de columnas, IQR, nulos geo, categorias, fechas
    transformaciones banderas (es_sin_geo / es_atipico_precio) y winsorizacion
    validacion      criterios post-limpieza comparados contra lo verificado
    bitacora        registro de decisiones (que, cuanto, como y por que)
    graficos        figuras listas para exponer (matplotlib)
    pipeline        orquestador de las fases del flujo de limpieza
"""

__version__ = "1.0.0"