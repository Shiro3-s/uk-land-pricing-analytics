"""Bitacora de decisiones: registro estructurado de cada transformacion.

Registro con estado: acumula decisiones y las devuelve como tabla o JSON,
que es lo que el pipeline entrega como evidencia de cada ejecucion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Bitacora:
    """Registra cada correccion (que, cuanto, como y por que)."""

    registros: list[dict] = field(default_factory=list)

    def anotar(self, columna: str, problema: str, cantidad: int,
               metodo: str, razon: str, impacto: str) -> None:
        """Agrega una decision de limpieza/faltantes/outliers."""
        self.registros.append({
            "columna": columna,
            "problema": problema,
            "cantidad": int(cantidad),
            "metodo": metodo,
            "razon": razon,
            "impacto": impacto,
        })

    def tabla(self) -> pd.DataFrame:
        """Devuelve la bitacora como tabla lista para mostrar/guardar."""
        return pd.DataFrame(self.registros)

    def a_json(self, ruta) -> None:
        """Exporta la bitacora a JSON (para trazabilidad externa)."""
        ruta.write_text(json.dumps(self.registros, ensure_ascii=False, indent=2), encoding="utf-8")