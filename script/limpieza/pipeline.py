"""Orquestador del pipeline de limpieza: flujo estructurado de las 7 fases.

Orquestador con estado: encapsula df, df_limpio y la bitacora durante la corrida.
El orden importa: duplicados -> categorias -> nulos -> outliers -> validar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config, graficos
from .bitacora import Bitacora
from .cargar import cargar_dataset
from .diagnostico import contar_fuera_de_iqr, limites_iqr, nulos_geo, resumen_categorias, tabla_cuartiles
from .transformaciones import aplicar_bandera_atipicos, crear_bandera_sin_geo, winsorizar_price
from .validacion import construir_criterios, validar_criterios


def mostrar_titulo(numero: int, titulo: str) -> None:
    """Separador visual entre partes (para leer la corrida por etapas)."""
    print(f"\n{'=' * 72}\nPARTE {numero} · {titulo}\n{'=' * 72}")


def formato(numero: float | int) -> str:
    """Miles con comas y sin decimales (texto de consola)."""
    return f"{numero:,.0f}"


class PipelineLimpieza:
    """Ejecuta en orden las 7 fases y deja la version limpia exportada en parquet."""

    def __init__(self, ruta_origen: Path | None = None, ruta_salida: Path | None = None) -> None:
        """Configura rutas (o usa las de config) y el estado vacio del pipeline."""
        self.ruta_origen = Path(ruta_origen) if ruta_origen else config.RUTA_ORIGEN
        self.ruta_salida = Path(ruta_salida) if ruta_salida else config.RUTA_SALIDA
        self.bitacora = Bitacora()
        self.df: pd.DataFrame | None = None
        self.df_limpio: pd.DataFrame | None = None
        self.limites: dict | None = None

    # ----------------------------- fases -----------------------------
    def fase_1_cargar(self) -> pd.DataFrame:
        """Carga el origen y verifica dimensiones y filas esperadas."""
        mostrar_titulo(1, "CARGA DEL ORIGEN (intocable)")
        print(f"archivo   : {self.ruta_origen}")
        print(f"tamano    : {self.ruta_origen.stat().st_size / 1e6:,.0f} MB")
        self.df = cargar_dataset(self.ruta_origen)
        print(f"dtypes    : {len(config.DTYPE_CATEGORICOS)} categoricas + fecha parseada + llave string")
        assert self.df.shape == (config.FILAS_VERIFICADAS, 20), "Filas/columnas no verificadas"
        return self.df

    def fase_2_duplicados(self) -> None:
        """Verifica que la base venga unica (exactos y por transaction_id)."""
        mostrar_titulo(2, "DUPLICADOS")
        dup_exactos = int(self.df.duplicated().sum())
        dup_pk = int(self.df[config.CLAVE_PRIMARIA].duplicated().sum())
        print(f"duplicados exactos      : {formato(dup_exactos)}")
        print(f"duplicados por llave    : {formato(dup_pk)}")
        assert dup_exactos == 0 and dup_pk == 0, "Existen duplicados (base no unica)"
        print("-> base unica, no se elimina ninguna fila")

    def fase_3_categorias(self) -> None:
        """Reporta cardinalidad de las categoricas y los pares esperados del pais."""
        mostrar_titulo(3, "CATEGORICAS")
        resumen = resumen_categorias(self.df, [c for c in self.df.select_dtypes("category").columns])
        print(resumen.to_string(index=False))
        paises = self.df["country"].value_counts()
        print("\npaises: " + " · ".join(f"{pais} ({formato(v)})" for pais, v in paises.items()))
        assert self.df["country"].nunique() == 2, "country debe ser England/Wales"

    def fase_4_faltantes(self) -> None:
        """Clasifica nulos geo como MCAR simultaneos y crea la bandera es_sin_geo."""
        mostrar_titulo(4, "FALTANTES (clasificacion MCAR / MAR / MNAR)")
        geo = nulos_geo(self.df)
        assert geo["simultaneo"], "Los nulos geo NO son simultaneos (revisar MCAR)"
        assert geo["todas"] == config.NULOS_GEO_VERIFICADOS, f"Nulos geo != {config.NULOS_GEO_VERIFICADOS}"
        sin_geo = self.df[config.COLUMNAS_GEO[0]].isna()
        mediana_con = self.df.loc[~sin_geo, "price"].median()
        mediana_sin = self.df.loc[sin_geo, "price"].median()
        tasa = 100 * float(sin_geo.mean())
        print(f"filas con geo nula (5 cols juntas) : {formato(geo['todas'])} ({tasa:.2f} %)")
        print(f"mediana price · con geo / sin geo   : {formato(mediana_con)} / {formato(mediana_sin)}")
        print("-> clasificacion MCAR: la ausencia no depende del valor (fallo de geocodificacion)")
        self.df_limpio = crear_bandera_sin_geo(self.df, self.bitacora)
        print(f"-> bandera es_sin_geo creada ({formato(geo['todas'])} filas marcadas, NINGUN dato imputado)")

    def fase_5_outliers(self) -> None:
        """Expone los datos a tratar, los cuartiles (Q1/Q2/Q3) y la regla IQR sobre price."""
        mostrar_titulo(5, "OUTLIERS · CUARTILES + REGLA IQR (Tukey, paso a paso)")
        price_original = self.df["price"].astype("float64")
        l = limites_iqr(price_original)
        self.limites = l
        fuera = contar_fuera_de_iqr(price_original, self.limites)
        assert fuera == config.FUERA_IQR_VERIFICADOS, f"Fuera de IQR != {config.FUERA_IQR_VERIFICADOS}"
        media, desv, maximo = price_original.mean(), price_original.std(), price_original.max()

        print("datos que entran a este analisis:")
        print("   columnas          : price (objetivo), latitude/longitude, town_city/county/country")
        print(f"   columna analizada : price · unica numerica continua · {formato(price_original.min())} .. {formato(maximo)} GBP\n")
        cuartiles = tabla_cuartiles(price_original)
        print("PASO 1   los 4 cuartiles de price (columna objetivo):")
        print(cuartiles.to_string(index=False))
        print("         -> cada cuartil pesa ~25 %; Q1 + Q2 (hasta la mediana " +
              f"{formato(price_original.median())} £) concentran el mercado tipico")
        print(f"PASO 2   cuartiles de corte para el IQR: Q1 = {formato(l['q1'])} · "
              f"Q2 mediana = {formato(l['q2'])} · Q3 = {formato(l['q3'])}")
        print(f"         por que cuartiles y no media: el maximo {formato(maximo)} esta a "
              f"z = {(maximo - media) / desv:.1f} y contamina la media {formato(media)}; los cuartiles no.")
        print(f"PASO 3   IQR = Q3 - Q1 = {formato(l['iqr'])}")
        print(f"PASO 4   limite inferior = {formato(l['limite_inferior'])}   limite superior = {formato(l['limite_superior'])}")
        print(f"PASO 5   fuera de limites = {formato(fuera)} ({100 * fuera / len(price_original):.2f} %) — todos por arriba")
        print(f"PASO 6   decision: WINSORIZAR price al limite superior {formato(config.PRECIO_LIM_SUP)}")
        self.df_limpio = aplicar_bandera_atipicos(self.df_limpio, price_original, self.limites, self.bitacora)
        winsorizado = winsorizar_price(price_original, self.limites["limite_superior"])
        ruta_cuartiles = config.PROYECTO / "md" / "figuras" / "cuartiles_price.png"
        graficos.guardar_figura(
            graficos.cuartiles_barras(cuartiles, price_original.median()), ruta_cuartiles)
        ruta_box = config.PROYECTO / "md" / "figuras" / "boxplot_price.png"
        graficos.guardar_figura(graficos.box_antes_despues(price_original, winsorizado), ruta_box)
        print(f"-> figuras guardadas: {ruta_box}\n                       {ruta_cuartiles}")
        print(f"-> bandera es_atipico_precio + price recortado ({formato(fuera)} filas ajustadas, 0 eliminadas)")

    def fase_6_validar(self) -> None:
        """Compara los criterios post-transformacion contra los valores verificados."""
        mostrar_titulo(6, "VALIDACION POST-LIMPIEZA")
        criterios = construir_criterios(self.df_limpio)
        for fila in criterios.itertuples(index=False):
            print(f"   {fila.criterio:<28}: real {fila.real:,}  (esperado {fila.esperado:,})")
        validar_criterios(criterios)

    def fase_7_exportar(self) -> Path:
        """Escribe la version limpia en parquet (nueva salida; el original no se toca)."""
        mostrar_titulo(7, "EXPORTAR · DATASET LIMPIO")
        print(f"shape : original {self.df.shape} -> limpio {self.df_limpio.shape} "
              f"(+{len(config.COLUMNAS_BANDERAS)} banderas)")
        print(f"price : min {formato(self.df_limpio['price'].min())}  "
              f"max {formato(self.df_limpio['price'].max())}")
        self.ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        self.df_limpio.to_parquet(self.ruta_salida, index=False)
        print(f"guardado en {self.ruta_salida} ({self.ruta_salida.stat().st_size / 1e6:,.0f} MB)")
        return self.ruta_salida

    # ----------------------------- flujo -----------------------------
    def run(self) -> pd.DataFrame:
        """Ejecuta todas las fases en orden y muestra los resultados por partes."""
        self.fase_1_cargar()
        self.fase_2_duplicados()
        self.fase_3_categorias()
        self.fase_4_faltantes()
        self.fase_5_outliers()
        self.fase_6_validar()
        self.fase_7_exportar()
        mostrar_titulo(8, "BITACORA DE DECISIONES (evidencia del proceso)")
        print(self.bitacora.tabla().to_string(index=False))
        return self.df_limpio