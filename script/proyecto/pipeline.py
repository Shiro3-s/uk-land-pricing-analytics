"""Pipeline 4 fuentes -> cruce maestro_joven_uk en 11 partes (estado real por clase).

El flujo materializa las fases del proyecto (KDD + CRISP-DM):
  Fase 2 · Fuentes y tipos de datos   -> parte 1 (carga de origenes intocables).
  Fase 3 · Limpieza y cruzamiento      -> partes 2 a 11 (entregable de este avance).
  Fase 4 · Analisis y modelado         -> pendiente (no cubierta por este pipeline).
Orden del flujo: orígenes -> duplicados -> tipos/categorias -> nulos (MCAR/MAR/MNAR)
-> outliers (IQR vs z-score) -> transformaciones -> cruce -> indicadores ->
clima vs precios -> validacion -> exportar (figuras + bitacora).

Reutiliza la limpieza previa de F1 (data/limpia/uk_property_price_limpio.parquet).
Salidas en data/{limpia,integrado}, figuras en md/figuras y bitacora en md/.
"""
from __future__ import annotations

import pandas as pd

from . import config, graficos
from . import transformaciones as tr
from .bitacora import BitacoraProyecto
from . import cargar, cruces, diagnostico, indicadores, limpiar, validacion


def mostrar_titulo(numero: int, titulo: str) -> None:
    """Imprime el encabezado de una parte del pipeline (num + titulo)."""
    print(f"\n{'=' * 72}\nPARTE {numero} · {titulo}\n{'=' * 72}")


def formato(numero: float) -> str:
    """Da formato de miles (1234 -> '1,234'). Para etiquetas impresas."""
    return f"{numero:,.0f}"


class PipelineProyecto:
    """Orquesta las 11 partes del analisis de accesibilidad joven (UK)."""

    def __init__(self) -> None:
        """Crea el pipeline con estado vacio; cada parte llena sus atributos."""
        self.bitacora = BitacoraProyecto()
        self.hipotecas: pd.DataFrame | None = None
        self.clima: pd.DataFrame | None = None
        self.precios: pd.DataFrame | None = None
        self.ingreso: pd.DataFrame | None = None
        self.transacciones: pd.DataFrame | None = None
        self.precios_region: pd.DataFrame | None = None
        self.precios_ciudad: pd.DataFrame | None = None
        self.anom_nacional: pd.DataFrame | None = None
        self.anom_region: pd.DataFrame | None = None
        self.maestro: pd.DataFrame | None = None
        self.hipoteca_2022: pd.DataFrame | None = None
        self.historico: pd.DataFrame | None = None
        self.correlaciones: pd.DataFrame | None = None
        self.clasificacion: pd.DataFrame | None = None
        self.evidencia_iqr: dict | None = None

    # ----------------------------- partes -----------------------------
    def parte_1_origenes(self) -> None:
        """Paso 1 · Origines intocables: carga las 5 rutas y las copias crudas."""
        mostrar_titulo(1, "ORIGENES INTOCABLES (fase 2: fuentes y tipos de datos)")
        for ruta in [config.RUTA_TRANSACCIONES_LIMPIO, config.RUTA_HIPOTECAS,
                     config.RUTA_CLIMA, config.RUTA_PRECIOS_SALARIO,
                     config.RUTA_INGRESO_EDAD]:
            print(f"  {ruta.name:<45} {ruta.stat().st_size / 1e6:8,.1f} MB  {ruta}")
        self.transacciones = cargar.cargar_precios_limpios()
        self.hipotecas = cargar.cargar_hipotecas()
        self.clima = cargar.cargar_clima()
        self.precios = cargar.cargar_precios_salario()
        self.ingreso = cargar.cargar_ingreso_edad()
        # copias crudas para la clasificacion MCAR/MAR/MNAR (antes de tocar nada)
        self._raw = {"hipotecas": self.hipotecas.copy(),
                     "clima": self.clima.copy(),
                     "precios": self.precios.copy()}
        print(f"  F1 transacciones : {self.transacciones.shape[0]:,} filas (limpio)")
        print(f"  F2 hipotecas     : {self.hipotecas.shape[0]:,} productos (2022)")
        print(f"  F3 clima         : {self.clima.shape[0]:,} filas, "
              f"{self.clima['estacion'].nunique()} estaciones")
        print(f"  F4 precios/sal   : {self.precios.shape[0]} anios | "
              f"{self.ingreso.shape[0]} bandas de edad x genero")

    def parte_2_limpieza(self) -> None:
        """Paso 2 · Limpieza por fuente: duplicados, tipos y nulos (reglas en limpiar)."""
        mostrar_titulo(2, "LIMPIEZA POR FUENTE (fase 3: limpieza y cruzamiento)")
        self.hipotecas, d1 = limpiar.limpiar_hipotecas(self.hipotecas)
        self.clima, d2 = limpiar.limpiar_clima(self.clima)
        self.precios, d3 = limpiar.limpiar_precios_salario(self.precios)
        self.ingreso, d4 = limpiar.limpiar_ingreso_edad(self.ingreso)
        for d in d1 + d2 + d3 + d4:
            self.bitacora.anotar(**d)
            print(f"  [{d['fuente']}] {d['problema']}: {d['cantidad']:,} -> {d['metodo']}")
        print(f"  -> hipotecas {len(self.hipotecas):,} | clima {len(self.clima):,} "
              f"| precios {len(self.precios)}")

    def parte_3_faltantes(self) -> None:
        """Paso 3 · Faltantes: clasifica MCAR/MAR/MNAR por fuente y guarda la figura."""
        mostrar_titulo(3, "FALTANTES MCAR / MAR / MNAR (fase 3)")
        self.clasificacion = diagnostico.clasificacion_fuentes(self._raw)
        for r in self.clasificacion.to_dict("records"):
            print(f"  [{r['fuente']:<16}] {r['columna']:<22} nulos {r['nulos']:>5} "
                  f"({r['pct']:>5}%) -> {r['clasificacion']}")
            print(f"                 evidencia: {r['evidencia']}")
            print(f"                 tratamiento: {r['tratamiento']}")
            self.bitacora.anotar(r["fuente"], r["columna"] + " (faltante)",
                                 r["nulos"], r["tratamiento"],
                                 f"{r['clasificacion']}: {r['evidencia']}")
        graficos.guardar_figura(graficos.fig_nulos_fuentes(self.clasificacion),
                                config.MD_FIGURAS / "faltantes_por_fuente.png")
        print("  -> figura md/figuras/faltantes_por_fuente.png")

    def parte_4_outliers(self) -> None:
        """Paso 4 · Outliers: compara IQR vs z-score, winsoriza la tasa y marca atipicos."""
        mostrar_titulo(4, "OUTLIERS IQR (Tukey) vs Z-SCORE (fase 3: decision)")
        tasa = self.hipotecas["tasa_inicial_pct"].astype("float64")
        ev = diagnostico.comparar_iqr_zscore(tasa, "tasa_inicial_pct (F2)")
        self.evidencia_iqr = ev
        print(f"  F2 tasa inicial : n={ev['n']:,} | Q1={ev['q1']} mediana={ev['mediana']} "
              f"Q3={ev['q3']} | IQR={ev['iqr']}")
        print(f"  IQR -> limite superior {ev['lim_sup']}: fuera {ev['fuera_iqr']:,} "
              f"({ev['pct_iqr']} %)  |  z-score (>3): {ev['fuera_z']:,} ({ev['pct_z']} %)")
        print(f"  skew={ev['skew']} -> {ev['razon']}")
        limite = diagnostico.limites_iqr(tasa)["limite_superior"]
        self.hipotecas["tasa_inicial_pct"] = diagnostico.winsorizar(tasa, limite)
        self.hipotecas["es_atipico_tasa"] = (tasa > limite)
        self.bitacora.anotar("F2 hipotecas", "outliers de tasa_inicial (metodo)",
                             ev["fuera_iqr"],
                             f"winsorizar clip al limite IQR {ev['lim_sup']}",
                             f"IQR vs z-score: IQR detecta {ev['fuera_iqr']} y z-score "
                             f"{ev['fuera_z']} (skew {ev['skew']}); IQR no exige normalidad",
                             f"tasa media {tasa.mean():.2f} -> "
                             f"{self.hipotecas['tasa_inicial_pct'].mean():.2f} %")
        graficos.guardar_figura(graficos.fig_outliers_tasa(
            tasa, self.hipotecas["tasa_inicial_pct"], ev),
            config.MD_FIGURAS / "outliers_tasa_iqr.png")
        print("  -> figura md/figuras/outliers_tasa_iqr.png · bandera es_atipico_tasa")

    def parte_5_transformaciones(self) -> None:
        """Paso 5 · Transformaciones: agregados, anomalias climaticas y salario joven."""
        mostrar_titulo(5, "TRANSFORMACIONES (fase 3: agregados + anomalias + salario)")
        self.precios_region = tr.agregado_precios_region(self.transacciones)
        self.precios_ciudad = tr.agregado_precios_ciudad(self.transacciones)
        anual = tr.clima_anual(self.clima)
        self.anom_nacional, self.anom_region = tr.anomalias_climaticas(anual)
        joven, factor = tr.salario_joven(self.ingreso, self.precios)
        salario_region = tr.serie_salario_joven(self.precios, factor)
        self.historico = tr.historico_nacional(self.precios, factor)

        for r in [config.RUTA_PRECIOS_REGION, config.RUTA_PRECIOS_CIUDAD]:
            r.parent.mkdir(parents=True, exist_ok=True)
        self.precios_region.to_parquet(config.RUTA_PRECIOS_REGION, index=False)
        self.precios_ciudad.to_parquet(config.RUTA_PRECIOS_CIUDAD, index=False)
        self.anom_nacional.to_parquet(config.RUTA_ANOMALIAS_NACIONAL, index=False)
        self.anom_region.to_parquet(config.RUTA_ANOMALIAS_REGION, index=False)

        self._salario_region = salario_region
        self._factor_joven = factor
        print(f"  salario joven 18-34 : GBP {joven:,.0f} (factor {factor:.3f} vs mediana 2020)")
        print(f"  precios por region  : {len(self.precios_region):,} filas | "
              f"ciudad: {len(self.precios_ciudad):,}")
        print(f"  anomalias regionales: {len(self.anom_region):,} filas")
        self.bitacora.anotar("F4 salario joven",
                             "bandas de edad sin corte exacto en 34",
                             3, "media de medianas 18-21/22-29/30-39",
                             "la fuente publica bandas; 30-39 cubre 30-34 (aprox.)",
                             "salario joven = GBP %.0f" % joven)

    def parte_6_cruce(self) -> None:
        """Paso 6 · Cruce: arma maestro_joven_uk (anio x region) con clima y salario."""
        mostrar_titulo(6, f"CRUCE maestro_joven_uk (fase 3: anio x region, "
                          f"{config.VENTANA_CUADRO[0]}-{config.VENTANA_CUADRO[1]})")
        self.maestro = cruces.cruce_maestro(self.precios_region, self._salario_region,
                                            self.anom_region)
        print(self.maestro.head(10).to_string(index=False))
        print(f"  -> {len(self.maestro):,} filas | nulos: "
              f"{int(self.maestro.isna().sum().sum())}")

    def parte_7_indicadores(self) -> None:
        """Paso 7 · Indicadores: anos de salario y esfuerzo hipotecario 2022."""
        mostrar_titulo(7, "INDICADORES DE ACCESIBILIDAD (fase 3: anos_salario + hipoteca)")
        promedio = (self.maestro.groupby("region", observed=True)["anos_salario"]
                    .mean().sort_values())
        print("  Anios de salario joven para comprar (promedio 2015-2020):")
        print(promedio.round(1).to_string())
        self.hipoteca_2022 = indicadores.hipoteca_2022(
            self.precios_region, self.hipotecas, self._salario_region)
        print(f"\n  Esfuerzo hipotecario 2022 (tasa media winsorizada "
              f"{self.hipoteca_2022['tasa_inicial_pct'].iloc[0]:.2f} %, 90 % LTV, "
              f"25 anios, precio deflactado a GBP 2020):")
        for r in self.hipoteca_2022.itertuples():
            estado = "accesible" if r.accesible else "no accesible"
            print(f"    {r.region:<26} cuota GBP {r.cuota_mensual:>9,.0f} mes | "
                  f"{r.cuota_pct_salario * 100:5.1f} % del salario -> {estado}")
        self.bitacora.anotar("F2 hipotecas",
                             "precios 2022 nominales vs salario GBP 2020",
                             len(self.hipoteca_2022), "deflactar por CPI 2020->2022 (1.137)",
                             "homogeneizar moneda entre F1 (nominal) y F4 (real 2020)",
                             "cuota en GBP 2020")

    def parte_8_clima_precio(self) -> None:
        """Paso 8 · Analisis: correlacion clima-precio sobre el cuadro maestro."""
        mostrar_titulo(8, "ANALISIS CLIMA vs PRECIOS (fase 3)")
        self.correlaciones = indicadores.correlacion_clima_precio(self.maestro)
        print("  Correlacion de Pearson (variacion anual del precio vs anomalias):")
        print(self.correlaciones.round(3).to_string(index=False))
        media_lluvia = self.maestro["lluvia_anom"].mean()
        print(f"\n  -> anomalia media de lluvia en la ventana: {media_lluvia:+.2f} mm "
              f"(signo = mas/menos lluvia que la normal 1931-2020)")

    def parte_9_validacion(self) -> None:
        """Paso 9 · Validacion: asserts post-cruce (clave unica, sin nulos, IQR)."""
        mostrar_titulo(9, "VALIDACION POST-CRUCE (fase 3)")
        validacion.validar_maestro(self.maestro)
        validacion.validar_hipoteca(self.hipoteca_2022)
        print("  maestro   : clave anio+region unica, sin nulos de precio/salario")
        print("  hipoteca  : 10 regiones, cuotas y % positivos")
        print("  outliers  : tasa winsorizada <= limite IQR")
        assert self.hipotecas["tasa_inicial_pct"].max() <= self.evidencia_iqr["lim_sup"], \
            "tasa no quedo dentro del limite IQR"
        print("  -> OK, listo para exportar")

    def parte_10_exportar(self) -> None:
        """Paso 10 · Exportar: parquets, CSVs muestra, figuras PNG y bitacora."""
        mostrar_titulo(10, "EXPORTAR FIGURAS + BITACORA (fase 3)")
        rutas = {
            "maestro_joven_uk.parquet": config.RUTA_MAESTRO,
            "hipoteca_2022.parquet": config.RUTA_HIPOTECA_2022,
            "historico_nacional.parquet": config.RUTA_HISTORICO,
            "hipotecas_limpio.parquet": config.RUTA_HIPOTECAS_LIMPIO,
            "clima_limpio.parquet": config.RUTA_CLIMA_LIMPIO,
            "precios_salario_limpio.parquet": config.RUTA_PRECIOS_SALARIO_LIMPIO,
        }
        for nombre, ruta in rutas.items():
            ruta.parent.mkdir(parents=True, exist_ok=True)
        self.maestro.to_parquet(config.RUTA_MAESTRO, index=False)
        self.hipoteca_2022.to_parquet(config.RUTA_HIPOTECA_2022, index=False)
        self.historico.to_parquet(config.RUTA_HISTORICO, index=False)
        self.hipotecas.to_parquet(config.RUTA_HIPOTECAS_LIMPIO, index=False)
        self.clima.to_parquet(config.RUTA_CLIMA_LIMPIO, index=False)
        self.precios.to_parquet(config.RUTA_PRECIOS_SALARIO_LIMPIO, index=False)

        self.maestro.head(10).to_csv(config.RUTA_MAESTRO_MUESTRA, index=False)
        self.hipoteca_2022.to_csv(config.RUTA_HIPOTECA_2022_MUESTRA, index=False)
        self.historico.to_csv(config.RUTA_HISTORICO_MUESTRA, index=False)

        for nombre, ruta in rutas.items():
            print(f"  parquet {nombre:<32} {ruta.stat().st_size / 1e3:8,.0f} KB")
        for ruta in [config.RUTA_MAESTRO_MUESTRA, config.RUTA_HIPOTECA_2022_MUESTRA,
                     config.RUTA_HISTORICO_MUESTRA]:
            print(f"  csv muestra          {ruta.name:<28} {ruta.stat().st_size / 1e3:8,.0f} KB")

        graficos.guardar_figura(graficos.fig_fases(), config.MD_FIGURAS / "fases_proyecto.png")
        graficos.guardar_figura(graficos.fig_historico_nacional(self.historico),
                                config.MD_FIGURAS / "historico_precio_salario.png")
        graficos.guardar_figura(graficos.fig_precios_region(self.precios_region),
                                config.MD_FIGURAS / "precios_por_region.png")
        graficos.guardar_figura(graficos.fig_anos_salario(self.maestro),
                                config.MD_FIGURAS / "anos_salario_por_region.png")
        graficos.guardar_figura(graficos.fig_anomalia_lluvia(self.anom_nacional),
                                config.MD_FIGURAS / "anomalia_lluvia.png")
        graficos.guardar_figura(graficos.fig_clima_precio(self.maestro),
                                config.MD_FIGURAS / "clima_precio.png")
        graficos.guardar_figura(graficos.fig_hipoteca_2022(self.hipoteca_2022),
                                config.MD_FIGURAS / "hipoteca_2022.png")
        print("  figuras PNG -> md/figuras: fases_proyecto, faltantes_por_fuente,")
        print("      outliers_tasa_iqr, historico_precio_salario, precios_por_region,")
        print("      anos_salario_por_region, anomalia_lluvia, clima_precio, hipoteca_2022")

        self.bitacora.a_markdown(config.RUTA_BITACORA)
        print(f"  bitacora -> {config.RUTA_BITACORA}")

    # ----------------------------- flujo -----------------------------
    def run(self) -> pd.DataFrame:
        """Ejecuta las 10 partes y muestra la bitacora; devuelve el cuadro maestro."""
        self.parte_1_origenes()
        self.parte_2_limpieza()
        self.parte_3_faltantes()
        self.parte_4_outliers()
        self.parte_5_transformaciones()
        self.parte_6_cruce()
        self.parte_7_indicadores()
        self.parte_8_clima_precio()
        self.parte_9_validacion()
        self.parte_10_exportar()
        mostrar_titulo(11, "BITACORA DE DECISIONES (fase 3: cierre)")
        print(self.bitacora.tabla().to_string(index=False))
        return self.maestro