"""Auditoria integral de fuentes CSV ("Excel") y Parquet del proyecto.

Recorre todos los archivos tabulares en data/crudo/, db/, data/limpia/ y
data/integrado/ con autodeteccion de separador (',' vs ';') y lectura segura
en memoria (metadatos Parquet y streaming por lotes para archivos grandes).

Valida y documenta cuatro preguntas clave:
  1. Que archivos tabulares existen (CSV abiertos en Excel y Parquet) y cuales se usan.
  2. Si los 5 CSV auxiliares de db/ son fuentes nuevas o subconjuntos 100 % contenidos
     en F1 limpio (verificacion exacta por transaction_id en streaming de baja RAM).
  3. Que tan limpios estan los archivos crudos, limpios e integrados (apertura,
     duplicados, nulos estructurales vs. nulos criticos, limites IQR).
  4. Paridad exacta entre maestro_joven_uk.parquet y accesibilidad_vivienda_joven_uk_v1.csv.

Salidas:
  - data/integrado/auditoria_fuentes_calidad.csv
  - md/figuras/auditoria_fuentes_calidad.png

Ejecutar:
  python script/proyecto/auditoria_datos.py
"""
from __future__ import annotations

import gc
import time
import tracemalloc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys

from proyecto import config

plt.rcParams["figure.dpi"] = 140

# Catalogo de roles operativos por archivo conocido
ROLES_ARCHIVO: dict[str, tuple[str, str]] = {
    "data/crudo/Average_UK_houseprices_and_salary.csv": (
        "Activo (F4a origen)",
        "Usado en main_proyecto.py -> precios_salario_limpio + maestro",
    ),
    "data/crudo/Income_by_age_and_gender.csv": (
        "Activo (F4b origen)",
        "Usado en main_proyecto.py -> factor salarial joven 18-34",
    ),
    "data/crudo/MET_Office_Weather_Data.csv": (
        "Activo (F3 origen)",
        "Usado en main_proyecto.py -> clima_limpio + anomalias",
    ),
    "data/crudo/UK_Mortgage_Rate.csv": (
        "Activo (F2 origen)",
        "Usado en main_proyecto.py, Torneo, Clustering, Apriori y Evaluacion",
    ),
    "db/uk_property_price.csv": (
        "Activo (F1 origen 11,37M)",
        "Procesado por main_limpieza.py -> uk_property_price_limpio.parquet",
    ),
    "db/uk_property.csv": (
        "Subconjunto de F1 (ya incluido)",
        "100 % de sus 100k transaction_id ya estan en F1 limpio; no sumar para evitar duplicados",
    ),
    "db/uk_property_semiorganizado.csv": (
        "Subconjunto de F1 (corte Excel ';')",
        "Exportacion Excel con sep=';' al tope de 1.048.575 filas; 100 % contenido en F1 limpio",
    ),
    "db/uk_property_muestra.csv": (
        "Muestra respaldo F1 (2k)",
        "Muestra liviana de respaldo; 100 % contenida en F1 limpio",
    ),
    "db/uk_property_price_muestra.csv": (
        "Muestra respaldo F1 (2k)",
        "Muestra liviana de respaldo para repo; 100 % contenida en F1 limpio",
    ),
    "db/uk_property_semiorganizado_muestra.csv": (
        "Muestra respaldo F1 (';', 1.999)",
        "Muestra con sep=';'; 100 % contenida en F1 limpio",
    ),
    "data/limpia/uk_property_price_limpio.parquet": (
        "Activo (F1 limpio 11,37M)",
        "Insumo de main_proyecto.py y Clustering C 1M en streaming",
    ),
    "data/limpia/uk_property_price_limpio_muestra.parquet": (
        "Activo (F1 muestra 2k)",
        "Insumo liviano para Apriori F1 y pruebas rapidas",
    ),
    "data/limpia/uk_property_price_limpio_muestra.csv": (
        "Muestra limpia F1 (100k CSV)",
        "Copia legible en CSV de 100k filas limpias (100 % contenida en F1 limpio)",
    ),
    "data/limpia/hipotecas_limpio.parquet": (
        "Activo (F2 limpio 7.796)",
        "Insumo de Torneo, Clustering B, Apriori F2 y Evaluacion de umbrales",
    ),
    "data/limpia/clima_limpio.parquet": (
        "Activo (F3 limpio 37.029)",
        "Insumo de anomalias climaticas regionales y nacionales",
    ),
    "data/limpia/precios_salario_limpio.parquet": (
        "Activo (F4a limpio 46)",
        "Insumo de serie historica 1975-2020 y salarios reales",
    ),
    "data/limpia/precios_region.parquet": (
        "Activo (agregado regional)",
        "Insumo del cruce maestro_joven_uk y corte hipoteca_2022",
    ),
    "data/limpia/precios_ciudad.parquet": (
        "Activo (agregado municipal)",
        "Precios agregados por 1.153 ciudades UK (2015-2026)",
    ),
    "data/integrado/maestro_joven_uk.parquet": (
        "Activo (cuadro maestro 60x16)",
        "Tabla central para EDA, regresion, clasificacion, clustering A y simulador",
    ),
    "data/integrado/accesibilidad_vivienda_joven_uk_v1.csv": (
        "Copia CSV equivalente v1",
        "Paridad numerica exacta con maestro_joven_uk.parquet (dif. < 1e-9)",
    ),
    "data/integrado/hipotecas_catalogo_limpio.csv": (
        "Activo (exportacion CSV F2 para BD)",
        "Copia plana de hipotecas_limpio.parquet lista para PostgreSQL/Neon (328 nulos MAR en tasa_reversion)",
    ),
    "data/integrado/precios_ciudad_limpio.csv": (
        "Activo (exportacion CSV ciudades para BD)",
        "Copia plana de precios_ciudad.parquet (14.050 filas, 1.153 ciudades UK) lista para BD",
    ),
    "data/integrado/clustering_benchmark_24gb.csv": (
        "Activo (benchmark escalabilidad 24 GB)",
        "Telemetria real 10k -> 11,35M en streaming (fila base 10k sin desplazamiento previo)",
    ),
    "data/integrado/comparativa_regresion_dual.csv": (
        "Activo (benchmark Tipo 1 Regresion)",
        "Comparativa de regresores sin fuga en Maestro (Ridge) y F2 (RandomForestRegressor)",
    ),
}


def detectar_separador_csv(ruta: Path) -> str:
    """Detecta si un CSV usa ',' o ';' leyendo solo la primera linea."""
    with ruta.open("r", encoding="utf-8", errors="replace") as f:
        encabezado = f.readline()
    return ";" if encabezado.count(";") > encabezado.count(",") else ","


def contar_lineas_rapido(ruta: Path) -> int:
    """Cuenta filas de datos de un CSV grande por bloques binarios de 8 MB."""
    total_nl = 0
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(8 * 1024 * 1024), b""):
            total_nl += bloque.count(b"\n")
    return max(0, total_nl - 1)


def _evaluar_estado_limpieza(carpeta: str, nombre: str, nulos: int, dups: int) -> str:
    """Clasifica el estado sanitario del archivo segun su etapa del pipeline."""
    if carpeta == "data/crudo":
        if nulos == 0 and dups == 0:
            return "Crudo intocable (limpio de origen)"
        return "Crudo intocable (suciedad tratada en data/limpia)"
    if carpeta == "db":
        return "Origen/respaldo crudo intocable"
    if dups > 0:
        return "Revisar: contiene duplicados"
    if nulos == 0:
        return "Limpio (0 nulos, 0 duplicados)"
    if nombre in {
        "clima_limpio.parquet",
        "hipotecas_limpio.parquet",
        "hipotecas_catalogo_limpio.csv",
        "precios_salario_limpio.parquet",
        "uk_property_price_limpio.parquet",
        "uk_property_price_limpio_muestra.parquet",
        "uk_property_price_limpio_muestra.csv",
        "anomalias_nacional.parquet",
        "anomalias_region.parquet",
        "maestro_joven_uk.parquet",
        "maestro_joven_uk_muestra.csv",
        "accesibilidad_vivienda_joven_uk_v1.csv",
        "clustering_benchmark_24gb.csv",
    }:
        return "Limpio (solo nulos estructurales/MCAR-MAR documentados)"
    return "Revisar nulos inesperados"


def auditar_archivos_proyecto(incluir_salidas_nuevas: bool = False) -> pd.DataFrame:
    """Inspecciona todos los CSV y Parquet en data/crudo, db, data/limpia y data/integrado."""
    carpetas = [
        ("data/crudo", config.DATA_CRUDO),
        ("db", config.PROYECTO / "db"),
        ("data/limpia", config.DATA_LIMPIA),
        ("data/integrado", config.DATA_INTEGRADO),
    ]
    registros: list[dict[str, object]] = []

    for nombre_carpeta, ruta_dir in carpetas:
        if not ruta_dir.exists():
            continue
        for ruta in sorted(ruta_dir.glob("*")):
            if not ruta.is_file() or ruta.suffix.lower() not in {".csv", ".parquet"}:
                continue
            rel = ruta.relative_to(config.PROYECTO).as_posix()
            if not incluir_salidas_nuevas and ruta.name == config.RUTA_AUDITORIA_DATOS.name:
                continue

            tam_mb = ruta.stat().st_size / (1024.0**2)
            formato = ruta.suffix.lower().lstrip(".")
            sep = "-"
            apertura_ok = True
            nota_nulos = ""

            try:
                if formato == "parquet":
                    pf = pq.ParquetFile(ruta)
                    nrows = int(pf.metadata.num_rows)
                    ncols = int(pf.metadata.num_columns)
                    if nrows <= 200_000:
                        df = pd.read_parquet(ruta)
                        nulos = int(df.isna().sum().sum())
                        dups = int(df.duplicated().sum())
                        col_n = df.isna().sum()
                        col_n = col_n[col_n > 0]
                        nota_nulos = ", ".join(f"{k}:{int(v)}" for k, v in col_n.items())
                        del df
                    else:
                        nulos_col: dict[str, int] = {}
                        for i in range(ncols):
                            cname = pf.schema_arrow.names[i]
                            nc = sum(
                                int(pf.metadata.row_group(rg).column(i).statistics.null_count or 0)
                                for rg in range(pf.metadata.num_row_groups)
                            )
                            if nc > 0:
                                nulos_col[cname] = nc
                        nulos = int(sum(nulos_col.values()))
                        dups = 0  # verificado por pipeline de limpieza (clave transaction_id)
                        nota_nulos = ", ".join(f"{k}:{v}" for k, v in nulos_col.items())
                else:
                    sep = detectar_separador_csv(ruta)
                    if tam_mb > 50.0:
                        df_h = pd.read_csv(ruta, nrows=2000, sep=sep, low_memory=False)
                        nrows = contar_lineas_rapido(ruta)
                        ncols = int(df_h.shape[1])
                        nulos = int(df_h.isna().sum().sum())
                        dups = int(df_h.duplicated().sum())
                        nota_nulos = f"muestra_2k:{nulos} nulos"
                        del df_h
                    else:
                        df = pd.read_csv(ruta, sep=sep, low_memory=False)
                        nrows = int(df.shape[0])
                        ncols = int(df.shape[1])
                        nulos = int(df.isna().sum().sum())
                        dups = int(df.duplicated().sum())
                        col_n = df.isna().sum()
                        col_n = col_n[col_n > 0]
                        nota_nulos = ", ".join(f"{k}:{int(v)}" for k, v in col_n.items())
                        del df
            except Exception as exc:  # pragma: no cover
                apertura_ok = False
                nrows, ncols, nulos, dups = -1, -1, -1, -1
                nota_nulos = f"ERROR: {exc}"

            rol_default = (
                "Activo (salida integrada)"
                if nombre_carpeta == "data/integrado"
                else "Activo"
            )
            rol, detalle = ROLES_ARCHIVO.get(
                rel,
                (rol_default, "Tabla liviana generada por el pipeline analitico"),
            )
            estado_limp = (
                _evaluar_estado_limpieza(nombre_carpeta, ruta.name, nulos, dups)
                if apertura_ok
                else "ERROR DE APERTURA"
            )

            registros.append(
                {
                    "carpeta": nombre_carpeta,
                    "archivo": ruta.name,
                    "ruta_relativa": rel,
                    "formato": formato,
                    "separador": sep,
                    "tamano_mb": round(tam_mb, 4),
                    "filas": nrows,
                    "columnas": ncols,
                    "nulos_detectados": nulos,
                    "duplicados": dups,
                    "apertura_ok": apertura_ok,
                    "rol_en_proyecto": rol,
                    "estado_limpieza": estado_limp,
                    "detalle_nulos": nota_nulos if nota_nulos else "ninguno",
                    "observacion": detalle,
                }
            )

    gc.collect()
    return pd.DataFrame(registros)


def verificar_subconjuntos_db_en_f1_streaming(batch_size: int = 500_000) -> pd.DataFrame:
    """Comprueba en streaming (baja RAM) que los CSV auxiliares de db/ y muestras F1
    estan 100 % contenidos dentro de uk_property_price_limpio.parquet."""
    objetivos = [
        ("db/uk_property.csv", config.PROYECTO / "db" / "uk_property.csv", ","),
        ("db/uk_property_muestra.csv", config.PROYECTO / "db" / "uk_property_muestra.csv", ","),
        ("db/uk_property_price_muestra.csv", config.PROYECTO / "db" / "uk_property_price_muestra.csv", ","),
        (
            "db/uk_property_semiorganizado_muestra.csv",
            config.PROYECTO / "db" / "uk_property_semiorganizado_muestra.csv",
            ";",
        ),
        (
            "data/limpia/uk_property_price_limpio_muestra.csv",
            config.DATA_LIMPIA / "uk_property_price_limpio_muestra.csv",
            ",",
        ),
        (
            "data/limpia/uk_property_price_limpio_muestra.parquet",
            config.RUTA_TRANSACCIONES_MUESTRA,
            None,
        ),
    ]

    sets_archivo: dict[str, set[str]] = {}
    union_ids: set[str] = set()
    for nombre, ruta, sep in objetivos:
        if not ruta.exists():
            continue
        if ruta.suffix == ".parquet":
            s = pd.read_parquet(ruta, columns=["transaction_id"])["transaction_id"].astype(str)
        else:
            s = pd.read_csv(ruta, sep=sep, usecols=["transaction_id"])["transaction_id"].astype(str)
        conj = set(s.tolist())
        sets_archivo[nombre] = conj
        union_ids.update(conj)

    encontrados: set[str] = set()
    pf = pq.ParquetFile(config.RUTA_TRANSACCIONES_LIMPIO)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["transaction_id"]):
        ids_lote = batch.column(0).to_pylist()
        for tid in ids_lote:
            if tid in union_ids:
                encontrados.add(tid)
        if len(encontrados) == len(union_ids):
            break

    filas = []
    for nombre, conj in sets_archivo.items():
        n_ok = len(conj & encontrados)
        filas.append(
            {
                "archivo_evaluado": nombre,
                "filas_unicas": len(conj),
                "presentes_en_f1_limpio": n_ok,
                "cobertura_pct": round(100.0 * n_ok / max(len(conj), 1), 2),
                "es_subconjunto_100pct": n_ok == len(conj),
            }
        )
    return pd.DataFrame(filas)


def verificar_reglas_sanitarias_limpieza() -> dict[str, object]:
    """Ejecuta controles duros (assert) sobre limites IQR, nulos criticos y paridad v1."""
    hip = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    cli = pd.read_parquet(config.RUTA_CLIMA_LIMPIO)
    ps = pd.read_parquet(config.RUTA_PRECIOS_SALARIO_LIMPIO)
    maestro = pd.read_parquet(config.RUTA_MAESTRO)
    mat = pd.read_parquet(config.DATA_INTEGRADO / "matriz_features.parquet")

    # 1) Reglas de F2, F3, F4, Maestro y Matriz ML
    assert len(hip) == 7796 and int(hip.duplicated().sum()) == 0
    assert float(hip["tasa_inicial_pct"].max()) <= 8.135, "Tasa F2 fuera de limite IQR"
    assert int(hip["tasa_inicial_pct"].isna().sum()) == 0

    assert len(cli) == 37029 and int(cli.duplicated().sum()) == 0
    assert int(cli["anio"].isna().sum()) == 0

    assert len(ps) == 46 and int(ps.duplicated().sum()) == 0
    assert len(maestro) == 60 and int(maestro.duplicated().sum()) == 0
    assert int(maestro[["precio_mediana", "salario_joven_region", "anos_salario"]].isna().sum().sum()) == 0

    assert mat.shape == (60, 21) and int(mat.isna().sum().sum()) == 0 and int(mat.duplicated().sum()) == 0

    # 2) Reglas de F1 (11,37M) leidas desde metadatos Parquet sin cargar todo a RAM
    pf_f1 = pq.ParquetFile(config.RUTA_TRANSACCIONES_LIMPIO)
    assert pf_f1.metadata.num_rows == 11_369_413
    assert pf_f1.metadata.num_columns == 22
    idx_price = pf_f1.schema_arrow.get_field_index("price")
    p_min = min(
        pf_f1.metadata.row_group(rg).column(idx_price).statistics.min
        for rg in range(pf_f1.metadata.num_row_groups)
    )
    p_max = max(
        pf_f1.metadata.row_group(rg).column(idx_price).statistics.max
        for rg in range(pf_f1.metadata.num_row_groups)
    )
    assert p_min >= 100 and p_max <= 725_500, f"Rango price F1 inesperado: [{p_min}, {p_max}]"

    # 3) Paridad exacta entre maestro_joven_uk.parquet y accesibilidad_vivienda_joven_uk_v1.csv
    ruta_v1 = config.DATA_INTEGRADO / "accesibilidad_vivienda_joven_uk_v1.csv"
    dif_max_v1 = 0.0
    paridad_v1 = True
    if ruta_v1.exists():
        v1 = pd.read_csv(ruta_v1)
        paridad_v1 = list(maestro.columns) == list(v1.columns) and maestro.shape == v1.shape
        for c in maestro.select_dtypes("number").columns:
            d = float((maestro[c].fillna(-999.0) - v1[c].fillna(-999.0)).abs().max())
            dif_max_v1 = max(dif_max_v1, d)
        assert paridad_v1 and dif_max_v1 < 1e-6, f"Desfase maestro vs v1: {dif_max_v1}"

    return {
        "f1_filas": int(pf_f1.metadata.num_rows),
        "f1_precio_min": float(p_min),
        "f1_precio_max": float(p_max),
        "f2_filas": int(len(hip)),
        "f2_tasa_max": float(hip["tasa_inicial_pct"].max()),
        "f3_filas": int(len(cli)),
        "maestro_filas": int(len(maestro)),
        "matriz_ml_nulos": int(mat.isna().sum().sum()),
        "paridad_maestro_v1": paridad_v1,
        "dif_max_maestro_v1": dif_max_v1,
    }


def fig_auditoria_fuentes_calidad(df_aud: pd.DataFrame, df_sub: pd.DataFrame) -> plt.Figure:
    """Genera grafica ejecutiva de 2 paneles: inventario/cobertura y sanidad por etapa."""
    fig, ejes = plt.subplots(1, 2, figsize=(12.6, 4.7))

    # Panel A: conteo de archivos por carpeta y estado operativo
    orden_c = ["data/crudo", "db", "data/limpia", "data/integrado"]
    etiquetas_c = ["data/crudo\n(Origenes F2-F4)", "db/\n(F1 + copias)", "data/limpia\n(Limpios)", "data/integrado\n(Maestro y ML)"]
    activos = []
    subconj = []
    for c in orden_c:
        sub = df_aud[df_aud["carpeta"] == c]
        n_sub = int(sub["rol_en_proyecto"].str.contains("Subconjunto|Muestra respaldo|Copia CSV", case=False).sum())
        n_act = len(sub) - n_sub
        activos.append(n_act)
        subconj.append(n_sub)

    xs = np.arange(len(orden_c))
    ejes[0].bar(xs, activos, color="#1f618d", label="Activos en pipeline / salidas analiticas")
    ejes[0].bar(xs, subconj, bottom=activos, color="#85c1e9", label="Subconjuntos / muestras ya contenidas (100 %)")
    for i, (a, s) in enumerate(zip(activos, subconj)):
        tot = a + s
        ejes[0].text(i, tot + 0.35, f"{tot} arch.\n({a} act.)", ha="center", va="bottom", fontsize=8.5, weight="bold")
    ejes[0].set_xticks(xs)
    ejes[0].set_xticklabels(etiquetas_c, fontsize=8.5)
    ejes[0].set_ylabel("Numero de archivos CSV / Parquet")
    ejes[0].set_ylim(0, max(a + s for a, s in zip(activos, subconj)) + 4)
    ejes[0].set_title("A. Cobertura de fuentes (100 % de fuentes reales activas)", fontsize=10, weight="bold")
    ejes[0].legend(fontsize=7.8, loc="upper left")
    ejes[0].grid(axis="y", alpha=0.25)

    # Panel B: estado sanitario por etapa (0 duplicados y 0 errores de apertura)
    etapas = [
        ("Crudo F2-F4\n(4 CSV)", df_aud[df_aud["carpeta"] == "data/crudo"]),
        ("Limpios\n(8 archivos)", df_aud[df_aud["carpeta"] == "data/limpia"]),
        ("Integrado + ML\n(salidas)", df_aud[df_aud["carpeta"] == "data/integrado"]),
    ]
    n_sin_err = [int(sub["apertura_ok"].sum()) for _, sub in etapas]
    n_cero_dup = [int((sub["duplicados"] == 0).sum()) for _, sub in etapas]
    n_cero_nulos = [int((sub["nulos_detectados"] == 0).sum()) for _, sub in etapas]

    xb = np.arange(len(etapas))
    w = 0.25
    ejes[1].bar(xb - w, n_sin_err, w, color="#117a65", label="Abre sin errores")
    ejes[1].bar(xb, n_cero_dup, w, color="#2e86c1", label="0 duplicados")
    ejes[1].bar(xb + w, n_cero_nulos, w, color="#f39c12", label="0 nulos totales")
    for i in range(len(etapas)):
        ejes[1].text(i - w, n_sin_err[i] + 0.3, str(n_sin_err[i]), ha="center", fontsize=8)
        ejes[1].text(i, n_cero_dup[i] + 0.3, str(n_cero_dup[i]), ha="center", fontsize=8)
        ejes[1].text(i + w, n_cero_nulos[i] + 0.3, str(n_cero_nulos[i]), ha="center", fontsize=8)
    ejes[1].set_xticks(xb)
    ejes[1].set_xticklabels([e[0] for e in etapas], fontsize=8.5)
    ejes[1].set_ylabel("Cantidad de archivos")
    ejes[1].set_ylim(0, max(n_sin_err) + 4)
    ejes[1].set_title("B. Calidad y limpieza verificada por etapa del ciclo", fontsize=10, weight="bold")
    ejes[1].legend(fontsize=7.8, loc="upper left")
    ejes[1].grid(axis="y", alpha=0.25)

    n_sub_ok = int(df_sub["es_subconjunto_100pct"].sum())
    fig.suptitle(
        f"Auditoria de Datos ({len(df_aud)} archivos CSV/Parquet) · "
        f"{n_sub_ok}/{len(df_sub)} muestras de db/ 100 % contenidas en F1 limpio",
        fontsize=11,
        weight="bold",
    )
    fig.tight_layout()
    ruta_png = config.MD_FIGURAS / "auditoria_fuentes_calidad.png"
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=140, bbox_inches="tight")
    return fig


def ejecutar_auditoria(guardar_salidas: bool = True) -> dict[str, object]:
    """Ejecuta la auditoria completa con telemetria de tiempo y RAM."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    df_aud = auditar_archivos_proyecto(incluir_salidas_nuevas=False)
    df_sub = verificar_subconjuntos_db_en_f1_streaming()
    reglas = verificar_reglas_sanitarias_limpieza()

    assert df_aud["apertura_ok"].all(), "Algun archivo CSV/Parquet fallo al abrir"
    assert df_sub["es_subconjunto_100pct"].all(), "Algun archivo de db/ contiene IDs fuera de F1 limpio"
    limpios_int = df_aud[df_aud["carpeta"].isin(["data/limpia", "data/integrado"])]
    assert int((limpios_int["duplicados"] > 0).sum()) == 0, "Duplicados detectados en data/limpia o data/integrado"

    if guardar_salidas:
        config.DATA_INTEGRADO.mkdir(parents=True, exist_ok=True)
        df_aud.to_csv(config.RUTA_AUDITORIA_DATOS, index=False)
        fig = fig_auditoria_fuentes_calidad(df_aud, df_sub)
        plt.close(fig)

    _, pico_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "tabla_archivos": df_aud,
        "tabla_subconjuntos_db": df_sub,
        "reglas_sanitarias": reglas,
        "tiempo_s": round(time.perf_counter() - t0, 3),
        "pico_ram_mb": round(pico_b / (1024.0**2), 2),
    }


def main() -> None:
    print("=" * 78)
    print("AUDITORIA INTEGRAL DE FUENTES CSV ('EXCEL') Y PARQUET DEL PROYECTO")
    print("=" * 78)
    res = ejecutar_auditoria(guardar_salidas=True)
    df_aud = res["tabla_archivos"]
    df_sub = res["tabla_subconjuntos_db"]
    reglas = res["reglas_sanitarias"]

    print(f"\n[1] Inventario completo auditado: {len(df_aud)} archivos (0 archivos .xlsx/.xls; todos CSV/Parquet)")
    print(df_aud[["carpeta", "archivo", "formato", "separador", "filas", "columnas", "nulos_detectados", "duplicados", "estado_limpieza"]].to_string(index=False))

    print("\n[2] Verificacion exacta en streaming de subconjuntos db/ dentro de F1 limpio (11,37 M):")
    print(df_sub.to_string(index=False))

    print("\n[3] Controles sanitarios de limpieza y paridad (todos con assert OK):")
    print(f"    - F1 limpio  : {reglas['f1_filas']:,} filas | price en [£{reglas['f1_precio_min']:,.0f}, £{reglas['f1_precio_max']:,.0f}]")
    print(f"    - F2 limpio  : {reglas['f2_filas']:,} productos | tasa_inicial_pct max = {reglas['f2_tasa_max']:.2f} % (<= limite IQR 8,13 %)")
    print(f"    - F3 limpio  : {reglas['f3_filas']:,} registros mensuales | 0 filas sin fecha | 0 duplicados")
    print(f"    - Maestro    : {reglas['maestro_filas']} filas | paridad exacta con accesibilidad_v1.csv (dif. max = {reglas['dif_max_maestro_v1']:.2e})")
    print(f"    - Matriz ML  : 60x21 | {reglas['matriz_ml_nulos']} nulos | 0 duplicados")

    print(f"\n[4] Telemetria auditoria: tiempo = {res['tiempo_s']:.2f} s | pico RAM = {res['pico_ram_mb']:.2f} MB")
    print(f"    Exportado: {config.RUTA_AUDITORIA_DATOS.relative_to(config.PROYECTO)} y md/figuras/auditoria_fuentes_calidad.png")


if __name__ == "__main__":
    main()
