"""Preparacion de la matriz X·y para el modelado (transformacion y reduccion).

Aplica el metodo del proyecto sobre el cuadro maestro (60 filas):
  - objetivo y: esfuerzo hipotecario por region y anio (cuota PMT sobre el
    precio mediano / salario joven mensual; tasa winsorizada y comisiones de
    la oferta 2022, 90 % LTV y 25 anios);
  - rasgos: 9 numericos estandarizados con StandardScaler (media 0, desv 1);
  - region nominal: one-hot con OneHotEncoder y drop_first (9 columnas);
  - nulos neutrales: sol_anom y precio_var_pct (primer anio) a 0;
  - fuera de X: anio y region (agrupadores), n_transacciones, precio_medio y
    las escalas derivadas del salario;
  - seccion [7]: comparacion empirica de metodos con cifras (perfil de los
    rasgos, distorsion de valores extremos, PCA por escalador, one-hot vs
    ordinal con VIF y R2, discretizacion KBins y diagnostico supervisado con
    validacion cruzada de 5 pliegues en pipeline);
  - salida: data/integrado/matriz_features.parquet (+ muestra de 10 filas).

El pipeline usa ColumnTransformer (escala + one-hot en un solo estimador), el
mismo patron del modelado donde el ajuste ira dentro de cada pliegue (sin fuga
de datos). Insumo: config.RUTA_MAESTRO y RUTA_HIPOTECAS_LIMPIO.
"""
# Ejecutar:  python script/proyecto/matriz_xy.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, shapiro, skew, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

if __package__ in (None, ""):  # lanzado como script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    del sys, Path

from proyecto import config, indicadores

RASGOS_NUMERICOS = [
    "precio_mediana", "salario_joven_region", "tmax_anom", "tmin_anom",
    "lluvia_anom", "sol_anom", "af_anom", "precio_var_pct", "anos_salario",
]


def cargar_insumo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el cuadro maestro (60 filas) y la oferta hipotecaria ya limpia."""
    maestro = pd.read_parquet(config.RUTA_MAESTRO).sort_values(["region", "anio"])
    hipotecas = pd.read_parquet(config.RUTA_HIPOTECAS_LIMPIO)
    return maestro, hipotecas


def agregar_esfuerzo(maestro: pd.DataFrame, hipotecas: pd.DataFrame
                     ) -> tuple[pd.DataFrame, float, float]:
    """Objetivo y: cuota mensual PMT sobre el precio mediano / salario joven
    mensual. Usa la tasa inicial media winsorizada y las comisiones promedio
    de la oferta 2022 (config: 90 % LTV, 25 anios)."""
    tasa = float(hipotecas["tasa_inicial_pct"].mean())
    comisiones = float(hipotecas["comisiones_total"].mean())
    df = maestro.copy()
    cuota = df["precio_mediana"].apply(
        lambda p: indicadores.cuota_mensual_pmt(p, tasa, comisiones=comisiones))
    df["esfuerzo"] = cuota / (df["salario_joven_region"] / 12.0)
    return df, tasa, comisiones


def completar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 0 como valor neutral: anomalia nula (sin anomalia) o variacion
    nula en el primer anio de la serie (no hay anio previo que comparar)."""
    m = df.copy()
    for col in ("sol_anom", "precio_var_pct"):
        m[col] = m[col].fillna(0.0)
    return m


def construir_matriz(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Arma X con ColumnTransformer (StandardScaler + OneHotEncoder) y y
    (esfuerzo): escala y codifica en un solo estimador, el mismo patron que
    se reajustara dentro de cada pliegue en el modelado.

    El one-hot usa `drop="first"`: sin el, las 10 columnas dummy de region
    suman siempre 1 y quedan linealmente dependientes entre si (VIF
    infinito). Con 9, la referencia es implicita y los coeficientes de la
    regresion se pueden interpretar. Seccion [7] de este modulo mide el
    contraste: VIF infinito con 10 dummies frente a VIF maximo 1,8 con 9.
    """
    preprocesador = ColumnTransformer(
        transformers=[
            ("escala", StandardScaler(), RASGOS_NUMERICOS),
            ("region", OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                                     drop="first"), ["region"]),
        ],
        remainder="drop",
    )
    salida = preprocesador.fit_transform(df[RASGOS_NUMERICOS + ["region"]])
    # con drop="first" la salida tiene 9 columnas pero categories_ conserva las
    # 10: hay que quitar la categoria descartada para que los nombres cuadren.
    codificador = preprocesador.named_transformers_["region"]
    categorias = list(codificador.categories_[0])
    if getattr(codificador, "drop_idx_", None) is not None:
        categorias.pop(int(codificador.drop_idx_[0]))
    col_region = [f"region_{v}" for v in categorias]
    x = pd.DataFrame(salida,
                     columns=[f"z_{c}" for c in RASGOS_NUMERICOS] + col_region,
                     index=df.index)
    y = df["esfuerzo"].astype(float).rename("esfuerzo")
    return x, y


def guardar_matriz(x: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> None:
    """Valida la matriz y la guarda en parquet (completa y muestra de 10)."""
    assert len(x) == len(y) == len(df) == 60, "la matriz no tiene 60 filas"
    assert int(x.isna().sum().sum()) == 0, "la matriz X tiene nulos"
    assert bool(y.gt(0).all()), "el esfuerzo debe ser positivo"
    # con drop="first" el one-hot aporta n_regiones - 1 columnas, no n_regiones
    esperado = len(RASGOS_NUMERICOS) + df["region"].nunique() - 1
    assert x.shape[1] == esperado, (
        f"numero de columnas (rasgos + one-hot) incorrecto: "
        f"esperado {esperado}, obtenido {x.shape[1]}")
    ruta = config.DATA_INTEGRADO / "matriz_features.parquet"
    ruta_muestra = config.DATA_INTEGRADO / "matriz_features_muestra.parquet"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    matriz = pd.concat([df[["region", "anio"]], x, y], axis=1)
    matriz.to_parquet(ruta, index=False)
    matriz.head(10).to_parquet(ruta_muestra, index=False)
    print(f"  guardado {str(ruta.relative_to(config.PROYECTO))} "
          f"({matriz.shape})")
    print(f"  guardado {str(ruta_muestra.relative_to(config.PROYECTO))} "
          f"({matriz.head(10).shape})")


# ---------------------------------------------------------------------------
# [7] Comparacion empirica de metodos: probar y comparar con cifras
# ---------------------------------------------------------------------------


def _vif_columnas(X: pd.DataFrame) -> pd.Series:
    """VIF sin statsmodels: 1/(1-R2) de regresar la columna j sobre las demas.
    R2 = 1 (dummy trap) implica VIF infinito."""
    vifs = {}
    for col in X.columns:
        otros = X.drop(columns=[col])
        r2 = LinearRegression().fit(otros, X[col]).score(otros, X[col])
        vifs[col] = float("inf") if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return pd.Series(vifs, name="VIF")


def comparar_escaladores(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Reune en UNA tabla las tres pruebas que justifican elegir un escalador.

    1. PERFIL:     ¿las variables se parecen a una normal? (Shapiro-Wilk, alpha=0,05)
    2. DISTORSION: ¿un extremo destruye la escala? Se inyecta precio x100 en una
                   fila y se mide cuanto se comprime el rango de las 59 normales.
    3. PREDICCION: ¿cual escala mejor el modelo? R2 de CV-5 (sin fuga) por modelo.

    Devuelve una tabla con una fila por escalador, para leer de un vistazo cual
    gana en cada prueba. Las tres pruebas MIRAN COSAS DISTINTAS: por eso el
    ganador puede cambiar de una a otra, y eso hay que decirlo explicitamente.
    """
    dist = probar_distorsion(df).set_index("escalador")
    diag = diagnostico_supervisado(df, y)
    perfil = perfilar_rasgos(df)

    filas = []
    for nombre in ("MinMaxScaler", "StandardScaler", "RobustScaler"):
        corto = {"MinMaxScaler": "MinMax", "StandardScaler": "Standard",
                 "RobustScaler": "Robust"}[nombre]
        sub = diag[(diag["escalador"] == corto) & (diag["codificacion"] == "one-hot")]
        if sub.empty:
            continue
        filas.append({
            "escalador": nombre,
            "compresion_extremo_pct": float(dist.loc[nombre, "compresion_pct"]),
            "r2_knn": float(sub[sub["modelo"] == "KNN5"]["r2_cv5"].iloc[0]),
            "r2_rl": float(sub[sub["modelo"] == "RL"]["r2_cv5"].iloc[0]),
            "rmse_knn": float(sub[sub["modelo"] == "KNN5"]["rmse_cv5"].iloc[0]),
        })

    # "sin escalar" no pasa por probar_distorsion (no escala), pero si aparece
    # en diagnostico_supervisado y es el dato honesto del comparativo.
    sub = diag[(diag["escalador"] == "sin escalar") & (diag["codificacion"] == "one-hot")]
    if not sub.empty:
        filas.append({
            "escalador": "sin escalar",
            "compresion_extremo_pct": float("nan"),
            "r2_knn": float(sub[sub["modelo"] == "KNN5"]["r2_cv5"].iloc[0]),
            "r2_rl": float(sub[sub["modelo"] == "RL"]["r2_cv5"].iloc[0]),
            "rmse_knn": float(sub[sub["modelo"] == "KNN5"]["rmse_cv5"].iloc[0]),
        })

    tabla = pd.DataFrame(filas).set_index("escalador")

    # Conclusion explicita, no deja que el lector tenga que deducirla.
    mejor_robustez = tabla["compresion_extremo_pct"].idxmin()
    mejor_knn = tabla["r2_knn"].idxmax()
    mejor_rl = tabla["r2_rl"].idxmax()
    tabla.attrs["conclusion"] = (
        f"Mas robusto ante un extremo: {mejor_robustez} | "
        f"Mejor en KNN: {mejor_knn} | Mejor en regresion: {mejor_rl}")
    tabla.attrs["n_rasgos_normales"] = int((perfil["shapiro_p"] >= 0.05).sum())
    tabla.attrs["n_rasgos"] = int(len(perfil))
    return tabla


def perfilar_rasgos(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil estadistico de los 9 rasgos: asimetria (skewness de Pearson),
    curtosis de exceso (normal=0), % de outliers Tukey (1.5 x IQR) y p-valor
    de Shapiro-Wilk (normalidad)."""
    filas = []
    for col in RASGOS_NUMERICOS:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        filas.append({
            "rasgo": col,
            "asimetria": float(skew(s)),
            "curtosis": float(kurtosis(s)),
            "outliers_tukey": n_out,
            "pct_tukey": 100 * n_out / len(s),
            "shapiro_p": float(shapiro(s)[1]) if s.nunique() > 1 else float("nan"),
        })
    return pd.DataFrame(filas).set_index("rasgo")


def probar_distorsion(df: pd.DataFrame) -> pd.DataFrame:
    """Prueba de distorsion: inyecta un valor extremo artificial (precio_mediana
    x100 en 1 fila) y mide cuanto se comprime el rango que queda para las 59
    filas normales con cada escalador (compresion % = 1 - rango_despues /
    rango_antes, en la columna escalada)."""
    X = df[RASGOS_NUMERICOS].copy()
    X_ext = X.copy()
    X_ext.loc[X_ext.index[0], "precio_mediana"] *= 100.0
    normales = X.index[1:]
    filas = []
    for nombre, escala in [("MinMaxScaler", MinMaxScaler()),
                           ("StandardScaler", StandardScaler()),
                           ("RobustScaler", RobustScaler())]:
        a = pd.DataFrame(escala.fit_transform(X), columns=RASGOS_NUMERICOS,
                         index=X.index)
        b = pd.DataFrame(escala.fit_transform(X_ext), columns=RASGOS_NUMERICOS,
                         index=X.index)
        col = "precio_mediana"
        rango_antes = float(a.loc[normales, col].max() - a.loc[normales, col].min())
        rango_despues = float(b.loc[normales, col].max() - b.loc[normales, col].min())
        filas.append({
            "escalador": nombre,
            "rango_antes": rango_antes,
            "rango_despues": rango_despues,
            "compresion_pct": 100 * (1 - rango_despues / rango_antes),
            "distorsion_media": float((b.loc[normales, col]
                                       - a.loc[normales, col]).abs().mean()),
        })
    return pd.DataFrame(filas)


def comparar_pca(df: pd.DataFrame) -> pd.DataFrame:
    """PCA sobre los 9 rasgos con cada forma de escalar: varianza explicada por
    la PC1, acumulada de 2 y 3 componentes (%) y minimos de componentes para
    alcanzar 80 % y 90 % de varianza acumulada."""
    filas = []
    for nombre, escala in [("sin escalar", "passthrough"),
                           ("MinMaxScaler", MinMaxScaler()),
                           ("StandardScaler", StandardScaler()),
                           ("RobustScaler", RobustScaler())]:
        if nombre == "sin escalar":
            Z = df[RASGOS_NUMERICOS].to_numpy(dtype=float)
        else:
            Z = escala.fit_transform(df[RASGOS_NUMERICOS])
        razones = PCA().fit(Z).explained_variance_ratio_
        acum = np.cumsum(razones)
        filas.append({
            "escalador": nombre,
            "pc1_pct": 100 * float(razones[0]),
            "acum2_pct": 100 * float(acum[1]),
            "acum3_pct": 100 * float(acum[2]),
            "comp_80": int(np.searchsorted(acum, 0.80) + 1),
            "comp_90": int(np.searchsorted(acum, 0.90) + 1),
        })
    return pd.DataFrame(filas)


def comparar_codificacion(df: pd.DataFrame, y: pd.Series
                          ) -> tuple[pd.DataFrame, dict]:
    """One-hot vs ordinal para la region: (1) r de Pearson del codigo ordinal
    con y bajo dos ordenamientos (muestra el orden falso); (2) VIF de las 10
    dummies (dummy trap) vs 9 con la primera eliminada; (3) R2 y RMSE de
    regresion lineal con validacion cruzada de 5 pliegues por codificacion."""
    regiones = df["region"].astype(str)
    codes = pd.Categorical(regiones).codes.astype(float)
    r_asc = float(np.corrcoef(codes, y)[0, 1])
    r_desc = float(np.corrcoef(-codes, y)[0, 1])

    dummies = pd.get_dummies(regiones, dtype=float)
    vif_10 = _vif_columnas(dummies)
    vif_9 = _vif_columnas(dummies.drop(columns=[dummies.columns[0]]))

    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    filas = []
    for nombre, enc in [("one-hot", OneHotEncoder(sparse_output=False,
                                                  handle_unknown="ignore")),
                        ("ordinal", OrdinalEncoder())]:
        pipe = Pipeline([
            ("pre", ColumnTransformer([("cat", enc, ["region"])])),
            ("rl", LinearRegression()),
        ])
        r2 = cross_val_score(pipe, df[["region"]], y, cv=kf, scoring="r2")
        rmse = np.sqrt(-cross_val_score(pipe, df[["region"]], y, cv=kf,
                                        scoring="neg_mean_squared_error"))
        filas.append({"codificacion": nombre, "r2_cv5": float(r2.mean()),
                      "r2_desv": float(r2.std()), "rmse_cv5": float(rmse.mean())})

    extras = {
        "r_ordinal_asc": r_asc,
        "r_ordinal_desc": r_desc,
        "vif_max_10": float(vif_10.max()),
        "vif_max_9": float(vif_9.max()),
        "vif_mediana_9": float(vif_9.median()),
    }
    return pd.DataFrame(filas), extras


def comparar_discretizacion(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """KBinsDiscretizer en 4 tramos, tres estrategias (uniform, quantile,
    kmeans) sobre dos rasgos: cuanto del valor original conserva el tramo
    (rho de Spearman, rho^2 en %) y cuanta informacion sobre y retiene la
    version discreta vs la continua (mutual information, %)."""
    filas = []
    for col in ["precio_mediana", "anos_salario"]:
        s = df[[col]]
        mi_continua = float(mutual_info_regression(s, y, random_state=config.SEED)[0])
        for estrategia in ["uniform", "quantile", "kmeans"]:
            kb = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy=estrategia,
                                  subsample=None)
            bins = kb.fit_transform(s).ravel()
            rho = float(spearmanr(s[col], bins).statistic)
            mi_bins = float(mutual_info_regression(
                bins.reshape(-1, 1), y, random_state=config.SEED)[0])
            filas.append({
                "rasgo": col,
                "estrategia": estrategia,
                "rho_spearman": rho,
                "rho2_pct": 100 * rho ** 2,
                "mi_continua": mi_continua,
                "mi_tramo": mi_bins,
                "mi_retenida_pct": 100 * mi_bins / mi_continua,
            })
    return pd.DataFrame(filas)


def diagnostico_supervisado(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """La prueba de fuego: 4 escaladores x 2 codificaciones x 2 modelos, todos
    en pipeline (el escalador se ajusta dentro de cada pliegue: sin fuga),
    validacion cruzada de 5 pliegues con semilla fija. Reporta R2 medio +/- desv
    y RMSE medio; al final, la variante Standard + PCA(80 %) + one-hot + RL."""
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    escaladores = {"MinMax": MinMaxScaler(), "Standard": StandardScaler(),
                   "Robust": RobustScaler(), "sin escalar": "passthrough"}
    encoders = {"one-hot": OneHotEncoder(sparse_output=False,
                                         handle_unknown="ignore"),
                "ordinal": OrdinalEncoder()}
    modelos = {"RL": LinearRegression(), "KNN5": KNeighborsRegressor(n_neighbors=5)}
    columnas = RASGOS_NUMERICOS + ["region"]
    filas = []
    for ne, esc in escaladores.items():
        for nc, enc in encoders.items():
            pre = ColumnTransformer([("num", esc, RASGOS_NUMERICOS),
                                     ("cat", enc, ["region"])])
            for nm, modelo in modelos.items():
                pipe = Pipeline([("pre", pre), ("modelo", modelo)])
                r2 = cross_val_score(pipe, df[columnas], y, cv=kf, scoring="r2")
                rmse = np.sqrt(-cross_val_score(pipe, df[columnas], y, cv=kf,
                                                scoring="neg_mean_squared_error"))
                filas.append({"escalador": ne, "codificacion": nc, "modelo": nm,
                              "r2_cv5": float(r2.mean()), "r2_desv": float(r2.std()),
                              "rmse_cv5": float(rmse.mean())})

    pre_pca = ColumnTransformer([("num", StandardScaler(), RASGOS_NUMERICOS),
                                 ("cat", OneHotEncoder(sparse_output=False,
                                                       handle_unknown="ignore"),
                                  ["region"])])
    pipe_pca = Pipeline([("pre", pre_pca), ("pca", PCA(n_components=0.80)),
                         ("modelo", LinearRegression())])
    r2 = cross_val_score(pipe_pca, df[columnas], y, cv=kf, scoring="r2")
    rmse = np.sqrt(-cross_val_score(pipe_pca, df[columnas], y, cv=kf,
                                    scoring="neg_mean_squared_error"))
    filas.append({"escalador": "Standard+PCA80%", "codificacion": "one-hot",
                  "modelo": "RL", "r2_cv5": float(r2.mean()),
                  "r2_desv": float(r2.std()), "rmse_cv5": float(rmse.mean())})

    return (pd.DataFrame(filas)
            .sort_values("r2_cv5", ascending=False)
            .reset_index(drop=True))


def seccion_comparacion(df: pd.DataFrame, y: pd.Series) -> None:
    """Ejecuta y anuncia la seccion [7] completa de comparacion empirica."""
    print("\n" + "=" * 72)
    print("[7] COMPARACION EMPIRICA DE METODOS (probar y comparar con cifras)")
    print("=" * 72)

    print("\n[7.1] Perfil de los 9 rasgos (skewness, curtosis, Tukey %, Shapiro)")
    perfil = perfilar_rasgos(df)
    print(perfil.round(3).to_string())
    n_norm = int((perfil["shapiro_p"] >= 0.05).sum())
    n_out = int((perfil["outliers_tukey"] > 0).sum())
    print(f"  -> asimetria media {perfil['asimetria'].abs().mean():.2f} | "
          f"Shapiro p>=0.05 en {n_norm}/{len(perfil)} rasgos | "
          f"con outliers Tukey: {n_out}/{len(perfil)}")

    print("\n[7.2] Distorsion de un valor extremo inyectado (precio x100, 1 fila)")
    dist = probar_distorsion(df)
    print(dist.round(3).to_string(index=False))
    peor = dist.loc[dist["compresion_pct"].idxmax()]
    mejor = dist.loc[dist["compresion_pct"].idxmin()]
    print(f"  -> peor: {peor['escalador']} comprime el rango normal "
          f"{peor['compresion_pct']:.1f} % | mejor: {mejor['escalador']} solo "
          f"{mejor['compresion_pct']:.1f} %")

    print("\n[7.3] PCA por escalador (varianza explicada, %)")
    pca_tab = comparar_pca(df)
    print(pca_tab.round(1).to_string(index=False))

    print("\n[7.4] Codificacion de la region: one-hot vs ordinal")
    cod, extras = comparar_codificacion(df, y)
    print(f"  r(ordinal alfabetico, y) = {extras['r_ordinal_asc']:+.3f} | "
          f"con el orden invertido = {extras['r_ordinal_desc']:+.3f} "
          f"(mismo dato, signo opuesto: el orden es inventado)")
    print(f"  VIF con las 10 dummies: max {extras['vif_max_10']:.0f} (dummy trap) | "
          f"con 9 (drop_first): max {extras['vif_max_9']:.1f}, "
          f"mediana {extras['vif_mediana_9']:.1f}")
    print(cod.round(3).to_string(index=False))

    print("\n[7.5] Discretizacion KBins (4 tramos):informacion retenida")
    disc = comparar_discretizacion(df, y)
    print(disc.round(2).to_string(index=False))

    print("\n[7.6] Diagnostico supervisado: pipelines con CV-5 (sin fuga)")
    sup = diagnostico_supervisado(df, y)
    print(sup.round(3).to_string(index=False))
    mejor, peor = sup.iloc[0], sup.iloc[-1]
    delta = 100 * (mejor["r2_cv5"] - peor["r2_cv5"])
    print(f"  -> mejor: {mejor['escalador']} + {mejor['codificacion']} + "
          f"{mejor['modelo']} (R2 {mejor['r2_cv5']:.3f}) | peor: {peor['escalador']} "
          f"+ {peor['codificacion']} + {peor['modelo']} (R2 {peor['r2_cv5']:.3f}): "
          f"diferencia {delta:.1f} puntos de R2")


def main() -> None:
    """Ejecuta la preparacion de la matriz X·y (metodo usado en el modelado)."""
    print("=" * 72)
    print("PREPARACION DE LA MATRIZ X·y (para el modelado:"
          " transformacion y reduccion)")
    print("=" * 72)

    # 1. insumo y nulos
    maestro, hipotecas = cargar_insumo()
    print("\n[1] Insumo maestro_joven_uk")
    print(f"  filas {len(maestro)} | anios {maestro['anio'].min()}-"
          f"{maestro['anio'].max()} | regiones {maestro['region'].nunique()}")
    columnas = ["tmax_anom", "tmin_anom", "lluvia_anom", "sol_anom",
                "af_anom", "precio_var_pct", "anos_salario"]
    print("  nulos en candidatos a rasgos:")
    print(maestro[columnas].isna().sum().to_string())

    df = completar_nulos(maestro)
    df, tasa, comisiones = agregar_esfuerzo(df, hipotecas)
    print(f"  tasa winsorizada {tasa:.2f} % | comisiones GBP {comisiones:,.0f} | "
          f"90 % LTV, 25 anios")

    # 2. objetivo y
    print("\n[2] Objetivo y = esfuerzo (cuota mensual / salario joven mensual)")
    for region, valor in (df.groupby("region", observed=True)["esfuerzo"]
                          .mean().sort_values().items()):
        estado = ("accesible" if valor <= config.UMBRAL_ACCESIBILIDAD
                  else "no accesible")
        print(f"  {region:<26} esfuerzo medio {valor:.3f} -> {estado}")
    print(f"  umbral {config.UMBRAL_ACCESIBILIDAD:.0%} "
          f"(config.UMBRAL_ACCESIBILIDAD)")

    # 3. metodo: escalado + one-hot
    print("\n[3] Metodo: ColumnTransformer (StandardScaler + OneHotEncoder)")
    x, y = construir_matriz(df)
    z = x[[f"z_{c}" for c in RASGOS_NUMERICOS]].to_numpy(dtype=float)
    print(f"  StandardScaler tras ajustar: media {z.mean():.4f} / "
          f"desv {z.std():.3f} (media 0, desv 1)")
    print(f"  one-hot de region: {int(x.shape[1]) - len(RASGOS_NUMERICOS)} "
          f"columnas (OneHotEncoder)")

    # 4. matriz X·y
    print("\n[4] Matriz X·y construida")
    print(f"  X {x.shape} | y {y.shape[0]} | nulos en X: "
          f"{int(x.isna().sum().sum())}")
    print(f"  esfuerzo: media {y.mean():.3f} | rango "
          f"{y.min():.3f}-{y.max():.3f}")
    print("\n  Matriz y (esfuerzo) por region y anio: 10 regiones x 6 anios")
    tab = y.to_frame().assign(region=df["region"], anio=df["anio"].astype(int))
    print(tab.pivot(index="region", columns="anio", values="esfuerzo")
          .round(3).to_string())
    print("\n  Distribucion del esfuerzo (y):")
    tramos_y = pd.cut(y, bins=[0, 0.3, 0.5, 0.7, 1.0, 2.0], right=False)
    print(tramos_y.value_counts(normalize=True).sort_index().round(3).to_string())

    # 5. resultados clave
    print("\n[5] Resultados clave")
    print(f"  Matriz X·y: {x.shape[0]} filas x {x.shape[1]} columnas X "
          f"(9 z_ + 10 region_) | nulos {int(x.isna().sum().sum())}")
    print(f"  y: media {y.mean():.3f} | mediana {y.median():.3f} | "
          f"Q1 {y.quantile(0.25):.3f} | Q3 {y.quantile(0.75):.3f} | "
          f"rango {y.min():.3f}-{y.max():.3f}")
    print("\n  Esfuerzo medio por anio (todas las regiones):")
    print(df.groupby("anio")["esfuerzo"].mean().round(3).to_string())
    print("\n  Zonas de mayor esfuerzo (top 3):")
    print(df.nlargest(3, "esfuerzo").loc[:, ["region", "anio", "esfuerzo"]]
          .round(3).to_string(index=False))
    print("  Zonas de menor esfuerzo (bottom 3):")
    print(df.nsmallest(3, "esfuerzo").loc[:, ["region", "anio", "esfuerzo"]]
          .round(3).to_string(index=False))
    print(f"  Frecuencias de regiones-anio: >=0.7 -> {(y >= 0.7).sum()} | "
          f">=1.0 -> {(y >= 1.0).sum()} (de {len(y)})")

    # 6. clima y zona (asociacion Pearson, no causalidad)
    print("\n[6] Clima y zona: asociacion con la hipoteca y el precio")
    clima = ["tmax_anom", "tmin_anom", "lluvia_anom", "sol_anom", "af_anom"]
    corr_clima = pd.DataFrame({
        "hipoteca (esfuerzo)": [df["esfuerzo"].corr(df[c]) for c in clima],
        "precio_var_pct": [df["precio_var_pct"].corr(df[c]) for c in clima],
    }, index=clima).round(3)
    print("  Correlacion de cada anomalia climatica (Pearson):")
    print(corr_clima.to_string())
    signos = [(region, g["lluvia_anom"].corr(g["precio_var_pct"]))
              for region, g in df.groupby("region", observed=True)]
    rz = pd.DataFrame(signos, columns=["region", "r_lluvia"])
    neg = int((rz["r_lluvia"] < 0).sum())
    print(f"\n  Lluvia vs variacion del precio: signo negativo en {neg} de "
          f"{len(rz)} zonas")
    print(rz.sort_values("r_lluvia").round(3).to_string(index=False))
    print("\n  Precio mediano por zona (media 2015-2020):")
    print(df.groupby("region", observed=True)["precio_mediana"].mean()
          .sort_values().round(0).to_string())

    # 7. comparacion empirica de metodos (probar y comparar con cifras)
    seccion_comparacion(df, y)

    guardar_matriz(x, y, df)


if __name__ == "__main__":
    main()