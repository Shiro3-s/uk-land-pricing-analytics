# Accesibilidad de vivienda para jóvenes (18–34) en el Reino Unido

**Proyecto Integral de Minería de Datos · Ciclo KDD + CRISP-DM (Fases 1 a 12)**  
**Autores:** Juan Pablo Angel Quitian · Joel Sebastian Bueno Medina

**Propósito.** Estimar **cuántos años de salario necesita un joven (18–34) para comprar una vivienda** en cada región del Reino Unido, **qué esfuerzo hipotecario** exige cada mercado, **qué papel juega el clima** en la evolución de precios, **qué algoritmo predictivo conviene adoptar según si es con etiqueta o sin etiqueta y su cantidad de aciertos**, **qué segmentos naturales** estructuran la oferta inmobiliaria e hipotecaria (escalando hasta **11,35 millones de transacciones en *streaming*** bajo un presupuesto de 24 GB de RAM) y **qué patrones co-ocurrentes** revelan las reglas de asociación.

**Entregables principales y despliegue en producción:**
- **Simulador Web Público en Vivo (Netlify + Render + Azure PostgreSQL):** https://uk-vivienda-joven.netlify.app/ (soporte **£ GBP y $ COP**, **LTV/TLV**, plazo hasta **480 meses / 40 años**, escritura libre bilingüe **ES/EN**, 3 gráficas claras, visor en vivo de **Azure BD** y descarga **Excel `.xlsx` de 5 hojas**; presentación en https://uk-vivienda-joven.netlify.app/metodologia).
- **API Backend en Vivo (Render):** https://uk-vivienda-api.onrender.com (`/healthz`, `/api/contexto`, `/api/predecir`, `/api/datos`, `/api/datos/exportar.xlsx`).
- **Base de Datos en la Nube (Microsoft Azure PostgreSQL):** servidor `pg-actividad-db-uk26` · base de datos `db_actividad_mineria` (`19` tablas, `24.777` filas).
- **Excel Oficial Multi-Hoja en Azure Blob Storage (`stactividaduk26`):** https://stactividaduk26.blob.core.windows.net/excel-datos/vivienda-uk-azure-datos.xlsx (`5` hojas verificadas `200 OK`).
- **Libro Mayor Integral explicado tabla por tabla y gráfica por gráfica:** [`Cuaderno_Maestro_Proyecto_UK.ipynb`](Cuaderno_Maestro_Proyecto_UK.ipynb) y su espejo ejecutable [`main_maestro_completo.py`](main_maestro_completo.py).
- **Presentación Interactiva con Simulador de Predicción en Vivo (a escrita y por opciones) conectado a la BD/CSV:** [`Presentacion_Interactiva_Proyecto_UK.html`](Presentacion_Interactiva_Proyecto_UK.html) (servidor en vivo [`app_presentacion_interactiva.py`](app_presentacion_interactiva.py)) y presentación nativa PowerPoint **[`Presentacion_Proyecto_UK.pptx`](Presentacion_Proyecto_UK.pptx)**.
- **14 Cuadernos Modulares numerados (`00` a `13`):** en [`libros/`](libros/).
- **4 Documentos Técnicos Oficiales + Guía de Fase 3 Azure:** en [`md/`](md/) ([`MEMORIA_TECNICA.md`](md/MEMORIA_TECNICA.md), [`DICCIONARIO_DATOS.md`](md/DICCIONARIO_DATOS.md), [`BITACORA_DECISIONES.md`](md/BITACORA_DECISIONES.md), [`PROPUESTA_METODOLOGICA.md`](md/PROPUESTA_METODOLOGICA.md) y [`GUIA_FASE3_SUBIDA_AZURE.md`](md/GUIA_FASE3_SUBIDA_AZURE.md)).

---

## 1. ¿Hasta Qué Fase Llegamos? — Estado del Ciclo KDD + CRISP-DM

| Fase | Componente Analítico | Etapa KDD | Fase CRISP-DM | Estado | Evidencia en el Repositorio |
|---:|---|---|---|---|---|
| **1** | Problema, objetivos, hipótesis ($H_1, H_2, H_3$), Escalera DIKW y marcos | 1 · Selección | 1 · Comprensión del negocio | **✔ 100 % Completada** | `README.md`, `md/PROPUESTA_METODOLOGICA.md` |
| **2** | Ingesta de las 5 fuentes UK (`11,37 M` filas), formatos y auditoría (`0` duplicados) | 1 · Selección / Prep. I | 2 · Comprensión de datos | **✔ 100 % Completada** | `libros/00`, `libros/01`, `script/proyecto/auditoria_datos.py` |
| **3** | Calidad y limpieza científica (`MCAR/MAR/MNAR`, Tukey IQR vs. Z-Score, winsorización) | 2 · Preprocesamiento | 3 · Preparación de datos | **✔ 100 % Completada** | `libros/00_Diagnostico_y_Limpieza_F1.ipynb`, `libros/02`, `diagnostico.py` |
| **4** | Cruce regional (`60 × 16`), cuota francesa PMT ($90\%$ LTV, 25 años) y clima ($H_3$) | 2 · Preprocesamiento | 3 · Preparación de datos | **✔ 100 % Completada** | `libros/03`, `libros/04`, `libros/05`, `script/proyecto/indicadores.py` |
| **5** | Transformación: 3 Escaladores, One-Hot (`drop_first`) vs. Ordinal, VIF, Discretización y PCA | 3 · Transformación | 3 · Preparación de datos | **✔ 100 % Completada** | `libros/06_Transformacion_y_Matriz.ipynb`, `script/proyecto/matriz_xy.py` |
| **6** | **EDA Estadístico Completo:** 3 familias de medidas, **6 gráficas clásicas**, 3 variables derivadas y Checklist | 3–5 · Exploración | 2 · Exploración | **✔ 100 % Completada** | `libros/07_EDA_Accesibilidad.ipynb`, `script/proyecto/eda.py` |
| **7** | **Predicción CON ETIQUETA · Tipo 1 (Regresión Continua):** Contable ($R^2=1,000$) vs. Sin Fuga (`Ridge` y `RandomForest`) | 4 · Minería (Supervisada) | 4 · Modelado | **✔ 100 % Completada** | `libros/08_Modelado_Accesibilidad.ipynb`, `prediccion_dual.py` |
| **8** | **Predicción CON ETIQUETA · Tipo 2A (Clasificación I):** Árbol CART (`max_depth=4`), Gini, invarianza y reglas | 4 · Minería (Supervisada) | 4 · Modelado | **✔ 100 % Completada** | `libros/09_Arbol_Decision_Clasificacion.ipynb`, `arbol_clasificacion.py` |
| **9** | **Predicción CON ETIQUETA · Tipo 2B (Torneo II):** Bayes $2\times 2$, Distancia KNN, CART vs. NB vs. KNN y Ganador | 4–5 · Minería y Selección | 4–5 · Modelado y Evaluación | **✔ 100 % Completada** | `libros/10_Torneo_Clasificadores.ipynb`, `comparacion_clasificadores.py` |
| **10** | **Modelado SIN ETIQUETA · Tipos 3 y 4:** Clustering (`K-Means`, `Ward`, `11,35M` en *streaming* bajo 24 GB) y `Apriori` / `FP-Growth` | 4 · Minería (No Supervisada) | 4 · Modelado de Patrones | **✔ 100 % Completada** | `libros/11`, `libros/12`, `clustering_mercados.py`, `reglas_asociacion.py` |
| **11** | **Evaluación Forense (CRISP-DM 5), 3 Umbrales y Presentación Interactiva con Predicción en Vivo** | 5–6 · Evaluación y Conocimiento | 5 · Evaluación | **✔ 100 % Completada** | `libros/13`, `evaluacion_modelos.py`, `Presentacion_Interactiva_Proyecto_UK.html`, `Presentacion_Proyecto_UK.pptx` |
| **12** | **Despliegue Completo en la Nube:** Azure PostgreSQL (`pg-actividad-db-uk26`) + Render API + Netlify Frontend | 6 · Operacionalización | 6 · Despliegue en Nube | **✔ 100 % En Vivo** | `https://uk-vivienda-joven.netlify.app/`, `https://uk-vivienda-api.onrender.com`, `md/GUIA_FASE3_SUBIDA_AZURE.md` |

---

## 2. Taxonomía de Predicciones: ¿Con Etiqueta o Sin Etiqueta, de Qué Tipo son y Cuántos Aciertos Logra Cada Método?

| ¿Con o Sin Etiqueta? | Tipo de Modelo / Predicción | Pregunta que Responde | Etiqueta Objetivo ($y$) | Métodos Comparados | Método Ganador y Cantidad Exacta de Aciertos / Precisión | Cuándo Utilizar Este Método |
|---|---|---|---|---|---|---|
| **CON ETIQUETA** *(Supervisado)* | **Tipo 1A · Regresión Continua Regional (sin fuga)** | *«¿Qué porcentaje exacto del salario joven exige la hipoteca en esta región?»* | `esfuerzo` continuo ($y \in [0,416,\; 1,065]$, sin precio ni salario en $X$) | 1. **`Ridge (alpha=1.0)` en Pipeline**<br/>2. `RandomForestRegressor (200)`<br/>3. `KNNRegressor (k=5)`<br/>4. `DecisionTreeRegressor (d=4)`<br/>*(+ Control contable lineal con `anos_salario`: $R^2=1,000$)* | **`Ridge (alpha=1.0, Pipeline StandardScaler)`**<br/>• **$R^2_{\text{CV-5}} = 0,909 \pm 0,037$**<br/>• **Error medio $\text{MAE} = \pm 4,21\text{ pp}$** (`1,22 KB`) | Cuando se requiere estimar el **porcentaje exacto de esfuerzo salarial** en una región. |
| **CON ETIQUETA** *(Supervisado)* | **Tipo 1B · Regresión Continua Hipotecaria F2** | *«¿Qué tasa inicial exacta (%) tendrá un crédito hipotecario según sus condiciones?»* | `tasa_inicial_pct` continua ($y \in [3,09\%,\; 8,13\%]$ en $n=7.796$) | 1. **`RandomForestRegressor (200, d=10)`**<br/>2. `KNNRegressor (k=5)`<br/>3. `DecisionTreeRegressor (d=6)`<br/>4. `LinearRegression` | **`RandomForestRegressor (200, max_depth=10)`**<br/>• **$R^2_{\text{test}} = 0,964$** (`n_test = 1.949`)<br/>• **Error medio $\text{MAE} = \pm 0,169\text{ pp}$** | Cuando un joven comprador desea estimar la **tasa inicial exacta (%)** de una oferta hipotecaria. |
| **CON ETIQUETA** *(Supervisado)* | **Tipo 2A · Clasificación Binaria Regional (sin fuga)** | *«¿La región tiene presión hipotecaria Alta o Moderada y bajo qué reglas?»* | `clase_esfuerzo_alto`: $1=\text{Alto} \ge 0,5636$, $0=\text{Moderado}$ (`30/30` balanceado) | 1. **`Árbol CART (max_depth=4)`**<br/>2. `KNN escalado (k=5)` en Pipeline<br/>3. `Random Forest (200)`<br/>4. `Naive Bayes (GaussianNB)`<br/>5. *Control:* `KNN SIN escalar (k=5)` | **`Árbol de Decisión CART (max_depth=4)` [GANADOR `4,82/5,00`]**<br/>• **`17 / 18` aciertos en Test (`94,4 %`, F1=`0,941`)**<br/>• **`47 / 60` en CV-5 (`78,3 %`**, sube a **`48 / 60 = 80,0 %`** con clúster K-Means)<br/>• **`9 / 10` regiones reales acertadas (`90,0 %`)** | **Método principal recomendado:** explica cada veredicto con reglas `SI / ENTONCES`, es invariante a escalado (`assert`), **11× más rápido** y **48× más ligero (`1,35 KB`)** que KNN. (*NB cae a `8/18 = 44,4%`; KNN sin escalar cae a `12/18 = 66,7%`, $-27,7\text{ pp}$*). |
| **CON ETIQUETA** *(Supervisado)* | **Tipo 2B · Clasificación Hipotecaria F2 + 3 Umbrales Operativos** | *«¿Esta hipoteca tiene riesgo de tasa elevada ($\ge Q_3$) o atípica ($>8,13\%$) y qué alerta activar?»* | `tasa >= 6,62%` (`25 %` pos.) y `es_atipico_tasa > 8,13%` (**`9,77 %` pos. = `190` casos en test $n=1.949$**) | 1. **`Árbol CART (d=4)` + 3 umbrales (`0,05 / 0,30 / 0,40`)**<br/>2. `KNN escalado (k=5)` y `(k=15)`<br/>3. `Random Forest (200)`<br/>4. `Naive Bayes (GaussianNB)`<br/>5. *Control:* `Modelo Ingenuo (Dummy)` | **`Árbol CART (d=4)` con umbral $\tau = 0,30$** (`ROC-AUC = 0,955`)<br/>• En tasa $\ge Q_3$: CART acierta **`1.805 / 1.949` (`92,6 %`)** y KNN `k=5` acierta **`1.842 / 1.949` (`94,5 %`)**.<br/>• En tasa atípica ($>8,13\%$): con $\tau=0,30$ CART detecta **`188 / 190` (`99,0 %` recall)**; con $\tau=0,05$ detecta **`190 / 190` (`100 %`)**; con $\tau=0,40$ logra **`100 %` precisión (`0` FP)**. KNN `k=5` detecta **`158 / 190` (`83,2 %`, F1=`0,825`)**. | Para **alertas de crédito hipotecario**: desenmascara al Modelo Ingenuo (`90,25 %` accuracy con **`0/190` aciertos**) y a Naive Bayes (**`1.247` falsas alarmas**). |
| **SIN ETIQUETA** *(No Supervisado)* | **Tipo 3 · Agrupamiento / Clustering Multi-escala** | *«¿Qué segmentos naturales de regiones, hipotecas y viviendas existen sin etiquetas?»* | **Ninguna en $X$** (`esfuerzo` se excluye de $X$ y solo valida *post-hoc*) | 1. **`K-Means` ($k=3$, `StandardScaler`)**<br/>2. **`Clustering Jerárquico Ward` ($O(n^2)$)**<br/>3. **`MiniBatchKMeans` en *streaming* PyArrow `float32` ($k=4$)** | **`K-Means (k=3)` + `MiniBatchKMeans` *Streaming* ($k=4$)**<br/>• Separa **`100 %` del núcleo extremo** (London y South East, esfuerzo `0,90`) y mejora a CART CV-5 (`78,3% -> 80,0%`).<br/>• En F2: `silueta = 0,479`.<br/>• En F1: procesa **`11.346.500` ventas en `49,87 MB` de RAM (`0,20 %` de 24 GB)** con `silueta = 0,333`. | Para segmentar mercados territoriales, ofertas bancarias o millones de compraventas sin etiqueta previa. |
| **SIN ETIQUETA** *(No Supervisado)* | **Tipo 4 · Reglas de Asociación (Patrones Co-ocurrentes)** | *«¿Qué combinaciones de atributos ($A \to B$) disparan el sobre-esfuerzo o reducen la tasa?»* | **Ninguna etiqueta fija** (implicaciones $A \to B$ medidas por soporte, confianza y lift) | 1. **`Apriori` puro con `tidsets` y filtro de trivialidades**<br/>2. **`FP-Growth`** (referencia en 2 barridos sin candidatos) | **`Apriori` con ranking `score = lift × soporte`**<br/>• **`10 / 10` tickets verificados a mano (`100 %` exactitud `assert`)**<br/>• F1: **`195` reglas** (`Detached + Freehold -> Acceso Extremo >100%`, **confianza `68,0 %`, `lift = 2,00`**)<br/>• F2: **`481` reglas** (`Tipo:discounted -> Tasa:Baja_Q1`, **confianza `99,1 %`, `lift = 3,96`**) | Para recomendar o alertar sobre combinaciones de compra y crédito tras filtrar trivialidades legales (`Flat -> Leasehold`). |

---

## 3. Estructura Limpia y Organizada del Repositorio

```text
mineria/
├── frontend/                               # Simulador web público (Netlify: index.html, styles.css, app.js, metodologia.html)
├── backend/                                # API Flask/Waitress (Render: web_api.py, requirements.txt, render.yaml, exportar_excel_blob.py)
├── netlify.toml                            # Configuración de publicación Netlify (publish = "frontend" + proxy /api/* -> Render)
├── Cuaderno_Maestro_Proyecto_UK.ipynb      # LIBRO MAYOR INTEGRAL explicado tabla por tabla y gráfica por gráfica (Fases 1-12)
├── main_maestro_completo.py                # Espejo ejecutable en Python del Libro Mayor (macOS / Linux / Windows)
├── Presentacion_Interactiva_Proyecto_UK.html # [FASE 2] Presentación interactiva + Simulador de Predicción (a escrita y por opciones)
├── Presentacion_Proyecto_UK.pptx           # [FASE 2] Presentación ejecutiva y estadística en Microsoft PowerPoint (.pptx)
├── app_presentacion_interactiva.py         # [FASE 2] Servidor ligero que conecta la presentación con los CSV / Azure PostgreSQL
├── main_proyecto.py                        # Pipeline base (4 fuentes -> maestro_joven_uk)
├── main_limpieza.py                        # Pipeline de limpieza masiva de F1 (11,37 M filas)
├── README.md                               # Portada ejecutiva, taxonomía Con/Sin Etiqueta y guía de ejecución
│
├── libros/                                 # 14 CUADERNOS MODULARES NUMERADOS EN ORDEN (00 a 13)
│   ├── 00_Diagnostico_y_Limpieza_F1.ipynb
│   ├── 01_Carga_4_Fuentes.ipynb
│   ├── 02_Limpieza_4_Datasets.ipynb
│   ├── 03_Cruces_Affordability_Joven.ipynb
│   ├── 04_Hipoteca_Joven.ipynb
│   ├── 05_Clima_y_Precios.ipynb
│   ├── 06_Transformacion_y_Matriz.ipynb
│   ├── 07_EDA_Accesibilidad.ipynb
│   ├── 08_Modelado_Accesibilidad.ipynb
│   ├── 09_Arbol_Decision_Clasificacion.ipynb
│   ├── 10_Torneo_Clasificadores.ipynb
│   ├── 11_Clustering_Segmentacion.ipynb
│   ├── 12_Reglas_Asociacion_Apriori.ipynb
│   └── 13_Evaluacion_Modelos_y_Umbrales.ipynb
│
├── md/                                     # DOCUMENTACIÓN TÉCNICA OFICIAL Y FIGURAS ACTIVAS
│   ├── MEMORIA_TECNICA.md                  # Informe Técnico Integral completo (Fases 1-12, fórmulas, tablas y gráficas explicadas)
│   ├── DICCIONARIO_DATOS.md                # Diccionario completo de las 5 fuentes, matriz X·y y las 17 tablas de Azure PostgreSQL
│   ├── BITACORA_DECISIONES.md              # Registro auditable de decisiones de limpieza, transformación y modelado
│   ├── PROPUESTA_METODOLOGICA.md           # Problema, objetivos, hipótesis H1-H3 y marcos KDD + CRISP-DM
│   ├── GUIA_FASE3_SUBIDA_AZURE.md          # [FASE 3] Guía guardada paso a paso para subir la BD/CSV a Azure y compartir el link
│   └── figuras/                            # 50 figuras técnicas PNG activas del proyecto
│
├── data/
│   ├── crudo/                              # F2, F3, F4a y F4b completos intocables
│   ├── limpia/                             # Parquets limpios y muestras livianas
│   ├── integrado/                          # Cuadro maestro, matriz X·y y 17 tablas CSV limpias (1,83 MB) para BD
│   └── modelos/                            # 4 modelos .joblib + contrato_prediccion.json + esquema_bd.sql
│
├── db/                                     # Maestro local F1 (11,37 M) y muestras versionadas
└── script/
    ├── limpieza/                           # Paquete de limpieza de F1
    ├── proyecto/                           # 23 módulos Python del ciclo analítico y predicción dual
    ├── documentacion/                      # Generadores de Presentacion_Proyecto_UK.pptx/.html y Word APA 7
    └── azure/                              # Scripts de Fase 3 para Azure Database for PostgreSQL (roles, subida y predictor)
```

---

## 4. Comandos Principales (macOS, Linux y Windows)

| Tarea | Comando (desde la raíz del repositorio) |
|---|---|
| **1. Ejecutar el Libro Mayor completo en terminal** | `python main_maestro_completo.py` (o `--region London`) |
| **2. Abrir la Presentación Interactiva con Predicción en Vivo (a escrita y por opciones)** | Abrir `Presentacion_Interactiva_Proyecto_UK.html` en el navegador o ejecutar `python app_presentacion_interactiva.py` |
| **3. Regenerar la Presentación PowerPoint (`.pptx`) y Web (`.html`)** | `python script/documentacion/generar_presentacion_ppt.py` |
| **4. Predicción Dual por terminal (por opciones o a escrita)** | `python script/azure/predecir_azure.py --region London --metodo todos`<br/>`python script/azure/predecir_azure.py --texto "Quiero comprar en London como First time buyer con comision 999 a 24 meses fijo"` |
| **5. Fase 3 (cuando decidas subir las 17 tablas CSV a Azure PostgreSQL)** | Seguir [`md/GUIA_FASE3_SUBIDA_AZURE.md`](md/GUIA_FASE3_SUBIDA_AZURE.md): `python script/azure/subir_datos_azure.py` |

---

## Referencias

- Land Registry Price Paid Data (OGL v3.0): [uk-property-sale-prices-20152026-geo-enriched](https://www.kaggle.com/datasets/mansiaggarwal88/uk-property-sale-prices-20152026-geo-enriched?select=uk_property_price_paid_geo.csv)
- Oferta hipotecaria UK 2022: [uk-mortgage-rates-thousands-of-mortgage-products](https://www.kaggle.com/datasets/thedevastator/uk-mortgage-rates-thousands-of-mortgage-products)
- Met Office Historic Station Data: [uk-met-office-weather-data](https://www.kaggle.com/datasets/josephw20/uk-met-office-weather-data)
- Precios y salarios históricos UK (1975–2020): [uk-median-house-prices-and-salary-19752020](https://www.kaggle.com/datasets/samuelcortinhas/uk-median-house-prices-and-salary-19752020)
- Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). *From Data Mining to Knowledge Discovery in Databases*. AI Magazine.
- Wirth, R., & Hipp, J. (2000). *CRISP-DM: Towards a standard process model for data mining*.
- Agrawal, R., & Srikant, R. (1994). *Fast algorithms for mining association rules* (Apriori).
- Han, J., Pei, J., & Yin, Y. (2000). *Mining frequent patterns without candidate generation* (FP-Growth).
