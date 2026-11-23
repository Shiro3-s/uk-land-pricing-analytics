# Resumen Completo del Proyecto · Accesibilidad de Vivienda para Jóvenes (18–34 años) en el Reino Unido

**Autores:** Juan Pablo Angel Quitian · Joel Sebastian Bueno Medina  
**Metodología:** Ciclo integrado **KDD + CRISP-DM** (12 fases completadas y verificadas)  
**Estado general:** Fases 1 a 12 completadas y desplegadas en producción en la nube (**Netlify Frontend + Render Backend API + Azure Database for PostgreSQL**):
- **Sitio Web en Vivo (Netlify):** https://uk-vivienda-joven.netlify.app/ (y presentación interactiva en https://uk-vivienda-joven.netlify.app/metodologia)
- **API Backend en Vivo (Render):** https://uk-vivienda-api.onrender.com (`/healthz`, `/api/contexto`, `/api/predecir`, `/api/datos`, `/api/datos/exportar.xlsx`)
- **Base de Datos en la Nube (Azure PostgreSQL):** `pg-actividad-db-uk26.postgres.database.azure.com` · BD `db_actividad_mineria` (`19` tablas, `24.777` filas) · [Abrir en Azure Portal](https://portal.azure.com/#@uniminuto.edu/resource/subscriptions/055b858b-7fbc-4b45-bf00-e072f09c083c/resourceGroups/rg-actividad-db/providers/Microsoft.DBforPostgreSQL/flexibleServers/pg-actividad-db-uk26/databases)
- **Archivo Excel Físico en Azure Blob Storage (`stactividaduk26` · `excel-datos`):** [vivienda-uk-azure-datos.xlsx](https://stactividaduk26.blob.core.windows.net/excel-datos/vivienda-uk-azure-datos.xlsx) (`5` hojas verificadas `200 OK`) · [Ver en Azure Portal](https://portal.azure.com/#@uniminuto.edu/resource/subscriptions/055b858b-7fbc-4b45-bf00-e072f09c083c/resourceGroups/rg-actividad-db/providers/Microsoft.Storage/storageAccounts/stactividaduk26/containersList)

---

## 1. Ficha Ejecutiva, Problema, Hipótesis y Escalera DIKW

### Problema central
Estimar con rigor estadístico y financiero **cuántos años de salario** y **qué porcentaje de su ingreso mensual** necesita un joven británico (18–34 años) para adquirir la vivienda mediana de su región o ciudad mediante una hipoteca estándar (**90 % LTV a 25 años**, fórmula de amortización francesa `PMT`), contrastar si el clima incide en la variación de precios y desplegar un ecosistema predictivo completo (**Con Etiqueta y Sin Etiqueta**) conectado a **Azure Database for PostgreSQL** y utilizable también de forma **100 % autónoma**.

### Contraste de Hipótesis
| Hipótesis | Enunciado | Evidencia Medida | Veredicto |
| :--- | :--- | :--- | :--- |
| **$H_1$ (Umbral 30 %)** | Ninguna región del Reino Unido cumple la regla prudencial bancaria ($\le 30\%$ del salario joven mensual). | 60/60 observaciones (2015–2020) y 10/10 regiones en 2022 superan el 30 % (rango **43,0 % a 100,0 %**). | **Confirmada (100 %)** |
| **$H_2$ (Brecha Norte–Sur)** | El Norte y Gales exigen significativamente menos años de sueldo joven que London y el Sur. | Brecha territorial estructural de **2,3×**: desde **6,19 años** (`North East`) hasta **14,24 años** (`London`). | **Confirmada** |
| **$H_3$ (Clima vs. Precios)** | Las anomalías climáticas guardan relación débil con la variación anual del precio de la vivienda. | Correlación de Pearson $\|r\| \in [0,145;\; 0,399]$ ($R^2 \le 16\%$): contexto ambiental, no motor de precios. | **Confirmada** |

### Escalera DIKW aplicada al proyecto
1. **Dato:** `11.369.413` escrituras reales del *HM Land Registry* (F1), `7.796` productos hipotecarios bancarios (F2), `37.049` lecturas mensuales de 36 estaciones *Met Office* (F3) y series de precios/salarios *ONS/ASHE* (F4a–F4b).
2. **Información:** Salario joven estimado en **£25.353/año** (factor de brecha etaria `0,805`); tasa inicial hipotecaria media winsorizada en **6,04 %**; cuota mensual 2022 entre **£772/mes (43,0 %)** en `North East` y **£2.641/mes (100,0 %)** en `London`.
3. **Conocimiento:** Árbol CART (`max_depth=4`) clasifica con **`17/18` aciertos en Test (`94,44 %`)**, **`9/10` regiones (`90,0 %`)** y detecta **`188/190` hipotecas atípicas (`99,0 %` recall)** en $\tau = 0,30$; `Ridge` estima el esfuerzo salarial continuo sin fuga con **$R^2 = 0,9094$ ($\pm 4,21\text{ pp}$)** y `RandomForestRegressor` estima la tasa hipotecaria con **$R^2 = 0,9642$ ($\pm 0,169\text{ pp}$)**.
4. **Decisión:** Subsidio a cuota inicial en Clúster 0 (Norte/Gales), garantías estatales en Clúster 1 (Midlands/SW/East), intervención estructural de vivienda social en Clúster 2 (London/SE) y semáforo automático de 3 umbrales (`0,05 / 0,30 / 0,40`) frente a hipotecas abusivas ($> 8,13\%$).

---

## 2. Mapa Completo de Fases: Las 12 Fases KDD + CRISP-DM y las 3 Macro-Fases Ejecutadas

### 2.1 Las 12 Fases Metodológicas (KDD + CRISP-DM)
*(Se integraron KDD y CRISP-DM descartando SEMMA, ya que SEMMA omite la comprensión del negocio y el despliegue).*

| Fase | Componente Analítico | Etapa KDD | Fase CRISP-DM | Estado | Cuadernos y Scripts Asociados |
|---:|---|---|---|---|---|
| **1** | Problema, objetivos, hipótesis ($H_1..H_3$) y Escalera DIKW | 1 · Selección | 1 · Negocio | **Completada** | `README.md`, `md/PROPUESTA_METODOLOGICA.md` |
| **2** | Ingesta de 5 fuentes (`11,37 M` filas) y auditoría sanitaria | 1 · Selección / Prep. I | 2 · Datos | **Completada** | `libros/00`, `libros/01`, `auditoria_datos.py`, `cargar.py` |
| **3** | Limpieza científica: `MCAR/MAR/MNAR`, `Tukey IQR` vs. `Z-Score`, winsorización | 2 · Preprocesamiento | 3 · Preparación | **Completada** | `libros/00`, `libros/02`, `limpiar.py`, `diagnostico.py` |
| **4** | Cruce maestro (`60 × 16`), amortización francesa `PMT` y clima ($H_3$) | 2 · Preprocesamiento | 3 · Preparación | **Completada** | `libros/03`, `04`, `05`, `cruces.py`, `indicadores.py` |
| **5** | Transformación: 3 escaladores en `Pipeline`, One-Hot (`VIF=1,80`), discretización y `PCA` | 3 · Transformación | 3 · Preparación | **Completada** | `libros/06`, `matriz_xy.py`, `transformaciones.py` |
| **6** | **EDA Estadístico Completo:** 3 familias de medidas, 6 gráficas clásicas, 3 derivadas y checklist | 3–5 · Exploración | 2 · Exploración | **Completada** | `libros/07`, `eda.py` |
| **7** | **CON ETIQUETA · Tipo 1:** Regresión Continua (control contable vs. `Ridge` y `RF` sin fuga) | 4 · Minería Sup. | 4 · Modelado | **Completada** | `libros/08`, `modelado.py`, `prediccion_dual.py` |
| **8** | **CON ETIQUETA · Tipo 2A:** Árbol CART (`d=4`), Gini, invarianza al escalado y reglas | 4 · Minería Sup. | 4 · Modelado | **Completada** | `libros/09`, `arbol_clasificacion.py` |
| **9** | **CON ETIQUETA · Tipo 2B:** Torneo CART vs. Naive Bayes vs. KNN (escalado y crudo) | 4–5 · Selección | 4–5 · Evaluación | **Completada** | `libros/10`, `comparacion_clasificadores.py` |
| **10** | **SIN ETIQUETA · Tipos 3 y 4:** Clustering (`11,35M` en `49,87 MB` RAM) y Apriori (`tidsets`) | 4 · Minería No Sup. | 4 · Patrones | **Completada** | `libros/11`, `12`, `clustering_mercados.py`, `reglas_asociacion.py` |
| **11** | Evaluación Forense, Trampa del Accuracy, 3 Umbrales (`0,05/0,30/0,40`), Presentación Interactiva y PPTX | 5–6 · Evaluación | 5 · Evaluación | **Completada** | `libros/13`, `evaluacion_modelos.py`, `generar_presentacion_ppt.py` |
| **12** | Despliegue en **Azure Database for PostgreSQL (`pg-actividad-db-uk26`)**, 19 tablas (`24.777` filas) y enlace compartible | 6 · Operacionalización | 6 · Despliegue | **Completada (Servidor pausado)** | `script/azure/*`, `app_presentacion_interactiva.py`, `md/GUIA_FASE3_SUBIDA_AZURE.md` |

### 2.2 Las 3 Macro-Fases Ejecutadas en la Recta Final del Proyecto
1. **Macro-Fase 1 · Organización Total, Libro Mayor, EDA Completo y Explicación de Cada Tabla y Gráfica:**
   - Se eliminaron todos los rastros de ejemplos ajenos (`contexto/` con 40 archivos), borradores antiguos (`Avances/` con 5 archivos), duplicados (`README_LABORATORIO.md`, `md/FASE1..4.*`, `INFORME_AVANCE1.md`, `INFORME_DIAGNOSTICO_LIMPIEZA.md`), 9 figuras duplicadas (`rev_*.png`, `fase4_*.png`), 4 scripts obsoletos en `script/proyecto/` y `libros/Lectura.ipynb`.
   - Se numeraron los 14 cuadernos modulares de `libros/00` a `libros/13`.
   - Se consolidó el **Libro Mayor** (`Cuaderno_Maestro_Proyecto_UK.ipynb` — 33 celdas, 36 tablas, 21 gráficas embebidas) y su espejo ejecutable en consola (`main_maestro_completo.py`, ejecución íntegra en `8,71 s`).
   - Se verificó mediante auditoría automática que el **100 % de las celdas con tablas o gráficas en los 15 cuadernos (`Cuaderno_Maestro_Proyecto_UK.ipynb` + `libros/00..13.ipynb`) tienen inmediatamente debajo su explicación en Markdown**.
   - Se sincronizó toda la documentación oficial (`README.md`, `md/MEMORIA_TECNICA.md`, `md/DICCIONARIO_DATOS.md`, `md/BITACORA_DECISIONES.md`, `md/PROPUESTA_METODOLOGICA.md`).
2. **Macro-Fase 2 · Presentación Interactiva Web Autónoma/Conectada + Simulador con Gráfica Movible + PowerPoint (`.pptx`):**
   - Se creó `Presentacion_Interactiva_Proyecto_UK.html` (`2,45 MB`, 15 diapositivas, 18 gráficas en Base64 con visor Zoom Lightbox, pantalla completa, sin rótulos `.png`, simulador dual Por Opciones y A Escrita con 8 botones de ejemplo, gráfica SVG arrastrable con el mouse, explicación matemática paso a paso por método y panel de código de vinculación Azure ⇄ HTML en la última página).
   - Se creó `Presentacion_Proyecto_UK.pptx` (`1,38 MB`, 14 diapositivas panorámicas `16:9` OpenXML validadas con 14 gráficas PNG embebidas).
   - Se creó el servidor web ligero `app_presentacion_interactiva.py` (`http://localhost:8765`, endpoints `/api/contexto` y `/api/predecir`).
3. **Macro-Fase 3 · Subida a Azure Database for PostgreSQL (`pg-actividad-db-uk26`), Pruebas en Vivo y Pausa Controlada:**
   - Se creó la guía detallada `md/GUIA_FASE3_SUBIDA_AZURE.md` y los scripts `script/azure/configurar_roles_azure.py`, `subir_datos_azure.py`, `test_conexion_azure.py` y `predecir_azure.py`.
   - Se encendió el servidor `pg-actividad-db-uk26`, se aplicó `data/modelos/esquema_bd.sql`, se subieron vía `COPY` las **17 tablas CSV limpias + 2.000 transacciones georreferenciadas + bitácora `historial_predicciones` = 19 tablas (`24.777` registros)**, se validó la seguridad de los 3 roles y las predicciones en vivo contra Azure, y posteriormente se pausó el servidor (`state: Stopped`) para proteger los créditos de Azure hasta el día de la exposición.

---

## 3. Estructura Organizada del Repositorio

```text
mineria (3)/
├── frontend/                                   # Simulador web (Netlify: index.html, styles.css, app.js, metodologia.html)
├── backend/                                    # API en la nube (Render: web_api.py, requirements.txt, render.yaml, exportar_excel_blob.py)
├── netlify.toml                                # Publicación Netlify (publish = "frontend") + proxy /api/* -> Render
├── Cuaderno_Maestro_Proyecto_UK.ipynb          # Libro Mayor completo (33 celdas, 36 tablas, 21 figuras)
├── main_maestro_completo.py                    # Orquestador ejecutable en consola (espejo 1:1 del Libro Mayor)
├── Presentacion_Interactiva_Proyecto_UK.html   # Presentación 15 slides + Simulador (100% autónoma + vinculable a Azure)
├── Presentacion_Proyecto_UK.pptx               # Presentación Microsoft PowerPoint nativa (14 slides, 16:9)
├── app_presentacion_interactiva.py             # Servidor HTTP + API (/api/contexto, /api/predecir) conectado a Azure/CSV
├── main_limpieza.py / main_proyecto.py         # Puntos de entrada rápidos del pipeline de limpieza y proyecto
├── README.md                                   # Portada y guía general del repositorio
├── .env / .env.example / .gitignore            # Variables de Azure (credenciales reales solo en .env local)
├── libros/                                     # 14 cuadernos modulares numerados 00..13 (100% explicados)
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
├── script/
│   ├── proyecto/                               # 24 módulos Python (config, cargar, limpiar, eda, modelado, predecir, etc.)
│   ├── limpieza/                               # Pipeline de limpieza de F1 (11,37M filas)
│   ├── documentacion/                          # generar_presentacion_ppt.py, generar_documento_apa.py
│   ├── azure/                                  # configurar_roles_azure.py, subir_datos_azure.py, predecir_azure.py, test_conexion_azure.py
│   └── requirements/                           # requirements.txt y requirements-notebooks.txt
├── data/
│   ├── crudo/                                  # 4 CSV originales intocables (F2, F3, F4a, F4b)
│   ├── limpia/                                 # Parquets depurados (incluye F1 11.369.413 filas)
│   ├── integrado/                              # 17 tablas CSV limpias ("Excel", 1,83 MB) + Parquets integrados
│   └── modelos/                                # 4 modelos .joblib (< 5 ms), contrato_prediccion.json y esquema_bd.sql
├── db/                                         # Origen F1 completo y respaldos crudos (100% contenidos en F1 limpio)
└── md/                                         # Documentación oficial y 49 figuras PNG (md/figuras/)
    ├── RESUMEN_COMPLETO_PROYECTO_UK.md         # Este documento maestro consolidado
    ├── MEMORIA_TECNICA.md
    ├── DICCIONARIO_DATOS.md
    ├── BITACORA_DECISIONES.md
    ├── PROPUESTA_METODOLOGICA.md
    └── GUIA_FASE3_SUBIDA_AZURE.md
```

---

## 4. Fuentes, Auditoría Sanitaria, Limpieza Científica y Resultados Financieros

### 4.1 Auditoría integral (43 archivos verificados, 0 duplicados)
- **F1 (HM Land Registry Price Paid Data, 2015–2026):** `11.369.413` filas × `22` columnas. Auditoría en *streaming* verificó que los 5 CSV de respaldo en `db/` tienen el **100,0 % de sus `transaction_id` ya incluidos** en F1 limpio (sin duplicidad).
- **F2 (UK Mortgage Rates Catalog, 2022):** `7.796` productos bancarios × `14` columnas.
- **F3 (Met Office Historic Station Data, 36 estaciones, 1931–2020):** `37.049` lecturas crudas → `37.029` válidas tras eliminar 20 filas corruptas/footer sin fecha.
- **F4a (Average UK House Prices & Salary, 1975–2020):** `46` registros anuales en GBP reales 2020.
- **F4b (ONS ASHE Income by Age & Gender, 2021):** `12` bandas etarias (factor salarial joven 18–34 = `0,805` → **£25.353/año**).

### 4.2 Limpieza científica: Faltantes (`MCAR/MAR/MNAR`) y Outliers (`Tukey IQR` vs. `Z-Score`)
- **Faltantes (Little & Rubin, 2019):**
  - `tasa_reversion` en F2 (`328`, `4,2 %`): **MAR** (falta `0,8 %` en *fixed* vs. `39,5 %` en *variable*); se conserva `NaN`.
  - `tmax`, `tmin`, `af`, `lluvia` en F3 (`1.478–2.949`): **MCAR** (falla instrumental conjunta de estación: `1.467/1.599` coinciden en `tmax` y `tmin`).
  - `sol` en F3 (`9.483`) y `salario_mediana_real` en F4 (`24`): **MNAR estructural** (no publicados antes de 1931 ni antes de 1999); jamás se imputa historia inventada.
- **Outliers (`IQR` de Tukey vs. `Z-Score`):**
  - En F2 (`tasa_inicial_pct`): las cercas de Tukey ($Q_3 + 1{,}5 \cdot \text{IQR} = 6{,}62 + 1{,}5 \cdot 1{,}00 = 8{,}13\%$) detectan **`762` atípicos (`9,77 %`)**, mientras `|z| > 3` detecta **`0` (`0 %`)** por inflación de la varianza. Se aplica **winsorización a `8,13 %`** (la media baja de `6,11 %` a `6,04 %`, `0` filas eliminadas + bandera `es_atipico_tasa`).
  - En F1 (`price`): Tukey detecta **`691.509` ventas extremas (`6,08 %`, $> \text{£}725.500$)**; se winsoriza al rango `[£100, £725.500]` + bandera `es_atipico_precio`.

### 4.3 Fórmula de Amortización Francesa (`PMT`) y Resultados por Región
$$C = P \cdot \frac{r}{1 - (1 + r)^{-n}} + \frac{\text{comisiones}}{n}$$
con $P = 0{,}90 \cdot \text{precio\_mediana}$ (90 % LTV), $r = 0{,}06043 / 12$, $n = 300$ meses (25 años) y $\text{esfuerzo} = C / (\text{salario\_joven\_región} / 12)$.

| Región del Reino Unido | Años de Salario (2015–2020) | Cuota Mensual 2022 (£) | Esfuerzo Hipotecario 2022 (%) | ¿Cumple $\le 30\%$? |
|---|---:|---:|---:|:---:|
| **London** | 14,24 años | £2.641 / mes | **100,0 %** | NO |
| **South East** | 11,61 años | £1.937 / mes | **86,5 %** | NO |
| **East of England** | 10,74 años | £1.745 / mes | **81,8 %** | NO |
| **South West** | 10,17 años | £1.589 / mes | **79,1 %** | NO |
| **East Midlands** | 7,80 años | £1.233 / mes | **63,4 %** | NO |
| **West Midlands** | 8,00 años | £1.207 / mes | **62,8 %** | NO |
| **Wales** | 6,79 años | £1.028 / mes | **54,0 %** | NO |
| **Yorkshire and The Humber** | 6,96 años | £977 / mes | **52,5 %** | NO |
| **North West** | 6,72 años | £1.002 / mes | **52,1 %** | NO |
| **North East** | 6,19 años | £772 / mes | **43,0 %** | NO |

---

## 5. Transformación (Sin Fuga) y EDA Estadístico Completo

### Decisiones de Transformación Verificadas
- **Escalamiento anti-fuga:** `StandardScaler` siempre dentro de `make_pipeline(...)` (ajustado solo en `X_train`). Sin escalar, `n_transacciones` absorbe el **100,0 % de la distancia euclidiana** y KNN pierde `-27,7 pp`.
- **Codificación nominal y multicolinealidad (`VIF`):** `OneHotEncoder(drop='first')` produce 9 dummies (referencia `East Midlands`), bajando el `VIF` de $\infty$ a **`1,80` (`< 5`)** y superando a `OrdinalEncoder` ($R^2 = 0,891$ vs. $-0,051$).
- **PCA y Discretización:** 3 componentes PCA retienen `~81 %` de varianza pero ocultan el nombre de las variables en las reglas del Árbol CART; se mantienen las variables originales para modelado y la discretización en 4 tramos para EDA/Apriori.

### EDA Estadístico sobre `esfuerzo` ($n = 60$)
- **3 Familias de Medidas:**
  - *Tendencia central:* Media = `0,625`, **Mediana = `0,564`** (manda la mediana por asimetría positiva `+0,816` arrastrada por London).
  - *Dispersión:* Desviación estándar $s = 0,182$, $\text{IQR} = 0,284$, rango `[0,416; 1,065]`.
  - *Posición y outliers:* $Q_1 = 0,486$, $Q_2 = 0,564$, $Q_3 = 0,770$, `0` outliers globales por Tukey.
- **3 Variables Derivadas Validadas:**
  1. `indice_tension` (`precio_región / mediana_nacional_año`): Pearson **$r = +0,946$** con esfuerzo.
  2. `cv_esfuerzo` ($s / \bar{x}$ por región, `0,035` a `0,094`): Pearson **$r = +0,684$** con volatilidad de precios.
  3. `segmento_4_niveles` (*Moderado*, *Alto*, *Crítico*, *Extremo*): progresión monótona (`0,458 → 0,632 → 0,867 → 1,033`).
- **Hallazgo crítico anti-fuga:** `anos_salario` y `esfuerzo` tienen correlación **$r = +1,000$** porque comparten el cociente `precio / salario`. Por ello, todos los modelos predictivos sin fuga excluyen `precio_mediana`, `salario_joven_region` y `anos_salario` de $X$.

---

## 6. Taxonomía Maestra: Modelos CON Etiqueta vs. SIN Etiqueta y Cantidad Exacta de Aciertos

| Paradigma | Tipo de Modelo | Método Elegido vs. Comparados | Cantidad Exacta de Aciertos y Métricas Verificadas | Por Qué lo Elegimos / Cuándo Usarlo |
|---|---|---|---|---|
| **CON ETIQUETA** *(Supervisado)* | **Tipo 1A · Regresión Continua Regional (sin fuga)** | **Regresión Ridge (`alpha=1.0`, Pipeline) [GANADOR]** vs. RF (`0,761`), KNN (`0,759`), Árbol (`0,706`) | **$R^2_{\text{CV-5}} = 0,9094 \pm 0,0375$**<br/>**$\text{MAE} = \pm 4,21\text{ pp}$**, $\text{RMSE} = 5,08\text{ pp}$ | Penalización $L_2$ evita sobreajuste en $n=60$; estima el `%` exacto de salario joven consumido. |
| **CON ETIQUETA** *(Supervisado)* | **Tipo 1B · Regresión Continua Hipotecaria F2** | **RandomForestRegressor (`200`, `d=10`) [GANADOR]** vs. KNN (`0,941`), Árbol (`0,899`), Lineal (`0,632`) | **$R^2_{\text{test}} = 0,9642$** ($n_{\text{test}}=1.949$)<br/>**$\text{MAE} = \pm 0,1692\text{ pp}$**, $\text{RMSE} = 0,251\text{ pp}$ | Captura relaciones no lineales entre comisión, plazo, tipo y perfil comprador para cotizar la tasa exacta (`%`). |
| **CON ETIQUETA** *(Supervisado)* | **Tipo 2A · Clasificación Binaria Regional** (`Alto >= 56,4%`, 30/30) | **Árbol CART (`max_depth=4`) [GANADOR 4,82/5,00]** vs. KNN `k=5`, RF `200`, Naive Bayes, KNN crudo | **CART:** **`17/18` Test (`94,44 %`, F1=`0,941`)**, **`47/60` CV-5 (`78,3 %` → `48/60 = 80,0 %` con K-Means)**, **`9/10` regiones (`90,0 %`)**, `1,35 KB`.<br/>**KNN `k=5`:** `17/18` Test, **`54/60` CV-5 (`90,0 %`)**, `9/10` regiones.<br/>**NB:** `8/18` (`44,4 %`). **KNN crudo:** `12/18` (`66,7 %`). | Único modelo que entrega reglas `SI/ENTONCES` auditables, es invariante al escalado (`assert`), `11×` más rápido y `48×` más ligero que KNN. |
| **CON ETIQUETA** *(Supervisado)* | **Tipo 2B · Clasificación Tasa Elevada F2** ($\ge Q_3 = 6,62\%$) | **Árbol CART (`d=5`)** y **KNN (`k=5`)** vs. Naive Bayes | **CART:** **`1.805 / 1.949` (`92,61 %`, AUC=`0,943`)**<br/>**KNN `k=5`:** **`1.842 / 1.949` (`94,51 %`, AUC=`0,975`)**<br/>**Naive Bayes:** `945 / 1.949` (`48,49 %`, Brier=`0,505`) | Anticipar si una hipoteca cae en el cuartil más costoso del mercado. |
| **CON ETIQUETA** *(Supervisado)* | **Tipo 2C · Clasificación Tasa Atípica F2 + 3 Umbrales** ($> 8,13\%$, `190` casos reales en test) | **Árbol CART (`d=4`)** con umbral móvil ($\tau \in \{0,05; 0,30; 0,40\}$) vs. KNN `k=5`, RF, NB, Ingenuo | **Ingenuo:** `90,25 %` acc pero **`0/190` (`0 %` recall)**.<br/>**CART $\tau=0,40$:** `59/190` con **`100 %` precisión (`0` FP)**.<br/>**CART $\tau=0,30$:** **`188/190` (`99,0 %` recall, F1=`0,619`, Youden `0,859`)**.<br/>**CART $\tau=0,05$:** **`190/190` (`100 %` recall)**.<br/>**KNN `k=5`:** `158/190` (`83,2 %` recall, F1=`0,825`). **NB:** `1.247` FP. | Evita la trampa del *accuracy* y permite elegir el umbral según el costo de error (consumidor `0,05`, balance `0,30`, bloqueo sin falsas alarmas `0,40`). |
| **SIN ETIQUETA** *(No Supervisado)* | **Tipo 3 · Clustering Territorial, Hipotecario y Masivo F1** | **K-Means (`k=3`)**, **Ward** y **MiniBatchKMeans Streaming (`float32`, 24 GB)** | Separa esfuerzo post-hoc `0,50 → 0,60 → 0,90` (**100 % Alto** en C2 London/SE); F2 **silueta `0,479`**; F1 **`11.346.500` ventas en `49,87 MB` RAM (`0,20 %` de 24 GB)** con silueta `0,333` (Ward exigiría ~517 TB). | Descubre segmentos naturales de mercado sin contaminar $X$ con `esfuerzo` y mejora a CART en CV-5 (`78,3 % → 80,0 %`). |
| **SIN ETIQUETA** *(No Supervisado)* | **Tipo 4 · Reglas de Asociación** | **Apriori Puro Python (`tidsets`, `23,58 MB` RAM)** vs. FP-Growth | **`10/10` tickets manuales verificados (`assert`)**.<br/>**`195` reglas F1** (`Detached + Freehold → Extremo >100%`, `conf=68,0 %`, `lift=2,00`).<br/>**`481` reglas F2** (`Tipo:discounted → Tasa:Baja_Q1`, `conf=99,1 %`, `lift=3,96`). | Filtra trivialidades legales (`Flat → Leasehold`) y prioriza combinaciones accionables por `score = lift × soporte`. |

### Modelos preentrenados serializados (`data/modelos/`, inferencia `< 5 ms`)
1. `modelo_ganador_clasificacion.joblib` (`1,35 KB`) — Árbol CART Regional (`max_depth=4`).
2. `modelo_regresion_esfuerzo.joblib` (`1,22 KB`) — Pipeline `StandardScaler + Ridge`.
3. `modelo_clasificacion_tasa_f2.joblib` (`1,62 KB`) — Árbol CART Hipotecario F2 (`max_depth=4`).
4. `modelo_regresion_tasa_f2.joblib` (`2.563,95 KB`) — `RandomForestRegressor (200, max_depth=10)` F2.

---

## 7. Presentación Interactiva Web, Simulador Movible y PowerPoint

### 7.1 `Presentacion_Interactiva_Proyecto_UK.html` (`2,45 MB`, 15 Diapositivas)
- **Visualización de alta calidad:** **18 gráficas incrustadas en Base64** (nunca se rompen), sin nombres `.png` en los textos explicativos, visor modal Lightbox **`🔍 Agrandar gráfica`** (zoom `60 %–300 %` con botones o rueda del ratón), botón **`⛶ Pantalla Completa`** (tecla `F`) y botón **`📥 Descargar HTML Autónomo`**.
- **Simulador Interactivo Dual en la Diapositiva 15:**
  - **Opción 1 · Predicción por Opciones:** selectores de 10 regiones, **110 ciudades reales** de `precios_ciudad`, método predictivo, tipo de tasa, perfil comprador, plazo (`15–35 años`), **5 botones rápidos de casos base** y deslizadores en tiempo real (`oninput`) de Precio (`£80k–£750k`), Variación (`-15 %` a `+25 %`), Volumen de ventas, `LTV (50 %–95 %)`, Comisión (`£0–£3.000`), Clima y Umbral $\tau$ (`0,05–0,80`).
  - **Opción 2 · Predicción A Escrita:** caja de texto libre en español con reconocimiento en vivo de ciudad, región, precio en libras, entrada/`LTV`, plazo, hipoteca y método + **8 botones de ejemplo listos para un clic** (`London`, `Manchester`, `Newcastle / North East`, `South East`, `Oxford con CART`, `Cardiff con KNN`, `Birmingham con Ridge`, `Bristol con Apriori / K-Means`).
  - **Gráfica Interactiva Movible (`#svgInteractivo`, *Drag & Drop*):**
    - *Vista A (Mapa Precio vs. Esfuerzo %):* 10 burbujas regionales clicables, zonas de riesgo (`30 %` y `56,4 %`), curva PMT y punto arrastrable **`● TU CASO`** que al moverlo con el mouse modifica precio, variación, LTV, cuota mensual y recalcula los 7 métodos en vivo.
    - *Vista B (Curva Móvil de Umbral $\tau$):* línea vertical arrastrable de `0,05` a `0,80` mostrando en tiempo real `TP / FP / Recall / Precisión / F1` sobre los 190 casos atípicos de F2.
  - **Explicación paso a paso según el método elegido:** pestañas interactivas para `Árbol CART`, `Ridge + PMT`, `RF Tasa F2`, `KNN (5 vecinos históricos reales)`, `K-Means`, `Apriori` y `Naive Bayes`.
  - **Panel de Vinculación Nube Azure ⇄ HTML (al final de la Diapositiva 15):** incluye 3 pestañas de código real y copiable (**Python `psycopg`**, **JavaScript `fetch` + respaldo autónomo** y **Consulta SQL dinámica en vivo**) más el estado de conexión con `public.historial_predicciones`.
- **Funcionamiento Dual Garantizado:**
  - **Con servidor (`http://localhost:8765`):** consulta Azure PostgreSQL en vivo y guarda cada prueba en `public.historial_predicciones`.
  - **Sin servidor (descargando o pasando SOLO el archivo `.html` al profesor):** conmuta automáticamente al motor matemático `CEREBRO` embebido dentro del propio HTML y funciona al **100 % sin Python, sin Azure y sin internet**.

### 7.2 `Presentacion_Proyecto_UK.pptx` (`1,38 MB`)
- Presentación nativa de Microsoft PowerPoint (`14` diapositivas panorámicas `16:9` con `14` figuras PNG embebidas, generada por `script/documentacion/generar_presentacion_ppt.py`).

---

## 8. Azure Database for PostgreSQL (`pg-actividad-db-uk26`) y Enlace Compartible

### 8.1 Datos de Infraestructura y Roles de Seguridad
- **Grupo de recursos:** `rg-actividad-db`
- **Servidor Flexible PostgreSQL 16.15:** `pg-actividad-db-uk26.postgres.database.azure.com` (puerto `5432`, cifrado obligatorio `sslmode=require`)
- **Base de datos:** `db_actividad_mineria`
- **Estado actual:** **`Ready` (en línea)** vinculado a Render (`https://uk-vivienda-api.onrender.com`) y Netlify (`https://uk-vivienda-joven.netlify.app/`).
- **Sitio Público en Vivo (Netlify):** `https://uk-vivienda-joven.netlify.app/` (incluye botón con logo oficial de Microsoft Azure para consultar las tablas y descargar el libro Excel multi-hoja `.xlsx` de 5 hojas, soporte dual **Libras `£ GBP` y Pesos Colombianos `$ COP`**, `LTV/TLV`, plazo de `5` a `40 años` / `60` a `480 meses máximo`, escritura libre 100 % bilingüe **Español/Inglés** y 3 gráficas interactivas).
- **API Pública en Vivo (Render):** `https://uk-vivienda-api.onrender.com` (rama `deploy/backend` en `uk-land-pricing-analytics.git`, proxy transparente `/api/*` desde Netlify).
- **Acceso directo en Azure Portal:** `https://portal.azure.com/#@uniminuto.edu/resource/subscriptions/055b858b-7fbc-4b45-bf00-e072f09c083c/resourceGroups/rg-actividad-db/providers/Microsoft.DBforPostgreSQL/flexibleServers/pg-actividad-db-uk26/databases`

| Variable en `.env` | Rol PostgreSQL | Permisos | Propósito |
|---|---|---|---|
| `DATABASE_URL` | `pgadmin_uk` | `ALL PRIVILEGES` | Solo administración local (crear esquema y subir CSV vía `COPY`). **No compartir.** |
| **`DATABASE_URL_READONLY`** | **`lector_uk`** | **`SELECT` únicamente** (escritura bloqueada y verificada) | **Enlace Único Compartible** para consultas y predicciones de terceros/profesores. |
| `DATABASE_URL_APP` | `app_predicciones` | `SELECT` global + `INSERT` en `historial_predicciones` | Usado por `app_presentacion_interactiva.py` y `predecir_azure.py` para registrar pruebas. |

**Plantilla del Enlace Único Compartible (las contraseñas reales están guardadas únicamente en `.env`):**
```text
postgresql://lector_uk:<PASSWORD>@pg-actividad-db-uk26.postgres.database.azure.com:5432/db_actividad_mineria?sslmode=require
```

### 8.2 Inventario Exacto de las 19 Tablas Cargadas y Verificadas en Azure (`24.777` Registros)
| # | Tabla en `public` (Azure) | Filas Cargadas | Origen CSV / Parquet |
|---:|---|---:|---|
| 1 | `maestro_regional` | 60 | `data/integrado/accesibilidad_vivienda_joven_uk_v1.csv` |
| 2 | `precios_ciudad` | 14.050 | `data/integrado/precios_ciudad_limpio.csv` (`1.153` ciudades UK) |
| 3 | `hipotecas_catalogo` | 7.796 | `data/integrado/hipotecas_catalogo_limpio.csv` |
| 4 | `hipoteca_2022` | 10 | `data/integrado/hipoteca_2022.csv` |
| 5 | `historico_nacional` | 22 | `data/integrado/historico_nacional.csv` |
| 6 | `torneo_clasificadores` | 8 | `data/integrado/torneo_clasificadores_resumen.csv` |
| 7 | `comparativa_regresion_dual` | 8 | `data/integrado/comparativa_regresion_dual.csv` |
| 8 | `segmentos_territoriales` | 60 | `data/integrado/segmentos_territoriales.csv` |
| 9 | `segmentos_hipotecas` | 3 | `data/integrado/segmentos_hipotecas_muestra.csv` |
| 10 | `segmentos_transacciones_f1` | 4 | `data/integrado/segmentos_transacciones_1m.csv` |
| 11 | `clustering_benchmark_24gb` | 5 | `data/integrado/clustering_benchmark_24gb.csv` |
| 12 | `reglas_asociacion_inmobiliario` | 195 | `data/integrado/reglas_asociacion_inmobiliario.csv` |
| 13 | `reglas_asociacion_hipotecas` | 481 | `data/integrado/reglas_asociacion_hipotecas.csv` |
| 14 | `evaluacion_umbrales` | 11 | `data/integrado/evaluacion_umbrales_resumen.csv` |
| 15 | `comparativa_global_modelos` | 8 | `data/integrado/comparativa_global_modelos.csv` |
| 16 | `telemetria_rendimiento` | 13 | `data/integrado/telemetria_rendimiento.csv` |
| 17 | `auditoria_fuentes_calidad` | 43 | `data/integrado/auditoria_fuentes_calidad.csv` |
| 18 | `transacciones_f1` | 2.000 | `data/limpia/uk_property_price_limpio_muestra.parquet` |
| 19 | `historial_predicciones` | Dinámica | Bitácora en vivo de cada predicción ejecutada desde el HTML / CLI |

### 8.3 Líneas de Código que Vinculan Azure PostgreSQL con el HTML Interactivo
**Python (`script/azure/predecir_azure.py` + `app_presentacion_interactiva.py`):**
```python
with psycopg.connect(DATABASE_URL_READONLY, connect_timeout=4) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ciudad, region, anio, n_transacciones, precio_mediana "
            "FROM public.precios_ciudad WHERE LOWER(ciudad) = LOWER(%s) "
            "ORDER BY anio DESC LIMIT 1;",
            (ciudad,),
        )
        fila_bd = cur.fetchone()
```
**JavaScript (`Presentacion_Interactiva_Proyecto_UK.html` — En Vivo ⇄ Autónomo):**
```javascript
if (window.location.protocol.startsWith("http")) {
  const resp = await fetch("/api/predecir?" + new URLSearchParams(parametros));
  if (resp.ok) { const datosAzure = await resp.json(); /* muestra historial_predicciones */ }
}
// Si se abre solo el .html descargado sin servidor, calcula localmente con CEREBRO embebido
```

---

## 9. Telemetría (13/13 Etapas `OK`) y Buenas Prácticas

- **Telemetría verificada:** las **13 etapas** del pipeline operan en estado **`OK`** por debajo del límite local de `220 MB` de RAM Python (auditoría de 43 archivos y `11,37M` filas: `137,79 MB`; clustering streaming de `11,35M` filas en `float32`: **`49,87 MB`**, equivalente al `0,20 %` de 24 GB).
- **Orquestación completa en consola:** `python main_maestro_completo.py` ejecuta los 11 bloques íntegros en **`8,71 s`**.
- **Controles de calidad con `assert`:** paridad `maestro_joven_uk.parquet` vs. CSV (`dif. max = 5,82e-11`), invarianza del Árbol CART a 4 escaladores, Teorema de Bayes `2 × 2`, cálculo manual de `F1` armónica y verificación exacta `10/10` tickets en Apriori.

---

## 10. Guía Rápida de Comandos para el Día de la Exposición

### Paso 1 · Encender Azure PostgreSQL (5 minutos antes de exponer)
```powershell
& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" postgres flexible-server start -g rg-actividad-db -n pg-actividad-db-uk26
```

### Paso 2 · Verificar conexión TLS y los 3 roles (opcional)
```powershell
python script/azure/test_conexion_azure.py
```

### Paso 3 · Iniciar la Presentación Interactiva conectada en vivo a Azure
```powershell
python app_presentacion_interactiva.py
```
- Abrir en el navegador: **`http://localhost:8765`**
- Pulsar **`⛶ Pantalla Completa`** (o tecla `F`).
- En la **Diapositiva 15**, probar predicciones **por opciones**, **a escrita** (o los 8 botones de ejemplo) y **arrastrando el punto en la gráfica**.

### Paso 4 · Probar predicciones por terminal conectadas a Azure (opcional)
```powershell
# Por opciones:
python script/azure/predecir_azure.py --region London --metodo todos

# A escrita (texto libre):
python script/azure/predecir_azure.py --texto "Quiero comprar vivienda en Oxford de 420000 libras como First time buyer con entrada de 15% a 30 anios y comision 999 explicado con KNN"
```

### Paso 5 · Compartir solo el HTML al profesor (funciona sin Azure y sin Python)
- Enviar directamente **`Presentacion_Interactiva_Proyecto_UK.html`** (o usar el botón **`📥 Descargar HTML Autónomo`**).

### Paso 6 · Pausar Azure al terminar la exposición (ahorrar créditos)
```powershell
& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" postgres flexible-server stop -g rg-actividad-db -n pg-actividad-db-uk26
```
*(Nota: Azure reactiva automáticamente el servidor tras 7 días de pausa; si deseas mantenerlo apagado más tiempo, vuelve a ejecutar el comando `stop`).*
