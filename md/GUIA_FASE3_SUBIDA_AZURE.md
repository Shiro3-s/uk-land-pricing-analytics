# Guía Completa de Fase 3 · Subida de Base de Datos y Archivos CSV («Excel») a Azure Database for PostgreSQL y Enlace Compartible de Predicción

**Proyecto:** Accesibilidad de Vivienda para Jóvenes (18–34 años) en el Reino Unido  
**Autores:** Juan Pablo Angel Quitian · Joel Sebastian Bueno Medina  
**Estado de la Fase 3:** Documentada y con scripts listos en `script/azure/` para ejecutar cuando se decida realizar la carga al servidor en la nube.

---

## 1. Propósito de la Fase 3 (Fase 6 de CRISP-DM: Despliegue en la Nube)

Esta fase tiene tres objetivos operativos:
1. **Subir el esquema relacional ([`data/modelos/esquema_bd.sql`](../data/modelos/esquema_bd.sql)) y las 17 tablas CSV limpias («Excel», `1,83 MB` en `data/integrado/*.csv`)** —más la muestra de `transacciones_f1`— a **Azure Database for PostgreSQL Flexible Server**.
2. **Habilitar un Enlace Único Compartible de Solo Lectura (`DATABASE_URL_READONLY` con rol `lector_uk` y cifrado TLS `sslmode=require`)** para que cualquier persona, evaluador o aplicación pueda conectarse, consultar las tablas reales en la nube y alimentar el motor de predicciones sin riesgo de modificar ni borrar datos.
3. **Permitir que cualquier usuario con el enlace ejecute predicciones (por opciones o a escrita)** viendo claramente **qué predicción utilizar según el método (Con Etiqueta vs. Sin Etiqueta) y su cantidad exacta de aciertos**.

---

## 2. Arquitectura de Seguridad y Roles en Azure PostgreSQL

El servidor Azure Database for PostgreSQL utiliza tres roles de mínimo privilegio configurados mediante [`script/azure/configurar_roles_azure.py`](../script/azure/configurar_roles_azure.py):

| Variable en `.env` | Rol PostgreSQL | Permisos en Esquema `public` | Quién lo usa y para qué |
|---|---|---|---|
| `DATABASE_URL` | `pgadmin_uk` | `ALL PRIVILEGES` (DDL + DML) | **Solo administrador local:** crea las tablas de `esquema_bd.sql` y sube los 17 archivos CSV con `script/azure/subir_datos_azure.py`. **No se comparte.** |
| **`DATABASE_URL_READONLY`** | **`lector_uk`** | **`SELECT` únicamente** (escritura/borrado bloqueados) | **Enlace Único Compartible:** es el link que se comparte para que cualquiera consulte las tablas desde Azure y ejecute predicciones en `predecir_azure.py` o `app_presentacion_interactiva.py`. |
| `DATABASE_URL_APP` | `app_predicciones` | `SELECT` en todas las tablas + `INSERT` en `historial_predicciones` | **Motor de Predicción / API Web:** lee los perfiles desde Azure y guarda cada consulta en la bitácora `historial_predicciones`. |

---

## 3. Inventario de las 17 Tablas CSV («Excel») + Tablas SQL que se Suben a Azure

Todas las tablas limpias e integradas viven en `data/integrado/` (0 duplicados verificados por `auditoria_datos.py`):

| # | Tabla en Azure PostgreSQL | Archivo CSV («Excel») Fuente | Filas | Peso (KB) | Qué almacena |
|---:|---|---|---:|---:|---|
| 1 | `maestro_regional` | `data/integrado/accesibilidad_vivienda_joven_uk_v1.csv` | 60 | 13,31 | Cuadro maestro 10 regiones × 6 años (2015–2020): precios, salarios jóvenes, años de salario y clima. |
| 2 | `precios_ciudad` | `data/integrado/precios_ciudad_limpio.csv` | 14.050 | 806,97 | Precio medio, mediano y volumen de ventas anual en **1.153 ciudades** del Reino Unido. |
| 3 | `hipotecas_catalogo` | `data/integrado/hipotecas_catalogo_limpio.csv` | 7.796 | 934,74 | Catálogo hipotecario limpio F2 (bancos, tasas winsorizadas $\le 8,13\%$, APR, comisiones, plazos, bandera atípica). |
| 4 | `hipoteca_2022` | `data/integrado/hipoteca_2022.csv` | 10 | 1,35 | Corte hipotecario regional 2022 (cuota francesa PMT al 90 % LTV y 25 años vs. umbral 30 %). |
| 5 | `historico_nacional` | `data/integrado/historico_nacional.csv` | 22 | 0,93 | Serie histórica nacional (1999–2020) de precios reales y salario joven en GBP 2020. |
| 6 | `torneo_clasificadores` | `data/integrado/torneo_clasificadores_resumen.csv` | 8 | 2,31 | Resultados del Torneo Supervisado (Árbol CART vs. Naive Bayes vs. KNN escalado y sin escalar). |
| 7 | `comparativa_regresion_dual` | `data/integrado/comparativa_regresion_dual.csv` | 8 | 0,97 | Resultados de Regresión Continua Sin Fuga (`Ridge`, `RandomForestRegressor`, `KNN`, `Árbol`). |
| 8 | `segmentos_territoriales` | `data/integrado/segmentos_territoriales.csv` | 60 | 13,68 | Segmentación K-Means ($k=3$) de mercados regionales con nombres y acciones de política pública. |
| 9 | `segmentos_hipotecas` | `data/integrado/segmentos_hipotecas_muestra.csv` | 3 | 0,54 | Perfiles de los 3 clústeres del catálogo hipotecario F2 en escala original. |
| 10 | `segmentos_transacciones_f1` | `data/integrado/segmentos_transacciones_1m.csv` | 4 | 0,53 | Los 4 macro-segmentos inmobiliarios descubiertos en *streaming* sobre los 11,35M de ventas F1. |
| 11 | `clustering_benchmark_24gb` | `data/integrado/clustering_benchmark_24gb.csv` | 5 | 0,36 | Telemetría real de escalabilidad `10k -> 100k -> 1M -> 5M -> 11,35M` dentro de 24 GB de RAM. |
| 12 | `reglas_asociacion_inmobiliario` | `data/integrado/reglas_asociacion_inmobiliario.csv` | 195 | 27,74 | Reglas Apriori sobre compraventas F1 clasificadas en accionables vs. triviales legales. |
| 13 | `reglas_asociacion_hipotecas` | `data/integrado/reglas_asociacion_hipotecas.csv` | 481 | 53,48 | Reglas Apriori sobre condiciones crediticias del catálogo hipotecario F2. |
| 14 | `evaluacion_umbrales` | `data/integrado/evaluacion_umbrales_resumen.csv` | 11 | 0,78 | Barrido de 11 umbrales (`0,05` a `0,80`) con conteo de `TP, FP, FN, TN`, Precisión, Recall, F1 y Youden $J$. |
| 15 | `comparativa_global_modelos` | `data/integrado/comparativa_global_modelos.csv` | 8 | 1,98 | Semáforo global de todos los algoritmos evaluados (Con Etiqueta y Sin Etiqueta). |
| 16 | `telemetria_rendimiento` | `data/integrado/telemetria_rendimiento.csv` | 13 | 1,73 | Tiempos de ejecución (s) y consumo pico de RAM (MB) de las 13 etapas del proyecto. |
| 17 | `auditoria_fuentes_calidad` | `data/integrado/auditoria_fuentes_calidad.csv` | 38 | 10,31 | Auditoría sanitaria completa de los archivos CSV y Parquet del repositorio. |
| 18 | `transacciones_f1` | `data/limpia/uk_property_price_limpio_muestra.parquet` | 2.000 *(o 11,37M)* | 135,00 | Muestra georreferenciada de transacciones individuales del Land Registry (escalable a 11,37M vía `COPY`). |
| 19 | `historial_predicciones` | *(Tabla dinámica de bitácora en Azure)* | Dinámica | — | Registra cada predicción realizada en vivo (fecha, entrada JSON, clasificación, regresión, umbral y latencia). |

---

## 4. Qué Predicción Utilizar según el Método (Con/Sin Etiqueta) y su Cantidad de Aciertos

Cuando un usuario se conecta mediante el enlace compartible de Azure (`DATABASE_URL_READONLY`), el sistema ofrece esta guía cuantitativa para elegir el método adecuado:

| Método / Algoritmo | ¿Con o Sin Etiqueta? | Tipo de Predicción | Cantidad Exacta de Aciertos y Precisión Validada | Cuándo Utilizar Este Método |
|---|---|---|---|---|
| **1. Árbol de Decisión CART (`max_depth=4`)** | **CON ETIQUETA** *(Supervisado)* | **Tipo 2 · Clasificación + Probabilidad + Reglas + 3 Umbrales** | • **`17 / 18` aciertos en Test Regional (`94,4 %`, F1=`0,941`)**<br/>• **`47 / 60` en CV-5 (`78,3 %`**, sube a **`48 / 60 = 80,0 %`** con clúster K-Means)<br/>• **`9 / 10` regiones reales acertadas (`90,0 %`)**<br/>• **`1.805 / 1.949` aciertos en F2 Tasa Elevada $\ge Q_3$ (`92,6 %`, AUC=`0,943`)**<br/>• **`188 / 190` hipotecas atípicas detectadas con umbral $\tau=0,30$ (`99,0 %` recall)**; **`190 / 190` (`100 %`)** con $\tau=0,05$; **`0` falsas alarmas (`100 %` precisión)** con $\tau=0,40$. | **GANADOR PRINCIPAL (`4,82 / 5,00`):** utilizar siempre que se necesite clasificar si una región o hipoteca es de riesgo **Alto o Moderado / Alerta** y **explicar al usuario las reglas `SI / ENTONCES`** que justifican la decisión. |
| **2. Regresión Ridge (`alpha=1.0`, Pipeline)** | **CON ETIQUETA** *(Supervisado)* | **Tipo 1A · Regresión Continua Regional (sin fuga)** | • **$R^2_{\text{CV-5}} = 0,909 \pm 0,037$**<br/>• **Error medio absoluto $\text{MAE} = \pm 4,21$ puntos porcentuales** de salario joven (`RMSE = 5,08 pp`) | **GANADOR REGRESIÓN REGIONAL:** utilizar cuando se requiere saber **qué porcentaje exacto del salario joven** consumirá la cuota hipotecaria en una región. |
| **3. RandomForestRegressor (`200 árboles, d=10`)** | **CON ETIQUETA** *(Supervisado)* | **Tipo 1B · Regresión Continua Hipotecaria F2** | • **$R^2_{\text{test}} = 0,964$** (sobre `1.949` hipotecas de prueba)<br/>• **Error medio absoluto $\text{MAE} = \pm 0,169$ puntos porcentuales de tasa** (`RMSE = 0,251 pp`) | **GANADOR REGRESIÓN HIPOTECAS:** utilizar para estimar **la tasa de interés inicial exacta (%)** que cobrará un banco según comisión, plazo, tipo y perfil comprador. |
| **4. KNN Escalado (`k=5`, Pipeline `StandardScaler`)** | **CON ETIQUETA** *(Supervisado)* | **Tipo 2 · Clasificación por Vecinos Más Cercanos** | • **`17 / 18` aciertos en Test Regional (`94,4 %`)**<br/>• **`54 / 60` aciertos en CV-5 Regional (`90,0 %`)**<br/>• **`9 / 10` regiones reales acertadas (`90,0 %`)**<br/>• **`1.842 / 1.949` aciertos en F2 Tasa Elevada (`94,5 %`, AUC=`0,975`)**<br/>• **`158 / 190` hipotecas atípicas detectadas en F2 (`83,2 %` recall, `81,9 %` precisión, F1=`0,825`)** | **RETADOR (*CHALLENGER*) DE PRECISIÓN BRUTA:** utilizar como **segunda opinión de máxima precisión local** (especialmente en validación cruzada y detección balanceada F1), sabiendo que es 11× más lento y no entrega reglas. |
| **5. Random Forest Clasificador (`200 árboles`)** | **CON ETIQUETA** *(Supervisado)* | **Tipo 2 · Clasificación por Ensamble** | • **`17 / 18` en Test Regional (`94,4 %`)** y **`9 / 10` regiones (`90,0 %`)**<br/>• **`ROC-AUC = 0,975` en F2 Atípico** con **`100 %` de precisión (`67/190` detectadas y `0` falsas alarmas en umbral `0,50`)** | **REFERENCIA DE ENSAMBLE:** utilizar cuando se priorice **cero falsas alarmas** y probabilidades suavizadas sin requerir árbol de reglas único. |
| **6. K-Means ($k=3$) + Ward + MiniBatchKMeans ($k=4$)** | **SIN ETIQUETA** *(No Supervisado)* | **Tipo 3 · Agrupamiento / Segmentación de Mercados** | • **Separa al `100 %` el núcleo metropolitano crítico** (London y South East, esfuerzo medio post-hoc `0,90`)<br/>• **Silueta `0,479`** en catálogo hipotecario F2 completo (`7.796` productos)<br/>• **Silueta `0,333`** sobre **`11.346.500` transacciones F1** en `49,87 MB` de RAM | **DESCUBRIMIENTO DE SEGMENTOS SIN ETIQUETA:** utilizar para identificar a qué **grupo estructural de mercado o tipo de oferta hipotecaria** pertenece una consulta. |
| **7. Reglas de Asociación Apriori (`score = lift × soporte`)** | **SIN ETIQUETA** *(No Supervisado)* | **Tipo 4 · Patrones Co-ocurrentes ($A \to B$)** | • **`10 / 10` tickets manuales verificados (`100 %` exactitud)**<br/>• **`195` reglas en F1** (`Detached + Freehold -> Acceso Extremo >100%` con **`68,0 %` confianza, `lift = 2,00`**)<br/>• **`481` reglas en F2** (`Tipo:discounted -> Tasa:Baja_Q1` con **`99,1 %` confianza, `lift = 3,96`**) | **RECOMENDADOR DE COMBINACIONES:** utilizar para advertir qué **combinaciones de vivienda o crédito** disparan el riesgo o garantizan tasas bajas. |
| **8. Controles Negativos (Naive Bayes / KNN sin escalar / Modelo Ingenuo)** | **CON ETIQUETA** *(Descartados)* | **Líneas base de control metodológico** | • **Naive Bayes:** cae a **`8 / 18` en Test (`44,4 %`)** y genera **`1.247` falsas alarmas** en F2 (`Brier = 0,505`)<br/>• **KNN sin escalar:** cae a **`12 / 18` (`66,7 %, -27,7 pp`)** por tiranía de `n_transacciones`<br/>• **Modelo Ingenuo:** `90,25 %` accuracy pero **`0 / 190` atípicas detectadas (`0 %` recall)** | **NO USAR PARA PREDECIR:** se muestran únicamente en la tabla comparativa para demostrar empíricamente por qué escalamos, por qué calibramos umbrales y por qué descartamos Naive Bayes. |

---

## 5. Paso a Paso para Ejecutar la Fase 3 Cuando Quieras Subir a Azure

### Paso 1 · Reanudar el servidor de Azure (si estaba detenido para ahorrar créditos) y verificar `.env`
```powershell
az postgres flexible-server start -g rg-actividad-db -n <TU_SERVIDOR_AZURE>
```
Verifica que tu archivo `.env` en la raíz tenga configurados `DATABASE_URL`, `DATABASE_URL_READONLY` y `DATABASE_URL_APP` (según la plantilla `.env.example`).

### Paso 2 · Configurar roles y subir las 17 tablas CSV («Excel») + esquema SQL a Azure
```powershell
python script/azure/configurar_roles_azure.py
python script/azure/subir_datos_azure.py
python script/azure/test_conexion_azure.py
```
- `subir_datos_azure.py` crea todas las tablas de `data/modelos/esquema_bd.sql`, sube las 17 tablas CSV de `data/integrado/` más las 2.000 filas de muestra en `transacciones_f1`, aplica permisos `SELECT` al rol `lector_uk` y verifica los conteos en vivo.

### Paso 3 · Habilitar acceso en el Firewall de Azure para quienes usen el enlace compartible
- Para compartir con una IP específica:
  ```powershell
  az postgres flexible-server firewall-rule create -g rg-actividad-db -n <TU_SERVIDOR_AZURE> --rule-name AllowColaborador --start-ip-address <IP> --end-ip-address <IP>
  ```
- O para permitir que cualquier persona con el enlace `DATABASE_URL_READONLY` consulte desde cualquier red (protegido por TLS `sslmode=require` y el rol `lector_uk` que tiene bloqueada toda escritura):
  ```powershell
  az postgres flexible-server firewall-rule create -g rg-actividad-db -n <TU_SERVIDOR_AZURE> --rule-name AllowPublicReadOnly --start-ip-address 0.0.0.0 --end-ip-address 255.255.255.255
  ```

### Paso 4 · Consultar y hacer predicciones usando el enlace compartible de Azure
Cualquier persona con el enlace `DATABASE_URL_READONLY` puede ejecutar:

1. **Predicción desde la Presentación Interactiva Web conectada a Azure:**
   ```powershell
   python app_presentacion_interactiva.py
   ```
   *(Abre `http://localhost:8765` en el navegador: en la última diapositiva permite predecir **por opciones** o **a escrita** consultando Azure PostgreSQL en vivo).*

2. **Predicción por terminal (por opciones o a escrita) conectada a Azure:**
   ```powershell
   # A) Por opciones (Región o Ciudad + Comparativa de todos los métodos y sus aciertos)
   python script/azure/predecir_azure.py --region London --metodo todos

   # B) Por opciones (Oferta hipotecaria F2 + 3 umbrales)
   python script/azure/predecir_azure.py --hipoteca --comisiones 999 --meses 24 --tipo fixed --subtitulo "First time buyer"

   # C) A escrita (Consulta en texto libre en español)
   python script/azure/predecir_azure.py --texto "Quiero comprar vivienda en London como First time buyer con hipoteca fixed a 24 meses y comision 999"

   # D) Pasando directamente el link compartido por línea de comandos
   python script/azure/predecir_azure.py --url "postgresql://lector_uk:<PASSWORD>@<SERVIDOR>.postgres.database.azure.com:5432/db_actividad_mineria?sslmode=require" --region "North East" --metodo todos
   ```
