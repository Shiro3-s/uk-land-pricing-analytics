# Backend API · Vivienda UK (Render + Azure PostgreSQL + Excel)

Servicio Flask/Waitress que conecta los modelos entrenados (`data/modelos/*.joblib`) con **Microsoft Azure Database for PostgreSQL** (`pg-actividad-db-uk26 / db_actividad_mineria`) y genera en vivo el libro Excel multi-hoja (`.xlsx`, 5 hojas).

## Archivos
- `web_api.py`: endpoints `/healthz`, `/api/contexto`, `/api/predecir`, `/api/datos` (`regional`, `ciudades`, `hipoteca_2022`, `metodos`) y `/api/datos/exportar.xlsx`.
- `exportar_excel_blob.py`: regenera `vivienda-uk-azure-datos.xlsx` y permite sincronizarlo con **Azure Blob Storage** (`stactividaduk26 / excel-datos`).
- `requirements.txt` y `render.yaml`: dependencias y manifiesto de despliegue en **Render** (`https://uk-vivienda-api.onrender.com`).

## Unión con Base de Datos y Modelos
Usa los módulos de `script/azure/`:
- `script/azure/datos_web.py`: consultas parametrizadas a Azure PostgreSQL con respaldo automático en `data/integrado/*.csv`.
- `script/azure/predecir_azure.py`: motor predictivo dual (bilingüe ES/EN, GBP y COP, LTV/TLV, plazo hasta 480 meses, KNN 5 vecinos en vivo).

## Ejecución local
```powershell
python backend/web_api.py
```
Variables de entorno (en `.env` local, nunca versionado): `DATABASE_URL_READONLY` y `DATABASE_URL_APP`.
