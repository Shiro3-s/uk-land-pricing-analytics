# Frontend · Vivienda UK (Netlify)

Interfaz web pública desplegada en **Netlify** (`https://uk-vivienda-joven.netlify.app/`), conectada a la API de **Render** mediante proxy transparente `/api/*` definido en `netlify.toml`.

## Archivos
- `index.html`: simulador de accesibilidad (modo *Por opciones* y *Escribir mi caso ES/EN*, botón `Restablecer`, selector GBP/COP, LTV/TLV, plazo hasta 480 meses y visor `Azure BD · Excel`).
- `styles.css`: diseño editorial responsive (escritorio `1440px` y móvil `390px`).
- `app.js`: cliente HTTP con reintentos anti-timeout ante arranque en frío de Render y generación de las 3 gráficas SVG + serie histórica.
- `metodologia.html`: presentación ejecutiva `16:9` de 13 diapositivas comparativas (generada desde `script/documentacion/generar_presentacion_ppt.py`).

## Publicación
Cada push a la rama `deploy/frontend` en `Shiro3-s/uk-land-pricing-analytics` dispara automáticamente `.github/workflows/publicar-frontend.yml` usando `publish = "frontend"` definido en `netlify.toml`.
