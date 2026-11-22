"use strict";

const byId = (id) => document.getElementById(id);
const gbp = new Intl.NumberFormat("es-ES", {style: "currency", currency: "GBP", currencyDisplay: "narrowSymbol", maximumFractionDigits: 0, useGrouping: "always"});
const decimal = new Intl.NumberFormat("es-ES", {maximumFractionDigits: 2});
const integer = new Intl.NumberFormat("es-ES", {maximumFractionDigits: 0, useGrouping: "always"});
const percent = (value) => decimal.format(value) + " %";
const money = (value) => gbp.format(value);
const defaultRegions = ["East Midlands", "East of England", "London", "North East", "North West", "South East", "South West", "Wales", "West Midlands", "Yorkshire and The Humber"];
let cities = [];
let writing = false;
let currentResult = null;
let waitingForPrediction = false;
let tableRequest = 0;
let chartRequest = 0;
let currentData = null;
let contextPromise;
let lastContextTime = 0;

function element(tag, value) {
  const el = document.createElement(tag);
  el.textContent = value;
  return el;
}

function filters(region, year) {
  const params = new URLSearchParams();
  if (region) params.set("region", region);
  if (year) params.set("anio", year);
  return params.toString() ? "?" + params.toString() : "";
}

function setRegionOptions(regions) {
  for (const id of ["region", "data-region"]) {
    const select = byId(id);
    const previous = select.value;
    select.replaceChildren();
    if (id === "data-region") select.add(new Option("Todas las regiones", ""));
    for (const name of regions) select.add(new Option(name, name));
    if (regions.includes(previous)) select.value = previous;
    else select.value = id === "region" ? "London" : "";
  }
}

function populateCities() {
  const region = byId("region").value;
  const select = byId("city");
  const previous = select.value;
  select.replaceChildren(new Option("Referencia regional", ""));
  const names = cities.filter((city) => city.region === region).map((city) => city.ciudad);
  for (const name of Array.from(new Set(names)).sort()) select.add(new Option(name, name));
  if (names.includes(previous)) select.value = previous;
}

function setMode(textMode) {
  writing = textMode;
  byId("mode-options").setAttribute("aria-pressed", String(!writing));
  byId("mode-text").setAttribute("aria-pressed", String(writing));
  byId("option-fields").hidden = byId("option-fields").disabled = writing;
  byId("text-fields").hidden = byId("text-fields").disabled = !writing;
}

async function readApi(path) {
  let failure;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(path, {cache:"no-store", signal:AbortSignal.timeout(28000)});
      if (!response.ok && response.status < 500) throw new Error("petición rechazada");
      if (!response.ok) throw new Error("servidor iniciándose");
      return await response.json();
    } catch (error) {
      failure = error;
      if (error.message === "petición rechazada") break;
      if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }
  throw failure;
}

function historyShape(rows) {
  const width = 500, height = 95, padding = 5;
  const values = rows.map((row) => Number(row.precio_mediana));
  const highest = Math.max(...values), lowest = Math.min(...values);
  const positions = values.map((value, index) => {
    const x = padding + (width - padding * 2) * index / Math.max(1, values.length - 1);
    const y = height - 10 - (height - 24) * (value - lowest) / (highest - lowest || 1);
    return [x, y];
  });
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 500 95");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", rows.map((r) => r.anio + ", " + money(r.precio_mediana)).join(". "));
  function shape(tag, attributes) {
    const child = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attributes)) child.setAttribute(key, value);
    svg.append(child);
  }
  for (const y of [15, 49, 85]) shape("line", {x1:0,y1:y,x2:width,y2:y,stroke:"#c4d5bc",opacity:".22","stroke-dasharray":"3 6"});
  const points = positions.map((p) => p.join(",")).join(" ");
  shape("polygon", {points:padding + "," + height + " " + points + " " + (width-padding) + "," + height, fill:"#bde29f",opacity:".14"});
  shape("polyline", {points, fill:"none",stroke:"#c6e7a4","stroke-width":"3","vector-effect":"non-scaling-stroke"});
  for (const [cx,cy] of positions) shape("circle",{cx,cy,r:3.5,fill:"#e1f3c8"});
  return svg;
}

async function showRegionalChart() {
  if (currentResult) return;
  const region = byId("region").value;
  const request = ++chartRequest;
  byId("overview-region").textContent = region;
  byId("overview-link").href = "#datos" + filters(region);
  byId("overview-price").textContent = "—";
  byId("overview-source").textContent = "Consultando datos regionales";
  byId("history-chart").textContent = "Cargando la serie histórica…";
  const notice = setTimeout(() => {
    if (request === chartRequest && !currentResult) byId("history-chart").textContent = "Conectando con la API. El primer acceso puede tardar un minuto.";
  }, 5000);
  try {
    const data = await readApi("/api/datos" + filters(region));
    if (request !== chartRequest || currentResult) return;
    const rows = data.filas.slice().sort((a,b) => a.anio - b.anio);
    if (!rows.length) throw new Error("sin registros");
    const latest = rows.at(-1);
    byId("overview-price").textContent = money(latest.precio_mediana);
    byId("overview-source").textContent = "Fuente: " + data.fuente + " · " + latest.anio;
    const axis = document.createElement("div");
    axis.className = "history-axis";
    axis.append(element("span", rows[0].anio), element("span", latest.anio));
    byId("history-chart").replaceChildren(historyShape(rows), axis);
  } catch {
    if (request === chartRequest) {
      byId("history-chart").textContent = "La serie histórica no está disponible en este momento.";
      byId("overview-source").textContent = "Datos pendientes de conexión";
    }
  } finally { clearTimeout(notice); }
}

function addDetail(list, name, value) {
  const wrapper = document.createElement("div");
  wrapper.append(element("dt",name), element("dd",value));
  list.append(wrapper);
}

function renderResult(data, payload) {
  const calc = data.calculo_financiero_acoplado;
  const regional = data.prediccion_regional_dual;
  const town = data.ciudad_consultada;
  const effort = Number(calc.esfuerzo_pmt_acoplado_pct);
  if (!Number.isFinite(effort) || !Number.isFinite(Number(calc.cuota_mensual_pmt_gbp))) throw new Error("Respuesta incompleta");
  const scale = Math.max(100, Math.ceil(effort / 25) * 25);
  const area = town?.ciudad ? town.ciudad + " · " + regional.region : regional.region;
  const manualPrice = Boolean(payload.precio_gbp || data.interpretacion_texto?.precio_gbp);
  const priceYear = data.anio_precio ?? (manualPrice ? null : town?.anio ?? data.anio_referencia);
  byId("monthly").textContent = money(calc.cuota_mensual_pmt_gbp);
  byId("result-place").textContent = area;
  byId("effort-value").textContent = percent(effort);
  byId("effort-bar").style.width = 100 * effort / scale + "%";
  byId("reference").style.left = 100 * 30 / scale + "%";
  byId("max-scale").textContent = percent(scale);
  byId("effort-chart").setAttribute("aria-label", "La cuota representa el " + percent(effort) + " del salario mensual. Referencia: 30 %.");
  const delta = Math.abs(effort - 30);
  byId("effort-summary").textContent = delta < .05 ? "La cuota coincide con el umbral del 30 %." : "La cuota queda " + decimal.format(delta) + " puntos porcentuales por " + (effort > 30 ? "encima" : "debajo") + " del umbral.";
  const context = byId("result-context");
  context.replaceChildren();
  addDetail(context, "Zona de referencia", area);
  addDetail(context, "Año del precio histórico", priceYear ?? (manualPrice ? "Precio indicado en el caso" : "Sin dato"));
  addDetail(context, "Precio de la vivienda", money(calc.precio_vivienda_gbp));
  addDetail(context, "Origen del precio", data.fuente_precio || (manualPrice ? "Introducido en el caso" : "Mediana histórica"));
  addDetail(context, "Salario joven anual de referencia", money(calc.salario_joven_anual_gbp));
  addDetail(context, "Año del salario regional", data.anio_salario ?? "Sin dato");
  addDetail(context, "Tasa inicial estimada por el modelo", percent(calc.tasa_hipotecaria_usada_pct));
  addDetail(context, "Financiación aplicada", percent(calc.ltv_pct));
  addDetail(context, "Plazo aplicado", calc.plazo_anios + " años");
  addDetail(context, "Fuente del contexto", data.conectado_azure ? "Azure PostgreSQL" : "CSV de respaldo");
  const link = byId("result-data");
  link.href = "#datos" + filters(regional.region);
  const explanation = byId("explanation");
  explanation.replaceChildren();
  explanation.append(element("h3","Cuota y esfuerzo"), element("p","La cuota se calcula con amortización francesa a partir del precio, la financiación, el plazo, las comisiones y la tasa estimada por Random Forest. El esfuerzo divide esa cuota entre el salario joven anual de referencia dividido por doce."));
  explanation.append(element("h3","Clasificación regional"), element("p","El árbol de decisión clasifica el esfuerzo histórico de la región como «" + regional.prediccion + "». La regresión Ridge estima un esfuerzo regional del " + percent(regional.tipo1_regresion_esfuerzo_pct) + "."));
  const neighbors = data.knn_vecinos_en_vivo?.vecinos || [];
  if (neighbors.length) {
    explanation.append(element("h3","Casos históricos similares"));
    const list = document.createElement("ul");
    for (const row of neighbors) list.append(element("li", row.region + " · " + row.anio + " · esfuerzo del " + percent(row.esfuerzo_real_pct)));
    explanation.append(list);
  }
  byId("overview").hidden = true;
  byId("source-line").hidden = true;
  byId("result").hidden = false;
  currentResult = data;
  byId("result-title").focus({preventScroll:true});
  if (window.matchMedia("(max-width:720px)").matches) byId("result").scrollIntoView({block:"start"});
}

async function predict(event) {
  event.preventDefault();
  if (waitingForPrediction) return;
  const payload = Object.fromEntries(new FormData(event.currentTarget));
  if (writing) {
    payload.texto = payload.texto.trim();
    if (!payload.texto) { byId("prediction-status").textContent = "Escribe un caso para calcular la predicción."; return; }
  }
  waitingForPrediction = true;
  byId("predict").disabled = true;
  byId("prediction-status").classList.remove("error");
  byId("prediction-status").textContent = "Calculando tu predicción…";
  const slow = setTimeout(() => { byId("prediction-status").textContent = "El servicio puede tardar un minuto en iniciarse."; }, 5000);
  try {
    // Warm up with a read-only request before sending the prediction once.
    await ensureContext();
    const response = await fetch("/api/predecir", {method:"POST", headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),signal:AbortSignal.timeout(35000)});
    if (!response.ok) throw new Error(response.status === 400 || response.status === 422 ? "Revisa el precio, el plazo y los datos del caso." : "La API no pudo confirmar la predicción. Vuelve a intentarlo en un momento.");
    const data = await response.json();
    renderResult(data, payload);
    byId("prediction-status").textContent = "Predicción calculada.";
  } catch (error) {
    byId("prediction-status").classList.add("error");
    byId("prediction-status").textContent = error.message.startsWith("Revisa") || error.message.startsWith("La API") ? error.message : "No se pudo confirmar la predicción. Comprueba la conexión y vuelve a intentarlo.";
  } finally {
    clearTimeout(slow);
    byId("predict").disabled = false;
    waitingForPrediction = false;
  }
}

async function showData() {
  const request = ++tableRequest;
  const query = filters(byId("data-region").value, byId("data-year").value);
  currentData = null;
  byId("data-status").textContent = "Consultando registros…";
  byId("data-rows").replaceChildren();
  byId("download").disabled = true;
  const slow = setTimeout(() => {
    if (request === tableRequest) byId("data-status").textContent = "Conectando con la API. El primer acceso puede tardar un minuto.";
  }, 5000);
  try {
    const data = await readApi("/api/datos" + query);
    if (request !== tableRequest) return;
    const rows = document.createDocumentFragment();
    for (const row of data.filas) {
      const tr = document.createElement("tr");
      for (const value of [row.anio, row.region, integer.format(row.n_transacciones), money(row.precio_mediana), money(row.salario_joven_region), decimal.format(row.anos_salario)]) tr.append(element("td", value));
      rows.append(tr);
    }
    byId("data-rows").replaceChildren(rows);
    byId("data-status").textContent = (data.total ? data.total + (data.total === 1 ? " registro" : " registros") : "Sin registros para esta selección") + " · Fuente: " + data.fuente;
    byId("download").disabled = !data.total;
    currentData = {query, total: data.total};
  } catch {
    if (request === tableRequest) byId("data-status").textContent = "No se pudieron cargar los datos. Cambia los filtros o vuelve a abrir esta vista.";
  } finally { clearTimeout(slow); }
}

async function downloadExcel() {
  if (!currentData?.total) return;
  const selection = currentData;
  const button = byId("download");
  button.disabled = true;
  try {
    const response = await fetch("/api/datos/exportar.xlsx" + selection.query, {cache:"no-store",signal:AbortSignal.timeout(30000)});
    if (!response.ok || !response.headers.get("Content-Type")?.includes("spreadsheetml")) throw new Error("descarga");
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    const params = new URLSearchParams(selection.query);
    link.download = ["vivienda-uk", params.get("region"), params.get("anio")].filter(Boolean).join("-") + ".xlsx";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch {
    if (currentData === selection) byId("data-status").textContent = "No se pudo descargar el Excel. Inténtalo de nuevo.";
  } finally {
    if (currentData === selection) button.disabled = false;
  }
}

function route(initial = false) {
  const parts = location.hash.slice(1).split("?");
  const dataView = parts[0] === "datos";
  const wasData = !byId("data-view").hidden;
  byId("simulador").hidden = dataView;
  byId("data-view").hidden = !dataView;
  document.title = (dataView ? "Datos de Azure" : "Simulador de vivienda") + " · Vivienda UK";
  byId("nav-sim").removeAttribute("aria-current");
  byId("nav-data").removeAttribute("aria-current");
  byId(dataView ? "nav-data" : "nav-sim").setAttribute("aria-current","page");
  if (dataView) {
    const params = new URLSearchParams(parts[1] || "");
    byId("data-region").value = params.get("region") || "";
    byId("data-year").value = params.get("anio") || "";
    showData();
    if (!initial && !wasData) { byId("data-title").focus({preventScroll:true}); window.scrollTo(0,0); }
  } else if (!initial && wasData) {
    byId("contenido").focus({preventScroll:true}); window.scrollTo(0,0);
    if (!currentResult) showRegionalChart();
  }
}

setRegionOptions(defaultRegions);
for (let year = 2020; year >= 2015; year--) byId("data-year").add(new Option(year, year));
byId("mode-options").addEventListener("click", () => setMode(false));
byId("mode-text").addEventListener("click", () => setMode(true));
byId("region").addEventListener("change", () => { populateCities(); showRegionalChart(); });
byId("prediction-form").addEventListener("submit", predict);
for (const id of ["data-region","data-year"]) byId(id).addEventListener("change", () => {
  location.hash = "datos" + filters(byId("data-region").value, byId("data-year").value);
});
byId("download").addEventListener("click", downloadExcel);
window.addEventListener("hashchange", () => route());
route(true);
if (byId("data-view").hidden) showRegionalChart();
// Share the initial warm-up across requests. After inactivity Render may need to start again.
function ensureContext() {
  if (contextPromise && (!lastContextTime || Date.now() - lastContextTime < 600000)) return contextPromise;
  lastContextTime = 0;
  contextPromise = readApi("/api/contexto").then((data) => {
    if (data.regiones_validas?.length) setRegionOptions(data.regiones_validas);
    cities = data.ciudades_destacadas || [];
    populateCities();
    lastContextTime = Date.now();
    return data;
  }).catch((error) => {
    contextPromise = null;
    throw error;
  });
  return contextPromise;
}
ensureContext().catch(() => {});
