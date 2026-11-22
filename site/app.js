"use strict";

const byId = (id) => document.getElementById(id);
const gbpFmt = new Intl.NumberFormat("es-ES", { style: "currency", currency: "GBP", maximumFractionDigits: 0, useGrouping: "always" });
const copFmt = new Intl.NumberFormat("es-CO", { style: "currency", currency: "COP", maximumFractionDigits: 0, useGrouping: "always" });
const decFmt = new Intl.NumberFormat("es-ES", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const intFmt = new Intl.NumberFormat("es-ES", { maximumFractionDigits: 0, useGrouping: "always" });

const moneyGBP = (v) => gbpFmt.format(Number(v) || 0);
const moneyCOP = (v) => copFmt.format(Number(v) || 0);
const pctFmt = (v) => decFmt.format(Number(v) || 0) + " %";

const defaultRegions = [
  "East Midlands", "East of England", "London", "North East", "North West",
  "South East", "South West", "Wales", "West Midlands", "Yorkshire and The Humber"
];

let cities = [];
let regionalPrices = {};
let writing = false;
let currency = "GBP";          // "GBP" | "COP"
let termUnit = "years";        // "years" (5-40) | "months" (60-480)
let currentResult = null;
let waitingForPrediction = false;
let tableRequest = 0;
let chartRequest = 0;
let currentData = null;
let currentTable = "regional";
let contextPromise = null;
let lastContextTime = 0;
let exchangeRate = 5100;

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function buildQuery(region, year, tabla) {
  const p = new URLSearchParams();
  if (tabla && tabla !== "regional") p.set("tabla", tabla);
  if (region) p.set("region", region);
  if (year && (!tabla || tabla === "regional")) p.set("anio", year);
  return p.toString() ? "?" + p.toString() : "";
}

function getRate() {
  const r = Number(byId("tasa-cop")?.value);
  return Number.isFinite(r) && r >= 1000 && r <= 20000 ? r : exchangeRate;
}

function setRegionOptions(regions) {
  for (const id of ["region", "data-region"]) {
    const sel = byId(id);
    const prev = sel.value;
    sel.replaceChildren();
    if (id === "data-region") sel.add(new Option("Todas las regiones", ""));
    for (const r of regions) sel.add(new Option(r, r));
    if (regions.includes(prev)) sel.value = prev;
    else sel.value = id === "region" ? "London" : "";
  }
}

function populateCities() {
  const reg = byId("region").value;
  const sel = byId("city");
  const prev = sel.value;
  sel.replaceChildren(new Option("Referencia regional", ""));
  const names = cities.filter((c) => c.region === reg).map((c) => c.ciudad);
  for (const name of Array.from(new Set(names)).sort()) sel.add(new Option(name, name));
  if (names.includes(prev)) sel.value = prev;
}

function setMode(textMode) {
  writing = Boolean(textMode);
  byId("mode-options").setAttribute("aria-pressed", String(!writing));
  byId("mode-text").setAttribute("aria-pressed", String(writing));
  byId("option-fields").hidden = byId("option-fields").disabled = writing;
  byId("text-fields").hidden = byId("text-fields").disabled = !writing;
}

function setCurrency(nextCurrency) {
  const prevCurrency = currency;
  currency = nextCurrency === "COP" ? "COP" : "GBP";
  byId("curr-gbp").setAttribute("aria-pressed", String(currency === "GBP"));
  byId("curr-cop").setAttribute("aria-pressed", String(currency === "COP"));

  const input = byId("price");
  const rate = getRate();
  const raw = Number(input.value);
  if (prevCurrency !== currency && Number.isFinite(raw) && raw > 0) {
    input.value = currency === "COP" ? Math.round(raw * rate) : Math.round(raw / rate);
  }
  if (currency === "COP") {
    byId("price-prefix").textContent = "$ COP";
    input.name = "precio_cop";
    input.min = "1000000";
    input.max = "500000000000";
    input.placeholder = "Ej: 1200000000 (o vacío para mediana)";
  } else {
    byId("price-prefix").textContent = "£";
    input.name = "precio_gbp";
    input.min = "1";
    input.max = "100000000";
    input.placeholder = "Usar la mediana histórica de la zona";
  }
  updateFormHelpers();
}

function setTermUnit(nextUnit) {
  const prev = termUnit;
  termUnit = nextUnit === "months" ? "months" : "years";
  byId("term-years-btn").setAttribute("aria-pressed", String(termUnit === "years"));
  byId("term-months-btn").setAttribute("aria-pressed", String(termUnit === "months"));
  const input = byId("term-input");
  const currentVal = Number(input.value) || (prev === "years" ? 25 : 300);

  if (termUnit === "months") {
    input.min = "60";
    input.max = "480";
    byId("term-unit-label").textContent = "meses";
    if (prev === "years") input.value = Math.min(480, Math.max(60, Math.round(currentVal * 12)));
  } else {
    input.min = "5";
    input.max = "40";
    byId("term-unit-label").textContent = "años";
    if (prev === "months") input.value = Math.min(40, Math.max(5, Math.round(currentVal / 12)));
  }
  updateFormHelpers();
}

function getReferencePriceGBP() {
  const city = byId("city").value;
  if (city) {
    const found = cities.find((c) => c.ciudad === city && c.region === byId("region").value);
    if (found?.precio_mediana) return Number(found.precio_mediana);
  }
  const regPrice = regionalPrices[byId("region").value];
  if (regPrice) return Number(regPrice);
  return 250000;
}

function updateFormHelpers() {
  const rate = getRate();
  const rawPrice = Number(byId("price").value);
  const priceBadge = byId("price-conversion");
  let priceGBP = getReferencePriceGBP();

  if (Number.isFinite(rawPrice) && rawPrice > 0) {
    if (currency === "COP") {
      priceGBP = rawPrice / rate;
      priceBadge.textContent = "≈ " + moneyGBP(priceGBP) + " GBP";
    } else {
      priceGBP = rawPrice;
      priceBadge.textContent = "≈ " + moneyCOP(priceGBP * rate);
    }
  } else {
    priceBadge.textContent = "Mediana zona (" + moneyGBP(priceGBP) + ")";
  }

  const ltv = Math.min(100, Math.max(1, Number(byId("ltv").value) || 90));
  const downPct = Math.max(0, 100 - ltv);
  const downGBP = priceGBP * (downPct / 100);
  byId("downpayment-help").textContent =
    "Cuota inicial: " + decFmt.format(downPct) + " % (" + moneyGBP(downGBP) + " · " + moneyCOP(downGBP * rate) + ")";

  const tVal = Number(byId("term-input").value) || (termUnit === "years" ? 25 : 300);
  if (termUnit === "years") {
    const m = Math.round(tVal * 12);
    byId("term-help").textContent = m + " meses · Estándar 25 años (300 m) · Máx. 40 años (480 m)";
  } else {
    const y = (tVal / 12).toFixed(1).replace(".0", "");
    byId("term-help").textContent = "≈ " + y + " años · Máximo permitido: 480 meses (40 años)";
  }
}

function applyPreset(key) {
  setMode(false);
  if (key === "london") {
    setCurrency("GBP");
    byId("region").value = "London";
    populateCities();
    byId("city").value = "";
    byId("price").value = "";
    byId("ltv").value = "90";
    setTermUnit("years");
    byId("term-input").value = "25";
    byId("tipo").value = "fixed";
    byId("meses-inicial").value = "24";
    byId("comisiones").value = "999";
    byId("subtitulo").value = "First time buyer";
  } else if (key === "manchester") {
    setCurrency("GBP");
    byId("region").value = "North West";
    populateCities();
    byId("city").value = "MANCHESTER";
    byId("price").value = "250000";
    byId("ltv").value = "85";
    setTermUnit("years");
    byId("term-input").value = "30";
    byId("tipo").value = "fixed";
    byId("meses-inicial").value = "24";
    byId("comisiones").value = "999";
    byId("subtitulo").value = "First time buyer";
  } else if (key === "cop") {
    setCurrency("COP");
    byId("region").value = "West Midlands";
    populateCities();
    byId("city").value = "";
    byId("price").value = "1200000000";
    byId("ltv").value = "85";
    setTermUnit("months");
    byId("term-input").value = "360";
    byId("tipo").value = "fixed";
    byId("meses-inicial").value = "24";
    byId("comisiones").value = "999";
    byId("subtitulo").value = "First time buyer";
  } else if (key === "northeast40") {
    setCurrency("GBP");
    byId("region").value = "North East";
    populateCities();
    byId("city").value = "";
    byId("price").value = "155000";
    byId("ltv").value = "90";
    setTermUnit("months");
    byId("term-input").value = "480";
    byId("tipo").value = "fixed";
    byId("meses-inicial").value = "60";
    byId("comisiones").value = "0";
    byId("subtitulo").value = "First time buyer";
  }
  updateFormHelpers();
  byId("prediction-form").requestSubmit();
}

async function readApi(path) {
  let failure;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(path, { cache: "no-store", signal: AbortSignal.timeout(35000) });
      if (!response.ok && response.status < 500) {
        let msg = "petición rechazada";
        try { const j = await response.json(); if (j?.error) msg = j.error; } catch {}
        throw new Error(msg);
      }
      if (!response.ok) throw new Error("servidor iniciándose");
      return await response.json();
    } catch (error) {
      failure = error;
      if (error.message !== "servidor iniciándose" && error.name !== "TimeoutError" && error.name !== "TypeError") break;
      if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }
  throw failure;
}

function historyShape(rows, lightTheme = false) {
  const width = 520, height = 96, padding = 8;
  const values = rows.map((r) => Number(r.precio_mediana));
  const highest = Math.max(...values), lowest = Math.min(...values);
  const positions = values.map((val, i) => {
    const x = padding + (width - padding * 2) * i / Math.max(1, values.length - 1);
    const y = height - 14 - (height - 28) * (val - lowest) / (highest - lowest || 1);
    return [x, y];
  });
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 520 96");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", rows.map((r) => r.anio + ": " + moneyGBP(r.precio_mediana)).join(". "));

  const strokeLine = lightTheme ? "#16483f" : "#c6e7a4";
  const fillArea = lightTheme ? "#16483f" : "#bde29f";
  const gridColor = lightTheme ? "#8ea095" : "#c4d5bc";
  const dotFill = lightTheme ? "#0078d4" : "#e1f3c8";

  function shape(tag, attrs) {
    const child = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) child.setAttribute(k, v);
    svg.append(child);
  }
  for (const y of [16, 48, 82]) shape("line", { x1: 0, y1: y, x2: width, y2: y, stroke: gridColor, opacity: ".25", "stroke-dasharray": "3 6" });
  const points = positions.map((p) => p.join(",")).join(" ");
  shape("polygon", { points: padding + "," + height + " " + points + " " + (width - padding) + "," + height, fill: fillArea, opacity: lightTheme ? ".11" : ".14" });
  shape("polyline", { points, fill: "none", stroke: strokeLine, "stroke-width": "3", "vector-effect": "non-scaling-stroke" });
  for (const [cx, cy] of positions) shape("circle", { cx, cy, r: 3.8, fill: dotFill });
  return svg;
}

async function renderHistoryInto(region, containerId, sourceId, lightTheme, priceId, copId) {
  const req = ++chartRequest;
  const box = byId(containerId);
  box.textContent = "Cargando serie histórica…";
  try {
    const data = await readApi("/api/datos" + buildQuery(region));
    if (req !== chartRequest && !lightTheme) return;
    const rows = (data.filas || []).slice().sort((a, b) => a.anio - b.anio);
    if (!rows.length) throw new Error("sin datos");
    const latest = rows.at(-1);
    const rate = getRate();
    if (priceId) byId(priceId).textContent = moneyGBP(latest.precio_mediana);
    if (copId) byId(copId).textContent = "≈ " + moneyCOP(Number(latest.precio_mediana) * rate) + " (tasa 1 £ = " + intFmt.format(rate) + " COP)";
    if (sourceId) byId(sourceId).textContent = "Fuente: " + data.fuente + " · " + latest.anio;
    const axis = el("div", null, "history-axis");
    axis.append(
      el("span", rows[0].anio + " (" + moneyGBP(rows[0].precio_mediana) + ")"),
      el("span", latest.anio + " (" + moneyGBP(latest.precio_mediana) + ")")
    );
    box.replaceChildren(historyShape(rows, lightTheme), axis);
  } catch {
    box.textContent = "La serie histórica se cargará al conectar con la API.";
  }
}

function showRegionalChart() {
  if (currentResult) return;
  const region = byId("region").value;
  byId("overview-region").textContent = region;
  byId("overview-link").href = "#datos" + buildQuery(region);
  byId("overview-price").textContent = "—";
  renderHistoryInto(region, "history-chart", "overview-source", false, "overview-price", "overview-price-cop");
}

function renderRegionsComparisonSVG(regionsList, userEffort, userLabel) {
  const items = (regionsList || []).map((r) => ({
    label: r.region,
    val: Number(r.esfuerzo_pct),
    isUser: false,
  }));
  items.push({ label: "★ TU CASO (" + userLabel + ")", val: Number(userEffort), isUser: true });
  items.sort((a, b) => a.val - b.val);

  const maxVal = Math.max(115, Math.ceil(Math.max(...items.map((i) => i.val)) / 10) * 10 + 10);
  const labelW = 195;
  const barAreaW = 355;
  const valueW = 68;
  const width = labelW + barAreaW + valueW;
  const rowH = 26;
  const topPad = 24;
  const bottomPad = 14;
  const height = topPad + items.length * rowH + bottomPad;

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Comparativa del esfuerzo salarial de tu escenario frente a las 10 regiones del Reino Unido");

  const xPos = (pct) => labelW + (Math.min(maxVal, Math.max(0, pct)) / maxVal) * barAreaW;
  function add(tag, attrs, text) {
    const n = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text !== undefined) n.textContent = String(text);
    svg.append(n);
  }

  // Reference lines: 30% and 56.4%
  const x30 = xPos(30);
  const xCart = xPos(56.4);
  add("line", { x1: x30, y1: topPad - 6, x2: x30, y2: height - bottomPad, stroke: "#1f7a4c", "stroke-width": 1.6, "stroke-dasharray": "4 3" });
  add("text", { x: x30, y: 12, "text-anchor": "middle", fill: "#1f7a4c", "font-size": 10.5, "font-weight": 700 }, "30 % Ideal");

  add("line", { x1: xCart, y1: topPad - 6, x2: xCart, y2: height - bottomPad, stroke: "#c47a14", "stroke-width": 1.4, "stroke-dasharray": "4 3" });
  add("text", { x: xCart, y: 12, "text-anchor": "middle", fill: "#a16207", "font-size": 10.5, "font-weight": 700 }, "56,4 % CART");

  items.forEach((item, idx) => {
    const y = topPad + idx * rowH;
    const barW = Math.max(4, xPos(item.val) - labelW);
    if (item.isUser) {
      add("rect", { x: 2, y: y - 1, width: width - 4, height: rowH - 3, rx: 5, fill: "#e6f3fc", stroke: "#8ec5f2" });
    }
    const shortLbl = item.label.length > 26 ? item.label.slice(0, 25) + "…" : item.label;
    add("text", {
      x: labelW - 10,
      y: y + 15,
      "text-anchor": "end",
      fill: item.isUser ? "#005a9e" : "#2d4039",
      "font-size": item.isUser ? 11.3 : 11,
      "font-weight": item.isUser ? 700 : 500,
    }, shortLbl);

    const fillBar = item.isUser
      ? "#0078d4"
      : item.val <= 30 ? "#2f9e66" : item.val < 56.4 ? "#739c88" : "#9c5e4b";
    add("rect", { x: labelW, y: y + 4, width: barW, height: 14, rx: 4, fill: fillBar });
    add("text", {
      x: labelW + barW + 7,
      y: y + 15,
      fill: item.isUser ? "#005a9e" : "#30423b",
      "font-size": 11,
      "font-weight": item.isUser ? 700 : 600,
      stroke: item.isUser ? "#e6f3fc" : "#fffefa",
      "stroke-width": 3,
      "paint-order": "stroke",
    }, pctFmt(item.val));
  });

  byId("regions-chart").replaceChildren(svg);

  const cheaperCount = (regionsList || []).filter((r) => Number(r.esfuerzo_pct) > Number(userEffort)).length;
  byId("regions-takeaway").textContent =
    cheaperCount === 10
      ? "Tu escenario (" + pctFmt(userEffort) + ") resulta más accesible que las 10 medianas regionales históricas del Reino Unido."
      : cheaperCount === 0
      ? "Tu escenario (" + pctFmt(userEffort) + ") supera en exigencia salarial a las 10 medianas regionales históricas del Reino Unido."
      : "Tu escenario (" + pctFmt(userEffort) + ") exige menos proporción de salario que " + cheaperCount + " de las 10 regiones del Reino Unido.";
}

function addDetail(list, name, valueGBP, valueCOP) {
  const wrap = document.createElement("div");
  wrap.append(el("dt", name));
  const dd = el("dd", valueGBP);
  if (valueCOP) dd.append(el("span", valueCOP, "dd-sub"));
  wrap.append(dd);
  list.append(wrap);
}

function renderResult(data, payload) {
  const calc = data.calculo_financiero_acoplado;
  const reg = data.prediccion_regional_dual;
  const hip = data.prediccion_hipoteca_dual;
  const town = data.ciudad_consultada;
  const effort = Number(calc.esfuerzo_pmt_acoplado_pct);

  if (!Number.isFinite(effort) || !Number.isFinite(Number(calc.cuota_mensual_pmt_gbp))) {
    throw new Error("Respuesta incompleta de la API.");
  }

  const area = town?.ciudad ? town.ciudad + " · " + reg.region : reg.region;
  const langTag = data.interpretacion_texto?.idioma_detectado ? " · " + data.interpretacion_texto.idioma_detectado : "";
  byId("result-place").textContent = area + " · " + calc.plazo_meses + " meses (" + calc.plazo_anios + " años) · LTV " + pctFmt(calc.ltv_pct) + langTag;
  byId("monthly").textContent = moneyGBP(calc.cuota_mensual_pmt_gbp) + " / mes";
  byId("monthly-cop").textContent = "≈ " + moneyCOP(calc.cuota_mensual_pmt_cop) + " / mes";

  // Risk badge & plain-Spanish diagnosis
  const risk = calc.nivel_riesgo || "moderado";
  const riskLabels = {
    saludable: "Accesible (≤ 30 %)",
    moderado: "Esfuerzo moderado",
    alto: "Esfuerzo alto",
    critico: "Crítico (≥ 100 %)",
  };
  const pill = byId("risk-pill");
  pill.textContent = riskLabels[risk] || "Evaluado";
  pill.className = "risk-pill " + risk;
  byId("plain-diagnosis").textContent = calc.diagnostico_sencillo || "";

  // GRÁFICA 1: ¿Cómo queda tu salario mensual?
  const scale = Math.max(100, Math.ceil(effort / 25) * 25);
  byId("effort-value").textContent = pctFmt(effort) + " del sueldo";
  const quotaBar = byId("effort-bar");
  quotaBar.className = "budget-quota " + risk;
  quotaBar.style.width = Math.min(100, (100 * effort) / scale) + "%";
  const freePct = Math.max(0, 100 - effort);
  byId("free-bar").style.width = effort < 100 ? (100 * freePct) / scale + "%" : "0%";
  byId("reference").style.left = (100 * 30) / scale + "%";
  byId("reference-cart").style.left = (100 * 56.36) / scale + "%";
  byId("max-scale").textContent = pctFmt(scale);

  byId("kpi-salary-gbp").textContent = moneyGBP(calc.salario_joven_mensual_gbp);
  byId("kpi-salary-cop").textContent = moneyCOP(calc.salario_joven_mensual_cop);
  const saldoGBP = Number(calc.saldo_libre_mensual_gbp);
  const saldoCOP = Number(calc.saldo_libre_mensual_cop);
  const balBox = byId("kpi-balance-box");
  if (saldoGBP >= 0) {
    balBox.classList.remove("deficit");
    byId("kpi-balance-label").textContent = "Te queda libre / mes";
    byId("kpi-balance-gbp").textContent = moneyGBP(saldoGBP);
    byId("kpi-balance-cop").textContent = moneyCOP(saldoCOP);
  } else {
    balBox.classList.add("deficit");
    byId("kpi-balance-label").textContent = "Déficit mensual";
    byId("kpi-balance-gbp").textContent = "-" + moneyGBP(Math.abs(saldoGBP));
    byId("kpi-balance-cop").textContent = "-" + moneyCOP(Math.abs(saldoCOP));
  }
  byId("kpi-down-label").textContent = "Cuota inicial (" + decFmt.format(calc.cuota_inicial_pct) + " %)";
  byId("kpi-down-gbp").textContent = moneyGBP(calc.cuota_inicial_gbp);
  byId("kpi-down-cop").textContent = moneyCOP(calc.cuota_inicial_cop);

  byId("result-data").href = "#datos" + buildQuery(reg.region);

  // GRÁFICA 2: Comparativa 10 Regiones UK
  renderRegionsComparisonSVG(data.comparativa_10_regiones || [], effort, town?.ciudad || reg.region);

  // GRÁFICA 3A: Capital vs Intereses
  const pctInt = Number(calc.pct_intereses_del_total) || 0;
  const pctCap = Math.max(0, 100 - pctInt);
  byId("cost-term-label").textContent = calc.plazo_meses + " meses (" + calc.plazo_anios + " años)";
  byId("cost-total-label").textContent = "Total pagado: " + moneyGBP(calc.total_pagado_gbp) + " (" + moneyCOP(calc.total_pagado_cop) + ")";
  byId("seg-principal").style.width = pctCap + "%";
  byId("seg-interest").style.width = pctInt + "%";
  byId("pct-principal-label").textContent = pctFmt(pctCap);
  byId("pct-interest-label").textContent = pctFmt(pctInt);
  byId("val-principal-gbp").textContent = moneyGBP(calc.monto_prestamo_gbp);
  byId("val-principal-cop").textContent = moneyCOP(calc.monto_prestamo_cop);
  byId("val-interest-gbp").textContent = moneyGBP(calc.total_intereses_gbp);
  byId("val-interest-cop").textContent = moneyCOP(calc.total_intereses_cop);

  // GRÁFICA 3B: Termómetro Tasa Hipotecaria F2
  const rateF2 = Number(calc.tasa_hipotecaria_usada_pct);
  const pinPct = Math.min(98, Math.max(2, ((rateF2 - 2.5) / (8.5 - 2.5)) * 100));
  byId("rate-pred-badge").textContent = pctFmt(rateF2) + " anual (" + calc.meses_inicial_f2 + " m iniciales)";
  byId("rate-pin").style.left = pinPct + "%";
  byId("rate-takeaway").textContent =
    rateF2 >= 8.13
      ? "Alerta Tukey: tasa atípica (≥ 8,13 %), por encima del 99 % del catálogo hipotecario UK."
      : rateF2 <= 5.62
      ? "Tasa competitiva: se ubica en el cuartil más bajo (≤ Q1 5,62 %) del catálogo de 7.796 hipotecas UK."
      : rateF2 <= 6.62
      ? "Tasa dentro del rango normal del mercado bancario UK (Q1 5,62 % – Q3 6,62 %, mediana 6,10 %)."
      : "Tasa por encima del tercer cuartil (Q3 6,62 %): conviene comparar comisiones o plazo inicial.";

  // GRÁFICA 4: Mantener visible la serie histórica 2015–2020
  byId("res-hist-region").textContent = reg.region;
  renderHistoryInto(reg.region, "result-history-chart", "res-hist-source", true);

  // Interpretación del texto libre (si aplica)
  const parsedBox = byId("parsed-summary");
  if (data.interpretacion_texto) {
    const p = data.interpretacion_texto;
    parsedBox.hidden = false;
    parsedBox.replaceChildren(el("strong", "Así interpretamos tu caso escrito (" + (p.idioma_detectado === "EN" ? "Inglés" : "Español") + "):"));
    const chips = el("div", null, "parsed-chips");
    const items = [
      "Zona: " + area,
      "Moneda: " + (data.moneda_entrada || "GBP"),
      "LTV / TLV: " + pctFmt(calc.ltv_pct),
      "Entrada: " + pctFmt(calc.cuota_inicial_pct),
      "Plazo: " + calc.plazo_meses + " meses (" + calc.plazo_anios + " años)",
      "Inicial F2: " + calc.meses_inicial_f2 + " m · " + p.tipo,
      "Perfil: " + p.subtitulo,
    ];
    for (const t of items) chips.append(el("span", t, "parsed-chip"));
    parsedBox.append(chips);
  } else {
    parsedBox.hidden = true;
  }

  // Resumen financiero completo en GBP + COP
  const ctxList = byId("result-context");
  ctxList.replaceChildren();
  addDetail(ctxList, "Precio de la vivienda", moneyGBP(calc.precio_vivienda_gbp), moneyCOP(calc.precio_vivienda_cop));
  addDetail(ctxList, "Cuota inicial (" + pctFmt(calc.cuota_inicial_pct) + ")", moneyGBP(calc.cuota_inicial_gbp), moneyCOP(calc.cuota_inicial_cop));
  addDetail(ctxList, "Monto financiado (LTV " + pctFmt(calc.ltv_pct) + ")", moneyGBP(calc.monto_prestamo_gbp), moneyCOP(calc.monto_prestamo_cop));
  addDetail(ctxList, "Salario joven anual (" + (data.anio_salario || 2020) + ")", moneyGBP(calc.salario_joven_anual_gbp), moneyCOP(calc.salario_joven_anual_cop));
  addDetail(ctxList, "Años de salario íntegro", decFmt.format(calc.anos_salario_requeridos) + " años", data.fuente_precio);
  addDetail(ctxList, "Plazo aplicado y límite UK", calc.plazo_meses + " meses (" + calc.plazo_anios + " años)", "Máx. UK: 480 m (40 años) · F2 inicial: " + calc.meses_inicial_f2 + " m");
  addDetail(ctxList, "Tasa de cambio usada", "1 £ = " + intFmt.format(calc.tasa_cambio_gbp_cop) + " COP", "Editable en opciones");
  addDetail(ctxList, "Fuente de datos en vivo", data.conectado_azure ? "Azure PostgreSQL (db_actividad_mineria)" : "CSV de respaldo", data.registro_id ? "Registro guardado en Azure #" + data.registro_id : "Consulta en vivo");

  // Tabla comparativa de métodos
  const mRows = byId("methods-rows");
  mRows.replaceChildren();
  const knn = data.knn_vecinos_en_vivo || {};
  const seg = data.segmento_clustering_sin_etiqueta || {};
  const methodSummary = [
    ["Árbol CART", "Esfuerzo regional: " + reg.prediccion + " (" + pctFmt((reg.probabilidad_alto || 0) * 100) + " prob. Alto)", "17/18 Test (94,4 %) · 9/10 regiones"],
    ["Regresión Ridge", "Esfuerzo regional proyectado: " + pctFmt(reg.tipo1_regresion_esfuerzo_pct), "R² CV-5 = 0,909 · MAE ±4,21 pp"],
    ["Random Forest (F2)", "Tasa hipotecaria estimada: " + pctFmt(hip.tipo1_tasa_estimada_pct), "R² = 0,964 · MAE ±0,169 pp (7.796 hipotecas)"],
    ["KNN (5 vecinos)", knn.prediccion_knn ? knn.prediccion_knn + " (" + knn.votos_alto + "/5 votos Alto)" : "—", "17/18 Test (94,4 %) · 54/60 CV-5 (90,0 %)"],
    ["K-Means (Sin etiqueta)", seg.segmento_nombre ? "Clúster " + seg.cluster_id + ": " + seg.segmento_nombre : "—", "Silhouette 0,321 (regiones) · 0,479 (F2)"],
    ["Apriori (Sin etiqueta)", (data.reglas_apriori_sin_etiqueta || [])[0] || "Reglas territoriales activas", "195 reglas F1 + 481 F2 · Lift hasta 3,96×"],
  ];
  for (const [m, r, a] of methodSummary) {
    const tr = document.createElement("tr");
    tr.append(el("td", m), el("td", r), el("td", a));
    mRows.append(tr);
  }

  // Explicación técnica desplegable + 5 vecinos KNN + Apriori
  const exp = byId("explanation");
  exp.replaceChildren();
  exp.append(
    el("h4", "Fórmula francesa (PMT) acoplada a ML"),
    el("p", "La cuota mensual combina el precio (" + moneyGBP(calc.precio_vivienda_gbp) + "), la financiación LTV (" + pctFmt(calc.ltv_pct) + "), el plazo exacto (" + calc.plazo_meses + " meses) y la tasa inicial predicha por Random Forest (" + pctFmt(rateF2) + ") sobre 7.796 productos hipotecarios británicos.")
  );
  if (knn.vecinos?.length) {
    exp.append(el("h4", "Los 5 vecinos históricos más cercanos (KNN en vivo)"));
    const ul = document.createElement("ul");
    for (const v of knn.vecinos) {
      ul.append(el("li", "#" + v.orden + " · " + v.region + " (" + v.anio + ") — Esfuerzo real " + pctFmt(v.esfuerzo_real_pct) + " · Precio " + moneyGBP(v.precio_mediana_gbp) + " · Clase: " + v.clase));
    }
    exp.append(ul);
  }
  if (data.reglas_apriori_sin_etiqueta?.length) {
    exp.append(el("h4", "Patrones territoriales descubiertos (Apriori sin etiqueta)"));
    const ul2 = document.createElement("ul");
    for (const rule of data.reglas_apriori_sin_etiqueta) ul2.append(el("li", rule));
    exp.append(ul2);
  }

  byId("overview").hidden = true;
  byId("source-line").hidden = true;
  byId("result").hidden = false;
  currentResult = data;
  byId("result-title").focus({ preventScroll: true });
  if (window.matchMedia("(max-width:920px)").matches) byId("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function predict(event) {
  event.preventDefault();
  if (waitingForPrediction) return;

  const payload = {};
  if (writing) {
    const txt = byId("free-text").value.trim();
    if (!txt) {
      byId("prediction-status").classList.add("error");
      byId("prediction-status").textContent = "Escribe un caso en español o inglés (o elige uno de los 6 ejemplos).";
      return;
    }
    payload.texto = txt;
    payload.tasa_cop = getRate();
  } else {
    payload.region = byId("region").value;
    if (byId("city").value) payload.ciudad = byId("city").value;
    payload.moneda = currency;
    payload.tasa_cop = getRate();
    const rawPrice = byId("price").value.trim();
    if (rawPrice) {
      if (currency === "COP") payload.precio_cop = Number(rawPrice);
      else payload.precio_gbp = Number(rawPrice);
    }
    payload.ltv_pct = Number(byId("ltv").value);
    const tVal = Number(byId("term-input").value);
    if (termUnit === "months") payload.plazo_meses = Math.round(tVal);
    else payload.plazo_anios = Math.round(tVal);
    payload.tipo = byId("tipo").value;
    payload.meses = Math.round(Number(byId("meses-inicial").value));
    payload.comisiones = Number(byId("comisiones").value);
    payload.subtitulo = byId("subtitulo").value;
  }

  waitingForPrediction = true;
  byId("predict").disabled = true;
  byId("prediction-status").classList.remove("error");
  byId("prediction-status").textContent = "Consultando modelos y Azure PostgreSQL…";
  const slow = setTimeout(() => {
    byId("prediction-status").textContent = "Activando el servidor en Render… el primer acceso puede tardar ~45 s.";
  }, 4500);

  try {
    await ensureContext();
    const response = await fetch("/api/predecir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(45000),
    });
    if (!response.ok) {
      let msg = "Revisa el precio, el LTV (5–100 %) y el plazo (5–40 años / 60–480 meses).";
      try { const errJson = await response.json(); if (errJson?.error) msg = errJson.error; } catch {}
      throw new Error(msg);
    }
    const data = await response.json();
    renderResult(data, payload);
    byId("prediction-status").textContent = "Predicción calculada en vivo (" + (data.conectado_azure ? "Azure PostgreSQL" : "respaldo CSV") + ").";
  } catch (error) {
    byId("prediction-status").classList.add("error");
    byId("prediction-status").textContent = error.message || "No se pudo confirmar la predicción. Inténtalo de nuevo.";
  } finally {
    clearTimeout(slow);
    byId("predict").disabled = false;
    waitingForPrediction = false;
  }
}

/* ===================== VISTA DE DATOS DE AZURE ===================== */
const TABLE_HEADERS = {
  regional: ["Año", "Región", "Transacciones", "Precio mediano (£)", "Salario joven anual (£)", "Años de salario"],
  ciudades: ["Año", "Región", "Ciudad", "Transacciones", "Precio mediano (£)", "Precio aprox. (COP)"],
  hipoteca_2022: ["Región", "Precio mediano (£)", "Salario joven anual (£)", "Tasa inicial (%)", "Cuota mensual (£)", "Esfuerzo (%)", "Accesible ≤30%"],
  metodos: ["Paradigma", "Tipo de Predicción", "Método", "Aciertos Exactos", "Métrica Clave", "Cuándo Usar"],
};

function renderDataHead(tabla) {
  const tr = document.createElement("tr");
  for (const h of TABLE_HEADERS[tabla] || TABLE_HEADERS.regional) tr.append(el("th", h));
  byId("data-head").replaceChildren(tr);
  const tbl = byId("data-rows").closest("table");
  tbl.classList.toggle("text-table", tabla === "metodos");
}

function updateTableTabs(tabla) {
  currentTable = tabla || "regional";
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.setAttribute("aria-selected", String(btn.dataset.tabla === currentTable));
  }
  byId("year-filter-wrap").hidden = currentTable !== "regional";
  byId("data-region").disabled = currentTable === "metodos";
}

async function showData() {
  const req = ++tableRequest;
  const reg = currentTable === "metodos" ? "" : byId("data-region").value;
  const yr = currentTable === "regional" ? byId("data-year").value : "";
  const query = buildQuery(reg, yr, currentTable);
  currentData = null;
  renderDataHead(currentTable);
  byId("data-status").textContent = "Consultando tabla en Azure PostgreSQL…";
  byId("data-rows").replaceChildren();
  byId("download").disabled = true;

  const slow = setTimeout(() => {
    if (req === tableRequest) byId("data-status").textContent = "Activando la API en Render para consultar Azure PostgreSQL…";
  }, 5000);

  try {
    const data = await readApi("/api/datos" + query);
    if (req !== tableRequest) return;
    const frag = document.createDocumentFragment();
    const rate = getRate();

    for (const row of data.filas || []) {
      const tr = document.createElement("tr");
      let cells = [];
      if (currentTable === "ciudades") {
        cells = [row.anio, row.region, row.ciudad, intFmt.format(row.n_transacciones), moneyGBP(row.precio_mediana), moneyCOP(Number(row.precio_mediana) * rate)];
      } else if (currentTable === "hipoteca_2022") {
        cells = [row.region, moneyGBP(row.precio_mediana), moneyGBP(row.salario_joven_region), pctFmt(row.tasa_inicial_pct), moneyGBP(row.cuota_mensual), pctFmt(Number(row.cuota_pct_salario) * 100), row.accesible ? "Sí" : "No"];
      } else if (currentTable === "metodos") {
        cells = [row.paradigma, row.tipo, row.metodo, row.aciertos_exactos, row.metrica_clave, row.cuando_usar];
      } else {
        cells = [row.anio, row.region, intFmt.format(row.n_transacciones), moneyGBP(row.precio_mediana), moneyGBP(row.salario_joven_region), decFmt.format(row.anos_salario)];
      }
      for (const val of cells) tr.append(el("td", val));
      frag.append(tr);
    }

    byId("data-rows").replaceChildren(frag);
    byId("data-status").textContent =
      (data.total ? data.total + (data.total === 1 ? " registro" : " registros") : "Sin registros para este filtro") +
      " · Fuente: " + data.fuente + " (" + (data.azure_base_datos || "db_actividad_mineria") + ")";
    byId("download").disabled = false;
    currentData = { query: buildQuery(reg, yr), total: data.total };
  } catch {
    if (req === tableRequest) byId("data-status").textContent = "No se pudieron cargar los datos en este momento. Inténtalo de nuevo.";
  } finally {
    clearTimeout(slow);
  }
}

async function downloadExcel(customQuery, triggerButton) {
  const query = customQuery !== undefined ? customQuery : (currentData?.query || "");
  const btn = triggerButton || byId("download");
  const prevText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Generando Excel (.xlsx)…";
  try {
    const response = await fetch("/api/datos/exportar.xlsx" + query, { cache: "no-store", signal: AbortSignal.timeout(45000) });
    if (!response.ok || !response.headers.get("Content-Type")?.includes("spreadsheetml")) throw new Error("descarga");
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    const params = new URLSearchParams(query);
    link.download = ["vivienda-uk-azure", params.get("region"), params.get("anio")].filter(Boolean).join("-") + ".xlsx";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch {
    byId("data-status").textContent = "No se pudo descargar el Excel. Inténtalo de nuevo.";
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

function route(initial = false) {
  const parts = location.hash.slice(1).split("?");
  const dataView = parts[0] === "datos";
  const wasData = !byId("data-view").hidden;
  byId("simulador").hidden = dataView;
  byId("data-view").hidden = !dataView;
  document.title = (dataView ? "Datos de Azure · Excel" : "Simulador de vivienda") + " · Vivienda UK";
  byId("nav-sim").removeAttribute("aria-current");
  byId("nav-data").removeAttribute("aria-current");
  byId(dataView ? "nav-data" : "nav-sim").setAttribute("aria-current", "page");

  if (dataView) {
    const params = new URLSearchParams(parts[1] || "");
    updateTableTabs(params.get("tabla") || "regional");
    byId("data-region").value = params.get("region") || "";
    byId("data-year").value = params.get("anio") || "";
    showData();
    if (!initial && !wasData) { byId("data-title").focus({ preventScroll: true }); window.scrollTo(0, 0); }
  } else if (!initial && wasData) {
    byId("contenido").focus({ preventScroll: true });
    window.scrollTo(0, 0);
    if (!currentResult) showRegionalChart();
  }
}

function ensureContext() {
  if (contextPromise && (!lastContextTime || Date.now() - lastContextTime < 600000)) return contextPromise;
  lastContextTime = 0;
  contextPromise = readApi("/api/contexto").then((data) => {
    if (data.regiones_validas?.length) setRegionOptions(data.regiones_validas);
    cities = data.ciudades_destacadas || [];
    regionalPrices = {};
    for (const r of data.comparativa_10_regiones || []) {
      regionalPrices[r.region] = r.precio_mediana_gbp;
    }
    if (data.tasa_cambio_gbp_cop_default) {
      exchangeRate = Number(data.tasa_cambio_gbp_cop_default);
      if (!byId("tasa-cop").value) byId("tasa-cop").value = String(exchangeRate);
    }
    const pill = byId("azure-pill");
    const pillText = byId("azure-pill-text");
    if (data.conectado_azure) {
      pill.classList.remove("offline");
      pillText.textContent = "Azure conectado · " + (data.azure_base_datos || "db_actividad_mineria");
    } else {
      pill.classList.add("offline");
      pillText.textContent = "Modo respaldo CSV · " + (data.fuente_datos || "Local");
    }
    populateCities();
    updateFormHelpers();
    lastContextTime = Date.now();
    return data;
  }).catch((error) => {
    contextPromise = null;
    throw error;
  });
  return contextPromise;
}

/* ===================== INICIALIZACIÓN ===================== */
setRegionOptions(defaultRegions);
for (let y = 2020; y >= 2015; y--) byId("data-year").add(new Option(String(y), String(y)));

byId("mode-options").addEventListener("click", () => setMode(false));
byId("mode-text").addEventListener("click", () => setMode(true));
byId("curr-gbp").addEventListener("click", () => setCurrency("GBP"));
byId("curr-cop").addEventListener("click", () => setCurrency("COP"));
byId("term-years-btn").addEventListener("click", () => setTermUnit("years"));
byId("term-months-btn").addEventListener("click", () => setTermUnit("months"));

for (const btn of document.querySelectorAll("[data-preset]")) {
  btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
}
for (const btn of document.querySelectorAll("[data-example]")) {
  btn.addEventListener("click", () => {
    setMode(true);
    byId("free-text").value = btn.dataset.example;
    byId("prediction-form").requestSubmit();
  });
}

byId("region").addEventListener("change", () => {
  populateCities();
  updateFormHelpers();
  showRegionalChart();
});
for (const id of ["city", "price", "ltv", "term-input", "tasa-cop"]) {
  byId(id).addEventListener("input", updateFormHelpers);
  byId(id).addEventListener("change", updateFormHelpers);
}
byId("prediction-form").addEventListener("submit", predict);

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => {
    const t = btn.dataset.tabla;
    location.hash = "datos" + buildQuery(t === "metodos" ? "" : byId("data-region").value, t === "regional" ? byId("data-year").value : "", t);
  });
}
for (const id of ["data-region", "data-year"]) {
  byId(id).addEventListener("change", () => {
    location.hash = "datos" + buildQuery(byId("data-region").value, byId("data-year").value, currentTable);
  });
}
byId("download").addEventListener("click", () => downloadExcel());
byId("result-excel").addEventListener("click", (e) => {
  const reg = currentResult?.prediccion_regional_dual?.region || "";
  downloadExcel(buildQuery(reg), e.currentTarget);
});

window.addEventListener("hashchange", () => route());
updateFormHelpers();
route(true);
if (byId("data-view").hidden) showRegionalChart();
ensureContext().catch(() => {});
