let rows = [];

async function forecastJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const details = Array.isArray(data.details) ? `: ${data.details.join("; ")}` : "";
    throw new Error((data.error || "Request failed") + details);
  }
  return data;
}

function showError(element, error) {
  element.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
}

function productOptions(products) {
  return products.map((product) =>
    `<option value="${product.StockCode}">${product.StockCode} — ${product.Description || "Description unavailable"}</option>`
  ).join("");
}

async function loadForecastProducts() {
  const data = await forecastJson("/api/product-catalog");
  const futureData = await forecastJson("/api/future-forecast-products");
  const historicalSelect = document.querySelector("#forecast-product");
  const futureSelect = document.querySelector("#future-product");
  historicalSelect.innerHTML = productOptions(data.products);
  futureSelect.innerHTML = futureData.products.map((product) => {
    const hint = product.forecast_ready ? product.readiness_hint : `Unavailable: ${product.readiness_hint}`;
    return `<option value="${product.StockCode}" ${product.forecast_ready ? "" : "disabled"} title="${hint}">${product.StockCode} — ${product.Description || "Description unavailable"}${product.forecast_ready ? "" : " (unavailable)"}</option>`;
  }).join("");
  const firstReady = futureData.products.find((product) => product.forecast_ready);
  if (firstReady) futureSelect.value = firstReady.StockCode;
  document.querySelector("#future-product-hint").textContent =
    `${futureData.ready_count} of ${futureData.count} products are ready. ${futureData.hint}`;
  historicalSelect.addEventListener("change", loadRows);
  await loadRows();
}

async function loadRows() {
  const select = document.querySelector("#forecast-row");
  try {
    const code = document.querySelector("#forecast-product").value;
    const data = await forecastJson(`/api/forecast-rows/${encodeURIComponent(code)}`);
    rows = data.rows;
    select.innerHTML = rows.map((row, index) => `<option value="${index}">${row.Date}</option>`).join("");
    document.querySelector("#forecast-button").disabled = !rows.length;
  } catch (error) {
    showError(document.querySelector("#forecast-status"), error);
  }
}

async function predict() {
  const status = document.querySelector("#forecast-status");
  const button = document.querySelector("#forecast-button");
  status.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Predicting...';
  button.disabled = true;
  try {
    const row = rows[Number(document.querySelector("#forecast-row").value)];
    const selectedDate = row.Date;
    const features = Object.fromEntries(Object.entries(row).filter(([key]) => key !== "Date"));
    const data = await forecastJson("/api/forecast", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(features)
    });
    document.querySelector("#forecast-result").innerHTML =
      `<div class="text-center"><p class="eyebrow">Predicted demand</p><strong class="display-4">${data.predicted_demand.toFixed(2)}</strong><p class="text-muted mt-2">StockCode ${data.StockCode} · historical row ${selectedDate}</p></div>`;
    renderHistoricalChart(features, data.predicted_demand);
    status.innerHTML = "";
  } catch (error) {
    showError(status, error);
  } finally {
    button.disabled = rows.length === 0;
  }
}

function renderHistoricalChart(features, prediction) {
  const chart = document.querySelector("#forecast-chart");
  if (!chart || !window.Plotly) return;
  chart.classList.remove("d-none");
  Plotly.newPlot(chart, [
    {x: ["28-day lag", "14-day lag", "7-day lag", "1-day lag"], y: [features.DemandLag28, features.DemandLag14, features.DemandLag7, features.DemandLag1], type: "scatter", mode: "lines+markers", name: "Historical lag demand"},
    {x: ["Prediction"], y: [prediction], type: "scatter", mode: "markers+text", name: "Predicted demand", text: [prediction.toFixed(2)], textposition: "top center"}
  ], {margin: {t: 20, r: 20, b: 55, l: 50}, yaxis: {title: "Demand units", rangemode: "tozero"}, xaxis: {title: "Historical input vs prediction"}, legend: {orientation: "h"}, paper_bgcolor: "transparent", plot_bgcolor: "transparent"}, {responsive: true, displayModeBar: false});
}

function futureRequestBody() {
  return {StockCode: document.querySelector("#future-product").value, target_date: document.querySelector("#future-date").value};
}

async function predictFuture() {
  const status = document.querySelector("#future-status");
  const button = document.querySelector("#future-button");
  status.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Forecasting recursively...';
  button.disabled = true;
  try {
    const data = await forecastJson("/api/forecast/future", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(futureRequestBody())
    });
    document.querySelector("#future-result").innerHTML =
      `<div class="text-center"><p class="eyebrow">Experimental predicted daily demand</p><strong class="display-4">${data.predicted_daily_demand.toFixed(2)}</strong><p class="text-muted">${data.StockCode} · ${data.target_date} · ${data.forecast_steps} recursive steps</p><p class="small text-warning">${data.warning}</p></div>`;
    status.innerHTML = "";
  } catch (error) {
    showError(status, error);
  } finally {
    button.disabled = false;
  }
}

async function predictFutureRange() {
  const status = document.querySelector("#future-range-status");
  const button = document.querySelector("#future-range-button");
  status.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Forecasting range...';
  button.disabled = true;
  try {
    const data = await forecastJson("/api/forecast/future/range", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({StockCode: document.querySelector("#future-product").value, start_date: document.querySelector("#future-start").value, end_date: document.querySelector("#future-end").value})
    });
    const chart = document.querySelector("#future-range-chart");
    if (chart && window.Plotly) {
      chart.classList.remove("d-none");
      Plotly.newPlot(chart, [{x: data.predictions.map((item) => item.date), y: data.predictions.map((item) => item.predicted_daily_demand), type: "scatter", mode: "lines+markers", name: "Recursive prediction"}], {margin: {t: 20, r: 20, b: 55, l: 50}, yaxis: {title: "Demand units", rangemode: "tozero"}, xaxis: {title: "Date"}, paper_bgcolor: "transparent", plot_bgcolor: "transparent"} , {responsive: true, displayModeBar: false});
    }
    status.innerHTML = `<div class="alert alert-warning">${data.warning}</div>`;
  } catch (error) {
    showError(status, error);
  } finally {
    button.disabled = false;
  }
}

document.querySelector("#forecast-button").addEventListener("click", predict);
document.querySelector("#future-button").addEventListener("click", predictFuture);
document.querySelector("#future-range-button").addEventListener("click", predictFutureRange);
loadForecastProducts().catch((error) => showError(document.querySelector("#forecast-status"), error));
