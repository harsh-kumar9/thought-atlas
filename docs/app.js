const fmt = new Intl.NumberFormat("en-US");
const pct = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });
const one = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

const COLORS = {
  verification: "#149b8f",
  backtracking: "#d99410",
  subgoal: "#2563eb",
  backward_chaining: "#8a95a3",
  Question_and_Answering: "#0f766e",
  Perspective_Shift: "#cf4b5d",
  Conflict_of_Perspectives: "#a85535",
  Reconciliation: "#188a52",
  cognitive: "#149b8f",
  conversational: "#cf4b5d",
};

const LABELS = {
  verification: "Verification",
  backtracking: "Backtracking",
  subgoal: "Subgoal",
  backward_chaining: "Backward chaining",
  Question_and_Answering: "Question & Answering",
  Perspective_Shift: "Perspective Shift",
  Conflict_of_Perspectives: "Conflict",
  Reconciliation: "Reconciliation",
  high_quality: "High quality",
  low_quality: "Low quality",
};

const state = {
  models: new Set(),
  domains: new Set(),
  families: new Set(["cognitive", "conversational"]),
  outcomes: new Set(["solved", "failed", "high_quality", "low_quality", "unknown"]),
  chartMode: "behavior",
  bin: 10,
  traceIndex: 0,
};

const store = {};

const $ = (id) => document.getElementById(id);

async function loadData() {
  const [manifest, summary, heartbeat, traces, distance] = await Promise.all([
    fetch("data/manifest.json").then((r) => r.json()),
    fetch("data/summary.json").then((r) => r.json()),
    fetch("data/heartbeat.json").then((r) => r.json()),
    fetch("data/trace_samples.json").then((r) => r.json()),
    fetch("data/distance.json").then((r) => r.json()),
  ]);
  Object.assign(store, {
    manifest,
    summary,
    heartbeat,
    traces: traces.traces,
    distance,
    behaviors: manifest.behaviors,
    domains: manifest.domains.map((d) => d.task_type),
    models: manifest.models.map((m) => m.gen_model),
  });
  state.models = new Set(store.models);
  state.domains = new Set(store.domains);
  $("progressSlider").max = manifest.bins - 1;
  state.bin = Math.round((manifest.bins - 1) * 0.43);
  $("progressSlider").value = state.bin;
  renderFilters();
  bindEvents();
  renderAll();
}

function familyFor(behavior) {
  return store.behaviors.find((b) => b.key === behavior)?.family || "cognitive";
}

function modelLabel(model) {
  return model.replace("qwen35_", "qwen3.5 ").replace("_", " ");
}

function titleCase(s) {
  return (LABELS[s] || s).replace(/\b\w/g, (m) => m.toUpperCase());
}

function shortId(id) {
  return id ? `${id.slice(0, 8)}…${id.slice(-4)}` : "—";
}

function renderFilters() {
  $("modelCount").textContent = store.models.length;
  $("domainCount").textContent = store.domains.length;
  makeChecks("modelFilters", store.models, state.models, modelLabel, countBy("models"));
  makeChecks("domainFilters", store.domains, state.domains, titleCase, countBy("domains"));
  makeChecks("familyFilters", ["cognitive", "conversational"], state.families, titleCase);
  makeChecks("outcomeFilters", ["solved", "failed", "high_quality", "low_quality", "unknown"], state.outcomes, titleCase);
}

function countBy(kind) {
  const map = new Map();
  if (kind === "models") {
    store.manifest.models.forEach((m) => map.set(m.gen_model, m.n_traces));
  } else {
    store.manifest.domains.forEach((d) => map.set(d.task_type, d.n_traces));
  }
  return map;
}

function makeChecks(targetId, items, selectedSet, labeler, counts = null) {
  const target = $(targetId);
  target.innerHTML = "";
  items.forEach((item) => {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = selectedSet.has(item);
    input.addEventListener("change", () => {
      if (input.checked) selectedSet.add(item);
      else selectedSet.delete(item);
      state.traceIndex = 0;
      renderAll();
    });
    const text = document.createElement("span");
    text.className = "label";
    text.textContent = labeler(item);
    label.append(input, text);
    if (counts) {
      const sub = document.createElement("span");
      sub.className = "subcount";
      sub.textContent = fmt.format(counts.get(item) || 0);
      label.append(sub);
    }
    target.appendChild(label);
  });
}

function bindEvents() {
  $("resetFilters").addEventListener("click", () => {
    state.models = new Set(store.models);
    state.domains = new Set(store.domains);
    state.families = new Set(["cognitive", "conversational"]);
    state.outcomes = new Set(["solved", "failed", "high_quality", "low_quality", "unknown"]);
    state.traceIndex = 0;
    renderFilters();
    renderAll();
  });
  $("progressSlider").addEventListener("input", (event) => {
    state.bin = Number(event.target.value);
    renderChart();
  });
  $("stepBack").addEventListener("click", () => {
    state.bin = Math.max(0, state.bin - 1);
    $("progressSlider").value = state.bin;
    renderChart();
  });
  $("stepForward").addEventListener("click", () => {
    state.bin = Math.min(store.manifest.bins - 1, state.bin + 1);
    $("progressSlider").value = state.bin;
    renderChart();
  });
  document.querySelectorAll("[data-chart-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-chart-mode]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      state.chartMode = button.dataset.chartMode;
      renderChart();
    });
  });
  $("prevTrace").addEventListener("click", () => {
    const traces = filteredTraces();
    state.traceIndex = (state.traceIndex - 1 + traces.length) % Math.max(1, traces.length);
    renderTrace();
  });
  $("nextTrace").addEventListener("click", () => {
    const traces = filteredTraces();
    state.traceIndex = (state.traceIndex + 1) % Math.max(1, traces.length);
    renderTrace();
  });
  $("distanceKind").addEventListener("change", renderDistance);
  $("toggleFilters").addEventListener("click", () => $("filtersPanel").classList.toggle("open"));
  document.querySelectorAll("[data-scroll]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
      button.classList.add("active");
      $(button.dataset.scroll).scrollIntoView({ block: "start" });
    });
  });
  document.querySelectorAll(".copy-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const trace = currentTrace();
      const key = button.dataset.copy;
      const text = key === "prompt" ? trace?.prompt?.text : key === "thinking" ? trace?.thinking?.text : trace?.answer?.text;
      if (!text) return;
      await navigator.clipboard?.writeText(text);
      button.textContent = "Copied";
      setTimeout(() => (button.textContent = "Copy"), 900);
    });
  });
}

function renderAll() {
  renderMetrics();
  renderChart();
  renderTrace();
  renderSummary();
  renderDistance();
}

function selectedCells() {
  return store.summary.cells.filter((cell) => state.models.has(cell.gen_model) && state.domains.has(cell.task_type));
}

function renderMetrics() {
  const cells = selectedCells();
  const traces = cells.reduce((sum, c) => sum + c.n_traces, 0);
  const completion = weighted(cells, "completed_rate", "n_traces");
  const medianTokens = median(cells.map((c) => c.median_new_tokens).filter(Number.isFinite));
  const success = average(cells.map((c) => c.success_rate).filter((v) => v !== null));
  const quality = average(cells.map((c) => c.quality_score).filter((v) => v !== null));
  const behavior = average(cells.map((c) => c.mean_behavior_count).filter((v) => v !== null));
  const metrics = [
    ["Traces", fmt.format(traces)],
    ["Completed", completion == null ? "—" : pct.format(completion)],
    ["Median Tokens", medianTokens == null ? "—" : fmt.format(Math.round(medianTokens))],
    ["Success / Quality", [success, quality].filter((v) => v != null).map((v) => pct.format(v)).join(" / ") || "—"],
    ["Behavior Count", behavior == null ? "—" : one.format(behavior)],
  ];
  const row = $("metricRow");
  row.innerHTML = "";
  const template = $("metricTemplate");
  metrics.forEach(([label, value]) => {
    const node = template.content.cloneNode(true);
    node.querySelector("span").textContent = label;
    node.querySelector("strong").textContent = value;
    row.appendChild(node);
  });
  $("chartSubtitle").textContent = `${state.models.size} model${state.models.size === 1 ? "" : "s"} · ${state.domains.size} domain${state.domains.size === 1 ? "" : "s"} · ${[...state.families].map(titleCase).join(" + ")}`;
}

function renderChart() {
  const svg = $("heartbeatChart");
  const width = svg.clientWidth || 820;
  const height = svg.clientHeight || 360;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const margin = { top: 18, right: 24, bottom: 40, left: 48 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const series = buildSeries();
  const maxY = Math.max(0.01, ...series.flatMap((s) => s.values.map((v) => v.freq)));
  const x = (bin) => margin.left + (bin / (store.manifest.bins - 1)) * plotW;
  const y = (value) => margin.top + plotH - (value / maxY) * plotH;

  drawGrid(svg, width, height, margin, plotW, plotH, maxY, x, y);

  series.forEach((s) => {
    const path = s.values
      .map((v, i) => `${i === 0 ? "M" : "L"} ${x(v.bin).toFixed(2)} ${y(v.freq).toFixed(2)}`)
      .join(" ");
    const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
    el.setAttribute("d", path);
    el.setAttribute("fill", "none");
    el.setAttribute("stroke", s.color);
    el.setAttribute("stroke-width", s.key === "backward_chaining" ? "1.5" : "2.2");
    el.setAttribute("opacity", s.key === "backward_chaining" ? "0.65" : "0.95");
    svg.appendChild(el);
  });

  const scrubX = x(state.bin);
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", scrubX);
  line.setAttribute("x2", scrubX);
  line.setAttribute("y1", margin.top);
  line.setAttribute("y2", margin.top + plotH);
  line.setAttribute("stroke", "#111827");
  line.setAttribute("stroke-width", "1.2");
  line.setAttribute("stroke-dasharray", "4 4");
  svg.appendChild(line);

  series.forEach((s) => {
    const point = s.values.find((v) => v.bin === state.bin);
    if (!point) return;
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", scrubX);
    dot.setAttribute("cy", y(point.freq));
    dot.setAttribute("r", "4");
    dot.setAttribute("fill", s.color);
    dot.setAttribute("stroke", "#fff");
    dot.setAttribute("stroke-width", "1.5");
    svg.appendChild(dot);
  });

  const progress = state.bin / (store.manifest.bins - 1);
  $("progressPct").textContent = `${Math.round(progress * 100)}%`;
  renderScrubTooltip(series, progress);
}

function buildSeries() {
  const groups = new Map();
  store.heartbeat.curves.forEach((row) => {
    if (!state.models.has(row.gen_model) || !state.domains.has(row.task_type) || !state.outcomes.has(row.outcome)) return;
    const family = familyFor(row.behavior);
    if (!state.families.has(family)) return;
    const key = state.chartMode === "family" ? family : row.behavior;
    if (!groups.has(key)) groups.set(key, Array.from({ length: store.manifest.bins }, () => ({ sum: 0, weight: 0 })));
    const bucket = groups.get(key)[row.bin];
    bucket.sum += row.freq * row.n_segments;
    bucket.weight += row.n_segments;
  });
  return [...groups.entries()]
    .map(([key, bins]) => ({
      key,
      label: titleCase(key),
      color: COLORS[key] || "#2563eb",
      values: bins.map((b, bin) => ({ bin, freq: b.weight ? b.sum / b.weight : 0 })),
    }))
    .sort((a, b) => (a.key > b.key ? 1 : -1));
}

function drawGrid(svg, width, height, margin, plotW, plotH, maxY, x, y) {
  const axisColor = "#9aa5b3";
  for (let i = 0; i <= 4; i += 1) {
    const yy = margin.top + (plotH / 4) * i;
    const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
    grid.setAttribute("x1", margin.left);
    grid.setAttribute("x2", width - margin.right);
    grid.setAttribute("y1", yy);
    grid.setAttribute("y2", yy);
    grid.setAttribute("stroke", "#e5ebf1");
    svg.appendChild(grid);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", margin.left - 8);
    label.setAttribute("y", yy + 4);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("font-size", "11");
    label.setAttribute("fill", axisColor);
    label.textContent = `${Math.round((maxY - (maxY / 4) * i) * 100)}%`;
    svg.appendChild(label);
  }
  [0, 0.25, 0.5, 0.75, 1].forEach((p) => {
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x(p * (store.manifest.bins - 1)));
    label.setAttribute("y", height - 12);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "11");
    label.setAttribute("fill", axisColor);
    label.textContent = `${Math.round(p * 100)}%`;
    svg.appendChild(label);
  });
}

function renderScrubTooltip(series, progress) {
  const rows = series
    .map((s) => {
      const v = s.values.find((p) => p.bin === state.bin)?.freq || 0;
      return { label: s.label, color: s.color, value: v };
    })
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);
  $("scrubTooltip").innerHTML = `<b>${Math.round(progress * 100)}% through trace</b>${rows
    .map((r) => `<div><span style="color:${r.color}">●</span> ${r.label}: ${pct.format(r.value)}</div>`)
    .join("")}`;
}

function filteredTraces() {
  return store.traces.filter((t) => state.models.has(t.gen_model) && state.domains.has(t.task_type) && state.outcomes.has(t.outcome));
}

function currentTrace() {
  const traces = filteredTraces();
  if (!traces.length) return null;
  state.traceIndex = Math.min(state.traceIndex, traces.length - 1);
  return traces[state.traceIndex];
}

function renderTrace() {
  const trace = currentTrace();
  if (!trace) {
    $("traceTitle").textContent = "Trace";
    $("traceMeta").textContent = "No sampled traces match the active filters.";
    $("traceFacts").innerHTML = "";
    $("promptText").textContent = "";
    $("thinkingText").textContent = "";
    $("answerText").textContent = "";
    return;
  }
  const traces = filteredTraces();
  $("traceTitle").textContent = `Trace @ ${Math.round((state.bin / (store.manifest.bins - 1)) * 100)}%`;
  $("traceMeta").textContent = `${state.traceIndex + 1} of ${traces.length} sampled traces · ${shortId(trace.trace_id)}`;
  const facts = [
    ["Model", modelLabel(trace.gen_model)],
    ["Domain", titleCase(trace.task_type)],
    ["Outcome", titleCase(trace.outcome)],
    ["Completed", trace.completed ? "yes" : "no"],
    ["Total Tokens", fmt.format(trace.n_new_tokens || 0)],
    ["Failure Mode", trace.failure_mode || "—"],
  ];
  $("traceFacts").innerHTML = facts.map(([k, v]) => `<div class="fact"><span>${k}</span><strong title="${v}">${v}</strong></div>`).join("");
  $("thinkingLabel").textContent = `Thinking (${fmt.format(trace.thinking.tokens_est || 0)} est. tokens${trace.thinking.truncated ? ", clipped" : ""})`;
  $("answerLabel").textContent = `Answer (${fmt.format(trace.answer.tokens_est || 0)} est. tokens${trace.answer.truncated ? ", clipped" : ""})`;
  $("promptText").textContent = trace.prompt.text || "—";
  $("thinkingText").textContent = trace.thinking.text || "No explicit thinking block for this trace.";
  $("answerText").textContent = trace.answer.text || "—";
}

function renderSummary() {
  const models = store.models.filter((m) => state.models.has(m));
  const domains = store.domains.filter((d) => state.domains.has(d));
  const byKey = new Map(store.summary.cells.map((c) => [`${c.gen_model}|${c.task_type}`, c]));
  let html = "<thead><tr><th>Model</th>";
  domains.forEach((d) => (html += `<th>${titleCase(d)}</th>`));
  html += "</tr></thead><tbody>";
  models.forEach((m) => {
    html += `<tr><td><strong>${modelLabel(m)}</strong></td>`;
    domains.forEach((d) => {
      const c = byKey.get(`${m}|${d}`);
      if (!c) {
        html += "<td>—</td>";
        return;
      }
      const score = c.success_rate != null ? pct.format(c.success_rate) : c.quality_score != null ? pct.format(c.quality_score) : "—";
      html += `<td><strong>${score}</strong><small>${pct.format(c.completed_rate)} done · ${fmt.format(Math.round(c.median_new_tokens || 0))} tok</small></td>`;
    });
    html += "</tr>";
  });
  html += "</tbody>";
  $("summaryTable").innerHTML = html;
}

function renderDistance() {
  const kind = $("distanceKind").value;
  const rows = store.distance[kind] || [];
  const models = store.models.filter((m) => state.models.has(m));
  const values = new Map();
  rows.forEach((r) => {
    values.set(`${r.model_a}|${r.model_b}`, Number(r.overall_dist));
    values.set(`${r.model_b}|${r.model_a}`, Number(r.overall_dist));
  });
  const allVals = [...values.values()].filter(Number.isFinite);
  const max = Math.max(0.01, ...allVals);
  const matrix = $("distanceMatrix");
  matrix.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "matrix-grid";
  grid.style.gridTemplateColumns = `120px repeat(${models.length}, minmax(86px, 1fr))`;
  grid.appendChild(cell("", "matrix-head"));
  models.forEach((m) => grid.appendChild(cell(modelLabel(m), "matrix-head")));
  models.forEach((row) => {
    grid.appendChild(cell(modelLabel(row), "matrix-head"));
    models.forEach((col) => {
      const v = row === col ? null : values.get(`${row}|${col}`);
      const c = cell(v == null ? "—" : v.toFixed(3), "matrix-cell");
      if (v != null) c.style.background = heat(v / max);
      grid.appendChild(c);
    });
  });
  matrix.appendChild(grid);
}

function cell(text, className) {
  const div = document.createElement("div");
  div.className = `matrix-cell ${className || ""}`;
  div.textContent = text;
  return div;
}

function heat(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const stops = [
    [64, 183, 168],
    [240, 198, 106],
    [217, 101, 111],
  ];
  const a = clamped < 0.5 ? stops[0] : stops[1];
  const b = clamped < 0.5 ? stops[1] : stops[2];
  const local = clamped < 0.5 ? clamped * 2 : (clamped - 0.5) * 2;
  const rgb = a.map((v, i) => Math.round(v + (b[i] - v) * local));
  return `rgb(${rgb.join(",")})`;
}

function weighted(rows, key, weightKey) {
  const ok = rows.filter((r) => r[key] != null && r[weightKey] > 0);
  const den = ok.reduce((s, r) => s + r[weightKey], 0);
  if (!den) return null;
  return ok.reduce((s, r) => s + r[key] * r[weightKey], 0) / den;
}

function average(values) {
  const ok = values.filter((v) => v != null && Number.isFinite(v));
  if (!ok.length) return null;
  return ok.reduce((s, v) => s + v, 0) / ok.length;
}

function median(values) {
  const ok = values.filter((v) => v != null && Number.isFinite(v)).sort((a, b) => a - b);
  if (!ok.length) return null;
  const mid = Math.floor(ok.length / 2);
  return ok.length % 2 ? ok[mid] : (ok[mid - 1] + ok[mid]) / 2;
}

loadData().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<main class="panel" style="margin:24px;padding:24px"><h1>Dashboard data failed to load</h1><p>${error.message}</p></main>`;
});
