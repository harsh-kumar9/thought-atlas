const fmt = new Intl.NumberFormat("en-US");
const pct = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });
const one = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

const LABELS = {
  verification: "Verification",
  backtracking: "Backtracking",
  subgoal: "Subgoal",
  backward_chaining: "Backward chaining",
  Question_and_Answering: "Question & Answering",
  Perspective_Shift: "Perspective Shift",
  Conflict_of_Perspectives: "Conflict of Perspectives",
  Reconciliation: "Reconciliation",
  cognitive: "Cognitive",
  conversational: "Conversational",
  solved: "Solved",
  failed: "Failed",
  high_quality: "High quality",
  low_quality: "Low quality",
  unknown: "Unknown",
};

const OUTCOME_GROUPS = {
  all: { label: "All traces", outcomes: ["solved", "failed", "high_quality", "low_quality", "unknown"] },
  positive: { label: "Solved / high-quality", outcomes: ["solved", "high_quality"] },
  negative: { label: "Failed / low-quality", outcomes: ["failed", "low_quality"] },
  solved: { label: "Solved only", outcomes: ["solved"] },
  failed: { label: "Failed only", outcomes: ["failed"] },
  high_quality: { label: "High-quality only", outcomes: ["high_quality"] },
  low_quality: { label: "Low-quality only", outcomes: ["low_quality"] },
  unknown: { label: "Unknown outcome", outcomes: ["unknown"] },
};

const LANE_COLORS = {
  a: { line: "#137a3f", band: "rgba(19, 122, 63, 0.13)" },
  b: { line: "#cf3f36", band: "rgba(207, 63, 54, 0.13)" },
};

const state = {
  modelA: null,
  domainA: null,
  outcomeA: "all",
  modelB: null,
  domainB: null,
  outcomeB: "all",
  behaviors: new Set(),
  selectedBehavior: null,
  bin: 10,
  traceLane: "a",
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

  initializeState();
  renderControls();
  bindEvents();
  renderAll();
}

function initializeState() {
  state.modelA = store.models.includes("reasoner") ? "reasoner" : store.models[0];
  state.modelB = store.models.includes("qwen35_27b") ? "qwen35_27b" : store.models[Math.min(1, store.models.length - 1)];
  state.domainA = store.domains.includes("math") ? "math" : store.domains[0];
  state.domainB = state.domainA;
  state.outcomeA = "all";
  state.outcomeB = "all";
  const conversational = store.behaviors.filter((b) => b.family === "conversational").map((b) => b.key);
  state.behaviors = new Set(conversational.length ? conversational : store.behaviors.map((b) => b.key));
  state.selectedBehavior = [...state.behaviors][0];
  state.bin = Math.round((store.manifest.bins - 1) * 0.43);
  $("progressSlider").max = store.manifest.bins - 1;
  $("progressSlider").value = state.bin;
}

function renderControls() {
  makeSelect("modelA", store.models, modelLabel, state.modelA);
  makeSelect("modelB", store.models, modelLabel, state.modelB);
  makeSelect("domainA", store.domains, titleCase, state.domainA);
  makeSelect("domainB", store.domains, titleCase, state.domainB);
  makeSelect("outcomeA", Object.keys(OUTCOME_GROUPS), (k) => OUTCOME_GROUPS[k].label, state.outcomeA);
  makeSelect("outcomeB", Object.keys(OUTCOME_GROUPS), (k) => OUTCOME_GROUPS[k].label, state.outcomeB);
  renderBehaviorFilters();
}

function makeSelect(id, values, labeler, selected) {
  const select = $(id);
  select.innerHTML = values.map((value) => `<option value="${escapeAttr(value)}">${labeler(value)}</option>`).join("");
  select.value = selected;
}

function renderBehaviorFilters() {
  $("behaviorCount").textContent = state.behaviors.size;
  const target = $("behaviorFilters");
  target.innerHTML = "";
  store.behaviors.forEach((behavior) => {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.behaviors.has(behavior.key);
    input.addEventListener("change", () => {
      if (input.checked) state.behaviors.add(behavior.key);
      else state.behaviors.delete(behavior.key);
      if (!state.behaviors.has(state.selectedBehavior)) {
        state.selectedBehavior = [...state.behaviors][0] || behavior.key;
      }
      renderBehaviorFilters();
      renderComparison();
      renderInspector();
    });
    const text = document.createElement("span");
    text.className = "label";
    text.textContent = titleCase(behavior.key);
    const family = document.createElement("span");
    family.className = "family";
    family.textContent = titleCase(behavior.family);
    label.append(input, text, family);
    target.appendChild(label);
  });
}

function bindEvents() {
  ["modelA", "modelB", "domainA", "domainB", "outcomeA", "outcomeB"].forEach((id) => {
    $(id).addEventListener("change", (event) => {
      state[id] = event.target.value;
      state.traceIndex = 0;
      renderAll();
    });
  });

  $("swapLanes").addEventListener("click", () => {
    [state.modelA, state.modelB] = [state.modelB, state.modelA];
    [state.domainA, state.domainB] = [state.domainB, state.domainA];
    [state.outcomeA, state.outcomeB] = [state.outcomeB, state.outcomeA];
    renderControls();
    renderAll();
  });

  $("resetControls").addEventListener("click", () => {
    initializeState();
    renderControls();
    renderAll();
  });

  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      const preset = button.dataset.preset;
      const selected = store.behaviors
        .filter((b) => preset === "all" || b.family === preset)
        .map((b) => b.key);
      state.behaviors = new Set(selected);
      state.selectedBehavior = selected[0] || state.selectedBehavior;
      renderBehaviorFilters();
      renderComparison();
      renderInspector();
    });
  });

  $("progressSlider").addEventListener("input", (event) => {
    state.bin = Number(event.target.value);
    renderComparison();
    renderInspector();
    renderTrace();
  });

  $("stepBack").addEventListener("click", () => {
    state.bin = Math.max(0, state.bin - 1);
    $("progressSlider").value = state.bin;
    renderComparison();
    renderInspector();
    renderTrace();
  });

  $("stepForward").addEventListener("click", () => {
    state.bin = Math.min(store.manifest.bins - 1, state.bin + 1);
    $("progressSlider").value = state.bin;
    renderComparison();
    renderInspector();
    renderTrace();
  });

  document.querySelectorAll("[data-trace-lane]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-trace-lane]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      state.traceLane = button.dataset.traceLane;
      state.traceIndex = 0;
      renderTrace();
      renderInspector();
    });
  });

  $("prevTrace").addEventListener("click", () => {
    const traces = filteredTracesForLane(state.traceLane);
    state.traceIndex = (state.traceIndex - 1 + traces.length) % Math.max(1, traces.length);
    renderTrace();
  });

  $("nextTrace").addEventListener("click", () => {
    const traces = filteredTracesForLane(state.traceLane);
    state.traceIndex = (state.traceIndex + 1) % Math.max(1, traces.length);
    renderTrace();
  });

  $("distanceKind").addEventListener("change", renderDistance);
  $("toggleControls").addEventListener("click", () => $("controlsPanel").classList.toggle("open"));

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
  renderComparison();
  renderInspector();
  renderTrace();
  renderSummary();
  renderDistance();
}

function renderComparison() {
  const progress = state.bin / (store.manifest.bins - 1);
  $("progressPct").textContent = `${Math.round(progress * 100)}%`;
  $("comparisonSubtitle").textContent = `${state.behaviors.size} behavior${state.behaviors.size === 1 ? "" : "s"} at ${Math.round(progress * 100)}% through trace`;
  $("laneReadout").innerHTML = `
    <div class="lane-chip a"><strong>${laneTitle("a")}</strong><span>${OUTCOME_GROUPS[state.outcomeA].label}</span></div>
    <div class="lane-chip b"><strong>${laneTitle("b")}</strong><span>${OUTCOME_GROUPS[state.outcomeB].label}</span></div>
  `;

  const behaviors = [...state.behaviors];
  const grid = $("comparisonGrid");
  grid.innerHTML = "";
  if (!behaviors.length) {
    grid.innerHTML = '<div class="empty-state">Select at least one behavior to draw trajectory comparisons.</div>';
    return;
  }

  behaviors.forEach((behavior) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `mini-chart ${behavior === state.selectedBehavior ? "selected" : ""}`;
    card.addEventListener("click", () => {
      state.selectedBehavior = behavior;
      renderComparison();
      renderInspector();
    });

    const curveA = aggregateCurve("a", behavior);
    const curveB = aggregateCurve("b", behavior);
    const pointA = curveA.values[state.bin]?.freq || 0;
    const pointB = curveB.values[state.bin]?.freq || 0;
    const delta = pointA - pointB;
    card.innerHTML = `
      <div class="mini-chart-head">
        <div><h3>${titleCase(behavior)}</h3><small>${titleCase(familyFor(behavior))}</small></div>
        <span class="delta-pill">${signedPct(delta)}</span>
      </div>
      <svg role="img" aria-label="${titleCase(behavior)} trajectory comparison"></svg>
    `;
    drawMiniChart(card.querySelector("svg"), curveA, curveB);
    grid.appendChild(card);
  });
}

function renderInspector() {
  const behavior = state.selectedBehavior || [...state.behaviors][0];
  if (!behavior) {
    $("inspectorTitle").textContent = "Inspector";
    $("inspectorSubtitle").textContent = "Select a behavior to inspect.";
    $("deltaFacts").innerHTML = "";
    $("annotationNote").textContent = "";
    $("sampleList").innerHTML = "";
    return;
  }

  const curveA = aggregateCurve("a", behavior);
  const curveB = aggregateCurve("b", behavior);
  const progress = state.bin / (store.manifest.bins - 1);
  const atA = curveA.values[state.bin]?.freq || 0;
  const atB = curveB.values[state.bin]?.freq || 0;
  const aucA = average(curveA.values.map((v) => v.freq));
  const aucB = average(curveB.values.map((v) => v.freq));
  const earlyA = average(curveA.values.filter((v) => v.bin / (store.manifest.bins - 1) <= 0.4).map((v) => v.freq));
  const earlyB = average(curveB.values.filter((v) => v.bin / (store.manifest.bins - 1) <= 0.4).map((v) => v.freq));
  const lateA = average(curveA.values.filter((v) => v.bin / (store.manifest.bins - 1) >= 0.6).map((v) => v.freq));
  const lateB = average(curveB.values.filter((v) => v.bin / (store.manifest.bins - 1) >= 0.6).map((v) => v.freq));

  $("inspectorTitle").textContent = titleCase(behavior);
  $("inspectorSubtitle").textContent = `${Math.round(progress * 100)}% through trace · ${titleCase(familyFor(behavior))} behavior`;
  $("deltaFacts").innerHTML = [
    ["Lane A at cursor", pct.format(atA)],
    ["Lane B at cursor", pct.format(atB)],
    ["Cursor delta", signedPct(atA - atB)],
    ["AUC delta", signedPct(aucA - aucB)],
    ["Early delta (0-40%)", signedPct(earlyA - earlyB)],
    ["Late delta (60-100%)", signedPct(lateA - lateB)],
  ]
    .map(([label, value]) => `<div class="delta-fact"><span>${label}</span><strong class="${value.startsWith("-") ? "negative" : "positive"}">${value}</strong></div>`)
    .join("");

  renderSampleInspector(behavior);
}

function renderSampleInspector(behavior) {
  const tracesA = filteredTracesForLane("a").slice(0, 3);
  const tracesB = filteredTracesForLane("b").slice(0, 3);
  const annotated = [...tracesA, ...tracesB].some((trace) => Array.isArray(trace.annotations) && trace.annotations.length);
  $("annotationNote").textContent = annotated
    ? "Sentence-level behavior annotations are shown for sampled traces when the export includes them."
    : "Current static samples expose raw prompt/thinking/answer text plus per-trace behavior counts. Re-run the dashboard export after the annotation upgrade to show sentence-level behavior spans.";

  const cards = [
    ...tracesA.map((trace) => sampleCard(trace, "Lane A", behavior)),
    ...tracesB.map((trace) => sampleCard(trace, "Lane B", behavior)),
  ];
  $("sampleList").innerHTML = cards.join("") || '<div class="empty-state">No sampled traces match the selected comparison lanes.</div>';
}

function sampleCard(trace, lane, behavior) {
  const count = trace.behavior_counts?.[behavior] || 0;
  const matchingAnnotations = (trace.annotations || []).filter((a) => (a.behaviors || []).includes(behavior)).slice(0, 2);
  const snippet = matchingAnnotations.length
    ? matchingAnnotations.map((a) => `${Math.round(a.norm_pos * 100)}%: ${escapeHtml(a.text)}`).join("<br>")
    : escapeHtml(firstUsefulText(trace));
  return `
    <article class="sample-card">
      <h3>${lane} · ${modelLabel(trace.gen_model)} · ${titleCase(trace.task_type)}</h3>
      <p>${shortId(trace.trace_id)} · ${titleCase(trace.outcome)} · ${fmt.format(count)} ${titleCase(behavior)} mark${count === 1 ? "" : "s"}</p>
      <p>${snippet || "No text excerpt available for this sample."}</p>
      <div class="behavior-chips">${topBehaviorChips(trace)}</div>
    </article>
  `;
}

function drawMiniChart(svg, curveA, curveB) {
  const width = 300;
  const height = 165;
  const margin = { top: 12, right: 12, bottom: 25, left: 34 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const all = [...curveA.values, ...curveB.values];
  const maxY = Math.max(0.01, ...all.map((v) => v.upper || v.freq || 0)) * 1.05;
  const x = (bin) => margin.left + (bin / (store.manifest.bins - 1)) * plotW;
  const y = (value) => margin.top + plotH - (value / maxY) * plotH;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  drawMiniGrid(svg, width, height, margin, plotW, plotH, maxY, x);
  drawBoundary(svg, curveA.boundary, curveB.boundary, x, margin, plotH);
  drawBand(svg, curveA.values, x, y, LANE_COLORS.a.band);
  drawBand(svg, curveB.values, x, y, LANE_COLORS.b.band);
  drawLine(svg, curveA.values, x, y, LANE_COLORS.a.line, false);
  drawLine(svg, curveB.values, x, y, LANE_COLORS.b.line, true);

  const scrubX = x(state.bin);
  svg.appendChild(svgEl("line", {
    x1: scrubX,
    x2: scrubX,
    y1: margin.top,
    y2: margin.top + plotH,
    stroke: "#111827",
    "stroke-width": "1",
    "stroke-dasharray": "3 3",
  }));

  [curveA, curveB].forEach((curve, idx) => {
    const point = curve.values[state.bin];
    if (!point) return;
    svg.appendChild(svgEl("circle", {
      cx: scrubX,
      cy: y(point.freq || 0),
      r: "3.2",
      fill: idx === 0 ? LANE_COLORS.a.line : LANE_COLORS.b.line,
      stroke: "#fff",
      "stroke-width": "1.4",
    }));
  });
}

function drawBoundary(svg, boundaryA, boundaryB, x, margin, plotH) {
  const ranges = [boundaryA, boundaryB].filter(Boolean);
  if (!ranges.length) return;
  const q25 = Math.min(...ranges.map((r) => r.q25));
  const q75 = Math.max(...ranges.map((r) => r.q75));
  const x1 = x(q25 * (store.manifest.bins - 1));
  const x2 = x(q75 * (store.manifest.bins - 1));
  svg.appendChild(svgEl("rect", {
    x: x1,
    y: margin.top,
    width: Math.max(1, x2 - x1),
    height: plotH,
    fill: "rgba(17, 24, 39, 0.06)",
    stroke: "none",
  }));
}

function drawMiniGrid(svg, width, height, margin, plotW, plotH, maxY, x) {
  const yAxis = (value) => margin.top + plotH - (value / maxY) * plotH;
  [0, 0.5, 1].forEach((p) => {
    const yy = yAxis(maxY * p);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, stroke: "#e7edf3", "stroke-width": "1" }));
    const label = svgEl("text", { x: margin.left - 6, y: yy + 3, "text-anchor": "end", "font-size": "9.5", fill: "#7a8696", stroke: "none" });
    label.textContent = p === 0 ? "0" : `${Math.round(maxY * p * 100)}%`;
    svg.appendChild(label);
  });
  [0, 0.5, 1].forEach((p) => {
    const label = svgEl("text", { x: x(p * (store.manifest.bins - 1)), y: height - 8, "text-anchor": "middle", "font-size": "9.5", fill: "#7a8696", stroke: "none" });
    label.textContent = `${Math.round(p * 100)}%`;
    svg.appendChild(label);
  });
}

function drawBand(svg, values, x, y, color) {
  if (!values.length) return;
  const upper = values.map((v, i) => `${i === 0 ? "M" : "L"} ${x(v.bin).toFixed(2)} ${y(v.upper).toFixed(2)}`).join(" ");
  const lower = values
    .slice()
    .reverse()
    .map((v) => `L ${x(v.bin).toFixed(2)} ${y(v.lower).toFixed(2)}`)
    .join(" ");
  svg.appendChild(svgEl("path", { d: `${upper} ${lower} Z`, fill: color, stroke: "none" }));
}

function drawLine(svg, values, x, y, color, dashed) {
  if (!values.length) return;
  const path = values.map((v, i) => `${i === 0 ? "M" : "L"} ${x(v.bin).toFixed(2)} ${y(v.freq || 0).toFixed(2)}`).join(" ");
  svg.appendChild(svgEl("path", {
    d: path,
    fill: "none",
    stroke: color,
    "stroke-width": "2.2",
    "stroke-dasharray": dashed ? "5 4" : "",
  }));
}

function aggregateCurve(laneKey, behavior) {
  const lane = laneConfig(laneKey);
  const outcomes = new Set(OUTCOME_GROUPS[lane.outcome].outcomes);
  const bins = Array.from({ length: store.manifest.bins }, (_, bin) => ({ bin, sum: 0, weight: 0, traces: 0 }));
  store.heartbeat.curves.forEach((row) => {
    if (row.gen_model !== lane.model || row.task_type !== lane.domain || row.behavior !== behavior || !outcomes.has(row.outcome)) return;
    const bucket = bins[row.bin];
    bucket.sum += row.freq * row.n_segments;
    bucket.weight += row.n_segments;
    bucket.traces += row.n_traces || 0;
  });
  return {
    lane: laneKey,
    boundary: aggregateBoundary(lane),
    values: bins.map((bucket) => {
      const freq = bucket.weight ? bucket.sum / bucket.weight : 0;
      const se = bucket.weight ? Math.sqrt(Math.max(0, freq * (1 - freq)) / bucket.weight) : 0;
      return {
        bin: bucket.bin,
        freq,
        n: bucket.weight,
        lower: Math.max(0, freq - 1.96 * se),
        upper: Math.min(1, freq + 1.96 * se),
      };
    }),
  };
}

function aggregateBoundary(lane) {
  const rows = store.heartbeat.answer_boundaries || [];
  if (!rows.length) return null;
  const outcomes = new Set(OUTCOME_GROUPS[lane.outcome].outcomes);
  const matches = rows.filter((row) => row.gen_model === lane.model && row.task_type === lane.domain && outcomes.has(row.outcome));
  const total = matches.reduce((sum, row) => sum + (row.n_traces || 0), 0);
  if (!total) return null;
  return {
    q25: matches.reduce((sum, row) => sum + row.q25 * (row.n_traces || 0), 0) / total,
    q75: matches.reduce((sum, row) => sum + row.q75 * (row.n_traces || 0), 0) / total,
  };
}

function laneConfig(laneKey) {
  return laneKey === "a"
    ? { model: state.modelA, domain: state.domainA, outcome: state.outcomeA }
    : { model: state.modelB, domain: state.domainB, outcome: state.outcomeB };
}

function laneTitle(laneKey) {
  const lane = laneConfig(laneKey);
  return `${modelLabel(lane.model)} · ${titleCase(lane.domain)}`;
}

function renderTrace() {
  const trace = currentTrace();
  if (!trace) {
    $("traceTitle").textContent = "Raw Trace";
    $("traceMeta").textContent = "No sampled traces match the active raw trace lane.";
    $("traceFacts").innerHTML = "";
    $("promptText").textContent = "";
    $("thinkingText").textContent = "";
    $("answerText").textContent = "";
    return;
  }

  const traces = filteredTracesForLane(state.traceLane);
  $("traceTitle").textContent = `${state.traceLane === "a" ? "Lane A" : "Lane B"} Raw Trace`;
  $("traceMeta").textContent = `${state.traceIndex + 1} of ${traces.length} sampled traces · ${shortId(trace.trace_id)}`;
  const facts = [
    ["Model", modelLabel(trace.gen_model)],
    ["Domain", titleCase(trace.task_type)],
    ["Outcome", titleCase(trace.outcome)],
    ["Completed", trace.completed ? "yes" : "no"],
    ["Total Tokens", fmt.format(trace.n_new_tokens || 0)],
    ["Failure Mode", trace.failure_mode || "-"],
  ];
  $("traceFacts").innerHTML = facts.map(([k, v]) => `<div class="fact"><span>${k}</span><strong title="${escapeAttr(v)}">${escapeHtml(v)}</strong></div>`).join("");
  $("thinkingLabel").textContent = `Thinking (${fmt.format(trace.thinking.tokens_est || 0)} est. tokens${trace.thinking.truncated ? ", clipped" : ""})`;
  $("answerLabel").textContent = `Answer (${fmt.format(trace.answer.tokens_est || 0)} est. tokens${trace.answer.truncated ? ", clipped" : ""})`;
  $("promptText").textContent = trace.prompt.text || "-";
  $("thinkingText").textContent = trace.thinking.text || "No explicit thinking block for this trace.";
  $("answerText").textContent = trace.answer.text || "-";
}

function filteredTracesForLane(laneKey) {
  const lane = laneConfig(laneKey);
  const outcomes = new Set(OUTCOME_GROUPS[lane.outcome].outcomes);
  return store.traces.filter((trace) => trace.gen_model === lane.model && trace.task_type === lane.domain && outcomes.has(trace.outcome));
}

function currentTrace() {
  const traces = filteredTracesForLane(state.traceLane);
  if (!traces.length) return null;
  state.traceIndex = Math.min(state.traceIndex, traces.length - 1);
  return traces[state.traceIndex];
}

function renderSummary() {
  const domains = store.domains;
  const byKey = new Map(store.summary.cells.map((c) => [`${c.gen_model}|${c.task_type}`, c]));
  let html = "<thead><tr><th>Model</th>";
  domains.forEach((d) => (html += `<th>${titleCase(d)}</th>`));
  html += "</tr></thead><tbody>";
  store.models.forEach((model) => {
    html += `<tr><td><strong>${modelLabel(model)}</strong></td>`;
    domains.forEach((domain) => {
      const cell = byKey.get(`${model}|${domain}`);
      if (!cell) {
        html += "<td>-</td>";
        return;
      }
      const score = cell.success_rate != null ? pct.format(cell.success_rate) : cell.quality_score != null ? pct.format(cell.quality_score) : "-";
      html += `<td><strong>${score}</strong><small>${pct.format(cell.completed_rate)} done · ${fmt.format(Math.round(cell.median_new_tokens || 0))} tok · ${one.format(cell.mean_behavior_count || 0)} beh</small></td>`;
    });
    html += "</tr>";
  });
  html += "</tbody>";
  $("summaryTable").innerHTML = html;
}

function renderDistance() {
  const kind = $("distanceKind").value;
  const rows = store.distance[kind] || [];
  const models = store.models;
  const values = new Map();
  rows.forEach((row) => {
    values.set(`${row.model_a}|${row.model_b}`, Number(row.overall_dist));
    values.set(`${row.model_b}|${row.model_a}`, Number(row.overall_dist));
  });
  const allVals = [...values.values()].filter(Number.isFinite);
  const max = Math.max(0.01, ...allVals);
  const grid = document.createElement("div");
  grid.className = "matrix-grid";
  grid.style.gridTemplateColumns = `120px repeat(${models.length}, minmax(86px, 1fr))`;
  grid.appendChild(cell("", "matrix-head"));
  models.forEach((model) => grid.appendChild(cell(modelLabel(model), "matrix-head")));
  models.forEach((rowModel) => {
    grid.appendChild(cell(modelLabel(rowModel), "matrix-head"));
    models.forEach((colModel) => {
      const value = rowModel === colModel ? null : values.get(`${rowModel}|${colModel}`);
      const div = cell(value == null ? "-" : value.toFixed(3), "matrix-cell");
      if (value != null) div.style.background = heat(value / max);
      grid.appendChild(div);
    });
  });
  $("distanceMatrix").replaceChildren(grid);
}

function cell(text, className) {
  const div = document.createElement("div");
  div.className = `matrix-cell ${className || ""}`;
  div.textContent = text;
  return div;
}

function topBehaviorChips(trace) {
  return Object.entries(trace.behavior_counts || {})
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([behavior, count]) => `<span class="behavior-chip">${titleCase(behavior)} ${fmt.format(count)}</span>`)
    .join("");
}

function firstUsefulText(trace) {
  const text = trace.thinking?.text || trace.answer?.text || trace.prompt?.text || "";
  return text.length > 240 ? `${text.slice(0, 240)}...` : text;
}

function familyFor(behavior) {
  return store.behaviors.find((b) => b.key === behavior)?.family || "cognitive";
}

function modelLabel(model) {
  return (model || "")
    .replace("qwen35_", "Qwen3.5 ")
    .replace("reasoner", "DeepSeek R1-Distill")
    .replace("anchor", "Llama 3.1 Anchor")
    .replace(/_/g, " ");
}

function titleCase(value) {
  return (LABELS[value] || value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function signedPct(value) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)}pp`;
}

function shortId(id) {
  return id ? `${id.slice(0, 8)}...${id.slice(-4)}` : "-";
}

function average(values) {
  const ok = values.filter((v) => v != null && Number.isFinite(v));
  if (!ok.length) return 0;
  return ok.reduce((sum, value) => sum + value, 0) / ok.length;
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

function svgEl(name, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => {
    if (value !== "") el.setAttribute(key, value);
  });
  return el;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}

loadData().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<main class="panel" style="margin:24px;padding:24px"><h1>Dashboard data failed to load</h1><p>${escapeHtml(error.message)}</p></main>`;
});
