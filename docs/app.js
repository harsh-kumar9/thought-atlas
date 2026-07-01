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

const MODEL_COLORS = {
  anchor: "#64748b",
  qwen35_4b: "#0891b2",
  qwen35_9b: "#7c3aed",
  qwen35_27b: "#d97706",
  reasoner: "#2563eb",
};

const FAMILY_META = {
  conversational: {
    label: "Conversational Behaviors",
    short: "Conversational",
    description: "Questioning, perspective moves, conflict, and reconciliation.",
  },
  cognitive: {
    label: "Cognitive Behavior Markers",
    short: "Cognitive",
    description: "Verification, backtracking, subgoals, and backward chaining.",
  },
};

const BEHAVIOR_DETAILS = {
  verification: "Checks or validates a claim, calculation, assumption, feasibility constraint, or intermediate result.",
  backtracking: "Revises course after detecting a weak path, mistake, contradiction, or unproductive line of reasoning.",
  subgoal: "Breaks the task into intermediate objectives, steps, milestones, or local targets before continuing.",
  backward_chaining: "Reasons backward from a desired answer, condition, proof target, or success criterion to needed premises.",
  Question_and_Answering: "Frames uncertainty as explicit questions, then answers or partially answers them inside the trace.",
  Perspective_Shift: "Switches viewpoint, representation, strategy, stakeholder frame, or interpretation of the task.",
  Conflict_of_Perspectives: "Surfaces tension between competing hypotheses, constraints, values, options, or interpretations.",
  Reconciliation: "Integrates competing considerations into a compromise, synthesis, final choice, or resolved direction.",
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
  viewMode: "full",
  traceLane: "a",
  traceIndex: 0,
};

const store = {};
let activeTooltipTarget = null;
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
  state.viewMode = "full";
  state.traceLane = "a";
  state.traceIndex = 0;
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
  syncButtonStates();
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
    text.className = "label behavior-label has-tooltip";
    text.textContent = titleCase(behavior.key);
    text.title = behaviorDescription(behavior.key);
    text.dataset.tooltip = behaviorDescription(behavior.key);
    text.tabIndex = 0;
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

  document.querySelectorAll("[data-recipe]").forEach((button) => {
    button.addEventListener("click", () => {
      applyRecipe(button.dataset.recipe);
      renderControls();
      renderAll();
    });
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

  document.querySelectorAll("[data-view-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      state.viewMode = button.dataset.viewMode;
      syncButtonStates();
      renderComparison();
      renderInspector();
      renderTrace();
    });
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
      state.traceLane = button.dataset.traceLane;
      state.traceIndex = 0;
      syncButtonStates();
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

  bindBehaviorTooltips();
}

function bindBehaviorTooltips() {
  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest?.(".has-tooltip[data-tooltip]");
    if (target) showBehaviorTooltip(target);
  });
  document.addEventListener("focusin", (event) => {
    const target = event.target.closest?.(".has-tooltip[data-tooltip]");
    if (target) showBehaviorTooltip(target);
  });
  document.addEventListener("mouseout", (event) => {
    const target = event.target.closest?.(".has-tooltip[data-tooltip]");
    if (target && !target.contains(event.relatedTarget)) hideBehaviorTooltip(target);
  });
  document.addEventListener("focusout", (event) => {
    const target = event.target.closest?.(".has-tooltip[data-tooltip]");
    if (target) hideBehaviorTooltip(target);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideBehaviorTooltip();
  });
  window.addEventListener("scroll", () => hideBehaviorTooltip(), true);
  window.addEventListener("resize", () => hideBehaviorTooltip());
}

function tooltipElement() {
  let tooltip = $("behaviorTooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "behaviorTooltip";
    tooltip.className = "floating-tooltip";
    tooltip.setAttribute("role", "tooltip");
    document.body.appendChild(tooltip);
  }
  return tooltip;
}

function showBehaviorTooltip(target) {
  const detail = target.dataset.tooltip;
  if (!detail) return;
  if (activeTooltipTarget && activeTooltipTarget !== target) activeTooltipTarget.removeAttribute("aria-describedby");
  activeTooltipTarget = target;
  const tooltip = tooltipElement();
  tooltip.textContent = detail;
  tooltip.classList.add("visible");
  target.setAttribute("aria-describedby", tooltip.id);
  positionBehaviorTooltip(target, tooltip);
}

function positionBehaviorTooltip(target, tooltip) {
  const margin = 10;
  const gap = 8;
  const targetRect = target.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  let left = targetRect.left;
  let top = targetRect.bottom + gap;
  if (left + tooltipRect.width > window.innerWidth - margin) {
    left = window.innerWidth - tooltipRect.width - margin;
  }
  if (top + tooltipRect.height > window.innerHeight - margin) {
    top = targetRect.top - tooltipRect.height - gap;
  }
  tooltip.style.left = `${Math.max(margin, left)}px`;
  tooltip.style.top = `${Math.max(margin, top)}px`;
}

function hideBehaviorTooltip(target = activeTooltipTarget) {
  const tooltip = $("behaviorTooltip");
  if (target) target.removeAttribute("aria-describedby");
  if (tooltip) tooltip.classList.remove("visible");
  if (!target || target === activeTooltipTarget) activeTooltipTarget = null;
}

function renderAll() {
  renderComparison();
  renderInspector();
  renderTrace();
  renderSummary();
  renderDistance();
}

function syncButtonStates() {
  document.querySelectorAll("[data-view-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.viewMode === state.viewMode);
  });
  document.querySelectorAll("[data-trace-lane]").forEach((button) => {
    button.classList.toggle("active", button.dataset.traceLane === state.traceLane);
  });
}

function applyRecipe(recipe) {
  const qwenSmall = store.models.find((m) => m.includes("4b")) || store.models[0];
  const qwenLarge = store.models.find((m) => m.includes("27b")) || store.models[store.models.length - 1];
  const reasoner = store.models.find((m) => m === "reasoner") || store.models[0];
  const base = store.models.find((m) => m.includes("27b")) || store.models.find((m) => m !== reasoner) || store.models[0];

  if (recipe === "outcome") {
    state.modelB = state.modelA;
    state.domainB = state.domainA;
    state.outcomeA = "positive";
    state.outcomeB = "negative";
  } else if (recipe === "scale") {
    state.modelA = qwenSmall;
    state.modelB = qwenLarge;
    state.domainB = state.domainA;
    state.outcomeA = "all";
    state.outcomeB = "all";
  } else if (recipe === "reasoner") {
    state.modelA = reasoner;
    state.modelB = base === reasoner ? store.models[0] : base;
    state.domainB = state.domainA;
    state.outcomeA = "all";
    state.outcomeB = "all";
  }
  state.traceIndex = 0;
}

function renderComparison() {
  const progress = state.bin / (store.manifest.bins - 1);
  const styleA = laneStyle("a");
  const styleB = laneStyle("b");
  document.documentElement.style.setProperty("--lane-a-color", styleA.line);
  document.documentElement.style.setProperty("--lane-b-color", styleB.line);
  $("progressPct").textContent = `${Math.round(progress * 100)}%`;
  $("comparisonSubtitle").textContent = `${state.behaviors.size} behavior${state.behaviors.size === 1 ? "" : "s"} at ${Math.round(progress * 100)}% through trace · ${viewModeLabel(state.viewMode)}`;
  $("laneReadout").innerHTML = `
    <div class="lane-chip a" style="border-left-color:${styleA.line}">
      <strong><i class="model-dot" style="background:${styleA.line}"></i>${modelLabel(state.modelA)}</strong>
      <span>${titleCase(state.domainA)}</span>
      <em class="outcome-chip ${outcomeTone(state.outcomeA)}">${OUTCOME_GROUPS[state.outcomeA].label}</em>
    </div>
    <div class="lane-chip b" style="border-left-color:${styleB.line}">
      <strong><i class="model-dot" style="background:${styleB.line}"></i>${modelLabel(state.modelB)}</strong>
      <span>${titleCase(state.domainB)}</span>
      <em class="outcome-chip ${outcomeTone(state.outcomeB)}">${OUTCOME_GROUPS[state.outcomeB].label}</em>
    </div>
  `;

  const behaviors = orderedBehaviors([...state.behaviors]);
  const grid = $("comparisonGrid");
  grid.innerHTML = "";
  grid.className = "comparison-grid";
  if (!behaviors.length) {
    grid.innerHTML = '<div class="empty-state">Select at least one behavior to draw trajectory comparisons.</div>';
    return;
  }

  const groups = behaviorGroups(behaviors);
  if (groups.length > 1) {
    grid.className = "comparison-grid grouped";
    groups.forEach((group) => {
      const section = document.createElement("section");
      section.className = `behavior-family-group ${group.family}`;
      section.innerHTML = `
        <div class="family-group-head">
          <div>
            <h3>${escapeHtml(group.meta.label)}</h3>
            <p>${escapeHtml(group.meta.description)}</p>
          </div>
          <span>${group.behaviors.length} selected</span>
        </div>
        <div class="family-chart-grid"></div>
      `;
      const familyGrid = section.querySelector(".family-chart-grid");
      group.behaviors.forEach((behavior) => familyGrid.appendChild(behaviorCard(behavior)));
      grid.appendChild(section);
    });
  } else {
    behaviors.forEach((behavior) => grid.appendChild(behaviorCard(behavior)));
  }
}

function behaviorCard(behavior) {
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
  const pointA = curveA.values[state.bin] || {};
  const pointB = curveB.values[state.bin] || {};
  const delta = (pointA.freq || 0) - (pointB.freq || 0);
  card.innerHTML = `
    <div class="mini-chart-head">
      <div>
        <h3 class="has-tooltip" title="${escapeAttr(behaviorDescription(behavior))}" data-tooltip="${escapeAttr(behaviorDescription(behavior))}">${titleCase(behavior)}</h3>
        <small>${familyShortLabel(behavior)} · ${sampleSizeLabel(pointA.n, pointB.n)}</small>
      </div>
      <span class="delta-pill ${delta >= 0 ? "delta-up" : "delta-down"}">${signedPct(delta)}</span>
    </div>
    <svg role="img" aria-label="${titleCase(behavior)} trajectory comparison"></svg>
  `;
  drawMiniChart(card.querySelector("svg"), curveA, curveB);
  return card;
}

function renderInspector() {
  const behavior = state.selectedBehavior || [...state.behaviors][0];
  if (!behavior) {
    $("inspectorTitle").textContent = "Inspector";
    $("inspectorSubtitle").textContent = "Select a behavior to inspect.";
    $("deltaFacts").innerHTML = "";
    $("divergencePanel").innerHTML = "";
    $("hypothesisList").innerHTML = "";
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
    .map(([label, value]) => `<div class="delta-fact"><span>${label}</span><strong class="${value.startsWith("-") ? "delta-down" : "delta-up"}">${value}</strong></div>`)
    .join("");

  renderDivergencePanel();
  renderSampleInspector(behavior);
}

function renderDivergencePanel() {
  const behaviors = orderedBehaviors([...state.behaviors]);
  if (behaviors.length < 2) {
    $("divergencePanel").innerHTML = "";
    $("hypothesisList").innerHTML = "";
    return;
  }

  const divergences = behaviors
    .map((behavior) => {
      const curveA = aggregateCurve("a", behavior);
      const curveB = aggregateCurve("b", behavior);
      const atA = curveA.values[state.bin]?.freq || 0;
      const atB = curveB.values[state.bin]?.freq || 0;
      const earlyDelta = average(curveA.values.slice(0, Math.max(1, state.bin + 1)).map((v, idx) => (v.freq || 0) - (curveB.values[idx]?.freq || 0)));
      const lateDelta = average(curveA.values.slice(state.bin).map((v, idx) => (v.freq || 0) - (curveB.values[state.bin + idx]?.freq || 0)));
      return {
        behavior,
        family: familyFor(behavior),
        atA,
        atB,
        delta: atA - atB,
        earlyDelta,
        lateDelta,
      };
    })
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  const top = divergences.slice(0, 3);
  $("divergencePanel").innerHTML = `
    <div class="inspector-section-title">
      <span>Divergence at Cursor</span>
      <em>${Math.round((state.bin / (store.manifest.bins - 1)) * 100)}%</em>
    </div>
    <div class="divergence-list">
      ${top
        .map(
          (row) => `
            <button class="divergence-card ${row.behavior === state.selectedBehavior ? "selected" : ""}" data-behavior="${escapeAttr(row.behavior)}">
              <strong class="has-tooltip" title="${escapeAttr(behaviorDescription(row.behavior))}" data-tooltip="${escapeAttr(behaviorDescription(row.behavior))}">${titleCase(row.behavior)}</strong>
              <span>${familyShortLabel(row.behavior)} · ${signedPct(row.delta)}</span>
            </button>
          `,
        )
        .join("")}
    </div>
  `;
  $("divergencePanel").querySelectorAll("[data-behavior]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedBehavior = button.dataset.behavior;
      renderComparison();
      renderInspector();
    });
  });

  renderHypothesisList(divergences);
}

function renderHypothesisList(divergences) {
  const top = divergences[0];
  if (!top) {
    $("hypothesisList").innerHTML = "";
    return;
  }

  const familyRows = ["conversational", "cognitive"]
    .map((family) => {
      const rows = divergences.filter((row) => row.family === family);
      return rows.length ? { family, delta: average(rows.map((row) => row.delta)) } : null;
    })
    .filter(Boolean)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  const prompts = [];
  prompts.push(
    `${higherLane(top.delta)} is higher on ${titleCase(top.behavior)} by ${signedPct(Math.abs(top.delta)).replace("+", "")} at the cursor; use Reveal mode to see whether that gap appears before or after the answer boundary.`,
  );

  if (familyRows.length) {
    const family = familyRows[0];
    prompts.push(
      `${FAMILY_META[family.family]?.short || titleCase(family.family)} markers lean toward ${higherLane(family.delta)} by ${signedPct(Math.abs(family.delta)).replace("+", "")} on average at this point.`,
    );
  }

  if (state.modelA !== state.modelB && state.domainA === state.domainB) {
    prompts.push(`Model hypothesis: keep ${titleCase(state.domainA)} fixed and switch outcomes to test whether the model gap survives performance stratification.`);
  } else if (state.outcomeA !== state.outcomeB && state.modelA === state.modelB && state.domainA === state.domainB) {
    prompts.push(`Outcome hypothesis: this isolates performance for ${modelLabel(state.modelA)} on ${titleCase(state.domainA)}; check if the divergence grows close to the answer phase.`);
  } else {
    prompts.push("Domain hypothesis: pin one model and one outcome group, then sweep domains to see whether this shape is task-specific.");
  }

  $("hypothesisList").innerHTML = `
    <div class="inspector-section-title"><span>Hypothesis Prompts</span></div>
    ${prompts.map((prompt) => `<article>${escapeHtml(prompt)}</article>`).join("")}
  `;
}

function renderSampleInspector(behavior) {
  const tracesA = rankedTracesForInspector("a", behavior, 3);
  const tracesB = rankedTracesForInspector("b", behavior, 3);
  const annotated = [...tracesA, ...tracesB].some((trace) => Array.isArray(trace.annotations) && trace.annotations.length);
  $("annotationNote").textContent = annotated
    ? `Evidence prioritizes sentence annotations near the ${Math.round((state.bin / (store.manifest.bins - 1)) * 100)}% cursor, then falls back to the closest matching behavior spans.`
    : "Current static samples expose raw prompt/thinking/answer text plus per-trace behavior counts. Re-run the dashboard export after the annotation upgrade to show sentence-level behavior spans.";

  const cards = [
    ...tracesA.map((trace) => sampleCard(trace, "Lane A", behavior)),
    ...tracesB.map((trace) => sampleCard(trace, "Lane B", behavior)),
  ];
  $("sampleList").innerHTML = cards.join("") || '<div class="empty-state">No sampled traces match the selected comparison lanes.</div>';
}

function sampleCard(trace, lane, behavior) {
  const count = trace.behavior_counts?.[behavior] || 0;
  const matchingAnnotations = rankedAnnotations(trace, behavior, 3);
  const snippet = matchingAnnotations.length
    ? `<div class="annotation-snippets">${matchingAnnotations
        .map(
          (a) => `
            <div class="annotation-snippet">
              <span>${Math.round(a.norm_pos * 100)}% · ${titleCase(a.section_type)}</span>
              <p>${escapeHtml(a.text)}</p>
            </div>
          `,
        )
        .join("")}</div>`
    : escapeHtml(firstUsefulText(trace));
  return `
    <article class="sample-card">
      <h3>${lane} · ${modelLabel(trace.gen_model)} · ${titleCase(trace.task_type)}</h3>
      <p>${shortId(trace.trace_id)} · ${titleCase(trace.outcome)} · ${fmt.format(count)} <span class="has-tooltip inline-tooltip" title="${escapeAttr(behaviorDescription(behavior))}" data-tooltip="${escapeAttr(behaviorDescription(behavior))}">${titleCase(behavior)}</span> mark${count === 1 ? "" : "s"}</p>
      ${matchingAnnotations.length ? snippet : `<p>${snippet || "No text excerpt available for this sample."}</p>`}
      <div class="behavior-chips">${topBehaviorChips(trace)}</div>
    </article>
  `;
}

function rankedTracesForInspector(laneKey, behavior, limit) {
  return filteredTracesForLane(laneKey)
    .map((trace, index) => ({
      trace,
      index,
      score: traceEvidenceScore(trace, behavior),
    }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, limit)
    .map((row) => row.trace);
}

function traceEvidenceScore(trace, behavior) {
  const annotations = rankedAnnotations(trace, behavior, 3);
  const count = trace.behavior_counts?.[behavior] || 0;
  if (!annotations.length) return count * 0.1;
  const best = annotations[0];
  const nearBoost = Math.max(0, 1 - Math.abs((best.norm_pos || 0) - cursorProgress()) / cursorWindow());
  const behaviorBoost = (best.behaviors || []).includes(behavior) ? 2 : 0;
  return 1 + behaviorBoost + nearBoost + Math.min(1, count / 8);
}

function rankedAnnotations(trace, behavior, limit = 5, selectedOnly = true) {
  const annotations = Array.isArray(trace?.annotations) ? trace.annotations : [];
  const cursor = cursorProgress();
  const window = cursorWindow();
  const selected = state.behaviors.size ? state.behaviors : new Set(store.behaviors.map((b) => b.key));
  const rows = annotations
    .filter((annotation) => {
      const behaviors = annotation.behaviors || [];
      if (!selectedOnly) return true;
      return behaviors.includes(behavior) || behaviors.some((b) => selected.has(b));
    })
    .map((annotation) => {
      const dist = Math.abs((annotation.norm_pos || 0) - cursor);
      const behaviors = annotation.behaviors || [];
      const exact = behaviors.includes(behavior) ? 0 : 1;
      const near = dist <= window ? 0 : 1;
      return { ...annotation, _score: near * 4 + exact * 2 + dist };
    })
    .sort((a, b) => a._score - b._score)
    .slice(0, limit);

  return rows.length ? rows : annotations
    .map((annotation) => ({ ...annotation, _score: Math.abs((annotation.norm_pos || 0) - cursor) }))
    .sort((a, b) => a._score - b._score)
    .slice(0, limit);
}

function drawMiniChart(svg, curveA, curveB) {
  const styleA = laneStyle("a");
  const styleB = laneStyle("b");
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
  drawWindowHighlight(svg, x, margin, plotH);
  const visibleA = visibleCurveValues(curveA.values);
  const visibleB = visibleCurveValues(curveB.values);
  drawBand(svg, visibleA, x, y, styleA.band);
  drawBand(svg, visibleB, x, y, styleB.band);
  drawLine(svg, visibleA, x, y, styleA.line, false);
  drawLine(svg, visibleB, x, y, styleB.line, true);

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
      fill: idx === 0 ? styleA.line : styleB.line,
      stroke: "#fff",
      "stroke-width": "1.4",
    }));
  });
}

function visibleCurveValues(values) {
  if (state.viewMode === "reveal") return values.filter((value) => value.bin <= state.bin);
  if (state.viewMode === "window") return values.filter((value) => Math.abs(value.bin - state.bin) <= cursorBinWindow());
  return values;
}

function drawWindowHighlight(svg, x, margin, plotH) {
  if (state.viewMode !== "window") return;
  const left = Math.max(0, state.bin - cursorBinWindow());
  const right = Math.min(store.manifest.bins - 1, state.bin + cursorBinWindow());
  svg.appendChild(svgEl("rect", {
    x: x(left),
    y: margin.top,
    width: Math.max(1, x(right) - x(left)),
    height: plotH,
    fill: "rgba(20, 155, 143, 0.06)",
    stroke: "none",
  }));
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

function laneStyle(laneKey) {
  const lane = laneConfig(laneKey);
  const line = colorForModel(lane.model);
  return { line, band: alphaColor(line, 0.13) };
}

function colorForModel(model) {
  return MODEL_COLORS[model] || "#334155";
}

function alphaColor(hex, alpha) {
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function outcomeTone(outcomeKey) {
  if (["positive", "solved", "high_quality"].includes(outcomeKey)) return "good";
  if (["negative", "failed", "low_quality"].includes(outcomeKey)) return "bad";
  if (outcomeKey === "unknown") return "unknown";
  return "mixed";
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
    $("traceAnnotationRail").innerHTML = "";
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
  renderTraceAnnotationRail(trace);
  $("thinkingLabel").textContent = `Thinking (${fmt.format(trace.thinking.tokens_est || 0)} est. tokens${trace.thinking.truncated ? ", clipped" : ""})`;
  $("answerLabel").textContent = `Answer (${fmt.format(trace.answer.tokens_est || 0)} est. tokens${trace.answer.truncated ? ", clipped" : ""})`;
  $("promptText").textContent = trace.prompt.text || "-";
  $("thinkingText").textContent = trace.thinking.text || "No explicit thinking block for this trace.";
  $("answerText").textContent = trace.answer.text || "-";
}

function renderTraceAnnotationRail(trace) {
  const behavior = state.selectedBehavior || [...state.behaviors][0];
  const annotations = rankedAnnotations(trace, behavior, 5, false);
  if (!annotations.length) {
    $("traceAnnotationRail").innerHTML = "";
    return;
  }

  $("traceAnnotationRail").innerHTML = `
    <div class="trace-annotation-head">
      <strong>Cursor Evidence</strong>
      <span>${Math.round(cursorProgress() * 100)}% through trace · closest annotated spans</span>
    </div>
    <div class="trace-annotation-list">
      ${annotations
        .map(
          (annotation) => `
            <article class="trace-annotation ${Math.abs((annotation.norm_pos || 0) - cursorProgress()) <= cursorWindow() ? "near" : ""}">
              <div>
                <strong>${Math.round((annotation.norm_pos || 0) * 100)}%</strong>
                <span>${titleCase(annotation.section_type || "trace")}</span>
              </div>
              <p>${escapeHtml(annotation.text || "")}</p>
              <div class="behavior-chips">${(annotation.behaviors || [])
                .map((b) => behaviorChip(b))
                .join("")}</div>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
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
    .map(([behavior, count]) => behaviorChip(behavior, fmt.format(count)))
    .join("");
}

function behaviorChip(behavior, suffix = "") {
  const detail = behaviorDescription(behavior);
  return `<span class="behavior-chip has-tooltip" title="${escapeAttr(detail)}" data-tooltip="${escapeAttr(detail)}">${titleCase(behavior)}${suffix ? ` ${suffix}` : ""}</span>`;
}

function firstUsefulText(trace) {
  const text = trace.thinking?.text || trace.answer?.text || trace.prompt?.text || "";
  return text.length > 240 ? `${text.slice(0, 240)}...` : text;
}

function orderedBehaviors(behaviors) {
  const order = new Map(store.behaviors.map((behavior, index) => [behavior.key, index]));
  const familyOrder = new Map([
    ["conversational", 0],
    ["cognitive", 1],
  ]);
  return behaviors.slice().sort((a, b) => {
    const familyA = familyFor(a);
    const familyB = familyFor(b);
    return (familyOrder.get(familyA) ?? 9) - (familyOrder.get(familyB) ?? 9) || (order.get(a) ?? 999) - (order.get(b) ?? 999);
  });
}

function behaviorGroups(behaviors) {
  const groups = new Map();
  orderedBehaviors(behaviors).forEach((behavior) => {
    const family = familyFor(behavior);
    if (!groups.has(family)) groups.set(family, []);
    groups.get(family).push(behavior);
  });
  return ["conversational", "cognitive", ...groups.keys()]
    .filter((family, index, arr) => arr.indexOf(family) === index && groups.has(family))
    .map((family) => ({
      family,
      meta: FAMILY_META[family] || { label: titleCase(family), short: titleCase(family), description: "Behavior markers in this framework." },
      behaviors: groups.get(family),
    }));
}

function familyFor(behavior) {
  return store.behaviors.find((b) => b.key === behavior)?.family || "cognitive";
}

function familyShortLabel(behavior) {
  const family = familyFor(behavior);
  return FAMILY_META[family]?.short || titleCase(family);
}

function behaviorDescription(behavior) {
  return BEHAVIOR_DETAILS[behavior] || "Behavior marker detected in the trace annotation pipeline.";
}

function sampleSizeLabel(nA = 0, nB = 0) {
  return `A/B n ${fmt.format(Math.round(nA || 0))}/${fmt.format(Math.round(nB || 0))}`;
}

function viewModeLabel(mode) {
  if (mode === "reveal") return "reveal to cursor";
  if (mode === "window") return "local window";
  return "full curve";
}

function cursorProgress() {
  return state.bin / Math.max(1, store.manifest.bins - 1);
}

function cursorBinWindow() {
  return Math.max(2, Math.round((store.manifest.bins - 1) * 0.08));
}

function cursorWindow() {
  return cursorBinWindow() / Math.max(1, store.manifest.bins - 1);
}

function higherLane(delta) {
  return delta >= 0 ? "Lane A" : "Lane B";
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
