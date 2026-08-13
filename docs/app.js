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
  safe_response: "Safe response",
  harmful_compliance: "High harmful compliance",
  high_quality: "High quality",
  low_quality: "Low quality",
  unknown: "Unknown",
};

const OUTCOME_GROUPS = {
  all: { label: "All traces", outcomes: ["solved", "failed", "safe_response", "harmful_compliance", "high_quality", "low_quality", "unknown"] },
  positive: { label: "Solved / safe / high-quality", outcomes: ["solved", "safe_response", "high_quality"] },
  negative: { label: "Failed / harmful / low-quality", outcomes: ["failed", "harmful_compliance", "low_quality"] },
  solved: { label: "Solved only", outcomes: ["solved"] },
  failed: { label: "Failed only", outcomes: ["failed"] },
  safe_response: { label: "Safe response only", outcomes: ["safe_response"] },
  harmful_compliance: { label: "Harmful compliance only", outcomes: ["harmful_compliance"] },
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

const MODEL_DISPLAY_NAMES = {
  anchor: "Llama-3.1-8B-Instruct",
  reasoner: "DeepSeek-R1-Distill-Llama-8B",
  qwen35_4b: "Qwen3.5-4B",
  qwen35_9b: "Qwen3.5-9B",
  qwen35_27b: "Qwen3.5-27B",
};

const TIMING_CLASSES = {
  both: { label: "Amount + timing", tone: "both" },
  level_only: { label: "Amount gap", tone: "level" },
  timing_only: { label: "Timing shape", tone: "timing" },
  neither: { label: "No clear split", tone: "neither" },
  insufficient: { label: "Too sparse", tone: "insufficient" },
};

const MONITOR_FEATURES = {
  metadata: { label: "Task only", short: "Task", color: "#64748b", dash: "", description: "model, domain, and task metadata before reading the trace" },
  length: { label: "+ length", short: "Length", color: "#0891b2", dash: "4 4", description: "task metadata plus how much trace text is visible" },
  counts: { label: "+ behavior mix", short: "Behavior mix", color: "#2563eb", dash: "", description: "which behaviors have appeared so far, ignoring exact timing" },
  temporal: { label: "+ behavior timing", short: "Behavior timing", color: "#d97706", dash: "", description: "which behaviors appeared and where they appeared inside the visible prefix" },
  time_shuffled: { label: "Timing sanity check", short: "Timing check", color: "#7c3aed", dash: "6 4", description: "a timing-control baseline for analysis, hidden from the simple view" },
};

const MONITOR_VISIBLE_FEATURES = ["metadata", "length", "counts", "temporal"];

const LANE_DASHES = ["", "5 4", "2 3", "7 3 2 3", "1 4", "10 4 2 4"];

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

const BEHAVIOR_NUMBERS = {
  verification: 1,
  backtracking: 2,
  subgoal: 3,
  backward_chaining: 4,
  Question_and_Answering: 5,
  Perspective_Shift: 6,
  Conflict_of_Perspectives: 7,
  Reconciliation: 8,
};

const BEHAVIOR_COLORS = {
  verification: "rgb(255, 140, 140)",
  backtracking: "rgb(255, 209, 122)",
  subgoal: "rgb(255, 255, 132)",
  backward_chaining: "rgb(139, 255, 139)",
  Question_and_Answering: "rgb(137, 218, 255)",
  Perspective_Shift: "rgb(139, 139, 255)",
  Conflict_of_Perspectives: "rgb(206, 154, 255)",
  Reconciliation: "rgb(255, 164, 209)",
};

const state = {
  lanes: [],
  nextLaneIndex: 0,
  behaviors: new Set(),
  selectedBehavior: null,
  bin: 10,
  viewMode: "full",
  traceLane: null,
  traceIndex: 0,
  timing: {
    models: new Set(),
    domains: new Set(),
    families: new Set(["cognitive", "conversational"]),
    classes: new Set(Object.keys(TIMING_CLASSES)),
    selectedKey: null,
  },
  monitor: {
    split: "prompt_disjoint",
  },
  truncateAnnotations: true,
  renderLatex: true,
  annotationGridIndex: {}, // { [laneId]: index into sorted prompt list }
  showFullTrace: true, // false = annotated sentences only, true = full text with annotations inline
};

const store = {};
let activeTooltipTarget = null;
const $ = (id) => document.getElementById(id);

async function loadData() {
  const [manifest, summary, heartbeat, traces, distance, trackA, trace_sentences] = await Promise.all([
    fetch("data/manifest.json").then((r) => r.json()),
    fetch("data/summary.json").then((r) => r.json()),
    fetch("data/heartbeat.json").then((r) => r.json()),
    fetch("data/trace_samples.json").then((r) => r.json()),
    fetch("data/distance.json").then((r) => r.json()),
    fetch("data/trackA.json")
      .then((r) => (r.ok ? r.json() : { cells: [], families: [] }))
      .catch(() => ({ cells: [], families: [] })),
    fetch("data/trace_samples_by_prompt.json").then((r) => r.json()),
  ]);


  Object.assign(store, {
    manifest,
    summary,
    heartbeat,
    traces: traces.traces,
    distance,
    trackA,
    traceSentences: trace_sentences.traces || [],
    timingLevel: { meta: {}, pairs: [] },
    prefixMonitor: { meta: {}, metrics: [], deltas: [] },
    timingIndex: new Map(),
    trackAIndex: new Map((trackA.cells || []).map((row) => [trackAKey(row.gen_model, row.task_type, row.outcome_group, row.behavior), row])),
    trackAFamilyIndex: new Map((trackA.families || []).map((row) => [trackAKey(row.gen_model, row.task_type, row.outcome_group, row.family), row])),
    behaviors: manifest.behaviors,
    domains: manifest.domains.map((d) => d.task_type),
    models: manifest.models.map((m) => m.gen_model),
    modelLabels: new Map(
      manifest.models.map((m) => [
        m.gen_model,
        m.display_name || MODEL_DISPLAY_NAMES[m.gen_model] || shortModelName(m.gen_model_id) || titleCase(m.gen_model),
      ]),
    ),
  });

  ensureDashboardMarkup();
  initializeState();
  renderControls();
  bindEvents();
  renderAll();
  requestAnimationFrame(() => requestAnimationFrame(scrollToHashTarget));
}

function scrollToHashTarget() {
  if (!window.location.hash) return;
  const id = decodeURIComponent(window.location.hash.slice(1));
  scrollToPanel(id, "auto");
}

function scrollToPanel(id, behavior = "smooth") {
  const target = id ? $(id) : null;
  if (!target) return;
  const header = document.querySelector(".topbar");
  const offset = (header?.getBoundingClientRect().height || 62) + 12;
  const top = target.getBoundingClientRect().top + window.scrollY - offset;
  window.scrollTo({ top: Math.max(0, top), behavior });
}

function ensureDashboardMarkup() {
  const controls = $("controlsPanel");
  const headingCopy = controls?.querySelector(".control-heading p");
  if (headingCopy) headingCopy.textContent = "Add one or more model/domain lanes, then choose behaviors.";
  if (controls && !$("laneControls")) {
    controls.querySelectorAll(".lane-a, .lane-b").forEach((node) => node.remove());
    const laneMarkup = `
      <div id="laneControls" class="lane-stack"></div>
      <section class="control-group lane-toolbar">
        <button class="secondary-button" id="addLane">Add lane</button>
      </section>
    `;
    const heading = controls.querySelector(".control-heading");
    if (heading) heading.insertAdjacentHTML("afterend", laneMarkup);
    else controls.insertAdjacentHTML("afterbegin", laneMarkup);
  } else if ($("laneControls") && !$("addLane")) {
    $("laneControls").insertAdjacentHTML(
      "afterend",
      '<section class="control-group lane-toolbar"><button class="secondary-button" id="addLane">Add lane</button></section>',
    );
  }

  if (controls && !$("traceLaneButtons")) {
    const rawGroup = [...controls.querySelectorAll(".control-group")].find((section) => section.textContent.includes("Raw Trace Lane"));
    const oldButtons = rawGroup?.querySelector('[aria-label="Raw trace lane"]');
    if (oldButtons) {
      oldButtons.outerHTML = '<div id="traceLaneButtons" class="trace-lane-buttons" role="group" aria-label="Raw trace lane"></div>';
    } else if (rawGroup) {
      rawGroup.insertAdjacentHTML("beforeend", '<div id="traceLaneButtons" class="trace-lane-buttons" role="group" aria-label="Raw trace lane"></div>');
    }
  }

  const legend = document.querySelector(".compare-panel .legend");
  if (legend && !$("laneLegend")) {
    legend.id = "laneLegend";
    legend.innerHTML = `
      <span><i class="perf-good"></i>Solved / safe / high-quality</span>
      <span><i class="perf-bad"></i>Failed / harmful / low-quality</span>
      <span><i class="scrub-line"></i>progress</span>
    `;
  }
}

function initializeState() {
  const domain = store.domains.includes("math") ? "math" : store.domains[0];
  const primaryModel = store.models.includes("reasoner") ? "reasoner" : store.models[0];
  const secondaryModel = store.models.includes("qwen35_27b") ? "qwen35_27b" : store.models[Math.min(1, store.models.length - 1)];
  setLanes([
    { model: primaryModel, domain, outcome: "all" },
    { model: secondaryModel, domain, outcome: "all" },
  ]);
  const conversational = store.behaviors.filter((b) => b.family === "conversational").map((b) => b.key);
  state.behaviors = new Set(conversational.length ? conversational : store.behaviors.map((b) => b.key));
  state.selectedBehavior = [...state.behaviors][0];
  state.timing.models = new Set(store.models);
  state.timing.domains = new Set(store.domains);
  state.timing.families = new Set(["cognitive", "conversational"]);
  state.timing.classes = new Set(Object.keys(TIMING_CLASSES));
  state.timing.selectedKey = preferredTimingPair()?.key || null;
  state.monitor.split = (store.prefixMonitor?.meta?.splits || []).includes("prompt_disjoint")
    ? "prompt_disjoint"
    : (store.prefixMonitor?.meta?.splits || [])[0] || "prompt_disjoint";
  state.bin = Math.round((store.manifest.bins - 1) * 0.43);
  state.viewMode = "full";
  state.traceIndex = 0;
  $("progressSlider").max = store.manifest.bins - 1;
  $("progressSlider").value = state.bin;
}

function renderControls() {
  renderLaneControls();
  renderTraceLaneControls();
  renderBehaviorFilters();
  syncButtonStates();
}

function setLanes(configs) {
  state.nextLaneIndex = 0;
  state.lanes = configs.map((config) => createLane(config));
  state.traceLane = state.lanes[0]?.id || null;
  state.traceIndex = 0;
}

function createLane(config = {}) {
  const fallbackModel = store.models[state.nextLaneIndex % Math.max(1, store.models.length)] || store.models[0];
  const id = config.id || `lane-${state.nextLaneIndex}`;
  state.nextLaneIndex += 1;
  return {
    id,
    model: config.model || fallbackModel,
    domain: config.domain || store.domains[0],
    outcome: config.outcome || "all",
  };
}

function addLane() {
  const source = state.lanes[state.lanes.length - 1] || {};
  const model = store.models[state.lanes.length % Math.max(1, store.models.length)] || source.model || store.models[0];
  state.lanes.push(createLane({ model, domain: source.domain, outcome: source.outcome }));
  state.traceLane = state.traceLane || state.lanes[0].id;
}

function removeLane(laneId) {
  if (state.lanes.length <= 1) return;
  state.lanes = state.lanes.filter((lane) => lane.id !== laneId);
  if (!state.lanes.some((lane) => lane.id === state.traceLane)) state.traceLane = state.lanes[0]?.id || null;
  state.traceIndex = 0;
}

function duplicateLane(laneId) {
  const lane = laneConfig(laneId);
  if (!lane) return;
  state.lanes.push(createLane({ model: lane.model, domain: lane.domain, outcome: lane.outcome }));
}

function renderLaneControls() {
  const target = $("laneControls");
  target.innerHTML = state.lanes
    .map((lane, index) => {
      const style = laneStyle(lane.id);
      return `
        <section class="control-group lane dynamic-lane" data-lane-id="${escapeAttr(lane.id)}" style="--lane-control-color:${style.line}">
          <div class="lane-title">
            <i style="background:${style.line}"></i>
            <span>${laneLabel(lane.id)}</span>
            <div class="lane-title-actions">
              <button class="mini-action" data-duplicate-lane="${escapeAttr(lane.id)}" aria-label="Duplicate ${laneLabel(lane.id)}">Copy</button>
              <button class="mini-action" data-remove-lane="${escapeAttr(lane.id)}" ${state.lanes.length === 1 ? "disabled" : ""} aria-label="Remove ${laneLabel(lane.id)}">Remove</button>
            </div>
          </div>
          <label>Model <select data-lane-id="${escapeAttr(lane.id)}" data-lane-field="model">${selectOptions(store.models, modelLabel, lane.model)}</select></label>
          <label>Domain <select data-lane-id="${escapeAttr(lane.id)}" data-lane-field="domain">${selectOptions(store.domains, titleCase, lane.domain)}</select></label>
          <label>Outcome <select data-lane-id="${escapeAttr(lane.id)}" data-lane-field="outcome">${selectOptions(Object.keys(OUTCOME_GROUPS), (k) => OUTCOME_GROUPS[k].label, lane.outcome)}</select></label>
        </section>
      `;
    })
    .join("");
  $("addLane").textContent = "Add lane";
}

function renderTraceLaneControls() {
  const target = $("traceLaneButtons");
  target.innerHTML = state.lanes
    .map((lane) => {
      const style = laneStyle(lane.id);
      return `
        <button data-trace-lane="${escapeAttr(lane.id)}" class="${lane.id === state.traceLane ? "active" : ""}" style="--lane-control-color:${style.line}">
          <i style="background:${style.line}"></i>${laneLabel(lane.id)}
        </button>
      `;
    })
    .join("");
}

function selectOptions(values, labeler, selected) {
  return values.map((value) => `<option value="${escapeAttr(value)}" ${value === selected ? "selected" : ""}>${labeler(value)}</option>`).join("");
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
      renderTrackA();
      renderInspector();
    });
    const text = document.createElement("span");
    text.className = "label behavior-label has-tooltip";
    text.textContent = titleCase(behavior.key);
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
  $("laneControls").addEventListener("change", (event) => {
    const select = event.target.closest("select[data-lane-id][data-lane-field]");
    if (!select) return;
    const lane = laneConfig(select.dataset.laneId);
    if (!lane) return;
    lane[select.dataset.laneField] = select.value;
    state.traceIndex = 0;
    renderControls();
    renderAll();
  });

  $("laneControls").addEventListener("click", (event) => {
    const remove = event.target.closest("[data-remove-lane]");
    const duplicate = event.target.closest("[data-duplicate-lane]");
    if (remove) removeLane(remove.dataset.removeLane);
    if (duplicate) duplicateLane(duplicate.dataset.duplicateLane);
    if (!remove && !duplicate) return;
    renderControls();
    renderAll();
  });

  $("addLane").addEventListener("click", () => {
    addLane();
    renderControls();
    renderAll();
  });

  $("resetControls").addEventListener("click", () => {
    initializeState();
    renderControls();
    renderAll();
  });

  if ($("truncateAnnotationsToggle")) {
    $("truncateAnnotationsToggle").addEventListener("click", () => {
      state.truncateAnnotations = !state.truncateAnnotations;
      renderTraceAnnotationGrid();
    });
  }

  if ($("renderLatexToggle")) {
    $("renderLatexToggle").addEventListener("click", () => {
      state.renderLatex = !state.renderLatex;
      renderTraceAnnotationGrid();
    });
  }

  $("trace-annotation-grid").addEventListener("click", (event) => {
    const prev = event.target.closest("[data-annotation-prev]");
    const next = event.target.closest("[data-annotation-next]");
    if (prev) stepAnnotationPrompt(prev.dataset.annotationPrev, -1);
    if (next) stepAnnotationPrompt(next.dataset.annotationNext, 1);
  });

  if ($("showFullTraceToggle")) {
    $("showFullTraceToggle").addEventListener("click", () => {
      state.showFullTrace = !state.showFullTrace;
      renderTraceAnnotationGrid();
    });
  }

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
      renderTrackA();
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

  $("traceLaneButtons").addEventListener("click", (event) => {
    const button = event.target.closest("[data-trace-lane]");
    if (!button) return;
    state.traceLane = button.dataset.traceLane;
    state.traceIndex = 0;
    syncButtonStates();
    renderTrace();
    renderInspector();
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
  if ($("timingFilters")) {
    $("timingFilters").addEventListener("click", (event) => {
      const button = event.target.closest("[data-timing-filter]");
      if (!button) return;
      toggleTimingFilter(button.dataset.timingFilter, button.dataset.timingValue);
      renderTimingLevel();
    });
  }
  if ($("timingScatter")) {
    $("timingScatter").addEventListener("click", (event) => {
      const target = event.target.closest("[data-timing-key]");
      if (!target) return;
      selectTimingPair(target.dataset.timingKey);
    });
    $("timingScatter").addEventListener("keydown", (event) => {
      if (!["Enter", " "].includes(event.key)) return;
      const target = event.target.closest("[data-timing-key]");
      if (!target) return;
      event.preventDefault();
      selectTimingPair(target.dataset.timingKey);
    });
  }
  if ($("timingTable")) {
    $("timingTable").addEventListener("click", (event) => {
      const row = event.target.closest("[data-timing-key]");
      if (!row) return;
      selectTimingPair(row.dataset.timingKey);
    });
  }
  if ($("monitorSplitButtons")) {
    $("monitorSplitButtons").addEventListener("click", (event) => {
      const button = event.target.closest("[data-monitor-split]");
      if (!button) return;
      state.monitor.split = button.dataset.monitorSplit;
      renderMonitorability();
    });
  }

  document.querySelectorAll("[data-scroll]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
      button.classList.add("active");
      scrollToPanel(button.dataset.scroll);
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
    const target = tooltipTargetForEvent(event);
    if (target) showBehaviorTooltip(target);
  });
  document.addEventListener("mouseout", (event) => {
    const target = event.target.closest?.(".has-tooltip[data-tooltip]");
    if (target && !target.contains(event.relatedTarget)) hideBehaviorTooltip(target);
  });
  document.addEventListener("focusout", (event) => {
    const target = tooltipTargetForEvent(event);
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

function tooltipTargetForEvent(event) {
  const direct = event.target.closest?.(".has-tooltip[data-tooltip]");
  if (direct) return direct;
  if (event.type === "focusin") {
    return event.target.querySelector?.(".has-tooltip[data-tooltip]") || null;
  }
  return null;
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
  renderTrackA();
  renderInspector();
  renderTrace();
  renderSummary();
  renderDistance();
  renderTraceAnnotationGrid();
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
  const qwenMid = store.models.find((m) => m.includes("9b"));
  const qwenLarge = store.models.find((m) => m.includes("27b")) || store.models[store.models.length - 1];
  const reasoner = store.models.find((m) => m === "reasoner") || store.models[0];
  const anchor = store.models.find((m) => m === "anchor");
  const base = store.models.find((m) => m.includes("27b")) || store.models.find((m) => m !== reasoner) || store.models[0];
  const anchorLane = state.lanes[0] || { model: reasoner, domain: store.domains[0], outcome: "all" };

  if (recipe === "outcome") {
    setLanes([
      { model: anchorLane.model, domain: anchorLane.domain, outcome: "positive" },
      { model: anchorLane.model, domain: anchorLane.domain, outcome: "negative" },
    ]);
  } else if (recipe === "scale") {
    setLanes([qwenSmall, qwenMid, qwenLarge].filter(Boolean).map((model) => ({ model, domain: anchorLane.domain, outcome: "all" })));
  } else if (recipe === "reasoner") {
    setLanes([reasoner, base === reasoner ? store.models[0] : base, anchor].filter(Boolean).map((model) => ({ model, domain: anchorLane.domain, outcome: "all" })));
  }
  state.traceIndex = 0;
}

function renderComparison() {
  const progress = state.bin / (store.manifest.bins - 1);
  $("progressPct").textContent = `${Math.round(progress * 100)}%`;
  $("comparisonSubtitle").textContent = `${state.lanes.length} lane${state.lanes.length === 1 ? "" : "s"} · ${state.behaviors.size} behavior${state.behaviors.size === 1 ? "" : "s"} at ${Math.round(progress * 100)}% through trace · ${viewModeLabel(state.viewMode)}`;
  renderLaneLegend();
  $("laneReadout").innerHTML = state.lanes.map((lane) => laneChip(lane)).join("");

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

function renderLaneLegend() {
  const performanceLegend = `
    <span><i class="perf-good"></i>Solved / high-quality</span>
    <span><i class="perf-bad"></i>Failed / low-quality</span>
    <span><i class="scrub-line"></i>progress</span>
  `;
  const laneItems = state.lanes
    .map((lane) => {
      const style = laneStyle(lane.id);
      return `<span><i class="line-swatch" style="${lineSwatchStyle(style)}"></i>${laneLabel(lane.id)} · ${modelLabel(lane.model)}</span>`;
    })
    .join("");
  $("laneLegend").innerHTML = `${laneItems}${performanceLegend}`;
}

function laneChip(lane) {
  const style = laneStyle(lane.id);
  return `
    <div class="lane-chip" style="border-left-color:${style.line}">
      <strong><i class="model-dot" style="background:${style.line}"></i>${laneLabel(lane.id)} · ${modelLabel(lane.model)}</strong>
      <span>${titleCase(lane.domain)}</span>
      <em class="outcome-chip ${outcomeTone(lane.outcome)}">${OUTCOME_GROUPS[lane.outcome].label}</em>
    </div>
  `;
}

function behaviorCard(behavior) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = `mini-chart ${behavior === state.selectedBehavior ? "selected" : ""}`;
  card.addEventListener("click", () => {
    state.selectedBehavior = behavior;
    renderComparison();
    renderTrackA();
    renderInspector();
  });

  const curves = state.lanes.map((lane) => aggregateCurve(lane.id, behavior));
  const spread = curveSpread(curves, state.bin);
  card.innerHTML = `
    <div class="mini-chart-head">
      <div>
        <h3 class="has-tooltip" data-tooltip="${escapeAttr(behaviorDescription(behavior))}">${titleCase(behavior)}</h3>
        <small>${familyShortLabel(behavior)} · ${state.lanes.length} lane${state.lanes.length === 1 ? "" : "s"}</small>
      </div>
      <span class="delta-pill">${state.lanes.length === 1 ? pct.format(spread.maxValue) : `spread ${signedPct(spread.spread).replace("+", "")}`}</span>
    </div>
    <svg role="img" aria-label="${titleCase(behavior)} trajectory comparison"></svg>
  `;
  drawMiniChart(card.querySelector("svg"), curves);
  return card;
}

function renderTrackA() {
  if (!$("trackAGrid")) return;
  const behaviors = orderedBehaviors([...state.behaviors]);
  $("trackASubtitle").textContent = `${state.lanes.length} lane${state.lanes.length === 1 ? "" : "s"} · ${state.behaviors.size} behavior${state.behaviors.size === 1 ? "" : "s"} · whole-trace count and presence, no temporal binning`;
  renderTrackALegend();
  $("trackAReadout").innerHTML = state.lanes.map((lane) => laneChip(lane)).join("");

  const grid = $("trackAGrid");
  grid.innerHTML = "";
  grid.className = "tracka-grid";
  if (!store.trackA?.cells?.length) {
    grid.innerHTML = '<div class="empty-state">Whole-trace count data is not available in this dashboard export yet.</div>';
    $("trackADetail").innerHTML = "";
    return;
  }
  if (!behaviors.length) {
    grid.innerHTML = '<div class="empty-state">Select at least one behavior to compare whole-trace counts.</div>';
    $("trackADetail").innerHTML = "";
    return;
  }

  const groups = behaviorGroups(behaviors);
  grid.className = groups.length > 1 ? "tracka-grid grouped" : "tracka-grid";
  if (groups.length > 1) {
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
        <div class="tracka-card-grid"></div>
      `;
      const familyGrid = section.querySelector(".tracka-card-grid");
      group.behaviors.forEach((behavior) => familyGrid.appendChild(trackACard(behavior)));
      grid.appendChild(section);
    });
  } else {
    behaviors.forEach((behavior) => grid.appendChild(trackACard(behavior)));
  }
  renderTrackADetail(state.selectedBehavior || behaviors[0]);
}

function renderTrackALegend() {
  const laneItems = state.lanes
    .map((lane) => {
      const style = laneStyle(lane.id);
      return `<span><i class="line-swatch" style="${lineSwatchStyle(style)}"></i>${laneLabel(lane.id)} · ${modelLabel(lane.model)}</span>`;
    })
    .join("");
  $("trackALegend").innerHTML = `${laneItems}<span><i class="tracka-mean"></i>mean count</span><span><i class="tracka-presence"></i>presence rate</span>`;
}

function trackACard(behavior) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = `tracka-card ${behavior === state.selectedBehavior ? "selected" : ""}`;
  card.addEventListener("click", () => {
    state.selectedBehavior = behavior;
    renderComparison();
    renderTrackA();
    renderInspector();
  });

  const rows = state.lanes.map((lane) => trackAMetric(lane.id, behavior));
  const countSpread = metricSpread(rows, "meanCount");
  const presenceSpread = metricSpread(rows, "presenceRate");
  const maxCount = Math.max(0.1, ...rows.map((row) => row.upperCount || row.meanCount || 0));
  card.innerHTML = `
    <div class="mini-chart-head">
      <div>
        <h3 class="has-tooltip" data-tooltip="${escapeAttr(behaviorDescription(behavior))}">${titleCase(behavior)}</h3>
        <small>${familyShortLabel(behavior)} · whole trace</small>
      </div>
      <span class="delta-pill">${state.lanes.length === 1 ? one.format(countSpread.maxValue) : `spread ${one.format(countSpread.spread)}`}</span>
    </div>
    <div class="tracka-bars">
      ${rows
        .map((row) => {
          const style = laneStyle(row.lane.id);
          const width = Math.max(2, Math.min(100, (row.meanCount / maxCount) * 100));
          const presence = row.presenceRate == null ? "-" : pct.format(row.presenceRate);
          const countRange = row.nTraces ? `${one.format(row.lowerCount)}-${one.format(row.upperCount)}` : "-";
          return `
            <div class="tracka-bar-row" style="--bar-color:${style.line}">
              <div class="tracka-bar-label">
                <strong>${laneLabel(row.lane.id)}</strong>
                <span>${fmt.format(row.nTraces)} traces</span>
              </div>
              <div class="tracka-bar-shell"><i style="width:${width}%"></i></div>
              <div class="tracka-bar-values">
                <strong>${one.format(row.meanCount)}</strong>
                <span>${presence} present · CI ${countRange}</span>
              </div>
            </div>
          `;
        })
        .join("")}
    </div>
    <div class="tracka-card-foot">
      <span>Presence spread ${presenceSpread.spread == null ? "-" : signedPct(presenceSpread.spread).replace("+", "")}</span>
      <span>${laneLabel(countSpread.maxLane?.id)} highest</span>
    </div>
  `;
  return card;
}

function renderTrackADetail(behavior) {
  const target = $("trackADetail");
  if (!target || !behavior) return;
  const rows = state.lanes.map((lane) => trackAMetric(lane.id, behavior));
  const countSpread = metricSpread(rows, "meanCount");
  const presenceSpread = metricSpread(rows, "presenceRate");
  const familyRows = state.lanes.map((lane) => trackAFamilyMetric(lane.id, familyFor(behavior)));
  const familyMax = Math.max(0.1, ...familyRows.map((row) => row.meanCount || 0));
  const domainProfiles = state.lanes.map((lane) => ({
    lane,
    style: laneStyle(lane.id),
    metrics: store.domains.map((domain) => trackAMetricForConfig({ ...lane, domain }, behavior)),
  }));
  const profileMax = Math.max(0.1, ...domainProfiles.flatMap((profile) => profile.metrics.map((metric) => metric.meanCount || 0)));

  target.innerHTML = `
      <div class="tracka-detail-head">
        <h3 class="has-tooltip" data-tooltip="${escapeAttr(behaviorDescription(behavior))}">${titleCase(behavior)}</h3>
        <p>Static behavior counts across the full trace. These values ignore cursor position and answer-boundary timing.</p>
      </div>
    <div class="delta-facts tracka-facts">
      ${rows
        .map((row) => {
          const style = laneStyle(row.lane.id);
          return `<div class="delta-fact lane-fact" style="border-left-color:${style.line}"><span>${laneLabel(row.lane.id)} · ${modelLabel(row.lane.model)}</span><strong>${one.format(row.meanCount)}</strong><small>${pct.format(row.presenceRate || 0)} present · n=${fmt.format(row.nTraces)} · ${titleCase(row.lane.domain)}</small></div>`;
        })
        .join("")}
      ${
        state.lanes.length > 1
          ? `<div class="delta-fact spread-fact"><span>Static count spread</span><strong>${one.format(countSpread.spread)}</strong><small>${laneLabel(countSpread.maxLane?.id)} leads ${laneLabel(countSpread.minLane?.id)} · presence spread ${signedPct(presenceSpread.spread).replace("+", "")}</small></div>`
          : ""
      }
    </div>
    <div class="tracka-subsection">
      <div class="inspector-section-title"><span>${familyShortLabel(behavior)} Family Volume</span></div>
      <div class="tracka-family-bars">
        ${familyRows
          .map((row) => {
            const style = laneStyle(row.lane.id);
            return `<div class="tracka-family-row" style="--bar-color:${style.line}"><span>${laneLabel(row.lane.id)}</span><i><b style="width:${Math.max(2, Math.min(100, (row.meanCount / familyMax) * 100))}%"></b></i><strong>${one.format(row.meanCount)}</strong></div>`;
          })
          .join("")}
      </div>
    </div>
    <div class="tracka-subsection">
      <div class="inspector-section-title"><span>Domain Profile</span></div>
      <div class="tracka-domain-profiles">
        ${domainProfiles
          .map(
            (profile) => `
              <article style="--bar-color:${profile.style.line}">
                <strong>${laneLabel(profile.lane.id)} · ${modelLabel(profile.lane.model)}</strong>
                ${profile.metrics
                  .map(
                    (metric) => `
                      <div class="tracka-domain-row">
                        <span>${titleCase(metric.lane.domain)}</span>
                        <i><b style="width:${Math.max(2, Math.min(100, (metric.meanCount / profileMax) * 100))}%"></b></i>
                        <em>${one.format(metric.meanCount)}</em>
                      </div>
                    `,
                  )
                  .join("")}
              </article>
            `,
          )
          .join("")}
      </div>
    </div>
    <div class="hypothesis-list tracka-hypotheses">
      <div class="inspector-section-title"><span>Static Hypothesis Prompts</span></div>
      ${trackAHypotheses(behavior, rows, countSpread, presenceSpread).map((prompt) => `<article>${escapeHtml(prompt)}</article>`).join("")}
    </div>
  `;
}

function renderMonitorability() {
  if (!$("monitorCurve")) return;
  const metrics = store.prefixMonitor?.metrics || [];
  const deltas = store.prefixMonitor?.deltas || [];
  const splits = store.prefixMonitor?.meta?.splits || [];
  $("monitorSubtitle").textContent = metrics.length
    ? `${fmt.format(metrics.find((row) => row.split === state.monitor.split)?.n || metrics[0]?.n || 0)} traces · predicting final success/quality from the visible prefix`
    : "Prefix monitorability data is not available in this dashboard export yet.";
  renderMonitorSplitButtons(splits);
  const rows = metrics.filter((row) => row.split === state.monitor.split);
  const deltaRows = deltas.filter((row) => row.split === state.monitor.split);
  renderMonitorSummary(rows, deltaRows);
  drawMonitorCurve($("monitorCurve"), rows);
  renderMonitorPulseCards(rows, deltaRows);
  renderMonitorNotes();
}

function renderMonitorSplitButtons(splits) {
  const target = $("monitorSplitButtons");
  if (!target) return;
  target.innerHTML = (splits.length ? splits : [state.monitor.split])
    .map((split) => `<button class="${split === state.monitor.split ? "active" : ""}" data-monitor-split="${escapeAttr(split)}">${monitorSplitLabel(split)}</button>`)
    .join("");
}

function renderMonitorSummary(rows, deltaRows) {
  const target = $("monitorSummary");
  if (!target) return;
  if (!rows.length) {
    target.innerHTML = "<span>No monitorability results match the selected split.</span>";
    return;
  }
  const visible = monitorVisibleRows(rows);
  const prefixes = monitorPrefixes(visible);
  const earliest = prefixes.map((prefix) => bestMonitorRowAtPrefix(visible, prefix)).find((item) => item?.score >= 0.4) || bestMonitorRowAtPrefix(visible, prefixes[0]);
  const early = bestMonitorRowAtPrefix(visible, prefixes[0]);
  const full = bestMonitorRowAtPrefix(visible, prefixes.at(-1));
  const fullTimingDelta = monitorDelta(deltaRows, prefixes.at(-1), "counts");
  target.innerHTML = `
    <span><strong>${monitorSplitLabel(state.monitor.split)}</strong> split</span>
    <span><strong>${formatPrefix(earliest?.prefix)}</strong> first useful signal</span>
    <span><strong>${formatPredictabilityScore(early?.row)}</strong> by ${formatPrefix(early?.prefix)}</span>
    <span><strong>${MONITOR_FEATURES[full?.row?.feature_set]?.short || "Signal"}</strong> strongest at full trace</span>
    <span><strong>${formatPredictabilityDeltaFromAuroc(fullTimingDelta?.delta_auroc)}</strong> timing vs behavior mix</span>
  `;
}

function drawMonitorCurve(svg, rows) {
  const width = 760;
  const height = 340;
  const margin = { top: 34, right: 26, bottom: 50, left: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const visible = monitorVisibleRows(rows);
  const prefixes = monitorPrefixes(visible);
  const featureSets = MONITOR_VISIBLE_FEATURES.filter((feature) => visible.some((row) => row.feature_set === feature));
  const values = visible.map(predictabilityScore).filter(Number.isFinite);
  const ymax = Math.min(1, Math.max(0.6, Math.ceil((Math.max(...values, 0.45) + 0.08) * 10) / 10));
  const x = (prefix) => margin.left + Number(prefix || 0) * plotW;
  const y = (value) => margin.top + plotH - (Math.max(0, Math.min(ymax, value || 0)) / ymax) * plotH;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  svg.appendChild(svgEl("rect", { x: margin.left, y: margin.top, width: plotW, height: plotH, fill: "#fff", stroke: "#dce3eb", "stroke-width": "1" }));
  svg.appendChild(svgEl("rect", { x: margin.left, y: y(0.4), width: plotW, height: Math.max(0, y(0) - y(0.4)), fill: "#ecfdf5", stroke: "none", opacity: "0.68" }));
  [0, 0.25, 0.4, ymax].forEach((tick) => {
    const yy = y(tick);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, stroke: "#edf2f7", "stroke-width": "1" }));
    const label = svgEl("text", { x: margin.left - 8, y: yy + 4, "text-anchor": "end", "font-size": "10", fill: "#647084", stroke: "none" });
    label.textContent = tick === 0 ? "chance" : `${Math.round(tick * 100)}%`;
    svg.appendChild(label);
  });
  [0, ...prefixes].forEach((prefix) => {
    const xx = x(prefix);
    svg.appendChild(svgEl("line", { x1: xx, x2: xx, y1: margin.top, y2: height - margin.bottom, stroke: "#f1f5f9", "stroke-width": "1" }));
    const label = svgEl("text", { x: xx, y: height - 24, "text-anchor": "middle", "font-size": "10", fill: "#647084", stroke: "none" });
    label.textContent = formatPrefix(prefix);
    svg.appendChild(label);
  });
  const useful = svgEl("text", { x: width - margin.right - 8, y: y(0.4) - 7, "text-anchor": "end", "font-size": "10", fill: "#137a3f", stroke: "none", "font-weight": "800" });
  useful.textContent = "useful prediction zone";
  svg.appendChild(useful);
  const xLabel = svgEl("text", { x: margin.left + plotW / 2, y: height - 7, "text-anchor": "middle", "font-size": "11", fill: "#334155", stroke: "none" });
  xLabel.textContent = "How much of the trace is visible";
  svg.appendChild(xLabel);
  const yLabel = svgEl("text", { x: 15, y: margin.top + plotH / 2, transform: `rotate(-90 15 ${margin.top + plotH / 2})`, "text-anchor": "middle", "font-size": "11", fill: "#334155", stroke: "none" });
  yLabel.textContent = "Predictability above chance";
  svg.appendChild(yLabel);

  featureSets.forEach((feature) => {
    const meta = MONITOR_FEATURES[feature];
    const points = prefixes.map((prefix) => rows.find((row) => row.prefix === prefix && row.feature_set === feature)).filter(Boolean);
    if (!points.length) return;
    const path = points.map((row, i) => `${i ? "L" : "M"} ${x(row.prefix).toFixed(2)} ${y(predictabilityScore(row)).toFixed(2)}`).join(" ");
    svg.appendChild(svgEl("path", { d: path, fill: "none", stroke: meta.color, "stroke-width": feature === "temporal" ? "3" : "2.2", "stroke-dasharray": meta.dash }));
    points.forEach((row) => {
      const dot = svgEl("circle", { cx: x(row.prefix), cy: y(predictabilityScore(row)), r: feature === "temporal" ? "4.5" : "3.8", fill: meta.color, stroke: "#fff", "stroke-width": "1" });
      const title = svgEl("title", {});
      title.textContent = `${meta.label} · ${formatPrefix(row.prefix)} · ${formatPredictabilityScore(row)} above chance (${meta.description})`;
      dot.appendChild(title);
      svg.appendChild(dot);
    });
  });
  featureSets.forEach((feature, i) => {
    const meta = MONITOR_FEATURES[feature];
    const x0 = margin.left + (i % 3) * 210;
    const y0 = 12 + Math.floor(i / 3) * 14;
    svg.appendChild(svgEl("line", { x1: x0, x2: x0 + 24, y1: y0, y2: y0, stroke: meta.color, "stroke-width": "2.3", "stroke-dasharray": meta.dash }));
    const label = svgEl("text", { x: x0 + 30, y: y0 + 4, "font-size": "10", fill: "#334155", stroke: "none" });
    label.textContent = meta.label;
    svg.appendChild(label);
  });
}

function renderMonitorPulseCards(rows, deltaRows) {
  const target = $("monitorPulseCards");
  if (!target) return;
  const visible = monitorVisibleRows(rows);
  const prefixes = monitorPrefixes(visible);
  if (!prefixes.length) {
    target.innerHTML = '<div class="empty-state">No predictability results are available.</div>';
    return;
  }
  const fullPrefix = prefixes.at(-1);
  const fullTiming = monitorDelta(deltaRows, fullPrefix, "counts");
  const fullTimingDelta = (fullTiming?.delta_auroc || 0) * 2;
  const firstUseful = prefixes.map((prefix) => bestMonitorRowAtPrefix(visible, prefix)).find((item) => item?.score >= 0.4);
  $("monitorTakeaway").textContent = firstUseful
    ? `The final outcome is already ${predictabilityBand(firstUseful.score).toLowerCase()} by ${formatPrefix(firstUseful.prefix)}. In this export, exact timing ${Math.abs(fullTimingDelta) < 0.015 ? "mostly mirrors the behavior mix" : fullTimingDelta > 0 ? "adds extra signal beyond the behavior mix" : "does not improve on the behavior mix"}.`
    : "The visible prefixes remain close to chance in this split; use the task/model controls above to form more specific comparisons.";
  target.innerHTML = prefixes
    .map((prefix) => {
      const task = monitorRowAtPrefix(visible, prefix, "metadata");
      const counts = monitorRowAtPrefix(visible, prefix, "counts");
      const temporal = monitorRowAtPrefix(visible, prefix, "temporal");
      const best = bestMonitorRowAtPrefix(visible, prefix);
      const behaviorGain = predictabilityScore(counts) - predictabilityScore(task);
      const timingGain = predictabilityScore(temporal) - predictabilityScore(counts);
      return `
        <article>
          <div><strong>${formatPrefix(prefix)}</strong><span>${predictabilityBand(best?.score)}</span></div>
          <p>Best signal: <b>${MONITOR_FEATURES[best?.row?.feature_set]?.short || "n/a"}</b> at ${formatPredictabilityScore(best?.row)}.</p>
          <small>Behavior mix ${phraseScoreDelta(behaviorGain)} vs task; timing ${phraseScoreDelta(timingGain)} vs mix.</small>
        </article>
      `;
    })
    .join("");
}

function renderMonitorNotes() {
  const target = $("monitorNotes");
  if (!target) return;
  target.innerHTML = `
    <strong>Read gently</strong>
    <span>This predicts final outcome from partial behavior traces; it does not prove the behaviors caused success or failure.</span>
    <span>Percent prefixes are retrospective because final trace length is only known after generation.</span>
    <span>Safety uses a fixed safe/high-harm endpoint; moral and idea tasks use a high/low quality split.</span>
  `;
}

function monitorVisibleRows(rows) {
  return rows.filter((row) => MONITOR_VISIBLE_FEATURES.includes(row.feature_set));
}

function monitorPrefixes(rows) {
  return [...new Set(rows.map((row) => row.prefix).filter((prefix) => Number.isFinite(Number(prefix))))].sort((a, b) => a - b);
}

function monitorRowAtPrefix(rows, prefix, feature) {
  return rows.find((row) => row.prefix === prefix && row.feature_set === feature) || null;
}

function bestMonitorRowAtPrefix(rows, prefix) {
  const row = MONITOR_VISIBLE_FEATURES.map((feature) => monitorRowAtPrefix(rows, prefix, feature))
    .filter(Boolean)
    .sort((a, b) => predictabilityScore(b) - predictabilityScore(a))[0];
  return row ? { row, prefix, score: predictabilityScore(row) } : null;
}

function monitorDelta(rows, prefix, baseline) {
  return rows.find((row) => row.feature_set === "temporal" && row.baseline === baseline && row.prefix === prefix) || null;
}

function predictabilityScore(row) {
  const auroc = Number(row?.auroc);
  if (!Number.isFinite(auroc)) return 0;
  return Math.max(0, Math.min(1, (auroc - 0.5) / 0.5));
}

function formatPredictabilityScore(row) {
  return `${Math.round(predictabilityScore(row) * 100)}%`;
}

function formatPredictabilityDeltaFromAuroc(delta) {
  if (!Number.isFinite(Number(delta))) return "n/a";
  const pp = Math.round(Number(delta) * 200);
  if (Math.abs(pp) < 1) return "~0pp";
  return `${pp > 0 ? "+" : ""}${pp}pp`;
}

function phraseScoreDelta(delta) {
  if (!Number.isFinite(delta)) return "is unavailable";
  const pp = Math.round(delta * 100);
  if (Math.abs(pp) <= 1) return "is about the same";
  return pp > 0 ? `adds ${pp}pp` : `trails by ${Math.abs(pp)}pp`;
}

function predictabilityBand(score) {
  if (!Number.isFinite(score)) return "No signal";
  if (score >= 0.55) return "Strong";
  if (score >= 0.4) return "Useful";
  if (score >= 0.25) return "Emerging";
  return "Weak";
}

function monitorSplitLabel(split) {
  if (split === "prompt_disjoint") return "Prompt-disjoint";
  if (split === "random_trace") return "Random trace";
  return titleCase(split);
}

function formatPrefix(prefix) {
  return Number.isFinite(Number(prefix)) ? `${Math.round(Number(prefix) * 100)}%` : "n/a";
}

function renderTimingLevel() {
  if (!$("timingScatter")) return;
  const pairs = store.timingLevel?.pairs || [];
  $("timingSubtitle").textContent = pairs.length
    ? `${pairs.length} model × domain × behavior pairs · separates "how much" from "when" in good vs bad traces`
    : "Timing-vs-level data is not available in this dashboard export yet.";
  renderTimingLegend();
  renderTimingFilters();

  const filtered = timingFilteredPairs();
  const selected = ensureTimingSelection(filtered);
  renderTimingSummary(filtered);
  drawTimingScatter($("timingScatter"), filtered);
  renderTimingDetail(selected);
  renderTimingTable(filtered);
}

function renderTimingLegend() {
  const modelItems = store.models
    .map((model) => `<span><i class="model-dot" style="background:${colorForModel(model)}"></i>${modelLabel(model)}</span>`)
    .join("");
  $("timingLegend").innerHTML = `${modelItems}<span><i class="timing-shape circle"></i>Cognitive</span><span><i class="timing-shape diamond"></i>Conversational</span>`;
}

function renderTimingFilters() {
  const target = $("timingFilters");
  if (!target) return;
  const modelButtons = store.models.map((model) => timingFilterButton("models", model, modelLabel(model), state.timing.models.has(model))).join("");
  const domainButtons = store.domains.map((domain) => timingFilterButton("domains", domain, titleCase(domain), state.timing.domains.has(domain))).join("");
  const familyButtons = ["cognitive", "conversational"]
    .map((family) => timingFilterButton("families", family, FAMILY_META[family]?.short || titleCase(family), state.timing.families.has(family)))
    .join("");
  const classButtons = Object.entries(TIMING_CLASSES)
    .map(([key, meta]) => timingFilterButton("classes", key, meta.label, state.timing.classes.has(key), `class-${meta.tone}`))
    .join("");
  target.innerHTML = `
    <div class="timing-filter-group"><strong>Models</strong><div>${modelButtons}</div></div>
    <div class="timing-filter-group"><strong>Domains</strong><div>${domainButtons}</div></div>
    <div class="timing-filter-group"><strong>Family</strong><div>${familyButtons}</div></div>
    <div class="timing-filter-group"><strong>Class</strong><div>${classButtons}</div></div>
  `;
}

function timingFilterButton(kind, value, label, active, extraClass = "") {
  return `<button class="timing-filter-chip ${active ? "active" : ""} ${extraClass}" data-timing-filter="${kind}" data-timing-value="${escapeAttr(value)}">${escapeHtml(label)}</button>`;
}

function toggleTimingFilter(kind, value) {
  const set = state.timing[kind];
  if (!(set instanceof Set)) return;
  if (set.has(value)) {
    if (set.size <= 1) return;
    set.delete(value);
  } else {
    set.add(value);
  }
}

function timingFilteredPairs() {
  return (store.timingLevel?.pairs || []).filter((pair) => {
    return (
      state.timing.models.has(pair.gen_model) &&
      state.timing.domains.has(pair.task_type) &&
      state.timing.families.has(pair.family) &&
      state.timing.classes.has(pair.class)
    );
  });
}

function preferredTimingPair(rows = store.timingLevel?.pairs || []) {
  const tested = rows.filter((row) => row.status === "tested");
  const priority = { timing_only: 4, both: 3, level_only: 2, neither: 1, insufficient: 0 };
  const sorted = (tested.length ? tested : rows)
    .map((row) => ({ row, key: timingKey(row) }))
    .sort((a, b) => {
      const pa = priority[a.row.class] || 0;
      const pb = priority[b.row.class] || 0;
      return pb - pa || (b.row.timing?.I2_robust || 0) - (a.row.timing?.I2_robust || 0) || Math.abs(b.row.level?.dbar_robust_pp || 0) - Math.abs(a.row.level?.dbar_robust_pp || 0);
    });
  return sorted[0] || null;
}

function ensureTimingSelection(filtered) {
  const current = state.timing.selectedKey ? store.timingIndex.get(state.timing.selectedKey) : null;
  if (current && filtered.some((pair) => timingKey(pair) === state.timing.selectedKey)) return current;
  const next = preferredTimingPair(filtered);
  state.timing.selectedKey = next?.key || null;
  return next?.row || null;
}

function selectTimingPair(key) {
  state.timing.selectedKey = key;
  renderTimingLevel();
}

function renderTimingSummary(rows) {
  const counts = Object.fromEntries(Object.keys(TIMING_CLASSES).map((key) => [key, 0]));
  rows.forEach((row) => {
    counts[row.class] = (counts[row.class] || 0) + 1;
  });
  const tested = rows.filter((row) => row.status === "tested").length;
  $("timingSummary").innerHTML = `
    <span><strong>${fmt.format(tested)}</strong> behavior comparisons</span>
    <span class="class-both">${fmt.format(counts.both || 0)} both</span>
    <span class="class-level">${fmt.format(counts.level_only || 0)} amount gaps</span>
    <span class="class-timing">${fmt.format(counts.timing_only || 0)} timing shapes</span>
    <span>${fmt.format(counts.neither || 0)} no clear split</span>
    <span>${fmt.format(counts.insufficient || 0)} too sparse</span>
  `;
}

function drawTimingScatter(svg, rows) {
  const tested = rows.filter((row) => row.status === "tested");
  const width = 760;
  const height = 400;
  const margin = { top: 24, right: 24, bottom: 48, left: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const maxLevel = Math.max(1, ...tested.map((row) => Math.abs(row.level?.dbar_robust_pp || 0))) * 1.08;
  const x = (value) => margin.left + (Math.max(0, Math.min(maxLevel, value)) / maxLevel) * plotW;
  const y = (value) => margin.top + plotH - Math.max(0, Math.min(1, value || 0)) * plotH;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  svg.appendChild(svgEl("rect", { x: margin.left, y: margin.top, width: plotW, height: plotH, fill: "#fff", stroke: "#dce3eb", "stroke-width": "1" }));
  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    const yy = y(tick);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, stroke: "#edf2f7", "stroke-width": "1" }));
    const label = svgEl("text", { x: margin.left - 9, y: yy + 4, "text-anchor": "end", "font-size": "10", fill: "#647084", stroke: "none" });
    label.textContent = tick.toFixed(2).replace("0.", ".");
    svg.appendChild(label);
  });
  [0, 0.5, 1].forEach((tick) => {
    const xx = x(maxLevel * tick);
    svg.appendChild(svgEl("line", { x1: xx, x2: xx, y1: margin.top, y2: height - margin.bottom, stroke: "#f1f5f9", "stroke-width": "1" }));
    const label = svgEl("text", { x: xx, y: height - 24, "text-anchor": "middle", "font-size": "10", fill: "#647084", stroke: "none" });
    label.textContent = `${(maxLevel * tick).toFixed(tick ? 1 : 0)}pp`;
    svg.appendChild(label);
  });

  drawTimingReference(svg, x(0.5), margin.top, plotH, "amount gap gets visible", "vertical");
  drawTimingReference(svg, margin.left, y(0.25), plotW, "timing shape changes", "horizontal");
  const xLabel = svgEl("text", { x: margin.left + plotW / 2, y: height - 7, "text-anchor": "middle", "font-size": "11", fill: "#334155", stroke: "none" });
  xLabel.textContent = "How different the total amount is";
  svg.appendChild(xLabel);
  const yLabel = svgEl("text", { x: 16, y: margin.top + plotH / 2, transform: `rotate(-90 16 ${margin.top + plotH / 2})`, "text-anchor": "middle", "font-size": "11", fill: "#334155", stroke: "none" });
  yLabel.textContent = "How different the timing shape is";
  svg.appendChild(yLabel);

  if (!tested.length) {
    const empty = svgEl("text", { x: width / 2, y: height / 2, "text-anchor": "middle", "font-size": "13", fill: "#647084", stroke: "none" });
    empty.textContent = "No tested pairs match the current filters.";
    svg.appendChild(empty);
    return;
  }

  tested
    .slice()
    .sort((a, b) => (a.class === "neither") - (b.class === "neither"))
    .forEach((row) => {
      const key = timingKey(row);
      const xx = x(Math.abs(row.level?.dbar_robust_pp || 0));
      const yy = y(row.timing?.I2_robust || 0);
      const color = colorForModel(row.gen_model);
      const opacity = row.class === "neither" ? 0.38 : 0.9;
      const selected = key === state.timing.selectedKey;
      const attrs = {
        "data-timing-key": key,
        tabindex: "0",
        role: "button",
        "aria-label": timingAriaLabel(row),
        fill: color,
        stroke: selected ? "#111827" : "#fff",
        "stroke-width": selected ? "2.2" : "1.2",
        opacity,
      };
      const point = row.family === "conversational" ? timingDiamond(xx, yy, selected ? 6 : 5, attrs) : svgEl("circle", { ...attrs, cx: xx, cy: yy, r: selected ? "6" : "5" });
      const title = svgEl("title", {});
      title.textContent = timingTooltipText(row);
      point.appendChild(title);
      svg.appendChild(point);
    });
}

function drawTimingReference(svg, x1, y1, length, label, direction) {
  if (direction === "vertical") {
    svg.appendChild(svgEl("line", { x1, x2: x1, y1, y2: y1 + length, stroke: "#94a3b8", "stroke-width": "1", "stroke-dasharray": "4 4" }));
    const text = svgEl("text", { x: x1 + 6, y: y1 + 14, "font-size": "10", fill: "#647084", stroke: "none" });
    text.textContent = label;
    svg.appendChild(text);
  } else {
    svg.appendChild(svgEl("line", { x1, x2: x1 + length, y1, y2: y1, stroke: "#94a3b8", "stroke-width": "1", "stroke-dasharray": "4 4" }));
    const text = svgEl("text", { x: x1 + length - 6, y: y1 - 6, "text-anchor": "end", "font-size": "10", fill: "#647084", stroke: "none" });
    text.textContent = label;
    svg.appendChild(text);
  }
}

function timingDiamond(cx, cy, r, attrs) {
  return svgEl("path", { ...attrs, d: `M ${cx} ${cy - r} L ${cx + r} ${cy} L ${cx} ${cy + r} L ${cx - r} ${cy} Z` });
}

function renderTimingDetail(pair) {
  const target = $("timingDetail");
  if (!target) return;
  if (!pair) {
    target.innerHTML = '<div class="empty-state">Select a tested pair in the scatter or table.</div>';
    return;
  }
  const cls = TIMING_CLASSES[pair.class] || TIMING_CLASSES.neither;
  target.innerHTML = `
    <div class="timing-detail-head">
      <span class="timing-class-pill class-${cls.tone}">${cls.label}</span>
      <h3 class="has-tooltip" data-tooltip="${escapeAttr(behaviorDescription(pair.behavior))}">${titleCase(pair.behavior)}</h3>
      <p>${modelLabel(pair.gen_model)} · ${titleCase(pair.task_type)} · ${pair.status === "tested" ? `${pair.bins_used} time bins compared` : "too few examples for a timing shape"}</p>
    </div>
    <div class="delta-facts timing-facts">
      <div class="delta-fact"><span>Amount difference</span><strong>${formatPp(pair.level?.dbar_robust_pp)}</strong><small>positive means this behavior is more common in good traces</small></div>
      <div class="delta-fact"><span>Timing shape</span><strong>${formatNumber(pair.timing?.I2_robust, 2)}</strong><small>higher means the good/bad gap moves around the trace</small></div>
      <div class="delta-fact"><span>Trace classes</span><strong>${fmt.format(pair.n_traces?.good || 0)} / ${fmt.format(pair.n_traces?.bad || 0)}</strong><small>good vs bad traces</small></div>
    </div>
    <div class="timing-detail-chart">
      <svg id="timingDetailChart" role="img" aria-label="${escapeAttr(timingAriaLabel(pair))}"></svg>
    </div>
    <p class="timing-shape-sentence">${escapeHtml(timingShapeSentence(pair))}</p>
    <p class="timing-detail-note">In the small chart, values above zero mark parts of the trace where the behavior appears more in good traces; values below zero mark parts where it appears more in bad traces.</p>
  `;
  drawTimingDetailChart($("timingDetailChart"), pair);
}

function drawTimingDetailChart(svg, pair) {
  const width = 420;
  const height = 235;
  const margin = { top: 18, right: 16, bottom: 34, left: 42 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const values = pair.d_pp || [];
  const ses = pair.d_se_pp || [];
  const finite = values.filter((v) => Number.isFinite(v));
  const bandVals = values.flatMap((v, i) => (Number.isFinite(v) ? [v + 2 * (ses[i] || 0), v - 2 * (ses[i] || 0)] : []));
  const level = pair.level?.dbar_robust_pp;
  const maxAbs = Math.max(1, ...finite.map(Math.abs), ...bandVals.map(Math.abs), Math.abs(level || 0)) * 1.12;
  const x = (bin) => margin.left + (bin / Math.max(1, (store.timingLevel?.meta?.K || 24) - 1)) * plotW;
  const y = (value) => margin.top + plotH / 2 - (value / maxAbs) * (plotH / 2);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  svg.appendChild(svgEl("rect", { x: margin.left, y: margin.top, width: plotW, height: plotH, fill: "#fff", stroke: "#dce3eb", "stroke-width": "1" }));
  [-1, 0, 1].forEach((tick) => {
    const yy = y(maxAbs * tick);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, stroke: tick === 0 ? "#111827" : "#edf2f7", "stroke-width": tick === 0 ? "1.2" : "1" }));
    const label = svgEl("text", { x: margin.left - 7, y: yy + 3, "text-anchor": "end", "font-size": "9.5", fill: "#647084", stroke: "none" });
    label.textContent = `${Math.round(maxAbs * tick)}pp`;
    svg.appendChild(label);
  });
  const points = values.map((value, bin) => ({ value, bin, se: ses[bin] })).filter((point) => Number.isFinite(point.value));
  if (points.length) {
    const upper = points.map((p, i) => `${i ? "L" : "M"} ${x(p.bin).toFixed(2)} ${y(p.value + 2 * (p.se || 0)).toFixed(2)}`).join(" ");
    const lower = points
      .slice()
      .reverse()
      .map((p) => `L ${x(p.bin).toFixed(2)} ${y(p.value - 2 * (p.se || 0)).toFixed(2)}`)
      .join(" ");
    svg.appendChild(svgEl("path", { d: `${upper} ${lower} Z`, fill: alphaColor(colorForModel(pair.gen_model), 0.14), stroke: "none" }));
    svg.appendChild(svgEl("path", { d: points.map((p, i) => `${i ? "L" : "M"} ${x(p.bin).toFixed(2)} ${y(p.value).toFixed(2)}`).join(" "), fill: "none", stroke: colorForModel(pair.gen_model), "stroke-width": "2.3" }));
  }
  if (Number.isFinite(level)) {
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y(level), y2: y(level), stroke: "#334155", "stroke-width": "1.4", "stroke-dasharray": "5 4" }));
  }
  [0, 0.5, 1].forEach((tick) => {
    const label = svgEl("text", { x: x(tick * ((store.timingLevel?.meta?.K || 24) - 1)), y: height - 10, "text-anchor": "middle", "font-size": "9.5", fill: "#647084", stroke: "none" });
    label.textContent = `${Math.round(tick * 100)}%`;
    svg.appendChild(label);
  });
}

function timingShapeSentence(pair) {
  if (pair.status !== "tested") return "There are too few examples spread across the trace to compare timing shape for this pair.";
  const thirds = pair.thirds || {};
  const rows = [
    ["opening third", thirds.early || 0],
    ["middle third", thirds.mid || 0],
    ["closing third", thirds.late || 0],
  ].sort((a, b) => b[1] - a[1]);
  const top = rows[0];
  const direction = (pair.level?.dbar_robust_pp || 0) >= 0 ? "good traces" : "bad traces";
  return `The shape difference is most visible in the ${top[0]}. Positive sections mean the behavior appears more in good traces; negative sections mean it appears more in bad traces. Overall amount favors ${direction}.`;
}

function renderTimingTable(rows) {
  const sorted = rows
    .slice()
    .sort((a, b) => (b.timing?.I2_robust || 0) - (a.timing?.I2_robust || 0) || Math.abs(b.level?.dbar_robust_pp || 0) - Math.abs(a.level?.dbar_robust_pp || 0))
    .slice(0, 80);
  let html = "<thead><tr><th>Behavior</th><th>Pattern</th><th>Amount gap</th><th>Timing shape</th></tr></thead><tbody>";
  sorted.forEach((row) => {
    const cls = TIMING_CLASSES[row.class] || TIMING_CLASSES.neither;
    html += `
      <tr data-timing-key="${escapeAttr(timingKey(row))}" class="${timingKey(row) === state.timing.selectedKey ? "selected" : ""}" tabindex="0">
        <td><strong>${modelLabel(row.gen_model)}</strong><small>${titleCase(row.task_type)} · ${titleCase(row.behavior)}</small></td>
        <td><span class="timing-class-pill class-${cls.tone}">${cls.label}</span></td>
        <td><strong>${formatPp(row.level?.dbar_robust_pp)}</strong><small>${amountGapPhrase(row)}</small></td>
        <td><strong>${formatNumber(row.timing?.I2_robust, 2)}</strong><small>${timingShapePhrase(row)}</small></td>
      </tr>
    `;
  });
  html += "</tbody>";
  $("timingTable").innerHTML = sorted.length ? html : '<tbody><tr><td colspan="4">No pairs match the current filters.</td></tr></tbody>';
}

function timingKey(row) {
  return `${row.gen_model}|${row.task_type}|${row.behavior}`;
}

function timingAriaLabel(row) {
  return `${modelLabel(row.gen_model)} ${titleCase(row.task_type)} ${titleCase(row.behavior)}: ${TIMING_CLASSES[row.class]?.label || row.class}, amount gap ${formatPp(row.level?.dbar_robust_pp)}, timing shape ${formatNumber(row.timing?.I2_robust, 2)}`;
}

function timingTooltipText(row) {
  return `${modelLabel(row.gen_model)} · ${titleCase(row.task_type)} · ${titleCase(row.behavior)}\namount gap ${formatPp(row.level?.dbar_robust_pp)}\ntiming shape ${formatNumber(row.timing?.I2_robust, 2)}\n${TIMING_CLASSES[row.class]?.label || row.class}`;
}

function amountGapPhrase(row) {
  const value = row.level?.dbar_robust_pp || 0;
  if (Math.abs(value) < 0.5) return "similar totals";
  return value > 0 ? "more in good traces" : "more in bad traces";
}

function timingShapePhrase(row) {
  const value = row.timing?.I2_robust || 0;
  if (value >= 0.5) return "strong shape change";
  if (value >= 0.25) return "visible shape change";
  if (value > 0) return "small shape change";
  return "flat timing";
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

  const curves = state.lanes.map((lane) => aggregateCurve(lane.id, behavior));
  const progress = state.bin / (store.manifest.bins - 1);
  const stats = curves.map((curve) => laneStats(curve));
  const spread = curveSpread(curves, state.bin);

  $("inspectorTitle").textContent = titleCase(behavior);
  $("inspectorSubtitle").textContent = `${Math.round(progress * 100)}% through trace · ${titleCase(familyFor(behavior))} behavior`;
  $("deltaFacts").innerHTML = [
    ...stats.map((row) => {
      const style = laneStyle(row.lane.id);
      return `<div class="delta-fact lane-fact" style="border-left-color:${style.line}"><span>${laneLabel(row.lane.id)} · ${modelLabel(row.lane.model)}</span><strong>${pct.format(row.atCursor)}</strong><small>AUC ${pct.format(row.auc)} · early ${pct.format(row.early)} · late ${pct.format(row.late)}</small></div>`;
    }),
    state.lanes.length > 1
      ? `<div class="delta-fact spread-fact"><span>Cursor spread</span><strong>${signedPct(spread.spread).replace("+", "")}</strong><small>${laneLabel(spread.maxLane?.id)} leads ${laneLabel(spread.minLane?.id)}</small></div>`
      : "",
  ].join("");

  renderDivergencePanel();
  renderSampleInspector(behavior);
}

function renderDivergencePanel() {
  const behaviors = orderedBehaviors([...state.behaviors]);
  if (behaviors.length < 2 || state.lanes.length < 2) {
    $("divergencePanel").innerHTML = "";
    $("hypothesisList").innerHTML = "";
    return;
  }

  const divergences = behaviors
    .map((behavior) => {
      const curves = state.lanes.map((lane) => aggregateCurve(lane.id, behavior));
      const spread = curveSpread(curves, state.bin);
      const early = curveSpreadAtBins(curves, (value) => value.bin / (store.manifest.bins - 1) <= 0.4);
      const late = curveSpreadAtBins(curves, (value) => value.bin / (store.manifest.bins - 1) >= 0.6);
      return {
        behavior,
        family: familyFor(behavior),
        ...spread,
        earlySpread: early,
        lateSpread: late,
      };
    })
    .sort((a, b) => b.spread - a.spread);

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
              <strong class="has-tooltip" data-tooltip="${escapeAttr(behaviorDescription(row.behavior))}">${titleCase(row.behavior)}</strong>
              <span>${familyShortLabel(row.behavior)} · ${signedPct(row.spread).replace("+", "")}</span>
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
      renderTrackA();
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
      return rows.length ? { family, spread: average(rows.map((row) => row.spread)) } : null;
    })
    .filter(Boolean)
    .sort((a, b) => b.spread - a.spread);

  const prompts = [];
  prompts.push(
    `${laneLabel(top.maxLane?.id)} is highest and ${laneLabel(top.minLane?.id)} is lowest on ${titleCase(top.behavior)}, separated by ${signedPct(top.spread).replace("+", "")} at the cursor; use Reveal mode to see when that gap opens.`,
  );

  if (familyRows.length) {
    const family = familyRows[0];
    prompts.push(
      `${FAMILY_META[family.family]?.short || titleCase(family.family)} markers have the broadest average lane separation at ${signedPct(family.spread).replace("+", "")}.`,
    );
  }

  if (uniqueValues(state.lanes, "model").length > 1 && uniqueValues(state.lanes, "domain").length === 1) {
    prompts.push(`Model hypothesis: ${state.lanes.length} lanes share ${titleCase(state.lanes[0].domain)}; switch outcomes to test whether the model gap survives performance stratification.`);
  } else if (uniqueValues(state.lanes, "outcome").length > 1 && uniqueValues(state.lanes, "model").length === 1 && uniqueValues(state.lanes, "domain").length === 1) {
    prompts.push(`Outcome hypothesis: this isolates performance for ${modelLabel(state.lanes[0].model)} on ${titleCase(state.lanes[0].domain)}; check if separation grows near the answer phase.`);
  } else {
    prompts.push("Domain hypothesis: pin one model and one outcome group, then add domain lanes to see whether this shape is task-specific.");
  }

  $("hypothesisList").innerHTML = `
    <div class="inspector-section-title"><span>Hypothesis Prompts</span></div>
    ${prompts.map((prompt) => `<article>${escapeHtml(prompt)}</article>`).join("")}
  `;
}

function renderSampleInspector(behavior) {
  const limit = state.lanes.length > 3 ? 1 : 2;
  const laneSamples = state.lanes.flatMap((lane) => rankedTracesForInspector(lane.id, behavior, limit).map((trace) => ({ lane, trace })));
  const annotated = laneSamples.some(({ trace }) => Array.isArray(trace.annotations) && trace.annotations.length);
  $("annotationNote").textContent = annotated
    ? `Evidence prioritizes sentence annotations near the ${Math.round((state.bin / (store.manifest.bins - 1)) * 100)}% cursor, then falls back to the closest matching behavior spans.`
    : "Current static samples expose raw prompt/thinking/answer text plus per-trace behavior counts. Re-run the dashboard export after the annotation upgrade to show sentence-level behavior spans.";

  const cards = laneSamples.map(({ lane, trace }) => sampleCard(trace, laneLabel(lane.id), behavior));
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
      <p>${shortId(trace.trace_id)} · ${titleCase(trace.outcome)} · ${fmt.format(count)} <span class="has-tooltip inline-tooltip" data-tooltip="${escapeAttr(behaviorDescription(behavior))}">${titleCase(behavior)}</span> mark${count === 1 ? "" : "s"}</p>
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

function drawMiniChart(svg, curves) {
  const width = 300;
  const height = 165;
  const margin = { top: 12, right: 12, bottom: 25, left: 34 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const all = curves.flatMap((curve) => curve.values);
  const maxY = Math.max(0.01, ...all.map((v) => v.upper || v.freq || 0)) * 1.05;
  const x = (bin) => margin.left + (bin / (store.manifest.bins - 1)) * plotW;
  const y = (value) => margin.top + plotH - (value / maxY) * plotH;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  drawMiniGrid(svg, width, height, margin, plotW, plotH, maxY, x);
  drawBoundary(svg, curves.map((curve) => curve.boundary), x, margin, plotH);
  drawWindowHighlight(svg, x, margin, plotH);
  curves.forEach((curve) => {
    const style = laneStyle(curve.lane);
    const visible = visibleCurveValues(curve.values);
    if (curves.length <= 4) drawBand(svg, visible, x, y, style.band);
  });
  curves.forEach((curve) => {
    const style = laneStyle(curve.lane);
    drawLine(svg, visibleCurveValues(curve.values), x, y, style.line, style.dash);
  });

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

  curves.forEach((curve) => {
    const style = laneStyle(curve.lane);
    const point = curve.values[state.bin];
    if (!point) return;
    svg.appendChild(svgEl("circle", {
      cx: scrubX,
      cy: y(point.freq || 0),
      r: "3.2",
      fill: style.line,
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

function drawBoundary(svg, boundaries, x, margin, plotH) {
  const ranges = boundaries.filter(Boolean);
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

function drawLine(svg, values, x, y, color, dashArray) {
  if (!values.length) return;
  const path = values.map((v, i) => `${i === 0 ? "M" : "L"} ${x(v.bin).toFixed(2)} ${y(v.freq || 0).toFixed(2)}`).join(" ");
  svg.appendChild(svgEl("path", {
    d: path,
    fill: "none",
    stroke: color,
    "stroke-width": "2.2",
    "stroke-dasharray": dashArray || "",
  }));
}

function aggregateCurve(laneKey, behavior) {
  const lane = laneConfig(laneKey);
  if (!lane) {
    return { lane: laneKey, boundary: null, values: Array.from({ length: store.manifest.bins }, (_, bin) => ({ bin, freq: 0, n: 0, lower: 0, upper: 0 })) };
  }
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
  return state.lanes.find((lane) => lane.id === laneKey) || null;
}

function laneStyle(laneKey) {
  const lane = laneConfig(laneKey);
  if (!lane) return { line: "#334155", band: "rgba(51, 65, 85, 0.1)", dash: "" };
  const index = Math.max(0, state.lanes.findIndex((item) => item.id === laneKey));
  const line = colorForModel(lane.model);
  return { line, band: alphaColor(line, state.lanes.length > 3 ? 0.07 : 0.12), dash: LANE_DASHES[index % LANE_DASHES.length] };
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
  if (["positive", "solved", "safe_response", "high_quality"].includes(outcomeKey)) return "good";
  if (["negative", "failed", "harmful_compliance", "low_quality"].includes(outcomeKey)) return "bad";
  if (outcomeKey === "unknown") return "unknown";
  return "mixed";
}

function laneTitle(laneKey) {
  const lane = laneConfig(laneKey);
  return lane ? `${modelLabel(lane.model)} · ${titleCase(lane.domain)}` : "-";
}

function laneLabel(laneKey) {
  const index = state.lanes.findIndex((lane) => lane.id === laneKey);
  if (index < 0) return "Lane";
  const letter = index < 26 ? String.fromCharCode(65 + index) : `${index + 1}`;
  return `Lane ${letter}`;
}

function lineSwatchStyle(style) {
  return `border-top:3px ${style.dash ? "dashed" : "solid"} ${style.line};background:transparent`;
}

function trackAKey(model, domain, outcome, behavior) {
  return `${model}|${domain}|${outcome}|${behavior}`;
}

function trackAMetric(laneKey, behavior) {
  const lane = laneConfig(laneKey);
  if (!lane) return emptyTrackAMetric({ id: laneKey, model: "", domain: "", outcome: "all" }, behavior);
  return trackAMetricForConfig(lane, behavior);
}

function trackAMetricForConfig(config, behavior) {
  const row = store.trackAIndex?.get(trackAKey(config.model, config.domain, config.outcome, behavior));
  const lane = {
    id: config.id || `${config.model}-${config.domain}-${config.outcome}`,
    model: config.model,
    domain: config.domain,
    outcome: config.outcome,
  };
  if (!row) return emptyTrackAMetric(lane, behavior);
  return {
    lane,
    behavior,
    nTraces: row.n_traces || 0,
    totalCount: row.total_count || 0,
    meanCount: row.mean_count || 0,
    lowerCount: row.lower_count ?? row.mean_count ?? 0,
    upperCount: row.upper_count ?? row.mean_count ?? 0,
    presenceRate: row.presence_rate ?? 0,
    lowerPresence: row.lower_presence ?? row.presence_rate ?? 0,
    upperPresence: row.upper_presence ?? row.presence_rate ?? 0,
  };
}

function emptyTrackAMetric(lane, behavior) {
  return {
    lane,
    behavior,
    nTraces: 0,
    totalCount: 0,
    meanCount: 0,
    lowerCount: 0,
    upperCount: 0,
    presenceRate: 0,
    lowerPresence: 0,
    upperPresence: 0,
  };
}

function trackAFamilyMetric(laneKey, family) {
  const lane = laneConfig(laneKey);
  if (!lane) return emptyTrackAMetric({ id: laneKey, model: "", domain: "", outcome: "all" }, family);
  const row = store.trackAFamilyIndex?.get(trackAKey(lane.model, lane.domain, lane.outcome, family));
  if (row) {
    return {
      lane,
      behavior: family,
      nTraces: row.n_traces || 0,
      totalCount: row.total_count || 0,
      meanCount: row.mean_count || 0,
      lowerCount: row.lower_count ?? row.mean_count ?? 0,
      upperCount: row.upper_count ?? row.mean_count ?? 0,
      presenceRate: row.presence_rate ?? 0,
      lowerPresence: row.lower_presence ?? row.presence_rate ?? 0,
      upperPresence: row.upper_presence ?? row.presence_rate ?? 0,
    };
  }
  const familyBehaviors = store.behaviors.filter((behavior) => behavior.family === family).map((behavior) => behavior.key);
  const metrics = familyBehaviors.map((behavior) => trackAMetric(lane.id, behavior));
  return {
    lane,
    behavior: family,
    nTraces: Math.max(0, ...metrics.map((metric) => metric.nTraces || 0)),
    totalCount: metrics.reduce((sum, metric) => sum + metric.totalCount, 0),
    meanCount: metrics.reduce((sum, metric) => sum + metric.meanCount, 0),
    lowerCount: metrics.reduce((sum, metric) => sum + metric.lowerCount, 0),
    upperCount: metrics.reduce((sum, metric) => sum + metric.upperCount, 0),
    presenceRate: 0,
    lowerPresence: 0,
    upperPresence: 0,
  };
}

function metricSpread(rows, field) {
  const values = rows.filter((row) => row.lane && Number.isFinite(row[field]));
  if (!values.length) return { spread: 0, maxValue: 0, minValue: 0, maxLane: null, minLane: null };
  const max = values.reduce((best, row) => (row[field] > best[field] ? row : best), values[0]);
  const min = values.reduce((best, row) => (row[field] < best[field] ? row : best), values[0]);
  return { spread: max[field] - min[field], maxValue: max[field], minValue: min[field], maxLane: max.lane, minLane: min.lane };
}

function trackAHypotheses(behavior, rows, countSpread, presenceSpread) {
  if (!rows.some((row) => row.nTraces)) return ["No whole-trace count rows match the current lane configuration."];
  const prompts = [
    `${laneLabel(countSpread.maxLane?.id)} has the highest whole-trace ${titleCase(behavior)} volume, exceeding ${laneLabel(countSpread.minLane?.id)} by ${one.format(countSpread.spread)} marks per trace.`,
  ];
  if (presenceSpread.spread > 0.12) {
    prompts.push(`Presence varies by ${signedPct(presenceSpread.spread).replace("+", "")}, so the difference may be about whether traces use this behavior at all.`);
  } else if (countSpread.spread > 0.5) {
    prompts.push("Presence is comparatively stable, so the difference may be intensity among traces that already use the behavior.");
  }
  if (uniqueValues(state.lanes, "model").length > 1 && uniqueValues(state.lanes, "domain").length === 1) {
    prompts.push(`Model hypothesis: lanes share ${titleCase(state.lanes[0].domain)}; compare count volume with the trajectory view to see whether extra behavior is spread throughout the trace or localized.`);
  } else if (uniqueValues(state.lanes, "domain").length > 1 && uniqueValues(state.lanes, "model").length === 1) {
    prompts.push(`Domain hypothesis: ${modelLabel(state.lanes[0].model)} may invoke this behavior more often for some task families even before considering where it occurs.`);
  } else if (uniqueValues(state.lanes, "outcome").length > 1) {
    prompts.push("Outcome hypothesis: static count gaps can separate successful from failed traces even when their temporal shapes look similar.");
  }
  return prompts;
}

function laneStats(curve) {
  const lane = laneConfig(curve.lane);
  return {
    lane,
    atCursor: curve.values[state.bin]?.freq || 0,
    auc: average(curve.values.map((v) => v.freq)),
    early: average(curve.values.filter((v) => v.bin / (store.manifest.bins - 1) <= 0.4).map((v) => v.freq)),
    late: average(curve.values.filter((v) => v.bin / (store.manifest.bins - 1) >= 0.6).map((v) => v.freq)),
  };
}

function curveSpread(curves, bin) {
  const rows = curves
    .map((curve) => ({ lane: laneConfig(curve.lane), value: curve.values[bin]?.freq || 0 }))
    .filter((row) => row.lane);
  if (!rows.length) return { spread: 0, maxValue: 0, minValue: 0, maxLane: null, minLane: null };
  const max = rows.reduce((best, row) => (row.value > best.value ? row : best), rows[0]);
  const min = rows.reduce((best, row) => (row.value < best.value ? row : best), rows[0]);
  return { spread: max.value - min.value, maxValue: max.value, minValue: min.value, maxLane: max.lane, minLane: min.lane };
}

function curveSpreadAtBins(curves, predicate) {
  const spreads = Array.from({ length: store.manifest.bins }, (_, bin) => bin)
    .filter((bin) => predicate({ bin }))
    .map((bin) => curveSpread(curves, bin).spread);
  return average(spreads);
}

function uniqueValues(rows, key) {
  return [...new Set(rows.map((row) => row[key]))].filter((value) => value != null);
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
  $("traceTitle").textContent = `${laneLabel(state.traceLane)} Raw Trace`;
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
  if (!lane) return [];
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
  return `<span class="behavior-chip has-tooltip" data-tooltip="${escapeAttr(detail)}">${titleCase(behavior)}${suffix ? ` ${suffix}` : ""}</span>`;
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

function modelLabel(model) {
  return store.modelLabels?.get(model) || MODEL_DISPLAY_NAMES[model] || titleCase(model);
}

function shortModelName(modelId) {
  return modelId ? modelId.split("/").pop() : "";
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

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "-";
}

function formatSignedNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "-";
  const n = Number(value);
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
}

function formatQ(value) {
  if (!Number.isFinite(value)) return "-";
  if (value === 0) return "<.0001";
  if (value < 0.0001) return "<.0001";
  return value.toFixed(4).replace(/^0/, "");
}

function formatPp(value) {
  if (!Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${Number(value).toFixed(2)}pp`;
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

function promptsForLane(lane) {
  if (!lane) return [];
  return (store.traceSentences || [])
    .filter((t) => t.gen_model === lane.model && t.task_type === lane.domain)
    .slice()
    .sort((a, b) => (a.instance_id < b.instance_id ? -1 : a.instance_id > b.instance_id ? 1 : 0));
}

function currentPromptForLane(lane) {
  const prompts = promptsForLane(lane);
  if (!prompts.length) return { trace: null, index: 0, total: 0 };
  const raw = state.annotationGridIndex[lane.id] || 0;
  const index = ((raw % prompts.length) + prompts.length) % prompts.length; // wrap safely
  return { trace: prompts[index], index, total: prompts.length };
}

function stepAnnotationPrompt(laneId, delta) {
  const lane = laneConfig(laneId);
  if (!lane) return;
  const prompts = promptsForLane(lane);
  if (!prompts.length) return;
  const current = state.annotationGridIndex[laneId] || 0;
  state.annotationGridIndex[laneId] = ((current + delta) % prompts.length + prompts.length) % prompts.length;
  renderTraceAnnotationGrid();
}

function syncFullTraceToggle() {
  const toggle = $("showFullTraceToggle");
  if (toggle) toggle.classList.toggle("active", state.showFullTrace);
}

function annotationSentenceMarkup(a) {
  const primary = a.behaviors?.length ? a.behaviors[0] : null;
  const cls = primary || "no-behavior";
  const num = primary ? BEHAVIOR_NUMBERS[primary] : null;
  return `${num ? `<sup class="annotation-num ${cls}">${num}</sup>` : ""}<span class="trace-annotation-grid-sentence ${cls}">${escapeHtml(a.text)}</span>`;
}

function buildFullTraceHtml(trace) {
  const annotations = trace.annotations || [];
  const sections = [
    { type: "think", label: "Thinking", text: trace.thinking?.text || "" },
    { type: "answer", label: "Answer", text: trace.answer?.text || "" },
  ];

  return sections
    .filter((section) => section.text)
    .map((section) => {
      const sectionAnnotations = annotations.filter((a) =>
        section.type === "think" ? (a.section_type || "think") === "think" : (a.section_type || "answer") !== "think",
      );

      let cursor = 0;
      let html = "";
      sectionAnnotations.forEach((a) => {
        if (!a.text) return;
        const idx = section.text.indexOf(a.text, cursor);
        if (idx === -1) return; // sentence text didn't line up (whitespace/segmentation drift) — leave it unhighlighted
        html += escapeHtml(section.text.slice(cursor, idx)).replace(/\n/g, "<br>");
        html += annotationSentenceMarkup(a);
        cursor = idx + a.text.length;
      });
      html += escapeHtml(section.text.slice(cursor)).replace(/\n/g, "<br>");

      return `
        <div class="trace-annotation-fulltext-section">
          <div class="trace-annotation-fulltext-label">${section.label}</div>
          <p class="trace-annotation-grid-sentences trace-annotation-grid-sentences--full">${html}</p>
        </div>
      `;
    })
    .join("");
}

function renderTraceAnnotationGrid() {
  const target = $("trace-annotation-grid");
  if (!target) return;
  renderTraceAnnotationLegend();
  syncTruncateToggle();
  syncFullTraceToggle();

  target.innerHTML = state.lanes
    .map((lane) => {
      const style = laneStyle(lane.id);
      const { trace, index, total } = currentPromptForLane(lane);
      const nav = total > 1
        ? `
          <div class="trace-nav">
            <span class="trace-nav-count">Prompt ${index + 1} / ${total}</span>
            <button class="icon-button" data-annotation-prev="${escapeAttr(lane.id)}" aria-label="Previous prompt">
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 18l-6-6 6-6" /></svg>
            </button>
            <button class="icon-button" data-annotation-next="${escapeAttr(lane.id)}" aria-label="Next prompt">
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18l6-6-6-6" /></svg>
            </button>
          </div>
        `
        : "";
      const head = `
        <div class="trace-annotation-grid-head">
          <i style="background:${style.line}"></i>
          <strong>${laneLabel(lane.id)}</strong>
          <span>${modelLabel(lane.model)} · ${titleCase(lane.domain)}</span>
          ${nav}
        </div>
      `;

      if (!trace) {
        return `
          <div class="trace-annotation-grid-cell" style="--lane-control-color:${style.line}">
            ${head}
            <div class="empty-state">No sampled prompt for this lane.</div>
          </div>
        `;
      }

      const annotations = trace.annotations || [];
      const behaviorCounts = trace.behavior_counts || {};
      const totalBehaviorCount = Object.values(behaviorCounts).reduce((sum, n) => sum + (n || 0), 0);
      const hasBehaviorCounts = totalBehaviorCount > 0 && annotations.some((a) => a.behaviors?.length);
      const promptBlock = `<p class="trace-annotation-grid-prompt"><span class="prompt-label">Prompt</span>${escapeHtml(trace.prompt?.text || "")}</p>`;

      // No behaviours identified at all — always fall back to full raw text, regardless of toggle.
      if (!annotations.length || !hasBehaviorCounts) {
        const fallbackText = [trace.thinking?.text, trace.answer?.text].filter(Boolean).join("\n\n");
        return `
          <div class="trace-annotation-grid-cell" style="--lane-control-color:${style.line}">
            ${head}
            ${promptBlock}
            <div class="trace-annotation-status">No behaviors identified for this trace.</div>
            <p class="trace-annotation-grid-sentences trace-annotation-grid-sentences--unannotated">
              ${escapeHtml(fallbackText || "No trace text available.").replace(/\n/g, "<br>")}
            </p>
          </div>
        `;
      }

      // Full-text mode: reconstruct thinking/answer with annotated sentences highlighted inline.
      if (state.showFullTrace) {
        return `
          <div class="trace-annotation-grid-cell" style="--lane-control-color:${style.line}">
            ${head}
            ${promptBlock}
            ${buildFullTraceHtml(trace)}
          </div>
        `;
      }

      // Default mode: annotated sentences only (existing behavior).
      const { visible, hiddenCount } = truncateAnnotations(annotations);
      return `
        <div class="trace-annotation-grid-cell" style="--lane-control-color:${style.line}">
          ${head}
          ${promptBlock}
          <p class="trace-annotation-grid-sentences">
            ${visible
              .map((a) =>
                a._truncationMarker
                  ? `<span class="trace-annotation-truncation-marker">⋯ ${hiddenCount} sentence${hiddenCount === 1 ? "" : "s"} hidden ⋯</span>`
                  : annotationSentenceMarkup(a),
              )
              .join("")}
          </p>
        </div>
      `;
    })
    .join("");
  renderLatexInGrid(target);
}



function renderLatexInGrid(target) {
  if (!state.renderLatex || typeof window.renderMathInElement !== "function") return;
  window.renderMathInElement(target, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
}

function syncLatexToggle() {
  const toggle = $("renderLatexToggle");
  if (toggle) toggle.classList.toggle("active", state.renderLatex);
}

function wordCount(text) {
  return (text || "").trim().split(/\s+/).filter(Boolean).length;
}

function truncateAnnotations(annotations, headWordBudget = 75, tailWordBudget = 75) {
  if (!state.truncateAnnotations || !annotations.length) {
    return { visible: annotations, hiddenCount: 0 };
  }

  const counts = annotations.map((a) => wordCount(a.text));

  // Walk forward from the start, keeping whole sentences until the head budget is used up.
  // Always keep at least the first sentence, even if it alone exceeds the budget.
  let headEnd = 0;
  let headWords = 0;
  while (headEnd < annotations.length) {
    const w = counts[headEnd];
    if (headEnd > 0 && headWords + w > headWordBudget) break;
    headWords += w;
    headEnd += 1;
  }

  // Walk backward from the end, keeping whole sentences until the tail budget is used up.
  let tailStart = annotations.length;
  let tailWords = 0;
  while (tailStart > headEnd) {
    const w = counts[tailStart - 1];
    if (tailStart < annotations.length && tailWords + w > tailWordBudget) break;
    tailWords += w;
    tailStart -= 1;
  }

  const hiddenCount = tailStart - headEnd;
  if (hiddenCount <= 0) {
    return { visible: annotations, hiddenCount: 0 };
  }

  const head = annotations.slice(0, headEnd);
  const tail = annotations.slice(tailStart);
  return { visible: [...head, { _truncationMarker: true }, ...tail], hiddenCount };
}

function syncTruncateToggle() {
  const toggle = $("truncateAnnotationsToggle");
  if (toggle) toggle.classList.toggle("active", state.truncateAnnotations);
}

function renderTraceAnnotationLegend() {
  const target = $("trace-annotation-legend");
  if (!target) return;

  const groups = ["cognitive", "conversational"]
    .map((family) => ({
      family,
      meta: FAMILY_META[family],
      behaviors: Object.keys(BEHAVIOR_NUMBERS).filter((behavior) => familyFor(behavior) === family),
    }))
    .filter((group) => group.behaviors.length);

  target.innerHTML = groups
    .map(
      (group) => `
        <div class="trace-annotation-legend-group">
          <div class="trace-annotation-legend-group-title">${escapeHtml(group.meta?.short || titleCase(group.family))}</div>
          ${group.behaviors
            .map(
              (behavior) => `
                <div class="trace-annotation-legend-row">
                  <span class="trace-annotation-legend-swatch" style="background:${BEHAVIOR_COLORS[behavior]}">${BEHAVIOR_NUMBERS[behavior]}</span>
                  <span class="trace-annotation-legend-label has-tooltip" data-tooltip="${escapeAttr(behaviorDescription(behavior))}">${titleCase(behavior)}</span>
                </div>
              `,
            )
            .join("")}
        </div>
      `,
    )
    .join("");
}


loadData().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<main class="panel" style="margin:24px;padding:24px"><h1>Dashboard data failed to load</h1><p>${escapeHtml(error.message)}</p></main>`;
});
