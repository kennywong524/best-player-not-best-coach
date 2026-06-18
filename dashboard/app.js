const state = {
  data: null,
  tab: "project",
  runId: null,
  modelSet: "no_baseline",
  task: "tax_prep",
  mode: "augmentation",
  judge: "aggregate",
  selectedModel: null,
  textTab: "output",
  rubricFocus: null,
};

const taskOrder = ["counselling", "market_trends", "meal_plan", "operations_research", "tax_prep", "travel_planning", "tutoring"];
const taskLabels = {
  counselling: "Counseling",
  market_trends: "Market Trends",
  meal_plan: "Menu Planning",
  operations_research: "Operations Research",
  tax_prep: "Tax Prep",
  travel_planning: "Travel Agent",
  tutoring: "Tutoring",
};
const modeLabels = { augmentation: "Augmentation", automation: "Automation" };
const judgeLabels = {
  aggregate: "Aggregate",
  "gpt-4.1": "GPT-4.1",
  "anthropic/claude-opus-4-8": "Claude-Opus-4.8",
  "deepseek-ai/DeepSeek-V3.1": "DeepSeek-V3.1",
  "google/gemini-3.1-pro": "Gemini-3.1-Pro",
};
function canonicalJudge(name) {
  if (!name) return "";
  const s = String(name).toLowerCase();
  if (s.includes("claude") || s.includes("opus") || s.includes("anthropic")) return "claude";
  if (s.includes("deepseek")) return "deepseek";
  if (s.includes("gemini") || s.includes("google")) return "gemini";
  if (s.includes("gpt") || s.includes("openai") || s.includes("o3") || s.includes("o4")) return "gpt41";
  return s;
}

const canonicalJudgeLabels = {
  claude: "Claude-Opus-4.8",
  deepseek: "DeepSeek-V3.1",
  gemini: "Gemini-3.1-Pro",
  gpt41: "GPT-4.1",
};

function judgeDisplay(name) {
  return judgeLabels[name] || canonicalJudgeLabels[canonicalJudge(name)] || name;
}

const modelShort = {
  "GPT-5-Mini": "G5M",
  "GPT-4.1": "G4.1",
  "GPT-O4-Mini": "O4",
  "GPT-O3-Mini": "O3",
  "GPT-OSS-120B": "OSS",
  "DeepSeek-V3.1": "DS",
  "Claude-Opus-4.8": "Opus",
  "Claude-Sonnet-4.6": "Sonnet",
  "Gemini-3.1-Pro": "Gemini",
  "plain": "Plain",
  "GPT-3.5-Turbo": "3.5",
};
const validationChecks = [
  {
    id: "gpt_ladder_claude",
    title: "GPT-family automation ladder judged by Claude",
    judge: "anthropic/claude-opus-4-8",
    family: "GPT family",
    models: ["GPT-3.5-Turbo", "GPT-O3-Mini", "GPT-O4-Mini", "GPT-4.1", "GPT-5-Mini"],
    expectedBest: "GPT-5-Mini",
    note: "GPT-OSS-120B is omitted because it is an open-weight reference model rather than a clean generation step. Treat the GPT-family ordering as a reference capability gradient, not a perfectly chronological release ladder.",
  },
  {
    id: "claude_pair_gpt",
    title: "Claude-family automation pair judged by GPT-4.1",
    judge: "gpt-4.1",
    family: "Claude family",
    models: ["Claude-Sonnet-4.6", "Claude-Opus-4.8"],
    expectedBest: "Claude-Opus-4.8",
    note: "This is a two-model sanity check rather than a full generation ladder.",
  },
];
const rubricLabels = {
  counselling: {
    task_dimension_1: "Empathy and therapeutic tone",
    task_dimension_2: "Pattern recognition without overdiagnosis",
    task_dimension_3: "Evidence-informed psychological framing",
    task_dimension_4: "Actionable coping and goal-setting",
    task_dimension_5: "Ethical boundaries and escalation",
    task_dimension_6: "Coherence, warmth, and usefulness",
  },
  market_trends: {
    task_dimension_1: "Trend identification",
    task_dimension_2: "Economic mechanism quality",
    task_dimension_3: "Coverage of market drivers",
    task_dimension_4: "Analytical balance and nuance",
    task_dimension_5: "Client usefulness and actionability",
    task_dimension_6: "Conclusion quality",
  },
  meal_plan: {
    task_dimension_1: "Dietary safety",
    task_dimension_2: "Preference fit",
    task_dimension_3: "Affordability",
    task_dimension_4: "Nutritional adequacy and variety",
    task_dimension_5: "Simplicity and usability",
    task_dimension_6: "Completeness",
  },
  operations_research: {
    task_dimension_1: "Data and validation",
    task_dimension_2: "Analytical framework",
    task_dimension_3: "Recommendations",
    task_dimension_4: "Trade-offs and risks",
    task_dimension_5: "KPIs and communication",
    task_dimension_6: "Executive memo quality",
  },
  tax_prep: {
    task_dimension_1: "Rule accuracy",
    task_dimension_2: "Error detection",
    task_dimension_3: "Calculation quality",
    task_dimension_4: "Form guidance",
    task_dimension_5: "Dependent analysis",
    task_dimension_6: "Clear communication",
  },
  travel_planning: {
    task_dimension_1: "Completeness",
    task_dimension_2: "Cost realism and arithmetic",
    task_dimension_3: "Hotel and transport practicality",
    task_dimension_4: "Itinerary quality",
    task_dimension_5: "Travel-agent professionalism",
    task_dimension_6: "Handling uncertainty",
  },
  tutoring: {
    task_dimension_1: "Mathematical correctness",
    task_dimension_2: "Age-appropriate explanation and pacing",
    task_dimension_3: "Analogy quality",
    task_dimension_4: "Misconception handling",
    task_dimension_5: "Classroom practicality and engagement",
    task_dimension_6: "Assessment/checks for understanding",
  },
};
const rubricDetails = {
  counselling: {
    task_dimension_1: "How well the response validates the client's feelings, uses supportive language, and avoids minimizing distress.",
    task_dimension_2: "Whether it cautiously identifies possible patterns such as burnout, perfectionism, avoidance, low self-efficacy, or value mismatch without diagnosing.",
    task_dimension_3: "Whether it uses CBT, motivational interviewing, or positive psychology accurately and in plain language.",
    task_dimension_4: "Whether it gives concrete next steps for regulation, reframing, values clarification, and near-term goals.",
    task_dimension_5: "Whether it includes appropriate limits and encourages professional support if distress is severe or persistent.",
    task_dimension_6: "Whether the full single-session response is coherent, warm, useful, and appropriately bounded.",
  },
  market_trends: {
    task_dimension_1: "Exactly three bullish and exactly three bearish trends, clearly categorized.",
    task_dimension_2: "Causal mechanisms linking each trend to natural-gas prices through supply, demand, storage, LNG, infrastructure, policy, or expectations.",
    task_dimension_3: "Meaningful coverage of supply, demand, weather, and policy/regulation, including interactions among them.",
    task_dimension_4: "Nuance around uncertainty, competing forces, conditional scenarios, weather variability, production response, and policy/geopolitical risk.",
    task_dimension_5: "Usefulness to investors or energy firms through implications for risk management, timing, hedging, capacity, or strategy.",
    task_dimension_6: "A clear 2-3 sentence synthesis of the overall client-facing outlook.",
  },
  meal_plan: {
    task_dimension_1: "Strictly avoids shellfish, gluten, lactose-containing dairy, and onion.",
    task_dimension_2: "Fits meat, potatoes, Mexican, Thai, and Indian flavor preferences while avoiding very salty/sweet foods and strong seasoning.",
    task_dimension_3: "Uses affordable, common grocery-store ingredients.",
    task_dimension_4: "Provides variety and adequate proteins, carbs, fats, vegetables, fruits, and micronutrient coverage.",
    task_dimension_5: "Easy to follow, clearly structured, simple enough for the target user.",
    task_dimension_6: "Includes breakfast, lunch, dinner, snacks for all seven days, grocery list, and prep notes.",
  },
  operations_research: {
    task_dimension_1: "Identifies route, demand, fleet, warehouse, service, and cost data plus collection and validation methods.",
    task_dimension_2: "Uses suitable OR methods such as MILP, simulation, scenario analysis, routing, or allocation optimization.",
    task_dimension_3: "At least two feasible practical solutions with clear lines of action.",
    task_dimension_4: "Explicit cost, service, resource, implementation, and risk trade-offs.",
    task_dimension_5: "Clear KPIs and executive communication plan for both technical and non-technical audiences.",
    task_dimension_6: "Concise, self-contained 300-400 word executive memo with appropriate tone and structure.",
  },
  tax_prep: {
    task_dimension_1: "Correct federal and California tax rules for self-employment income, dependent rules, mortgage interest, deductions, and state/federal differences.",
    task_dimension_2: "Detects the known filing mistakes, including income mismatch, missing schedules, deduction errors, dependent-credit issues, and CA inconsistencies.",
    task_dimension_3: "Provides income, AGI, deductions, taxable income, self-employment tax, or tax-estimate calculations where possible.",
    task_dimension_4: "Gives clear guidance on Schedule C, Schedule SE, Schedule A vs. standard deduction, Form 1040, and California corrections.",
    task_dimension_5: "Handles the 22-year-old full-time student with $18k income cautiously and correctly.",
    task_dimension_6: "Explains dense tax rules in plain English so the client understands what changes and why.",
  },
  travel_planning: {
    task_dimension_1: "Covers clarifying questions, costs, hotels, transport, customs/regulations, itinerary, package logic, and budget confirmation.",
    task_dimension_2: "Flight, lodging, transport, food/activity estimates are plausible and correctly summed without exceeding budget.",
    task_dimension_3: "Provides viable hotel options, airport/transit guidance, and practical rail/pass advice.",
    task_dimension_4: "Five days of morning/afternoon/evening plans that are geographically sensible and fatigue-aware.",
    task_dimension_5: "Clear, customer-focused, travel-agent style with appropriate caveats around estimates.",
    task_dimension_6: "Asks clarifying questions for missing info while still providing a useful provisional plan.",
  },
  tutoring: {
    task_dimension_1: "Mathematically accurate treatment of improper fractions and mixed numbers with no concept/example errors.",
    task_dimension_2: "Clear Grade 3 pacing and language, avoiding abstract notation without visual support.",
    task_dimension_3: "Analogy is concrete and understandable for 8-year-olds.",
    task_dimension_4: "Identifies and corrects common misconceptions, ideally with a memorable check or trick.",
    task_dimension_5: "Immediately usable classroom segment with overt student participation and engagement.",
    task_dimension_6: "Specific formative checks such as thumbs, exit ticket, visual problem, or think-pair-share.",
  },
};
const generalRubricLabels = {
  general_instruction_following: "General: instruction following",
  general_accuracy_specificity: "General: accuracy and specificity",
  general_practical_usefulness: "General: practical usefulness",
  general_organization_readability: "General: organization and readability",
  general_tone_audience_fit: "General: tone and audience fit",
};
const generalRubricDetails = {
  general_instruction_following: "Satisfies explicit requirements and constraints in the task prompt.",
  general_accuracy_specificity: "Avoids false claims, vague filler, and unsupported assumptions.",
  general_practical_usefulness: "Gives concrete, usable guidance or outputs.",
  general_organization_readability: "Clear structure, easy to scan, appropriate formatting.",
  general_tone_audience_fit: "Matches the role, user need, and professional context.",
};

function runList() {
  return state.data?.meta?.replicate_runs || [{ id: state.data?.meta?.run_id, label: state.data?.meta?.run_id || "Run" }];
}

function activeData(runId = state.runId) {
  const id = runId || state.data?.meta?.default_run_id || state.data?.meta?.run_id;
  return state.data?.runs_by_id?.[id] || state.data;
}

function activeRunMeta(runId = state.runId) {
  const id = runId || state.data?.meta?.default_run_id || state.data?.meta?.run_id;
  return runList().find(r => r.id === id) || { id, label: id };
}

function modelAllowed(label) {
  const set = state.data.model_sets[state.modelSet];
  if (set.include) return set.include.includes(label);
  if (set.exclude) return !set.exclude.includes(label);
  return true;
}

function cleanTaskTitle(slug) {
  return taskLabels[slug] || slug;
}

function displayModel(label, mode = state.mode) {
  if (label === "gpt-3.5-turbo") return "GPT-3.5-Turbo";
  if (mode === "augmentation" && label === "plain") return "GPT-3.5-Turbo (plain)";
  return label || "";
}

function isOwnFamily(judge, modelLabel) {
  if (!judge || judge === "aggregate") return false;
  if (judge.startsWith("gpt-") || judge.includes("openai")) {
    return modelLabel === "plain" || modelLabel === "GPT-3.5-Turbo" || modelLabel.startsWith("GPT-");
  }
  if (judge.includes("claude") || judge.includes("anthropic")) {
    return modelLabel.startsWith("Claude-");
  }
  if (judge.includes("DeepSeek")) {
    return modelLabel.startsWith("DeepSeek");
  }
  if (judge.includes("gemini") || judge.includes("google")) {
    return modelLabel.startsWith("Gemini");
  }
  return false;
}

function visibleModels(mode, runId = state.runId) {
  return [...new Set(activeData(runId).aggregate
    .filter(d => d.mode === mode && modelAllowed(d.model_label))
    .map(d => d.model_label))];
}

function num(x) {
  return typeof x === "number" ? x : Number(x);
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function rubricName(dim) {
  return rubricLabels[state.task]?.[dim] || generalRubricLabels[dim] || dim;
}

function rubricTip(dim) {
  return rubricDetails[state.task]?.[dim] || generalRubricDetails[dim] || rubricName(dim);
}

function scoreColor(score) {
  const t = Math.max(0, Math.min(1, Number(score) / 10));
  if (t < 0.5) {
    const f = t / 0.5;
    const r = Math.round(185 + (228 - 185) * f);
    const g = Math.round(75 + (197 - 75) * f);
    const b = Math.round(72 + (90 - 72) * f);
    return `rgb(${r},${g},${b})`;
  }
  const f = (t - 0.5) / 0.5;
  const r = Math.round(228 + (37 - 228) * f);
  const g = Math.round(197 + (127 - 197) * f);
  const b = Math.round(90 + (99 - 90) * f);
  return `rgb(${r},${g},${b})`;
}

function records(mode, judge = "aggregate", runId = state.runId) {
  const bundle = activeData(runId);
  const source = judge === "aggregate" ? bundle.aggregate : bundle.by_judge;
  return source.filter(d =>
    d.mode === mode &&
    modelAllowed(d.model_label) &&
    (judge === "aggregate" || d.judge_model === judge) &&
    !isOwnFamily(judge, d.model_label)
  );
}

function rankOfRanks(mode, judge = "aggregate", runId = state.runId) {
  const rows = [];
  for (const task of taskOrder) {
    const sub = records(mode, judge, runId)
      .filter(d => d.task_slug === task)
      .sort((a, b) => num(a.rank_value) - num(b.rank_value) || num(b.score) - num(a.score) || a.model_label.localeCompare(b.model_label));
    sub.forEach((d, i) => rows.push({ ...d, display_rank: i + 1 }));
  }
  return rows;
}

function averageRanks(judge = "aggregate", runId = state.runId) {
  const out = {};
  for (const mode of ["augmentation", "automation"]) {
    for (const d of rankOfRanks(mode, judge, runId)) {
      out[d.model_label] ||= {};
      out[d.model_label][mode] ||= [];
      out[d.model_label][mode].push(d.display_rank);
    }
  }
  return Object.entries(out).map(([model, v]) => ({
    model,
    augmentation: v.augmentation ? avg(v.augmentation) : null,
    automation: v.automation ? avg(v.automation) : null,
  })).filter(d => d.augmentation !== null && d.automation !== null);
}

function avg(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function uniqueJudgmentCount(bundle) {
  const seen = new Set();
  Object.entries(bundle.runs).forEach(([taskMode, run]) => {
    run.judgments.forEach(j => {
      const a = Math.min(Number(j.left_idx), Number(j.right_idx));
      const b = Math.max(Number(j.left_idx), Number(j.right_idx));
      seen.add(`${taskMode}|${j.judge_model}|${a}|${b}`);
    });
  });
  return seen.size;
}

function heatColor(rank, maxRank) {
  if (!rank) return "#f2f4f7";
  const t = (rank - 1) / Math.max(1, maxRank - 1);
  const stops = [
    [37, 127, 99],
    [210, 232, 207],
    [244, 221, 124],
    [222, 105, 72],
    [158, 35, 42],
  ];
  const p = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(p));
  const f = p - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function renderHeatmap(el, mode) {
  const rows = rankOfRanks(mode, state.judge);
  const models = visibleModels(mode).sort((a, b) => {
    const avs = rows.filter(d => d.model_label === a).map(d => d.display_rank);
    const bvs = rows.filter(d => d.model_label === b).map(d => d.display_rank);
    const av = avs.length ? avg(avs) : 999;
    const bv = bvs.length ? avg(bvs) : 999;
    return av - bv;
  });
  const maxRank = models.length;
  const byKey = new Map(rows.map(d => [`${d.task_slug}|${d.model_label}`, d]));
  let html = `<table class="heat-table"><thead><tr><th>Task</th>${models.map(m => `<th>${displayModel(m, mode)}</th>`).join("")}</tr></thead><tbody>`;
  for (const task of taskOrder) {
    html += `<tr><td>${cleanTaskTitle(task)}</td>`;
    for (const model of models) {
      const d = byKey.get(`${task}|${model}`);
      const r = d?.display_rank;
      if (isOwnFamily(state.judge, model)) {
        html += `<td class="heat-na" title="${displayModel(model, mode)} is excluded from ${judgeLabels[state.judge] || state.judge} judging by leave-family-out.">N/A</td>`;
      } else {
        html += `<td title="${displayModel(model, mode)} / ${cleanTaskTitle(task)}: win rate ${d ? Number(d.score).toFixed(3) : "NA"}" style="background:${heatColor(r, maxRank)};color:${r > maxRank * .72 ? "white" : "#172033"}">${r || ""}</td>`;
      }
    }
    html += `</tr>`;
  }
  html += `<tr><td>Average</td>`;
  for (const model of models) {
    if (isOwnFamily(state.judge, model)) {
      html += `<td class="heat-na" title="Excluded by leave-family-out.">N/A</td>`;
    } else {
      const vals = rows.filter(d => d.model_label === model).map(d => d.display_rank);
      const av = vals.length ? avg(vals) : null;
      html += av ? `<td style="background:${heatColor(av, maxRank)}">${av.toFixed(1)}</td>` : `<td class="heat-na">N/A</td>`;
    }
  }
  html += `</tr></tbody></table>`;
  el.innerHTML = html;
}

function renderRoleScatter() {
  const data = averageRanks(state.judge);
  const size = 400, pad = 50;
  const maxRank = Math.max(...data.flatMap(d => [d.augmentation, d.automation]), 9);
  const x = v => pad + (v - 1) / (maxRank - 1) * (size - pad * 2);
  const y = v => size - pad - (v - 1) / (maxRank - 1) * (size - pad * 2);
  // Alternate label placement (right / left) to reduce overlap.
  const points = data.map((d, i) => {
    const cx = x(d.automation);
    const cy = y(d.augmentation);
    const short = modelShort[d.model] || d.model.slice(0, 6);
    const right = cx < size * 0.7;
    const lx = right ? cx + 9 : cx - 9;
    const anchor = right ? "start" : "end";
    return `<g><circle cx="${cx}" cy="${cy}" r="4.5" fill="#2f6fcb" stroke="white" stroke-width="1.2"><title>${displayModel(d.model)}: automate ${d.automation.toFixed(2)}, augment ${d.augmentation.toFixed(2)}</title></circle><text x="${lx}" y="${cy + 3}" font-size="10" font-weight="700" text-anchor="${anchor}" fill="#172033" stroke="white" stroke-width="2.6" paint-order="stroke" style="stroke-linejoin:round">${short}</text></g>`;
  }).join("");
  const legend = data.map(d => `<span><b>${modelShort[d.model] || d.model}</b> ${displayModel(d.model)}</span>`).join("");
  const ticks = Array.from({ length: Math.round(maxRank) }, (_, i) => i + 1).filter(t => t === 1 || t === Math.round(maxRank) || t % 2 === 0);
  document.getElementById("roleScatter").innerHTML = `<p class="chart-note">Lower-left is better in both modes. Distance from the diagonal indicates role specialization.</p><div class="svg-wrap"><svg viewBox="0 0 ${size} ${size}" role="img">
    <rect x="0" y="0" width="${size}" height="${size}" fill="white"/>
    ${ticks.map(t => `<line x1="${x(t)}" y1="${pad}" x2="${x(t)}" y2="${size-pad}" stroke="#eef1f5"/><line x1="${pad}" y1="${y(t)}" x2="${size-pad}" y2="${y(t)}" stroke="#eef1f5"/><text x="${x(t)}" y="${size-pad+18}" text-anchor="middle" font-size="10" fill="#657083">${t}</text><text x="${pad-12}" y="${y(t)+3}" text-anchor="end" font-size="10" fill="#657083">${t}</text>`).join("")}
    <line x1="${x(1)}" y1="${y(1)}" x2="${x(maxRank)}" y2="${y(maxRank)}" stroke="#9aa3b2" stroke-dasharray="6 5" stroke-width="1.4"/>
    ${points}
    <text x="${size/2}" y="${size-8}" text-anchor="middle" font-size="11" font-weight="700">Automation avg rank</text>
    <text x="15" y="${size/2}" text-anchor="middle" font-size="11" font-weight="700" transform="rotate(-90 15 ${size/2})">Augmentation avg rank</text>
  </svg></div><div class="role-legend">${legend}</div>`;
}

function renderMetrics() {
  const tasks = state.data.tasks.length;
  const bundle = activeData();
  const outputs = Object.values(bundle.runs).reduce((s, r) => s + r.outputs.length, 0);
  const judgments = uniqueJudgmentCount(bundle);
  const models = new Set(bundle.aggregate.map(d => d.model_label)).size;
  document.getElementById("metricRow").innerHTML = [
    [`${activeRunMeta().label}`, "selected run"],
    [`${tasks}`, "tasks"],
    [`${models}`, "candidate conditions"],
    [`${outputs}`, "saved outputs"],
    [`${judgments.toLocaleString()}`, "unique pairwise judgments"],
  ].map(([v, l]) => `<div class="metric"><b>${v}</b><span>${l}</span></div>`).join("");
}

function renderLeaderboard() {
  const rows = rankOfRanks(state.mode, state.judge).filter(d => d.task_slug === state.task);
  document.getElementById("rankTitle").textContent = `${cleanTaskTitle(state.task)} · ${modeLabels[state.mode]} Leaderboard`;
  const maxScore = Math.max(...rows.map(d => Number(d.score)), 1);
  document.getElementById("leaderboard").innerHTML = `<p style="color:var(--muted);font-size:12px;margin:0 0 10px">Rank badges order candidates by average output rank; bars show pairwise win rate. In replicated tasks these can diverge when one model has uneven replicates.</p>` + rows
    .sort((a, b) => a.display_rank - b.display_rank)
    .map(d => {
      const rank = Number(d.display_rank);
      const win = Number(d.score);
      const rawRank = num(d.rank_value);
      return `<div class="bar-row"><div><button class="${state.selectedModel === d.model_label ? "active" : ""}" data-rankmodel="${d.model_label}"><span class="rank-badge ${rank <= 3 ? "top" : ""}">${rank}</span><span>${displayModel(d.model_label, state.mode)}<span class="leader-meta">${modeLabels[state.mode]} · ${cleanTaskTitle(state.task)} · ${state.judge === "aggregate" ? "panel aggregate" : judgeLabels[state.judge] || state.judge}${Number.isFinite(rawRank) ? ` · avg output rank ${rawRank.toFixed(2)}` : ""}</span></span></button></div><div class="bar-track" title="Pairwise win rate"><div class="bar-fill" style="width:${win / maxScore * 100}%;background:${scoreColor(win * 10)}"></div></div><div title="Pairwise win rate">${win.toFixed(2)}</div></div>`;
    })
    .join("");
  document.querySelectorAll("[data-rankmodel]").forEach(b => b.addEventListener("click", () => {
    state.selectedModel = b.dataset.rankmodel;
    renderAll();
  }));
}

function renderRubric() {
  const dims = [
    ...Object.keys(rubricLabels[state.task] || {}),
    ...Object.keys(generalRubricLabels),
  ];
  const run = currentRun();
  const ranked = rankOfRanks(state.mode, state.judge).filter(d => d.task_slug === state.task);
  const allowed = new Set(ranked.map(d => d.model_label));
  const outputs = run.outputs.filter(o => allowed.has(o.model_label));
  const models = [...new Set(outputs.map(d => d.model_label))];
  const model = state.selectedModel && models.includes(state.selectedModel) ? state.selectedModel : models[0];
  const output = outputs.find(o => o.model_label === model);
  const grouped = new Map();
  if (output) {
    run.judgments
      .filter(j => state.judge === "aggregate" || j.judge_model === state.judge)
      .forEach(j => {
        const scores = j.left_idx === output.idx ? j.option_1_scores : (j.right_idx === output.idx ? j.option_2_scores : null);
        if (!scores) return;
        dims.forEach(dim => {
          if (scores[dim] === undefined || scores[dim] === null) return;
          const arr = grouped.get(dim) || [];
          arr.push(Number(scores[dim]));
          grouped.set(dim, arr);
        });
      });
  }
  const sub = dims
    .filter(d => grouped.has(d))
    .map(d => ({ dimension: d, mean_score: avg(grouped.get(d)) }));
  const max = 10;
  const nScores = grouped.size && sub.length ? grouped.get(sub[0].dimension)?.length || 0 : 0;
  const intro = `<h3>${displayModel(model || "", state.mode)}</h3><p style="color:var(--muted);font-size:12px;margin:6px 0 12px">Click any leaderboard model to inspect that selected response's rubric profile. Values average its pairwise appearances across ${state.judge === "aggregate" ? "included judges" : judgeLabels[state.judge] || state.judge}${nScores ? ` (n=${nScores})` : ""}.</p>`;
  const body = sub.length
    ? sub.map(d => `<div class="bar-row" data-rubricdim="${d.dimension}"><div><span class="rubric-label">${esc(rubricName(d.dimension))}</span></div><div class="bar-track"><div class="bar-fill" style="width:${Number(d.mean_score) / max * 100}%;background:${scoreColor(d.mean_score)}"></div></div><div>${Number(d.mean_score).toFixed(1)}</div></div>`).join("")
    : `<p style="color:var(--muted);font-size:13px">No rubric-score rows found for this response under the current judge filter. Try Aggregate or another judge.</p>`;
  const focus = state.rubricFocus && sub.some(d => d.dimension === state.rubricFocus) ? state.rubricFocus : sub[0]?.dimension;
  document.getElementById("rubricChart").innerHTML = intro + body + (focus ? `<div class="rubric-detail"><b>${esc(rubricName(focus))}</b><p>${esc(rubricTip(focus))}</p></div>` : "");
  document.querySelectorAll("[data-rubricdim]").forEach(el => {
    const setFocus = () => {
      state.rubricFocus = el.dataset.rubricdim;
      const detail = document.querySelector("#rubricChart .rubric-detail");
      if (detail) detail.innerHTML = `<b>${esc(rubricName(state.rubricFocus))}</b><p>${esc(rubricTip(state.rubricFocus))}</p>`;
    };
    el.addEventListener("mouseenter", setFocus);
    el.addEventListener("click", setFocus);
  });
}

function renderJudgeScatter() {
  const bundle = activeData();
  const judges = [...new Set(bundle.by_judge.map(d => d.judge_model))].filter(Boolean);
  const pairs = [];
  for (let i = 0; i < judges.length; i++) {
    for (let j = i + 1; j < judges.length; j++) pairs.push([judges[i], judges[j]]);
  }
  const size = 420, pad = 50, maxRank = 10;
  const x = v => pad + (v - 1) / (maxRank - 1) * (size - pad * 2);
  const y = v => size - pad - (v - 1) / (maxRank - 1) * (size - pad * 2);
  const stat = (pts, a, b) => {
    const diffs = pts.map(d => Math.abs(Number(d[a]) - Number(d[b])));
    const within1 = diffs.filter(d => d <= 1).length / Math.max(1, diffs.length);
    const meanDiff = diffs.reduce((s, d) => s + d, 0) / Math.max(1, diffs.length);
    const ca = canonicalJudge(a);
    const cb = canonicalJudge(b);
    const corr = bundle.correlations.find(d => d.scope === "all" && d.method === "spearman" && (
      (canonicalJudge(d.judge_a) === ca && canonicalJudge(d.judge_b) === cb) ||
      (canonicalJudge(d.judge_a) === cb && canonicalJudge(d.judge_b) === ca)
    ));
    return { within1, meanDiff, rho: corr ? Number(corr.correlation) : null };
  };
  const cardHtml = pairs.map(([a, b]) => {
    const pts = bundle.scatter_points.filter(d => d[a] !== null && d[b] !== null && modelAllowed(d.model_label));
    const s = stat(pts, a, b);
    return `<div class="metric"><b>${s.rho === null ? "NA" : s.rho.toFixed(2)}</b><span>${judgeLabels[a]} × ${judgeLabels[b]} rank correlation<br>${Math.round(s.within1 * 100)}% close calls within 1 rank<br>average rank gap ${s.meanDiff.toFixed(1)}</span></div>`;
  }).join("");
  document.getElementById("judgeAgreementCards").innerHTML = cardHtml;
  const panels = pairs.map(([a, b]) => {
    const pts = bundle.scatter_points.filter(d => d[a] !== null && d[b] !== null && modelAllowed(d.model_label));
    return `<svg viewBox="0 0 ${size} ${size}">
      <rect width="${size}" height="${size}" fill="white"/>
      <text x="${size/2}" y="24" text-anchor="middle" font-weight="700">${judgeLabels[a]} vs ${judgeLabels[b]}</text>
      <text x="${size/2}" y="42" text-anchor="middle" font-size="11" fill="#657083">One point = one model in one task/mode</text>
      ${[1,5,10].map(t => `<text x="${x(t)}" y="${size-pad+22}" text-anchor="middle" font-size="11">${t}</text><text x="${pad-12}" y="${y(t)+4}" text-anchor="end" font-size="11">${t}</text>`).join("")}
      ${Array.from({ length: 10 }, (_, i) => i + 1).map(t => `<line x1="${x(t)}" y1="${pad}" x2="${x(t)}" y2="${size-pad}" stroke="#e6e9ee"/><line x1="${pad}" y1="${y(t)}" x2="${size-pad}" y2="${y(t)}" stroke="#e6e9ee"/>`).join("")}
      <polygon points="${x(1)},${y(2)} ${x(9)},${y(10)} ${x(10)},${y(9)} ${x(2)},${y(1)}" fill="#2f6fcb" opacity=".08"/>
      <line x1="${x(1)}" y1="${y(1)}" x2="${x(10)}" y2="${y(10)}" stroke="#111" stroke-dasharray="7 6"/>
      ${pts.map((d, i) => `<circle cx="${x(d[a]) + ((i % 7) - 3) * 1.8}" cy="${y(d[b]) + (((i / 7) | 0) % 7 - 3) * 1.8}" r="3.8" fill="${d.mode === "augmentation" ? "#2f6fcb" : "#d96f31"}" opacity=".52"><title>${cleanTaskTitle(d.task)} ${d.mode}: ${displayModel(d.model_label, d.mode)}</title></circle>`).join("")}
      <text x="${size/2}" y="${size-10}" text-anchor="middle" font-size="12">${judgeLabels[a]} rank</text>
      <text x="14" y="${size/2}" text-anchor="middle" font-size="12" transform="rotate(-90 14 ${size/2})">${judgeLabels[b]} rank</text>
    </svg>`;
  }).join("");
  document.getElementById("judgeScatter").innerHTML = `<p style="color:var(--muted);font-size:13px;margin:0 0 12px">Dots near the dashed diagonal mean the two judges assigned similar ranks. The pale blue band marks rankings within one rank of each other. Blue dots are augmentation entries; orange dots are automation entries.</p><div class="two-col judge-grid">${panels}</div><h3 style="margin-top:18px">Task-Level Agreement Summary</h3><div id="judgeAgreementTable"></div><h3 style="margin-top:18px">Largest Individual Rank Disagreements</h3><div id="judgeDisagreementTable"></div>`;
  renderJudgeAgreementTable(pairs);
  renderJudgeDisagreementTable(pairs);
  renderCorrTable();
}

function renderJudgeAgreementTable(pairs) {
  const bundle = activeData();
  const rows = [];
  for (const task of taskOrder) {
    for (const mode of ["augmentation", "automation"]) {
      const cells = pairs.map(([a, b]) => {
        const pts = bundle.scatter_points.filter(d => d.task === task && d.mode === mode && d[a] !== null && d[b] !== null && modelAllowed(d.model_label));
        if (!pts.length) return "N/A";
        const diffs = pts.map(d => Math.abs(Number(d[a]) - Number(d[b])));
        const meanGap = diffs.reduce((s, d) => s + d, 0) / diffs.length;
        const within1 = diffs.filter(d => d <= 1).length / diffs.length;
        return `${meanGap.toFixed(1)} gap / ${Math.round(within1 * 100)}% close`;
      });
      rows.push({ task, mode, cells });
    }
  }
  const headers = pairs.map(([a, b]) => `${judgeLabels[a]} × ${judgeLabels[b]}`);
  document.getElementById("judgeAgreementTable").innerHTML = `<p style="color:var(--muted);font-size:12px;margin:6px 0 10px">Each cell reports average absolute rank difference, then the share of model ranks within one position of each other. Lower gap and higher close-share indicate better agreement.</p><table class="heat-table"><thead><tr><th>Task / Mode</th>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr><td>${cleanTaskTitle(r.task)} / ${modeLabels[r.mode]}</td>${r.cells.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function renderJudgeDisagreementTable(pairs) {
  const bundle = activeData();
  const rows = [];
  for (const [a, b] of pairs) {
    bundle.scatter_points
      .filter(d => d[a] !== null && d[b] !== null && modelAllowed(d.model_label))
      .forEach(d => {
        rows.push({
          pair: `${judgeLabels[a]} × ${judgeLabels[b]}`,
          task: d.task,
          mode: d.mode,
          model: d.model_label,
          aRank: Number(d[a]),
          bRank: Number(d[b]),
          gap: Math.abs(Number(d[a]) - Number(d[b])),
        });
      });
  }
  rows.sort((a, b) => b.gap - a.gap || a.task.localeCompare(b.task));
  const top = rows.slice(0, 12);
  document.getElementById("judgeDisagreementTable").innerHTML = `<p style="color:var(--muted);font-size:12px;margin:6px 0 10px">These are the individual model-task-mode entries where two judges disagreed most about rank. Useful for targeted qualitative audit.</p><table class="heat-table"><thead><tr><th>Pair</th><th>Task / Mode</th><th>Model</th><th>Ranks</th><th>Gap</th></tr></thead><tbody>${top.map(r => `<tr><td>${r.pair}</td><td>${cleanTaskTitle(r.task)} / ${modeLabels[r.mode]}</td><td>${displayModel(r.model, r.mode)}</td><td>${r.aRank} vs ${r.bRank}</td><td>${r.gap}</td></tr>`).join("")}</tbody></table>`;
}

function renderCorrTable() {
  const rows = activeData().correlations.filter(d => d.method === "spearman");
  document.getElementById("corrTable").innerHTML = `<p style="color:var(--muted);font-size:12px;margin:0 0 10px">Spearman is a rank-order correlation. It is computed only on entries both judges were eligible to score after leave-family-out exclusions.</p><table class="heat-table"><thead><tr><th>Scope</th><th>Pair</th><th>Spearman</th><th>Shared ranks</th></tr></thead><tbody>${rows.map(d => `<tr><td>${d.scope}</td><td>${judgeDisplay(d.judge_a)} × ${judgeDisplay(d.judge_b)}</td><td>${Number(d.correlation).toFixed(3)}</td><td>${d.n_pairs}</td></tr>`).join("")}</tbody></table>`;
}

function std(xs) {
  if (!xs.length) return 0;
  const m = avg(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length);
}

function sem(xs) {
  if (!xs.length) return 0;
  return std(xs) / Math.sqrt(xs.length);
}

function replicateRankRows(mode, judge = state.judge) {
  return runList().flatMap(run => rankOfRanks(mode, judge, run.id).map(d => ({
    ...d,
    run_id: run.id,
    run_label: run.label,
  })));
}

function replicateCellStats(mode, judge = state.judge) {
  const rows = replicateRankRows(mode, judge);
  const byCell = new Map();
  rows.forEach(d => {
    const key = `${d.task_slug}|${d.model_label}`;
    const arr = byCell.get(key) || [];
    arr.push(Number(d.display_rank));
    byCell.set(key, arr);
  });
  return { rows, byCell };
}

function renderReplicateHeatmap(el, mode) {
  const { rows, byCell } = replicateCellStats(mode);
  const models = [...new Set(rows.map(d => d.model_label))].sort((a, b) => {
    const av = avg(rows.filter(d => d.model_label === a).map(d => Number(d.display_rank)));
    const bv = avg(rows.filter(d => d.model_label === b).map(d => Number(d.display_rank)));
    return av - bv;
  });
  const maxRank = Math.max(models.length, 1);
  let html = `<table class="heat-table replicate-heat"><thead><tr><th>Task</th>${models.map(m => `<th>${displayModel(m, mode)}</th>`).join("")}</tr></thead><tbody>`;
  for (const task of taskOrder) {
    html += `<tr><td>${cleanTaskTitle(task)}</td>`;
    for (const model of models) {
      const vals = byCell.get(`${task}|${model}`) || [];
      if (!vals.length) {
        html += `<td class="heat-na">N/A</td>`;
      } else {
        const m = avg(vals);
        const se = sem(vals);
        html += `<td title="${displayModel(model, mode)} / ${cleanTaskTitle(task)} across ${vals.length} runs (mean rank ± standard error)" style="background:${heatColor(m, maxRank)};color:${m > maxRank * .72 ? "white" : "#172033"}"><b>${m.toFixed(1)}</b><small>±${se.toFixed(1)}</small></td>`;
      }
    }
    html += `</tr>`;
  }
  html += `<tr><td>Average</td>`;
  for (const model of models) {
    const vals = rows.filter(d => d.model_label === model).map(d => Number(d.display_rank));
    const m = vals.length ? avg(vals) : null;
    const se = vals.length ? sem(vals) : null;
    html += m === null ? `<td class="heat-na">N/A</td>` : `<td title="mean rank ± standard error across ${vals.length} run-task cells" style="background:${heatColor(m, maxRank)}"><b>${m.toFixed(1)}</b><small>±${se.toFixed(1)}</small></td>`;
  }
  html += `</tr></tbody></table>`;
  el.innerHTML = html;
}

function replicateModelStats(mode) {
  const rows = replicateRankRows(mode);
  const models = [...new Set(rows.map(d => d.model_label))];
  return models.map(model => {
    const perRun = runList().map(run => {
      const vals = rows.filter(d => d.model_label === model && d.run_id === run.id).map(d => Number(d.display_rank));
      return { run: run.label, value: vals.length ? avg(vals) : null };
    }).filter(d => d.value !== null);
    const allTaskRanks = rows.filter(d => d.model_label === model).map(d => Number(d.display_rank));
    const values = perRun.map(d => d.value);
    return {
      model,
      mean: values.length ? avg(values) : null,
      sd: values.length ? std(values) : null,
      best: values.length ? Math.min(...values) : null,
      worst: values.length ? Math.max(...values) : null,
      top3: allTaskRanks.length ? allTaskRanks.filter(x => x <= 3).length / allTaskRanks.length : null,
      perRun,
    };
  }).filter(d => d.mean !== null).sort((a, b) => a.mean - b.mean);
}

function renderReplicateStability(el, mode) {
  const stats = replicateModelStats(mode);
  el.innerHTML = `<table class="heat-table stability-table"><thead><tr><th>Model</th><th>Mean avg rank</th><th>SD</th><th>Best run</th><th>Worst run</th><th>Top-3 task cells</th><th>Per-run averages</th></tr></thead><tbody>${stats.map(d => `<tr><td>${displayModel(d.model, mode)}</td><td>${d.mean.toFixed(2)}</td><td>${d.sd.toFixed(2)}</td><td>${d.best.toFixed(2)}</td><td>${d.worst.toFixed(2)}</td><td>${Math.round(d.top3 * 100)}%</td><td>${d.perRun.map(r => `${r.run}: ${r.value.toFixed(2)}`).join(" · ")}</td></tr>`).join("")}</tbody></table>`;
}

function renderReplicateSummary() {
  renderReplicateHeatmap(document.getElementById("repHeatAug"), "augmentation");
  renderReplicateHeatmap(document.getElementById("repHeatAuto"), "automation");
  renderReplicateStability(document.getElementById("repStabilityAug"), "augmentation");
  renderReplicateStability(document.getElementById("repStabilityAuto"), "automation");
  const aug = replicateModelStats("augmentation");
  const auto = replicateModelStats("automation");
  const stable = xs => xs.length ? avg(xs.map(d => d.sd)).toFixed(2) : "NA";
  document.getElementById("replicateMetrics").innerHTML = [
    [`${runList().length}`, "replicate runs"],
    [displayModel(aug[0]?.model || "", "augmentation"), "best mean augmentation rank"],
    [displayModel(auto[0]?.model || "", "automation"), "best mean automation rank"],
    [`${stable(aug)} / ${stable(auto)}`, "mean rank SD: aug / auto"],
  ].map(([v, l]) => `<div class="metric"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join("");
}

function validationRows(check) {
  const rows = [];
  for (const run of runList()) {
    const bundle = activeData(run.id);
    for (const task of taskOrder) {
      const sub = bundle.by_judge
        .filter(d => d.mode === "automation" && d.task_slug === task && d.judge_model === check.judge && check.models.includes(d.model_label))
        .filter(d => Number.isFinite(Number(d.rank_value)) && Number.isFinite(Number(d.score)))
        .sort((a, b) => Number(a.rank_value) - Number(b.rank_value) || Number(b.score) - Number(a.score) || check.models.indexOf(a.model_label) - check.models.indexOf(b.model_label));
      sub.forEach((d, i) => rows.push({
        run_id: run.id,
        run_label: run.label,
        task: task,
        model: d.model_label,
        score: Number(d.score),
        source_rank: Number(d.rank_value),
        family_rank: i + 1,
      }));
    }
  }
  return rows;
}

function validationOrderStats(check, rows) {
  let wins = 0;
  let ties = 0;
  let total = 0;
  for (const run of runList()) {
    for (const task of taskOrder) {
      const sub = rows.filter(d => d.run_id === run.id && d.task === task);
      const rankByModel = new Map(sub.map(d => [d.model, d.family_rank]));
      for (let i = 0; i < check.models.length; i++) {
        for (let j = i + 1; j < check.models.length; j++) {
          const older = check.models[i];
          const newer = check.models[j];
          if (!rankByModel.has(older) || !rankByModel.has(newer)) continue;
          total += 1;
          const olderRank = rankByModel.get(older);
          const newerRank = rankByModel.get(newer);
          if (newerRank < olderRank) wins += 1;
          else if (newerRank === olderRank) ties += 1;
        }
      }
    }
  }
  return {
    wins,
    ties,
    total,
    agreement: total ? (wins + ties * 0.5) / total : null,
  };
}

function validationModelStats(check, rows) {
  return check.models.map(model => {
    const sub = rows.filter(d => d.model === model);
    const ranks = sub.map(d => d.family_rank);
    const scores = sub.map(d => d.score);
    return {
      model,
      meanRank: ranks.length ? avg(ranks) : null,
      sdRank: ranks.length ? std(ranks) : null,
      meanScore: scores.length ? avg(scores) : null,
      n: sub.length,
    };
  }).filter(d => d.n);
}

function renderValidationHeat(check, rows) {
  const maxRank = check.models.length;
  const byCell = new Map();
  rows.forEach(d => {
    const key = `${d.task}|${d.model}`;
    const arr = byCell.get(key) || [];
    arr.push(d.family_rank);
    byCell.set(key, arr);
  });
  let html = `<table class="heat-table validation-heat"><thead><tr><th>Task</th>${check.models.map(m => `<th>${displayModel(m, "automation")}</th>`).join("")}</tr></thead><tbody>`;
  for (const task of taskOrder) {
    html += `<tr><td>${cleanTaskTitle(task)}</td>`;
    for (const model of check.models) {
      const vals = byCell.get(`${task}|${model}`) || [];
      if (!vals.length) {
        html += `<td class="heat-na">N/A</td>`;
      } else {
        const m = avg(vals);
        const s = std(vals);
        html += `<td style="background:${heatColor(m, maxRank)};color:${m > maxRank * .72 ? "white" : "#172033"}" title="${displayModel(model, "automation")} mean within-family rank across ${vals.length} runs"><b>${m.toFixed(1)}</b><small>±${s.toFixed(1)}</small></td>`;
      }
    }
    html += `</tr>`;
  }
  html += `</tbody></table>`;
  return html;
}

function renderValidationCheck(check) {
  const rows = validationRows(check);
  const order = validationOrderStats(check, rows);
  const models = validationModelStats(check, rows);
  const best = models.slice().sort((a, b) => a.meanRank - b.meanRank || b.meanScore - a.meanScore)[0];
  const modelTable = `<table class="heat-table stability-table validation-models"><thead><tr><th>Reference order</th><th>Model</th><th>Mean family rank</th><th>Rank SD</th><th>Mean win rate</th><th>Cells</th></tr></thead><tbody>${models.map((d, i) => `<tr><td>${i + 1}</td><td>${displayModel(d.model, "automation")}</td><td>${d.meanRank.toFixed(2)}</td><td>${d.sdRank.toFixed(2)}</td><td>${d.meanScore.toFixed(2)}</td><td>${d.n}</td></tr>`).join("")}</tbody></table>`;
  return `<div class="viz-card validation-card">
    <div class="validation-head">
      <div>
        <h3>${esc(check.title)}</h3>
        <p>Judge: <b>${esc(judgeLabels[check.judge] || check.judge)}</b> · Mode: <b>Automation only</b> · Runs: <b>${runList().map(r => r.label).join(", ")}</b></p>
      </div>
      <span class="validation-pill">${esc(check.family)}</span>
    </div>
    <div class="metric-row validation-metrics">
      <div class="metric"><b>${order.agreement === null ? "NA" : `${Math.round(order.agreement * 100)}%`}</b><span>expected-order agreement</span></div>
      <div class="metric"><b>${order.wins}/${order.total}</b><span>strict ordered-pair wins</span></div>
      <div class="metric"><b>${esc(displayModel(best?.model || "", "automation"))}</b><span>best mean family rank</span></div>
      <div class="metric"><b>${esc(displayModel(check.expectedBest, "automation"))}</b><span>expected strongest model</span></div>
    </div>
    <p class="validation-note">${esc(check.note)}</p>
    <div class="validation-cols">
      <div class="validation-table-block">
        <h3>Mean Within-Family Rank By Task</h3>
        ${renderValidationHeat(check, rows)}
      </div>
      <div class="validation-table-block">
        <h3>Model-Level Summary</h3>
        ${modelTable}
      </div>
    </div>
  </div>`;
}

function renderValidation() {
  const el = document.getElementById("validationChecks");
  if (!el) return;
  el.innerHTML = validationChecks.map(renderValidationCheck).join("");
}

function renderProjectStats() {
  const bundles = runList().map(r => activeData(r.id));
  const outputs = bundles.reduce((s, b) => s + Object.values(b.runs).reduce((x, r) => x + r.outputs.length, 0), 0);
  const judgments = bundles.reduce((s, b) => s + uniqueJudgmentCount(b), 0);
  const models = new Set(bundles.flatMap(b => b.aggregate.map(d => d.model_label))).size;
  document.getElementById("projectStats").innerHTML = [
    [`${runList().length}`, "replicate runs"],
    [`${state.data.tasks.length}`, "tasks"],
    [`${models}`, "candidate conditions"],
    [`${outputs.toLocaleString()}`, "outputs"],
    [`${judgments.toLocaleString()}`, "unique judgments"],
  ].map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join("");
}

function runKey() {
  return `${state.task}/${state.mode}`;
}

function currentRun() {
  return activeData().runs[runKey()];
}

function renderQualitative() {
  const run = currentRun();
  const ranked = rankOfRanks(state.mode, state.judge).filter(d => d.task_slug === state.task);
  const allowed = new Set(ranked.map(d => d.model_label));
  const outputs = run.outputs.filter(o => allowed.has(o.model_label));
  if (!state.selectedModel || !outputs.some(o => o.model_label === state.selectedModel)) state.selectedModel = outputs[0]?.model_label;
  document.getElementById("modelList").innerHTML = outputs.map(o => {
    const r = ranked.find(d => d.model_label === o.model_label);
    const role = state.mode === "augmentation"
      ? `assistant/scaffold: ${displayModel(o.assistant_model || o.model_label, state.mode)} · worker: ${displayModel(o.worker_model || "gpt-3.5-turbo", state.mode)}`
      : `worker/direct solver: ${displayModel(o.model_label, state.mode)}`;
    return `<button class="model-button ${o.model_label === state.selectedModel ? "active" : ""}" data-model="${o.model_label}"><span>${displayModel(o.model_label, state.mode)}<small class="model-role">${esc(role)}</small></span><span>rank ${r?.display_rank || "?"}</span></button>`;
  }).join("");
  document.querySelectorAll(".model-button").forEach(b => b.addEventListener("click", () => {
    state.selectedModel = b.dataset.model;
    renderAll();
  }));
  const out = outputs.find(o => o.model_label === state.selectedModel) || outputs[0];
  document.getElementById("qualTitle").textContent = out ? `${cleanTaskTitle(state.task)} · ${modeLabels[state.mode]} · ${displayModel(out.model_label, state.mode)}` : "No output";
  const assistant = state.mode === "augmentation" ? displayModel(out?.assistant_model || out?.model_label, state.mode) : "N/A";
  const worker = state.mode === "augmentation" ? displayModel(out?.worker_model || "gpt-3.5-turbo", state.mode) : displayModel(out?.model_label, state.mode);
  const condition = out?.condition || "";
  document.getElementById("roleStrip").innerHTML = out ? [
    ["Mode", modeLabels[state.mode]],
    [state.mode === "augmentation" ? "Assistant / Scaffold Model" : "Assistant", assistant],
    [state.mode === "augmentation" ? "Worker / Final Output Model" : "Worker / Direct Solver", worker],
    ["Condition", condition],
  ].map(([k, v]) => `<div class="role-chip"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join("") : "";
  let text = "";
  if (state.textTab === "output") text = out?.output || "";
  if (state.textTab === "scaffold") text = out?.scaffold_text || "No scaffold in automation/plain condition.";
  if (state.textTab === "scaffoldPrompt") {
    const task = state.data.tasks.find(t => t.slug === state.task);
    text = `SCAFFOLD PROMPT USED FOR THIS TASK\n\n${task?.scaffold_prompt || "No scaffold prompt found in dashboard data."}\n\n\nWORKER INSTRUCTION THAT CONSUMES THE SCAFFOLD\n\n${task?.worker_instruction || "No worker instruction found in dashboard data."}`;
  }
  if (state.textTab === "prompt") {
    const task = state.data.tasks.find(t => t.slug === state.task);
    text = `TASK PROMPT\n\n${task?.task_prompt || ""}\n\n\nSCAFFOLD PROMPT USED IN AUGMENTATION\n\n${task?.scaffold_prompt || "No scaffold prompt found in dashboard data."}\n\n\nWORKER INSTRUCTION\n\n${task?.worker_instruction || "No worker instruction found in dashboard data."}\n\n\nJUDGE RUBRIC / PROMPT\n\n${task?.rubric || ""}`;
  }
  document.getElementById("qualText").textContent = text;
  renderRationales(out);
}

function renderRationales(out) {
  if (!out) {
    document.getElementById("rationales").innerHTML = "";
    return;
  }
  const run = currentRun();
  const byIdx = new Map(run.outputs.map(o => [o.idx, o]));
  const rows = run.judgments
    .filter(j => j.left_idx === out.idx || j.right_idx === out.idx)
    .filter(j => state.judge === "aggregate" || j.judge_model === state.judge)
    .slice(0, 24);
  const dims = [
    ...Object.keys(rubricLabels[state.task] || {}),
    ...Object.keys(generalRubricLabels),
  ];
  const scoreTable = scores => `<div class="score-inline"><table class="mini-score">${dims
    .filter(dim => scores && scores[dim] !== undefined && scores[dim] !== null)
    .map(dim => {
      const score = Number(scores[dim]);
      return `<tr><td><span class="rubric-label" title="${esc(rubricTip(dim))}">${esc(rubricName(dim))}</span></td><td><div class="mini-track"><div class="mini-fill" style="width:${score * 10}%;background:${scoreColor(score)}"></div></div></td><td>${score.toFixed(1)}</td></tr>`;
    })
    .join("")}</table></div>`;
  document.getElementById("rationales").innerHTML = rows.map(j => {
    const left = byIdx.get(j.left_idx);
    const right = byIdx.get(j.right_idx);
    const selected = j.winner === "option_1" ? left : right;
    const leftWinner = selected?.idx === left?.idx;
    const rightWinner = selected?.idx === right?.idx;
    return `<div class="rationale">
      <div class="meta">${j.judge_label} · ${displayModel(left?.model_label, state.mode)} vs ${displayModel(right?.model_label, state.mode)}</div>
      <div class="contestants">
        <details class="contestant-detail ${leftWinner ? "winner" : ""}" ${leftWinner ? "open" : ""}>
          <summary><span>Option 1: ${esc(displayModel(left?.model_label, state.mode))}</span><span class="score-chip">avg ${Number(j.option_1_average).toFixed(1)}</span></summary>
          ${scoreTable(j.option_1_scores)}
        </details>
        <details class="contestant-detail ${rightWinner ? "winner" : ""}" ${rightWinner ? "open" : ""}>
          <summary><span>Option 2: ${esc(displayModel(right?.model_label, state.mode))}</span><span class="score-chip">avg ${Number(j.option_2_average).toFixed(1)}</span></summary>
          ${scoreTable(j.option_2_scores)}
        </details>
      </div>
      <p>${esc(j.short_rationale)}</p>
    </div>`;
  }).join("");
}

function populateControls() {
  const runSelect = document.getElementById("runSelect");
  runSelect.innerHTML = runList().map(r => `<option value="${r.id}">${r.label}</option>`).join("");
  runSelect.value = state.runId;
  const modelSet = document.getElementById("modelSet");
  modelSet.innerHTML = Object.entries(state.data.model_sets).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join("");
  modelSet.value = state.modelSet;
  document.getElementById("taskSelect").innerHTML = taskOrder.map(t => `<option value="${t}">${cleanTaskTitle(t)}</option>`).join("");
  document.getElementById("taskSelect").value = state.task;
  document.getElementById("modeSelect").value = state.mode;
  const judges = ["aggregate", ...new Set(activeData().by_judge.map(d => d.judge_model))];
  document.getElementById("judgeSelect").innerHTML = judges.map(j => `<option value="${j}">${judgeLabels[j] || j}</option>`).join("");
  if (!judges.includes(state.judge)) state.judge = "aggregate";
  document.getElementById("judgeSelect").value = state.judge;
}

const methodologyDetails = {
  input: {
    title: "Input: seven professional tasks",
    body: "Every condition starts from the same fixed task prompt for each of seven professional tasks: counseling, market trends analysis, weekly menu planning, operations research, tax preparation, travel planning, and tutoring. Prompts, rubrics, and model rosters are versioned in task YAML files, so every model sees identical inputs.",
    action: { label: "Read the exact task prompts", run: () => { state.textTab = "prompt"; syncTextTabs(); goTab("qualitative"); renderAll(); } },
  },
  worker: {
    title: "Worker model: the fixed executor",
    body: "In augmentation, a single low-cost worker — GPT-3.5-Turbo — always produces the final deliverable. Because the worker never changes, the only thing that varies between augmentation conditions is the guidance it receives, which isolates the value added by each assistant's scaffold. A plain, no-scaffold worker run serves as the baseline.",
    action: { label: "See worker deliverables", run: () => { setMode("augmentation"); state.textTab = "output"; syncTextTabs(); goTab("qualitative"); renderAll(); } },
  },
  assistant: {
    title: "Assistant model: the model under test",
    body: "Each frontier model writes a process-only scaffold — a 'Three-Phase Workflow' of roughly 200-250 words covering requirements checks, planning, and self-review. Scaffolds are validated automatically (no task content leakage, no stubs, length caps) and regenerated when they fail. The scaffold, not the assistant's own answer, is what reaches the worker.",
    action: { label: "Browse real scaffolds", run: () => { setMode("augmentation"); state.textTab = "scaffold"; syncTextTabs(); goTab("qualitative"); renderAll(); } },
  },
  automation: {
    title: "Automation regime: the model solves alone",
    body: "Each focal model receives the task prompt directly and produces the deliverable end-to-end. This measures innate capability: no scaffold, no intermediary. These outputs then compete against each other in the automation tournament.",
    action: { label: "View automation rankings", run: () => { setMode("automation"); goTab("rankings"); renderAll(); } },
  },
  augmentation: {
    title: "Augmentation regime: the model guides a fixed worker",
    body: "The focal model acts as an assistant: it writes a process scaffold, which is handed to the fixed GPT-3.5-Turbo worker as internal guidance alongside the client task. The worker's deliverable is what gets judged — so a model wins this regime by making its worker better, mirroring how AI assistance augments a human professional.",
    action: { label: "View augmentation rankings", run: () => { setMode("augmentation"); goTab("rankings"); renderAll(); } },
  },
  evaluator: {
    title: "Evaluator panel: blind pairwise judging",
    body: "A panel of LLM judges (GPT-4.1, Claude-Opus-4.8, DeepSeek-V3.1, and Gemini-3.1-Pro where available) compares outputs two at a time, blind to which model produced them and with option order randomized. Judges never score outputs from their own model family (leave-one-family-out). Each judgment returns a pairwise choice, a short rationale, and per-dimension rubric scores against the task-specific rubric.",
    action: { label: "Inspect judge agreement", run: () => goTab("judges") },
  },
  results: {
    title: "Results aggregation",
    body: "Pairwise wins become win rates per model, task, and regime. Win rates rank models within each task, and per-task ranks roll up into the rank-of-ranks heat maps (Figure 1) and the role-swap scatter (Figure 2) — so every model can be compared as a direct solver versus as an augmenting assistant.",
    action: { label: "Jump to Figure 1", run: () => { goTab("overview"); setTimeout(() => document.getElementById("heatAug")?.scrollIntoView({ behavior: "smooth", block: "center" }), 60); } },
  },
};

function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  state.selectedModel = null;
  state.rubricFocus = null;
}

function syncTextTabs() {
  document.querySelectorAll("[data-texttab]").forEach(x => x.classList.toggle("active", x.dataset.texttab === state.textTab));
}

function goTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x.dataset.tab === tab));
  document.querySelectorAll(".panel").forEach(x => x.classList.toggle("active", x.id === tab));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function bindMethodology() {
  const detail = document.getElementById("methodDetail");
  if (!detail) return;
  const blocks = document.querySelectorAll(".method-card [data-stage]");
  const show = key => {
    const d = methodologyDetails[key];
    if (!d) return;
    blocks.forEach(b => b.classList.toggle("active", b.dataset.stage === key));
    detail.innerHTML = `<b>${esc(d.title)}</b><p>${esc(d.body)}</p>${d.action ? `<button class="pill" type="button">${esc(d.action.label)} &#8594;</button>` : ""}`;
    const act = detail.querySelector("button");
    if (act && d.action) act.addEventListener("click", d.action.run);
  };
  blocks.forEach(b => b.addEventListener("click", () => show(b.dataset.stage)));
  show("augmentation");
}

function bind() {
  bindMethodology();
  document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => {
    state.tab = b.dataset.tab;
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
    document.querySelectorAll(".panel").forEach(x => x.classList.toggle("active", x.id === state.tab));
  }));
  document.getElementById("runSelect").addEventListener("change", e => { state.runId = e.target.value; state.selectedModel = null; state.rubricFocus = null; renderAll(); });
  document.getElementById("modelSet").addEventListener("change", e => { state.modelSet = e.target.value; renderAll(); });
  document.getElementById("taskSelect").addEventListener("change", e => { state.task = e.target.value; state.selectedModel = null; state.rubricFocus = null; renderAll(); });
  document.getElementById("modeSelect").addEventListener("change", e => { state.mode = e.target.value; state.selectedModel = null; state.rubricFocus = null; renderAll(); });
  document.getElementById("judgeSelect").addEventListener("change", e => { state.judge = e.target.value; state.rubricFocus = null; renderAll(); });
  document.querySelectorAll("[data-texttab]").forEach(b => b.addEventListener("click", () => {
    state.textTab = b.dataset.texttab;
    document.querySelectorAll("[data-texttab]").forEach(x => x.classList.toggle("active", x === b));
    renderQualitative();
  }));
}

function renderAll() {
  populateControls();
  renderProjectStats();
  renderMetrics();
  renderHeatmap(document.getElementById("heatAug"), "augmentation");
  renderHeatmap(document.getElementById("heatAuto"), "automation");
  renderRoleScatter();
  renderReplicateSummary();
  renderValidation();
  renderLeaderboard();
  renderRubric();
  renderJudgeScatter();
  renderQualitative();
}

fetch("dashboard-data.json")
  .then(r => r.json())
  .then(data => {
    state.data = data;
    state.runId = data.meta.default_run_id || data.meta.run_id;
    populateControls();
    bind();
    renderAll();
  })
  .catch(err => {
    document.body.innerHTML = `<main><div class="viz-card"><h1>Dashboard data failed to load</h1><pre>${err}</pre></div></main>`;
  });
