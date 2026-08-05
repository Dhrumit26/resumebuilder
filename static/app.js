const jobInput = document.getElementById("job-description");
const buildBtn = document.getElementById("build-btn");
const loading = document.getElementById("loading");
const errorEl = document.getElementById("error");
const results = document.getElementById("results");
const latexOutput = document.querySelector("#latex-output code");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");
const refineInput = document.getElementById("refine-input");
const refineBtn = document.getElementById("refine-btn");
const refineChat = document.getElementById("refine-chat");
const refineStatus = document.getElementById("refine-status");

let latestLatex = "";
let latestResult = null;

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function scoreColor(score) {
  if (score >= 80) return "var(--success)";
  if (score >= 65) return "var(--warning)";
  return "var(--error)";
}

function verdictFor(score) {
  if (score >= 85) return "strong match";
  if (score >= 70) return "good match";
  if (score >= 55) return "partial match";
  return "weak match — the posting wants experience this resume does not have";
}

function renderRoleTarget(jd) {
  const el = document.getElementById("role-target");
  if (!jd) { el.innerHTML = ""; return; }
  el.innerHTML = `
    <h3>Target Role</h3>
    <p><strong>${esc(jd.role_title || "—")}</strong> · ${esc(jd.domain || "—")} · ${esc(jd.seniority_level || "")}</p>
    <p class="positioning">${esc(jd.competitive_positioning || jd.ideal_summary_angle || "")}</p>
  `;
}

function renderBuildMeta(meta) {
  const el = document.getElementById("build-meta");
  if (!meta) { el.innerHTML = ""; return; }
  const lines = (meta.skills_lines || []).join(", ");
  el.innerHTML = `
    <h3>How This Was Built</h3>
    <p class="positioning">
      ${meta.llm_calls ?? "—"} model calls · summary ${meta.summary_words ?? "—"} words ·
      skills lines: ${esc(lines)}
      ${meta.refine_note ? `<br>${esc(meta.refine_note)}` : ""}
    </p>
  `;
}

// Bullets that failed verification and reverted to the plain fact text. Worth
// surfacing: it means the writer could not phrase that fact within the rules.
function renderFallbacks(meta) {
  const el = document.getElementById("fallbacks");
  const flagged = (meta.blocks || []).filter(b => (b.fell_back_to_fact || []).length);
  const summaryFell = meta.summary && meta.summary.fell_back;
  if (!flagged.length && !summaryFell) { el.innerHTML = ""; return; }

  const items = flagged.map(b =>
    `<li class="risk">${esc(b.block)}: ${esc((b.fell_back_to_fact || []).join(", "))}</li>`
  );
  if (summaryFell) {
    items.push('<li class="risk">summary: kept the original — the tailored draft could not pass verification</li>');
  }
  el.innerHTML =
    "<h3>Reverted to Plain Fact Text</h3>" +
    "<p class='positioning'>These came out as the raw fact rather than a tailored sentence. Usually means the fact is thin — adding detail in data/facts.yaml helps.</p>" +
    `<ul>${items.join("")}</ul>`;
}

function renderSelectedFacts(meta) {
  const el = document.getElementById("selected-facts");
  const selected = meta.selected_facts || {};
  const blocks = Object.keys(selected);
  if (!blocks.length) { el.innerHTML = ""; return; }

  el.innerHTML =
    "<h3>Facts Chosen For This Posting</h3>" +
    "<p class='positioning'>Ranked by relevance to this JD. The same fact bank produces a different order for a different posting.</p>" +
    blocks.map(name => `
      <div class="breakdown-item"><span>${esc(name)}</span><span></span></div>
      <div class="breakdown-detail">${
        selected[name].map(f => `${esc(f.id)} (${f.score})`).join(" · ")
      }</div>
    `).join("");
}

function renderMeasurement(measurement) {
  const score = measurement.score ?? 0;
  const overallEl = document.getElementById("overall-score");
  overallEl.textContent = score;
  overallEl.style.color = scoreColor(score);
  document.getElementById("verdict").textContent = verdictFor(score);

  const breakdown = measurement.breakdown || {};
  const kw = breakdown.keyword_coverage || {};
  const md = breakdown.metric_density || {};

  const covEl = document.getElementById("coverage-score");
  const covPct = kw.max ? Math.round((kw.score / kw.max) * 100) : 0;
  covEl.textContent = `${covPct}%`;
  covEl.style.color = scoreColor(covPct);
  document.getElementById("coverage-detail").textContent = kw.details || "";

  const mdEl = document.getElementById("metric-score");
  const mdPct = md.max ? Math.round((md.score / md.max) * 100) : 0;
  mdEl.textContent = `${mdPct}%`;
  mdEl.style.color = scoreColor(mdPct);
  document.getElementById("metric-detail").textContent = md.details || "";

  document.getElementById("score-breakdown").innerHTML =
    "<h3>Breakdown</h3>" +
    Object.entries(breakdown).map(([key, val]) => `
      <div class="breakdown-item">
        <span>${esc(key.replace(/_/g, " "))}</span>
        <span>${val.score}/${val.max}</span>
      </div>
      <div class="breakdown-detail">${esc(val.details || "")}</div>
    `).join("");

  const matched = measurement.matched_keywords || [];
  document.getElementById("matched-keywords").innerHTML = matched.length
    ? "<h3>JD Keywords On The Page</h3><div class='tag-list'>" +
      matched.map(k => `<span class="tag">${esc(k)}</span>`).join("") + "</div>"
    : "";

  const missing = measurement.missing_keywords || [];
  document.getElementById("missing-keywords").innerHTML = missing.length
    ? "<h3>Genuine Gaps</h3>" +
      "<p class='positioning'>The posting asks for these and your fact bank has no evidence of them. They are left off on purpose — if you do have this experience, add it to data/facts.yaml.</p>" +
      "<div class='tag-list'>" +
      missing.map(k => `<span class="tag missing">${esc(k)}</span>`).join("") + "</div>"
    : "";
}

// The gaps this posting exposed, asked back as questions. Answering one turns it
// into a fact, and every later posting can draw on it.
function renderGapQuestions(gaps) {
  const el = document.getElementById("gap-questions");
  const questions = (gaps && gaps.questions) || [];
  if (!questions.length) { el.innerHTML = ""; return; }
  el.innerHTML =
    "<h3>Worth Checking — Did You Do Any Of This?</h3>" +
    "<p class='positioning'>This posting wants things your fact bank does not cover. " +
    "Anything you answer yes to (with a number) becomes a new fact in data/facts.yaml and " +
    "raises every future build. Anything you did not do stays off the resume.</p>" +
    `<ul>${questions.map(q => `<li>${esc(q)}</li>`).join("")}</ul>`;
}

function renderResult(data, { appendChat } = {}) {
  latestResult = data;
  latestLatex = data.latex;
  latexOutput.textContent = data.latex;
  renderRoleTarget(data.jd_analysis);
  renderBuildMeta(data.meta || {});
  renderMeasurement(data.measurement || {});
  renderGapQuestions(data.gaps || {});
  renderFallbacks(data.meta || {});
  renderSelectedFacts(data.meta || {});
  if (appendChat) {
    appendRefineMessage("bot", appendChat);
  }
  show(results);
}

function appendRefineMessage(role, text) {
  const div = document.createElement("div");
  div.className = `refine-msg ${role}`;
  div.textContent = text;
  refineChat.appendChild(div);
  refineChat.scrollTop = refineChat.scrollHeight;
}

function clearRefineChat() {
  refineChat.innerHTML = "";
  refineInput.value = "";
  hide(refineStatus);
}

async function build(jobDescription) {
  const res = await fetch("/api/v2/build", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_description: jobDescription }),
  });
  if (!res.ok) {
    let detail = `Generation failed (HTTP ${res.status})`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail) || detail;
    } catch (_) { /* keep the status-code message */ }
    throw new Error(detail);
  }
  clearRefineChat();
  renderResult(await res.json());
}

async function refine(suggestion) {
  if (!latestResult || !latestResult.sections) {
    throw new Error("Generate a resume first, then suggest a rewrite.");
  }
  const res = await fetch("/api/v2/refine", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_description: jobInput.value.trim(),
      suggestion,
      sections: latestResult.sections,
      jd_analysis: latestResult.jd_analysis || null,
    }),
  });
  if (!res.ok) {
    let detail = `Rewrite failed (HTTP ${res.status})`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail) || detail;
    } catch (_) { /* keep the status-code message */ }
    throw new Error(detail);
  }
  const data = await res.json();
  const note = (data.meta && data.meta.refine_note) || "Updated the resume from your suggestion.";
  const changed = ((data.meta && data.meta.refine_changed) || []).join(", ");
  const botText = changed ? `${note} (updated: ${changed})` : note;
  renderResult(data, { appendChat: botText });
}

buildBtn.addEventListener("click", async () => {
  const jobDescription = jobInput.value.trim();
  if (jobDescription.length < 20) {
    errorEl.textContent = "Please paste a full job description (at least 20 characters).";
    show(errorEl);
    return;
  }

  hide(errorEl);
  hide(results);
  show(loading);
  buildBtn.disabled = true;

  try {
    await build(jobDescription);
  } catch (err) {
    errorEl.textContent = err.message;
    show(errorEl);
  } finally {
    hide(loading);
    buildBtn.disabled = false;
  }
});

refineBtn.addEventListener("click", async () => {
  const suggestion = refineInput.value.trim();
  if (suggestion.length < 3) {
    errorEl.textContent = "Write a short suggestion for what to rewrite.";
    show(errorEl);
    return;
  }
  hide(errorEl);
  appendRefineMessage("user", suggestion);
  refineInput.value = "";
  refineBtn.disabled = true;
  refineStatus.textContent = "Rewriting from your suggestion…";
  refineStatus.classList.add("refining");
  show(refineStatus);

  try {
    await refine(suggestion);
  } catch (err) {
    appendRefineMessage("bot", `Could not rewrite: ${err.message}`);
    errorEl.textContent = err.message;
    show(errorEl);
  } finally {
    hide(refineStatus);
    refineStatus.classList.remove("refining");
    refineBtn.disabled = false;
  }
});

refineInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    refineBtn.click();
  }
});

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(latestLatex);
  copyBtn.textContent = "Copied!";
  setTimeout(() => { copyBtn.textContent = "Copy LaTeX"; }, 2000);
});

downloadBtn.addEventListener("click", () => {
  const blob = new Blob([latestLatex], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "tailored_resume.tex";
  a.click();
  URL.revokeObjectURL(url);
});
