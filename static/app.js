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
const refineLoading = document.getElementById("refine-loading");

const RESULT_KEY = "resumeBuilder.latestResult";
const JD_KEY = "resumeBuilder.jobDescription";

let latestLatex = "";
let latestResult = null;

function normalizeJd(text) {
  return String(text ?? "").replace(/\s+/g, " ").trim().toLowerCase();
}

function currentJd() {
  return (jobInput && jobInput.value.trim()) || "";
}

// A stored resume belongs to the posting it was built from. Without this check a
// new posting inherits the previous one's domain through the restored result.
try {
  const saved = sessionStorage.getItem(RESULT_KEY);
  const parsed = saved ? JSON.parse(saved) : null;
  const savedJd = sessionStorage.getItem(JD_KEY);
  if (savedJd && jobInput && !jobInput.value.trim()) jobInput.value = savedJd;
  if (parsed && normalizeJd(parsed.built_from_jd) === normalizeJd(currentJd())) {
    latestResult = parsed;
  } else {
    sessionStorage.removeItem(RESULT_KEY);
  }
} catch (_) { /* ignore */ }

function show(el) { if (el) el.classList.remove("hidden"); }
function hide(el) { if (el) el.classList.add("hidden"); }
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function persistResult(data, jobDescription) {
  const builtFrom = jobDescription || data.built_from_jd || currentJd();
  latestResult = Object.assign({}, data, { built_from_jd: builtFrom });
  latestLatex = data.latex || "";
  try {
    sessionStorage.setItem(RESULT_KEY, JSON.stringify({
      latex: data.latex,
      sections: data.sections,
      jd_analysis: data.jd_analysis,
      meta: data.meta,
      measurement: data.measurement,
      gaps: data.gaps,
      built_from_jd: builtFrom,
    }));
    sessionStorage.setItem(JD_KEY, builtFrom);
  } catch (_) { /* quota / private mode */ }
}

function clearBuildState() {
  latestResult = null;
  latestLatex = "";
  try { sessionStorage.removeItem(RESULT_KEY); } catch (_) { /* ignore */ }
  if (latexOutput) latexOutput.textContent = "";
  hide(results);
  clearRefineChat();
}

// Rewrites apply to the resume as generated. Once the posting in the box no
// longer matches, a rewrite would splice the old domain into the new target.
function syncRefineAvailability() {
  if (!refineBtn) return;
  const stale =
    !!latestResult && normalizeJd(currentJd()) !== normalizeJd(latestResult.built_from_jd);
  refineBtn.disabled = stale;
  refineBtn.title = stale
    ? "The job description changed — generate again before suggesting rewrites."
    : "";
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
  if (!el) return;
  if (!jd) { el.innerHTML = ""; return; }
  el.innerHTML = `
    <h3>Target Role</h3>
    <p><strong>${esc(jd.role_title || "—")}</strong> · ${esc(jd.domain || "—")} · ${esc(jd.seniority_level || "")}</p>
    <p class="positioning">${esc(jd.competitive_positioning || jd.ideal_summary_angle || "")}</p>
  `;
}

function renderBuildMeta(meta) {
  const el = document.getElementById("build-meta");
  if (!el) return;
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
  if (!el) return;
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
  if (!el) return;
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
  if (!overallEl) return;
  overallEl.textContent = score;
  overallEl.style.color = scoreColor(score);
  const verdictEl = document.getElementById("verdict");
  if (verdictEl) verdictEl.textContent = verdictFor(score);

  const breakdown = measurement.breakdown || {};
  const kw = breakdown.keyword_coverage || {};
  const md = breakdown.metric_density || {};

  const covEl = document.getElementById("coverage-score");
  const covPct = kw.max ? Math.round((kw.score / kw.max) * 100) : 0;
  if (covEl) {
    covEl.textContent = `${covPct}%`;
    covEl.style.color = scoreColor(covPct);
  }
  const covDetail = document.getElementById("coverage-detail");
  if (covDetail) covDetail.textContent = kw.details || "";

  const mdEl = document.getElementById("metric-score");
  const mdPct = md.max ? Math.round((md.score / md.max) * 100) : 0;
  if (mdEl) {
    mdEl.textContent = `${mdPct}%`;
    mdEl.style.color = scoreColor(mdPct);
  }
  const mdDetail = document.getElementById("metric-detail");
  if (mdDetail) mdDetail.textContent = md.details || "";

  const breakdownEl = document.getElementById("score-breakdown");
  if (breakdownEl) {
    breakdownEl.innerHTML =
      "<h3>Breakdown</h3>" +
      Object.entries(breakdown).map(([key, val]) => `
        <div class="breakdown-item">
          <span>${esc(key.replace(/_/g, " "))}</span>
          <span>${val.score}/${val.max}</span>
        </div>
        <div class="breakdown-detail">${esc(val.details || "")}</div>
      `).join("");
  }

  const matched = measurement.matched_keywords || [];
  const matchedEl = document.getElementById("matched-keywords");
  if (matchedEl) {
    matchedEl.innerHTML = matched.length
      ? "<h3>JD Keywords On The Page</h3><div class='tag-list'>" +
        matched.map(k => `<span class="tag">${esc(k)}</span>`).join("") + "</div>"
      : "";
  }

  const missing = measurement.missing_keywords || [];
  const missingEl = document.getElementById("missing-keywords");
  if (missingEl) {
    missingEl.innerHTML = missing.length
      ? "<h3>Genuine Gaps</h3>" +
        "<p class='positioning'>The posting asks for these and your fact bank has no evidence of them. They are left off on purpose — if you do have this experience, add it to data/facts.yaml.</p>" +
        "<div class='tag-list'>" +
        missing.map(k => `<span class="tag missing">${esc(k)}</span>`).join("") + "</div>"
      : "";
  }
}

// The gaps this posting exposed, asked back as questions. Answering one turns it
// into a fact, and every later posting can draw on it.
function renderGapQuestions(gaps) {
  const el = document.getElementById("gap-questions");
  if (!el) return;
  const questions = (gaps && gaps.questions) || [];
  if (!questions.length) { el.innerHTML = ""; return; }
  el.innerHTML =
    "<h3>Worth Checking — Did You Do Any Of This?</h3>" +
    "<p class='positioning'>This posting wants things your fact bank does not cover. " +
    "Anything you answer yes to (with a number) becomes a new fact in data/facts.yaml and " +
    "raises every future build. Anything you did not do stays off the resume.</p>" +
    `<ul>${questions.map(q => `<li>${esc(q)}</li>`).join("")}</ul>`;
}

function renderResult(data, { appendChat, jobDescription } = {}) {
  persistResult(data, jobDescription);
  if (latexOutput) latexOutput.textContent = data.latex;
  renderRoleTarget(data.jd_analysis);
  renderBuildMeta(data.meta || {});
  renderMeasurement(data.measurement || {});
  renderGapQuestions(data.gaps || {});
  renderFallbacks(data.meta || {});
  renderSelectedFacts(data.meta || {});
  if (appendChat) appendRefineMessage("bot", appendChat);
  show(results);
  syncRefineAvailability();
}

function appendRefineMessage(role, text) {
  if (!refineChat) return;
  const div = document.createElement("div");
  div.className = `refine-msg ${role}`;
  div.textContent = text;
  refineChat.appendChild(div);
  refineChat.scrollTop = refineChat.scrollHeight;
}

function clearRefineChat() {
  if (refineChat) refineChat.innerHTML = "";
  if (refineInput) refineInput.value = "";
  hide(refineStatus);
  hide(refineLoading);
}

function setRefineBusy(busy) {
  if (refineBtn) {
    refineBtn.disabled = busy;
    refineBtn.textContent = busy ? "Rewriting…" : "Rewrite with suggestion";
  }
  if (busy) {
    show(refineLoading);
    if (refineStatus) {
      refineStatus.textContent = "Rewriting from your suggestion…";
      refineStatus.classList.add("refining");
    }
    show(refineStatus);
  } else {
    hide(refineLoading);
    hide(refineStatus);
    if (refineStatus) refineStatus.classList.remove("refining");
    syncRefineAvailability();
  }
}

async function build(jobDescription) {
  clearBuildState();
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
  renderResult(await res.json(), { jobDescription });
}

async function refine(suggestion) {
  if (!latestResult || !latestResult.sections) {
    throw new Error("Generate a resume first, then suggest a rewrite.");
  }
  const jd = latestResult.built_from_jd || "";
  const typed = currentJd();
  if (typed && normalizeJd(typed) !== normalizeJd(jd)) {
    throw new Error(
      "This resume was generated for a different job description. " +
      "Click Generate for the new posting, then suggest rewrites."
    );
  }
  if (jd.length < 20) {
    throw new Error("Generate the resume again for this posting, then retry the rewrite.");
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000);
  let res;
  try {
    res = await fetch("/api/v2/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        job_description: jd,
        suggestion,
        sections: latestResult.sections,
        jd_analysis: latestResult.jd_analysis || null,
      }),
    });
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new Error("Rewrite timed out after 3 minutes. Try again.");
    }
    throw new Error(
      "Could not reach the server. Restart it with: python3 run.py"
    );
  } finally {
    clearTimeout(timeout);
  }
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
  renderResult(data, { appendChat: botText, jobDescription: jd });
}

async function onBuildClick() {
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
}

async function onRefineClick() {
  try {
    const suggestion = (refineInput && refineInput.value.trim()) || "";
    if (suggestion.length < 3) {
      errorEl.textContent = "Write a short suggestion for what to rewrite.";
      show(errorEl);
      return;
    }
    hide(errorEl);
    appendRefineMessage("user", suggestion);
    if (refineInput) refineInput.value = "";
    setRefineBusy(true);
    await refine(suggestion);
  } catch (err) {
    appendRefineMessage("bot", `Could not rewrite: ${err.message}`);
    if (errorEl) {
      errorEl.textContent = err.message;
      show(errorEl);
    }
  } finally {
    setRefineBusy(false);
  }
}

if (buildBtn) buildBtn.addEventListener("click", onBuildClick);
if (jobInput) jobInput.addEventListener("input", syncRefineAvailability);
if (refineBtn) {
  refineBtn.addEventListener("click", (e) => {
    e.preventDefault();
    onRefineClick();
  });
} else {
  console.error("refine-btn not found — rewrite UI will not work");
}

if (refineInput) {
  refineInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      onRefineClick();
    }
  });
}

if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(latestLatex);
    copyBtn.textContent = "Copied!";
    setTimeout(() => { copyBtn.textContent = "Copy LaTeX"; }, 2000);
  });
}

if (downloadBtn) {
  downloadBtn.addEventListener("click", () => {
    const blob = new Blob([latestLatex], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "tailored_resume.tex";
    a.click();
    URL.revokeObjectURL(url);
  });
}

if (latestResult && latestResult.latex) {
  renderResult(latestResult, { jobDescription: latestResult.built_from_jd });
} else {
  syncRefineAvailability();
}
