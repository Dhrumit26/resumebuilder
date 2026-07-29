const jobInput = document.getElementById("job-description");
const buildBtn = document.getElementById("build-btn");
const loading = document.getElementById("loading");
const errorEl = document.getElementById("error");
const results = document.getElementById("results");
const latexOutput = document.querySelector("#latex-output code");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");

let latestLatex = "";

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function scoreColor(score) {
  if (score >= 80) return "var(--success)";
  if (score >= 65) return "var(--warning)";
  return "var(--error)";
}

function renderRoleTarget(jd) {
  const el = document.getElementById("role-target");
  if (!jd) { el.innerHTML = ""; return; }
  el.innerHTML = `
    <h3>Target Role</h3>
    <p><strong>${jd.role_title || "—"}</strong> · ${jd.seniority_level || "—"} · ${jd.company_type || "—"} · ${jd.years_experience_wanted || ""}</p>
    <p class="positioning">${jd.competitive_positioning || jd.ideal_summary_angle || ""}</p>
  `;
}

function renderRemakeMeta(meta) {
  const el = document.getElementById("remake-meta");
  if (!meta) { el.innerHTML = ""; return; }
  const history = (meta.history || [])
    .map(h => {
      const cand = (h.candidate_scores && h.candidate_scores.length > 1)
        ? ` [candidates: ${h.candidate_scores.join("/")} → kept #${h.picked_candidate}]` : "";
      const fixed = (h.fixed_sections && h.fixed_sections.length)
        ? ` [fixed: ${h.fixed_sections.join(", ")}]` : "";
      const fb = (h.fallback_used && h.fallback_used.length)
        ? ` (kept prev: ${h.fallback_used.join(", ")})` : "";
      return `${h.type} → ${h.score ?? "—"}${cand}${fixed}${fb}`;
    })
    .join(" · ");
  const passed = meta.passed_threshold ? "passed threshold" : "below threshold (best effort)";
  el.innerHTML = `
    <h3>Optimization Loop</h3>
    <p><strong>Final ATS:</strong> ${meta.final_score ?? "—"} · min ${meta.score_threshold ?? 90} · target ${meta.target_score ?? 95} (${passed})</p>
    <p class="positioning">Remakes: ${meta.remake_attempts ?? 0} · History: ${history || "—"} · Kept best version from pass ${meta.best_pass ?? 1} · LLM calls: ${meta.llm_calls ?? "—"}</p>
  `;
}

function renderScores(data) {
  const atsScorer = data.scores.ats_scorer;
  const atsReview = data.scores.ats_reviewer;
  const humanReview = data.scores.human_reviewer;

  const overall = atsScorer.overall_score ?? 0;
  const overallEl = document.getElementById("overall-score");
  overallEl.textContent = overall;
  overallEl.style.color = scoreColor(overall);

  document.getElementById("verdict").textContent =
    (atsScorer.verdict || "").replace(/_/g, " ");

  const vsTypical = atsScorer.vs_typical_applicant || humanReview.vs_typical_applicant || "";
  document.getElementById("vs-typical").textContent = vsTypical;

  document.getElementById("ats-pass").textContent =
    (atsReview.pass_likelihood || "—").toUpperCase();
  document.getElementById("ats-pass").style.color = scoreColor(atsReview.ats_score ?? 0);
  document.getElementById("ats-review-score").textContent =
    atsReview.ats_score != null ? `Score: ${atsReview.ats_score}` : "";

  const humanEl = document.getElementById("human-score");
  humanEl.textContent = humanReview.human_score ?? "—";
  humanEl.style.color = scoreColor(humanReview.human_score ?? 0);

  document.getElementById("interview-rec").textContent =
    (humanReview.interview_recommendation || "").replace(/_/g, " ");
  document.getElementById("competitive-rank").textContent =
    (humanReview.competitive_rank || "").replace(/_/g, " ");

  const advEl = document.getElementById("competitive-advantages");
  const advantages = atsScorer.competitive_advantages || humanReview.strengths || [];
  if (advantages.length) {
    advEl.innerHTML = "<h3>Why You Beat Typical Applicants</h3><ul>" +
      advantages.map(a => `<li>${a}</li>`).join("") + "</ul>";
  } else { advEl.innerHTML = ""; }

  const trigEl = document.getElementById("interview-triggers");
  const triggers = humanReview.interview_triggers || [];
  if (triggers.length) {
    trigEl.innerHTML = "<h3>Interview Triggers</h3><ul>" +
      triggers.map(t => `<li>${t}</li>`).join("") + "</ul>";
  } else { trigEl.innerHTML = ""; }

  const breakdownEl = document.getElementById("score-breakdown");
  if (atsScorer.breakdown) {
    breakdownEl.innerHTML = "<h3>Score Breakdown</h3>" +
      Object.entries(atsScorer.breakdown)
        .map(([key, val]) =>
          `<div class="breakdown-item">
            <span>${key.replace(/_/g, " ")}</span>
            <span>${val.score}/${val.max}</span>
          </div>
          <div class="breakdown-detail">${val.details || ""}</div>`
        ).join("");
  }

  const matchedEl = document.getElementById("matched-keywords");
  const matched = atsScorer.top_matched_skills || atsReview.matched_keywords || [];
  if (matched.length) {
    matchedEl.innerHTML = "<h3>Matched Skills & Keywords</h3><div class='tag-list'>" +
      matched.map(k => `<span class="tag">${k}</span>`).join("") + "</div>";
  }

  const missingEl = document.getElementById("missing-keywords");
  const missing = atsScorer.critical_gaps || atsReview.missing_keywords || [];
  if (missing.length) {
    missingEl.innerHTML = "<h3>Critical Gaps</h3><div class='tag-list'>" +
      missing.map(k => `<span class="tag missing">${k}</span>`).join("") + "</div>";
  }

  const risksEl = document.getElementById("rejection-risks");
  const risks = atsReview.rejection_risks || humanReview.red_flags || [];
  if (risks.length) {
    risksEl.innerHTML = "<h3>Rejection Risks</h3><ul>" +
      risks.map(r => `<li class="risk">${r}</li>`).join("") + "</ul>";
  } else { risksEl.innerHTML = ""; }

  const recsEl = document.getElementById("recommendations");
  const recs = [
    ...(atsReview.recommendations || []),
    ...(humanReview.improvement_suggestions || []),
  ];
  if (recs.length) {
    recsEl.innerHTML = "<h3>Improvements (Ranked by Impact)</h3><ul>" +
      recs.map(r => `<li>${r}</li>`).join("") + "</ul>";
  }
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
    const res = await fetch("/api/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_description: jobDescription }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Generation failed");

    latestLatex = data.latex;
    latexOutput.textContent = data.latex;
    renderRoleTarget(data.jd_analysis);
    renderRemakeMeta(data.meta);
    renderScores(data);
    show(results);
  } catch (err) {
    errorEl.textContent = err.message;
    show(errorEl);
  } finally {
    hide(loading);
    buildBtn.disabled = false;
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
