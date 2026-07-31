"""Agent pipeline: JD agent -> 4 parallel section agents -> reviewer -> targeted fixes.

Design guarantees:
- Sections the reviewer does not flag are NEVER re-sent to an LLM (byte-identical).
- Any agent failure degrades gracefully: retry once, then fall back to the best
  known version of that section (original resume on the first pass).
- The shipped resume is always the best-scoring version seen, never the last attempt.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from .config import (
    BEST_OF_N as CFG_BEST_OF_N,
    FLEXIBLE_EXPERIENCE_COMPANIES,
    MAX_REMAKE_ATTEMPTS,
    SCORE_SAMPLES as CFG_SCORE_SAMPLES,
    SCORE_THRESHOLD,
    TARGET_SCORE,
)
from .llm import TruncatedCompletion, call_llm, call_llm_json
from .resume_builder import (
    _debug_dump,
    assemble_full_resume,
    bold_keywords_in_bullets,
    bullet_rewrite_ratio,
    clean_generated_sections,
    clean_llm_latex,
    count_resume_items_per_block,
    fill_prompt,
    find_incomplete_bullets,
    is_valid_jake_experience,
    is_valid_jake_projects,
    is_valid_jake_skills,
    is_valid_jake_summary,
    architecture_fog_in_flexible_bullets,
    architecture_fog_in_text,
    languages_in_flexible_bullets,
    latex_to_plain,
    load_full_template,
    load_original_sections,
    load_prompt,
    strip_delimiter_artifacts,
    strip_markdown_artifacts,
    truncate_jd,
    unevidenced_tools,
    word_count,
)

SECTION_NAMES = ("summary", "experience", "projects", "skills")

SECTION_PROMPTS = {
    "summary": "agent_summary.txt",
    "experience": "agent_experience.txt",
    "projects": "agent_projects.txt",
    "skills": "agent_skills.txt",
}

SECTION_VALIDATORS = {
    "summary": is_valid_jake_summary,
    "experience": is_valid_jake_experience,
    "projects": is_valid_jake_projects,
    "skills": is_valid_jake_skills,
}

MAX_FIX_ROUNDS = max(1, min(MAX_REMAKE_ATTEMPTS, 5))

# Module-level so tests can dial these down to 1
BEST_OF_N = max(1, min(CFG_BEST_OF_N, 4))
SCORE_SAMPLES = max(1, min(CFG_SCORE_SAMPLES, 3))

# Temperature schedule for candidate diversity in best-of-N generation
CANDIDATE_TEMPERATURES = [0.25, 0.55, 0.75, 0.9]

# Section attempts share one budget: an invalid-LaTeX retry, a too-close-to-original
# retry, and a truncation retry-at-a-higher-cap all draw from it.
MAX_SECTION_ATTEMPTS = 3
SECTION_TOKEN_CAP = 2500
MAX_SECTION_TOKEN_CAP = 6000

_DEFAULT_JD_ANALYSIS = {
    "role_title": "",
    "seniority_level": "mid",
    "company_type": "unknown",
    "years_experience_wanted": "",
    "domain": "",
    "industry": "",
    "domain_practices": [],
    "must_have_skills": [],
    "nice_to_have_skills": [],
    "exact_keywords_for_ats": [],
    "tools": [],
    "concepts": [],
    "keyword_placement": {name: [] for name in SECTION_NAMES},
    "ideal_summary_angle": "",
    "competitive_positioning": "",
}


def sanitize_jd(job_description: str) -> str:
    """Neutralize template-injection tokens and normalize whitespace."""
    jd = (job_description or "").strip()
    jd = re.sub(r"\{\{|\}\}", "", jd)  # would corrupt fill_prompt placeholders
    jd = re.sub(r"[ \t]+", " ", jd)
    if len(jd) < 20:
        raise ValueError("Job description is too short — paste the full posting.")
    return truncate_jd(jd)


def _clamp(value, lo: int = 0, hi: int = 100):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(lo, min(hi, int(value)))


def _clamp_scores_bundle(bundle: dict) -> dict:
    """Clamp every numeric score the UI/loop consumes; drop garbage values.

    The overall score is DERIVED from the breakdown sum when available: the rubric
    defines it as the category sum, and models routinely report an eyeballed
    overall that contradicts their own arithmetic (observed: breakdown 97, overall 88).
    """
    if not isinstance(bundle, dict):
        return {}
    ats = bundle.get("ats_scorer")
    if isinstance(ats, dict):
        ats["overall_score"] = _clamp(ats.get("overall_score"))
        breakdown = ats.get("breakdown")
        if isinstance(breakdown, dict):
            cat_scores = []
            max_total = 0
            for cat in breakdown.values():
                if isinstance(cat, dict):
                    hi = cat.get("max") if isinstance(cat.get("max"), (int, float)) else 100
                    cat["score"] = _clamp(cat.get("score"), 0, int(hi))
                    if cat["score"] is not None:
                        cat_scores.append(cat["score"])
                        max_total += int(hi)
            # Trust the derived sum ONLY when the breakdown is complete: every
            # category scored AND the maxes add up to the full 100-point rubric.
            # The old ">= 5 of 7" test happily summed a partial breakdown, which
            # understated the score by up to 15 points and bought extra fix rounds.
            if cat_scores and len(cat_scores) == len(breakdown) and max_total >= 100:
                ats["overall_score"] = _clamp(sum(cat_scores))
    review = bundle.get("ats_reviewer")
    if isinstance(review, dict):
        review["ats_score"] = _clamp(review.get("ats_score"))
    human = bundle.get("human_reviewer")
    if isinstance(human, dict):
        human["human_score"] = _clamp(human.get("human_score"))
    return bundle


def overall_score(bundle: dict) -> int:
    ats = (bundle.get("ats_scorer") or {}).get("overall_score")
    if isinstance(ats, (int, float)):
        return int(ats)
    human = (bundle.get("human_reviewer") or {}).get("human_score")
    if isinstance(human, (int, float)):
        return int(human)
    return 0


def _normalize_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if v and str(v).strip()]


def run_jd_agent(jd: str) -> tuple[dict, bool]:
    """Analyze the JD once. Never fails the pipeline: degrades to a minimal analysis."""
    prompt = fill_prompt(load_prompt("agent_jd.txt"), JOB_DESCRIPTION=jd)
    try:
        raw = call_llm_json(prompt, temperature=0.0, max_tokens=2000)
    except Exception:
        _debug_dump("agent_jd_error", "JD agent failed twice; using default analysis")
        return dict(_DEFAULT_JD_ANALYSIS), False

    analysis = dict(_DEFAULT_JD_ANALYSIS)
    for key in analysis:
        if key in raw:
            analysis[key] = raw[key]
    for key in (
        "must_have_skills",
        "nice_to_have_skills",
        "exact_keywords_for_ats",
        "tools",
        "concepts",
        "domain_practices",
    ):
        analysis[key] = _normalize_str_list(analysis.get(key))
    placement = analysis.get("keyword_placement")
    if not isinstance(placement, dict):
        placement = {}
    analysis["keyword_placement"] = {
        name: _normalize_str_list(placement.get(name)) for name in SECTION_NAMES
    }
    _debug_dump("agent_jd_analysis", json.dumps(analysis, indent=2))
    return analysis, True


def jd_keywords_of(analysis: dict) -> list[str]:
    seen: list[str] = []
    for key in ("must_have_skills", "exact_keywords_for_ats", "tools"):
        for kw in analysis.get(key) or []:
            if kw and kw not in seen:
                seen.append(kw)
    return seen


def _placement_keywords(analysis: dict, name: str, banned: list[str]) -> str:
    kws = list(analysis.get("keyword_placement", {}).get(name) or [])
    if not kws:
        kws = (analysis.get("must_have_skills") or [])[:8]
    if name in ("experience", "projects"):  # never steer bullet agents toward banned tools
        banned_lower = {b.lower() for b in banned}
        kws = [k for k in kws if k.lower() not in banned_lower]
    return ", ".join(kws) if kws else "(use your judgment from the JD)"


def _build_section_prompt(
    name: str,
    jd: str,
    jd_analysis: dict,
    original_section: str,
    evidence: str,
    fix_context: dict | None,
) -> str:
    fix_block = ""
    if fix_context:
        fixes = "\n".join(f"- {f}" for f in fix_context["fixes"])
        fix_block = (
            "\n## PREVIOUS VERSION (base facts — REWRITE wording, do not copy-paste)\n"
            + fix_context["previous"]
            + "\n\n## REVIEWER FIXES TO APPLY (targeted edits; keep facts, change sentences)\n"
            + fixes
        )
    banned = unevidenced_tools(jd_keywords_of(jd_analysis), evidence.lower())
    return fill_prompt(
        load_prompt(SECTION_PROMPTS[name]),
        JD_ANALYSIS=json.dumps(jd_analysis, indent=1),
        JOB_DESCRIPTION=jd,
        ORIGINAL_SECTION=original_section,
        EVIDENCE=evidence,
        PLACEMENT_KEYWORDS=_placement_keywords(jd_analysis, name, banned),
        JD_TOOLS=", ".join(jd_analysis.get("tools") or []) or "(none identified)",
        BANNED_TOOLS=", ".join(banned) or "(none — all JD tools are evidenced)",
        FLEXIBLE_COMPANY=", ".join(FLEXIBLE_EXPERIENCE_COMPANIES) or "(none)",
        FIX_BLOCK=fix_block,
    )


def write_section(
    name: str,
    jd: str,
    jd_analysis: dict,
    original_section: str,
    evidence: str,
    fix_context: dict | None = None,
    base_temperature: float | None = None,
) -> tuple[str | None, str]:
    """One section agent call. Returns (latex or None, status).

    status: ok | retried | invalid | error
    """
    prompt = _build_section_prompt(name, jd, jd_analysis, original_section, evidence, fix_context)
    validator = SECTION_VALIDATORS[name]
    if base_temperature is not None:
        temperature = base_temperature
    else:
        temperature = 0.15 if fix_context else 0.25

    last_error = None
    token_cap = SECTION_TOKEN_CAP
    attempts = 0
    while attempts < MAX_SECTION_ATTEMPTS:
        attempts += 1
        try:
            raw = call_llm(prompt, temperature=temperature, max_tokens=token_cap)
        except TruncatedCompletion as exc:
            # The model ran out of room mid-bullet. Retrying at the same cap would
            # truncate again, so raise the ceiling instead of falling back.
            last_error = exc
            _debug_dump(f"agent_{name}_truncated", f"cap={token_cap}: {exc}")
            token_cap = min(token_cap * 2, MAX_SECTION_TOKEN_CAP)
            continue
        except Exception as exc:  # network/timeouts: retry, then give up
            last_error = exc
            continue
        latex = strip_markdown_artifacts(clean_llm_latex(raw))
        latex = strip_delimiter_artifacts(latex)
        if not validator(latex):
            prompt += (
                "\n\nWARNING: Your previous output was INVALID (wrong or missing Jake LaTeX macros). "
                "Copy the macro skeleton from the ORIGINAL section exactly. LaTeX only."
            )
            temperature = 0.05
            continue
        # Language scatter in the flexible role (a Java bullet, a C# bullet, a
        # Python bullet in one job) is the instant tell of a fake resume. One
        # retry with a targeted warning; the reviewer penalizes any survivor.
        if name == "experience":
            langs = languages_in_flexible_bullets(latex, FLEXIBLE_EXPERIENCE_COMPANIES)
            if len(langs) > 2 and attempts < MAX_SECTION_ATTEMPTS:
                _debug_dump("agent_experience_language_scatter", ", ".join(langs))
                prompt += (
                    f"\n\nWARNING: Your current-role bullets scatter {len(langs)} programming "
                    f"languages ({', '.join(langs)}) across ONE job — the instant tell of a "
                    "fake resume. Rewrite the current-role bullets around ONE system and ONE "
                    "primary language (plus at most one scripting language in a tooling role). "
                    "The JD's other languages belong in the Skills section only."
                )
                temperature = min(temperature, 0.2)
                continue
            fog = architecture_fog_in_flexible_bullets(latex, FLEXIBLE_EXPERIENCE_COMPANIES)
            if fog and attempts < MAX_SECTION_ATTEMPTS:
                _debug_dump("agent_experience_architecture_fog", ", ".join(fog))
                prompt += (
                    "\n\nWARNING: Architecture fog in the current-role bullets: "
                    + ", ".join(fog)
                    + ". SQL is a LANGUAGE, not a destination — name PostgreSQL, Azure SQL "
                    "Database, SQL Server, etc. Never write 'analytics stores' or 'data stores'. "
                    "Never leave Azure/AWS bare — name the full service (Azure Data Factory, "
                    "not bare 'Data Factory'; Azure SQL Database; S3; Lambda; RDS). "
                    "Never write 'Git-based CI/CD' — name GitHub Actions, Azure DevOps, "
                    "Jenkins, or GitLab CI. Rewrite those bullets."
                )
                temperature = min(temperature, 0.2)
                continue

        if name == "summary":
            fog = architecture_fog_in_text(latex)
            if fog and attempts < MAX_SECTION_ATTEMPTS:
                _debug_dump("agent_summary_architecture_fog", ", ".join(fog))
                prompt += (
                    "\n\nWARNING: Architecture fog in the summary: "
                    + ", ".join(fog)
                    + ". Name the real database/warehouse and the real cloud service. "
                    "SQL is a language, not a product. Rewrite."
                )
                temperature = min(temperature, 0.2)
                continue

        # Experience/projects must be real rewrites, not near-copies of ORIGINAL
        if name in ("experience", "projects") and not fix_context:
            ratio = bullet_rewrite_ratio(latex, original_section)
            if ratio < 0.6 and attempts == 1:
                prompt += (
                    "\n\nWARNING: Your bullets were too close to ORIGINAL (copy/paraphrase). "
                    "REWRITE every \\resumeItem with different wording and a JD angle. "
                    "For the flexible current role, write natively in the JD's domain. "
                    "For fixed-history jobs keep the same facts and metrics — change the sentences. "
                    "At least half the words in each bullet must differ from ORIGINAL."
                )
                temperature = min(0.55, max(temperature, 0.4))
                continue
        _debug_dump(f"agent_{name}", latex)
        return latex, ("ok" if attempts == 1 else "retried")

    status = "error" if last_error is not None else "invalid"
    _debug_dump(f"agent_{name}_failed", f"{status}: {last_error}")
    return None, status


def _summary_evidence(evidence: str, experience_latex: str | None) -> str:
    """Evidence for the Summary agent, extended with the resume's OWN tailored
    experience section. The summary is written to agree with the story the
    experience tells (same domain, same technologies) — without this, a flexible
    current role rewritten into the JD's domain contradicts a summary that only
    ever saw the original resume."""
    if not experience_latex:
        return evidence
    return (
        evidence
        + "\n\nTAILORED CURRENT-ROLE EXPERIENCE (the experience section appearing on"
        " THIS resume, already rewritten for this JD — the summary MUST tell the same"
        " story: same domain, same technologies, no contradicting metrics):\n"
        + latex_to_plain(experience_latex)
    )


def write_sections_parallel(
    names: list[str],
    jd: str,
    jd_analysis: dict,
    original_sections: dict[str, str],
    evidence: str,
    fix_contexts: dict[str, dict] | None = None,
    base_temperature: float | None = None,
    pool: ThreadPoolExecutor | None = None,
    current_experience: str | None = None,
) -> tuple[dict[str, str | None], dict[str, str]]:
    fix_contexts = fix_contexts or {}

    def submit_all(executor: ThreadPoolExecutor):
        return {
            name: executor.submit(
                write_section,
                name,
                jd,
                jd_analysis,
                original_sections[name],
                _summary_evidence(evidence, current_experience) if name == "summary" else evidence,
                fix_contexts.get(name),
                base_temperature,
            )
            for name in names
        }

    if pool is not None:
        futures = submit_all(pool)
    else:
        own_pool = ThreadPoolExecutor(max_workers=len(names))
        futures = submit_all(own_pool)

    results: dict[str, str | None] = {}
    statuses: dict[str, str] = {}
    for name, future in futures.items():
        try:
            latex, status = future.result()
        except Exception:
            latex, status = None, "error"
        results[name] = latex
        statuses[name] = status
    if pool is None:
        own_pool.shutdown(wait=False)
    return results, statuses


def run_reviewer(resume_text: str, jd: str, jd_analysis: dict) -> dict | None:
    """Single reviewer sample. Returns None if the reviewer is unusable."""
    prompt = fill_prompt(
        load_prompt("agent_reviewer.txt"),
        RESUME_TEXT=resume_text[:6000],
        JOB_DESCRIPTION=jd,
        JD_ANALYSIS=json.dumps(jd_analysis, indent=1),
        FLEXIBLE_COMPANY=", ".join(FLEXIBLE_EXPERIENCE_COMPANIES) or "(none)",
    )
    try:
        bundle = call_llm_json(prompt, temperature=0.0, max_tokens=3000)
    except Exception:
        return None
    bundle = _clamp_scores_bundle(bundle)
    if overall_score(bundle) == 0 and not bundle.get("ats_scorer"):
        return None
    return bundle


def _merge_reviewer_samples(samples: list[dict]) -> dict:
    """Average scores across samples; union section fixes and missing keywords.

    Averaging cancels scorer noise; unioning fixes gives the fix loop the most
    complete work order.
    """
    base = json.loads(json.dumps(samples[0]))
    if len(samples) == 1:
        return base

    scores = [overall_score(s) for s in samples]
    avg = round(sum(scores) / len(scores))
    ats = base.setdefault("ats_scorer", {})
    ats["overall_score"] = avg
    ats["score_samples"] = scores

    merged_fixes: dict[str, list[str]] = {}
    merged_missing: list[str] = []
    for s in samples:
        for name, items in (_extract_section_fixes(s)).items():
            bucket = merged_fixes.setdefault(name, [])
            for item in items:
                if item not in bucket and len(bucket) < 6:
                    bucket.append(item)
        for kw in _normalize_str_list((s.get("ats_reviewer") or {}).get("missing_keywords")):
            if kw not in merged_missing:
                merged_missing.append(kw)
    base["section_fixes"] = merged_fixes
    base.setdefault("ats_reviewer", {})["missing_keywords"] = merged_missing
    return base


def run_reviewer_stable(resume_text: str, jd: str, jd_analysis: dict) -> dict | None:
    """SCORE_SAMPLES parallel reviewer samples, merged. None only if ALL fail."""
    if SCORE_SAMPLES <= 1:
        return run_reviewer(resume_text, jd, jd_analysis)
    with ThreadPoolExecutor(max_workers=SCORE_SAMPLES) as pool:
        futures = [
            pool.submit(run_reviewer, resume_text, jd, jd_analysis)
            for _ in range(SCORE_SAMPLES)
        ]
        samples = []
        for f in futures:
            try:
                s = f.result()
            except Exception:
                s = None
            if s is not None:
                samples.append(s)
    if not samples:
        return None
    return _merge_reviewer_samples(samples)


def _extract_section_fixes(bundle: dict) -> dict[str, list[str]]:
    raw = bundle.get("section_fixes")
    fixes: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for name in SECTION_NAMES:  # ignore unknown section names from the model
            items = _normalize_str_list(raw.get(name))
            if items:
                fixes[name] = items
    return fixes


def _apply_incomplete_bullet_gate(bundle: dict, cleaned: dict) -> None:
    incomplete = find_incomplete_bullets(cleaned["experience"]) + find_incomplete_bullets(
        cleaned["projects"]
    )
    if not incomplete:
        return
    review = bundle.setdefault("ats_reviewer", {})
    recs = list(review.get("recommendations") or [])
    recs.insert(
        0,
        "CRITICAL: Some bullets are cut mid-sentence. Rewrite as FULL complete sentences: "
        + "; ".join(incomplete[:3]),
    )
    review["recommendations"] = recs
    fixes = bundle.setdefault("section_fixes", {})
    exp_fixes = _normalize_str_list(fixes.get("experience"))
    exp_fixes.insert(0, "Rewrite cut-off bullets into complete sentences: " + "; ".join(incomplete[:3]))
    fixes["experience"] = exp_fixes
    ats = bundle.setdefault("ats_scorer", {})
    if isinstance(ats.get("overall_score"), (int, float)):
        ats["overall_score"] = min(int(ats["overall_score"]), 89)


def _is_clean(cleaned: dict) -> bool:
    return not (
        find_incomplete_bullets(cleaned["experience"]) + find_incomplete_bullets(cleaned["projects"])
    )


def _assemble(
    template: str,
    original_sections: dict[str, str],
    drafts: dict[str, str | None],
    fallback_sections: dict[str, str] | None,
    jd_keywords: list[str],
) -> tuple[dict, str, str]:
    fb = fallback_sections or original_sections
    cleaned = clean_generated_sections(
        drafts.get("summary") or fb["summary"],
        drafts.get("experience") or fb["experience"],
        drafts.get("projects") or fb["projects"],
        drafts.get("skills") or fb["skills"],
        original_sections,
        fallback_sections=fallback_sections,
        jd_keywords=jd_keywords,
    )
    # Guarantee JD keywords are visually bolded (the model's own bolding is inconsistent).
    # The summary is deliberately left unbolded — see strip_bold() in resume_builder.
    cleaned["experience"] = bold_keywords_in_bullets(cleaned["experience"], jd_keywords)
    cleaned["projects"] = bold_keywords_in_bullets(cleaned["projects"], jd_keywords)
    full_latex = assemble_full_resume(
        template, cleaned["summary"], cleaned["experience"], cleaned["projects"], cleaned["skills"]
    )
    full_latex = strip_delimiter_artifacts(full_latex)
    return cleaned, full_latex, latex_to_plain(full_latex)


def _result_payload(
    best: dict,
    jd_analysis: dict,
    llm_calls: int,
    rounds: int,
    history: list[dict],
    jd_ok: bool,
    reviewer_ok: bool,
    final: bool,
) -> dict:
    """Build the response body. Shared by the streamed draft and the final result."""
    cleaned = best["cleaned"]
    scores = best["scores"]
    return {
        "latex": best["full_latex"],
        "sections": {n: cleaned[n] for n in SECTION_NAMES},
        "jd_analysis": jd_analysis,
        "meta": {
            "architecture": "agents-v2",
            # False on a streamed draft: the fix loop is still running and a
            # better version may replace this one.
            "final": final,
            "llm_calls": llm_calls,
            "score_threshold": SCORE_THRESHOLD,
            "target_score": TARGET_SCORE,
            "final_score": best["score"],
            "passed_threshold": best["score"] >= SCORE_THRESHOLD,
            "hit_target": best["score"] >= TARGET_SCORE,
            "remake_attempts": rounds,
            "best_pass": best["pass"],
            "history": history,
            "jd_agent_ok": jd_ok,
            "reviewer_ok": reviewer_ok,
            "incomplete_bullets": find_incomplete_bullets(cleaned["experience"])
            + find_incomplete_bullets(cleaned["projects"]),
            "bullet_counts_expected": {
                "experience": cleaned["expected_experience_bullets"],
                "projects": cleaned["expected_project_bullets"],
            },
            "bullet_counts_actual": {
                "experience": cleaned["actual_experience_bullets"],
                "projects": cleaned["actual_project_bullets"],
            },
            "summary_words": word_count(latex_to_plain(cleaned["summary"])),
            "approx_body_words": word_count(best["resume_text"]),
        },
        "scores": {
            "ats_scorer": scores.get("ats_scorer", {}),
            "ats_reviewer": scores.get("ats_reviewer", {}),
            "human_reviewer": scores.get("human_reviewer", {}),
        },
    }


def _emit(on_progress, event: str, payload: dict) -> None:
    """Fire a progress event. A broken consumer must never fail the build."""
    if on_progress is None:
        return
    try:
        on_progress(event, payload)
    except Exception:
        pass


def build_tailored_resume(job_description: str, on_progress=None) -> dict:
    """Agent pipeline entry point. API-compatible with the previous implementation.

    on_progress: optional callable(event: str, payload: dict). Receives "jd" once
    the JD is analyzed, "draft" as soon as stage 3 picks a winner (usable resume,
    ~1/3 of total wall-clock), and "pass" after each fix round. The fix loop is
    inherently serial — each round needs the previous round's scores — so this is
    how the wait becomes usable rather than shorter.
    """
    jd = sanitize_jd(job_description)
    original_sections = load_original_sections()
    template = load_full_template()
    evidence = latex_to_plain("\n".join(original_sections[n] for n in SECTION_NAMES))

    llm_calls = 0
    history: list[dict] = []

    # ---- Stage 1: JD agent -------------------------------------------------
    jd_analysis, jd_ok = run_jd_agent(jd)
    llm_calls += 1
    keywords = jd_keywords_of(jd_analysis)
    _emit(on_progress, "jd", {"jd_analysis": jd_analysis, "jd_agent_ok": jd_ok})

    # ---- Stage 2: best-of-N candidate resumes, section agents parallel ------
    # The summary is written AFTER its candidate's experience so it can tell the
    # same story: the flexible current role may be rewritten into the JD's domain,
    # and a summary drafted from the original resume alone would contradict it.
    n = BEST_OF_N
    body_sections = [name for name in SECTION_NAMES if name != "summary"]
    with ThreadPoolExecutor(max_workers=max(1, len(SECTION_NAMES) * n)) as pool:
        candidate_futures = []
        for c in range(n):
            temp = CANDIDATE_TEMPERATURES[min(c, len(CANDIDATE_TEMPERATURES) - 1)]
            futures = {
                name: pool.submit(
                    write_section, name, jd, jd_analysis,
                    original_sections[name], evidence, None, temp,
                )
                for name in body_sections
            }
            candidate_futures.append((temp, futures))

        # Collect body sections; launch each candidate's summary as soon as its
        # experience is known. Later candidates' bodies keep running meanwhile.
        partials = []
        for temp, futures in candidate_futures:
            drafts: dict[str, str | None] = {}
            statuses: dict[str, str] = {}
            for name, future in futures.items():
                try:
                    latex, status = future.result()
                except Exception:
                    latex, status = None, "error"
                drafts[name] = latex
                statuses[name] = status
            summary_future = pool.submit(
                write_section, "summary", jd, jd_analysis,
                original_sections["summary"],
                _summary_evidence(evidence, drafts.get("experience")),
                None, temp,
            )
            partials.append((drafts, statuses, summary_future))

        candidates = []
        for drafts, statuses, summary_future in partials:
            try:
                latex, status = summary_future.result()
            except Exception:
                latex, status = None, "error"
            drafts["summary"] = latex
            statuses["summary"] = status
            candidates.append((drafts, statuses))
    llm_calls += len(SECTION_NAMES) * n

    # ---- Stage 3: assemble + score every candidate, keep the strongest -----
    # Assembly is pure CPU, so do it up front; then score every candidate
    # CONCURRENTLY. Candidates are independent, so the old sequential loop paid a
    # full reviewer round per extra candidate for no reason.
    assembled = []
    for idx, (drafts, statuses) in enumerate(candidates):
        cleaned, full_latex, resume_text = _assemble(
            template, original_sections, drafts, None, keywords
        )
        assembled.append((idx, cleaned, full_latex, resume_text, statuses))

    if len(assembled) == 1:
        score_results = [run_reviewer_stable(assembled[0][3], jd, jd_analysis)]
    else:
        with ThreadPoolExecutor(max_workers=len(assembled)) as score_pool:
            score_futures = [
                score_pool.submit(run_reviewer_stable, item[3], jd, jd_analysis)
                for item in assembled
            ]
            score_results = []
            for future in score_futures:
                try:
                    score_results.append(future.result())
                except Exception:
                    score_results.append(None)
    llm_calls += SCORE_SAMPLES * len(assembled)

    scored_candidates = []
    reviewer_ok = False
    for (idx, cleaned, full_latex, resume_text, statuses), scores in zip(
        assembled, score_results
    ):
        if scores is None:
            scores = {}
        else:
            reviewer_ok = True
            _apply_incomplete_bullet_gate(scores, cleaned)
        scored_candidates.append({
            "cleaned": cleaned,
            "full_latex": full_latex,
            "resume_text": resume_text,
            "scores": scores,
            "score": overall_score(scores),
            "statuses": statuses,
            "candidate": idx + 1,
        })

    scored_candidates.sort(
        key=lambda c: (c["score"], _is_clean(c["cleaned"])), reverse=True
    )
    winner = scored_candidates[0]
    cleaned, full_latex, resume_text = winner["cleaned"], winner["full_latex"], winner["resume_text"]
    scores, score = winner["scores"], winner["score"]

    _debug_dump("pass1_scores", json.dumps(scores, indent=2))
    history.append({
        "pass": 1,
        "type": "generate",
        "score": score,
        "candidate_scores": [c["score"] for c in sorted(scored_candidates, key=lambda c: c["candidate"])],
        "picked_candidate": winner["candidate"],
        "section_statuses": winner["statuses"],
        "fallback_used": [k for k, v in cleaned["fallback_used"].items() if v],
        "gate_reverts": {k: v for k, v in (cleaned.get("gate_reverts") or {}).items() if v},
    })

    best = {
        "cleaned": cleaned,
        "full_latex": full_latex,
        "resume_text": resume_text,
        "scores": scores,
        "score": score,
        "pass": 1,
    }

    # A complete, usable resume exists right now. Ship it to the client before
    # spending the next couple of minutes in the serial fix loop.
    _emit(
        on_progress,
        "draft",
        _result_payload(
            best, jd_analysis, llm_calls, 0, history, jd_ok, reviewer_ok, final=False
        ),
    )

    # ---- Stage 4: targeted fix loop ----------------------------------------
    rounds = 0
    stagnant = 0
    while reviewer_ok and rounds < MAX_FIX_ROUNDS and best["score"] < TARGET_SCORE:
        fixes = _extract_section_fixes(best["scores"])
        if not fixes and best["score"] < SCORE_THRESHOLD:
            missing = ", ".join(
                _normalize_str_list((best["scores"].get("ats_reviewer") or {}).get("missing_keywords"))[:8]
            )
            if missing:
                fixes = {
                    "experience": [f"Naturally cover these missing JD keywords where truthful: {missing}"],
                    "skills": [f"Add missing JD tools: {missing}"],
                }
        if not fixes:
            break  # reviewer says every section is strong — nothing to do

        # Keep fix rounds surgical: at most 2 sections per round (most-flagged first).
        # Rewriting everything every round degrades strong sections (proven empirically).
        if len(fixes) > 2:
            ranked = sorted(fixes.items(), key=lambda kv: len(kv[1]), reverse=True)
            fixes = dict(ranked[:2])

        rounds += 1
        prev_sections = {n: best["cleaned"][n] for n in SECTION_NAMES}
        fix_contexts = {
            name: {"previous": prev_sections[name], "fixes": items}
            for name, items in fixes.items()
        }
        # Teach the agent about bullets the truthfulness gate reverted last time
        gate_reverts = best["cleaned"].get("gate_reverts") or {}
        for name, terms in gate_reverts.items():
            if not terms:
                continue
            warning = (
                "Your previous attempt was rejected for claiming tools the candidate "
                f"never used: {', '.join(terms)}. NEVER mention these in bullets — "
                "use the candidate's real technologies instead."
            )
            if name in fix_contexts:
                fix_contexts[name]["fixes"].append(warning)
            elif name in ("experience", "projects"):
                fix_contexts[name] = {"previous": prev_sections[name], "fixes": [warning]}

        fixed_drafts, fix_statuses = write_sections_parallel(
            list(fix_contexts.keys()), jd, jd_analysis, original_sections, evidence,
            fix_contexts, current_experience=prev_sections["experience"],
        )
        llm_calls += len(fix_contexts)

        # Untouched sections are carried over verbatim — structural guarantee
        merged: dict[str, str | None] = dict(prev_sections)
        for name, latex in fixed_drafts.items():
            if latex is not None:
                merged[name] = latex

        cleaned, full_latex, resume_text = _assemble(
            template, original_sections, merged, prev_sections, keywords
        )

        scores = run_reviewer_stable(resume_text, jd, jd_analysis)
        llm_calls += SCORE_SAMPLES
        if scores is None:
            history.append({
                "pass": rounds + 1, "type": "fix", "score": None,
                "fixed_sections": list(fix_contexts.keys()),
                "note": "reviewer unavailable — kept best version",
            })
            break
        _apply_incomplete_bullet_gate(scores, cleaned)
        score = overall_score(scores)
        _debug_dump(f"pass{rounds + 1}_scores", json.dumps(scores, indent=2))

        history.append({
            "pass": rounds + 1,
            "type": "fix",
            "score": score,
            "fixed_sections": list(fix_contexts.keys()),
            "section_statuses": fix_statuses,
            "fallback_used": [k for k, v in cleaned["fallback_used"].items() if v],
            "gate_reverts": {k: v for k, v in (cleaned.get("gate_reverts") or {}).items() if v},
        })

        improved = score > best["score"] or (
            score == best["score"] and _is_clean(cleaned) and not _is_clean(best["cleaned"])
        )
        if improved:
            best = {
                "cleaned": cleaned,
                "full_latex": full_latex,
                "resume_text": resume_text,
                "scores": scores,
                "score": score,
                "pass": rounds + 1,
            }
            stagnant = 0
            # A better version exists — push it so the client can swap it in.
            _emit(
                on_progress,
                "draft",
                _result_payload(
                    best, jd_analysis, llm_calls, rounds, history,
                    jd_ok, reviewer_ok, final=False,
                ),
            )
        else:
            stagnant += 1
            # keep best scores for next round's fixes; the failed attempt's fixes
            # may chase noise from a worse resume
        _emit(
            on_progress,
            "pass",
            {"pass": rounds + 1, "score": score, "improved": improved,
             "best_score": best["score"], "fixed_sections": list(fix_contexts.keys())},
        )
        if best["score"] >= TARGET_SCORE and _is_clean(best["cleaned"]):
            break
        if stagnant >= 2:
            break

    return _result_payload(
        best, jd_analysis, llm_calls, rounds, history, jd_ok, reviewer_ok, final=True
    )
