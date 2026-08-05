"""Fact-grounded pipeline.

    JD analysis -> select real facts (code) -> write bullets -> verify (code)
    -> targeted repair -> render into the LaTeX templates -> measure

Differences from the v1 pipeline that matter:
- Writers never invent. Every bullet is licensed by one fact in data/facts.yaml,
  and verify.py checks that mechanically instead of hunting for known lies.
- The model never emits LaTeX. skeleton.py fills the shipped templates, so the
  layout and the bullet counts are the templates', not the model's.
- Quality is MEASURED (keyword coverage, metric density, domain match), not
  scored by an LLM rubric that swung 26 points between identical runs.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from .facts import Fact, load_fact_bank
from .llm import call_llm_json
from .matching import Selection, select_facts, select_skills, theme_profile, tool_matches
from .pipeline import jd_keywords_of, run_jd_agent, sanitize_jd
from .resume_builder import (
    _debug_dump,
    assemble_full_resume,
    bold_keywords_in_bullets,
    bold_metrics_in_bullets,
    latex_to_plain,
    load_full_template,
    word_count,
)
from .skeleton import (
    latex_safe,
    load_template,
    parse_section,
    render_section,
    render_skills,
    render_summary,
    skills_line_count,
)
from .verify import (
    Issue,
    keyword_coverage,
    tech_lexicon,
    verify_bullet,
    verify_summary,
)

BULLET_TEMPERATURE = 0.35
REPAIR_TEMPERATURE = 0.2
MAX_REPAIR_ROUNDS = 2


def _fill(template: str, **kwargs: str) -> str:
    out = template
    for key, value in kwargs.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _load_v2_prompt(name: str) -> str:
    from .config import PROMPTS_DIR

    return (PROMPTS_DIR / "v2" / name).read_text(encoding="utf-8")


def _latex_text(text: str) -> str:
    """Make one written bullet or summary safe for the LaTeX template."""
    return latex_safe(text)


def _format_facts(facts: list[Fact]) -> str:
    lines: list[str] = []
    for i, fact in enumerate(facts, 1):
        lines.append(f"{i}. WHAT HAPPENED: {fact.core}")
        lines.append(f"   tools you may name: {', '.join(fact.tools) if fact.tools else 'none'}")
        lines.append(
            f"   numbers you may use: {', '.join(fact.metrics) if fact.metrics else 'none — write it without a number'}"
        )
        if fact.frozen:
            lines.append(f"   never change this pairing: {', '.join(fact.frozen)}")
        if fact.angles:
            lines.append(f"   honest reframings: {'; '.join(fact.angles)}")
        lines.append("")
    return "\n".join(lines).strip()


def _flex_rule(selection: Selection) -> str:
    if selection.kind == "project":
        return (
            "A personal project. Same contract as everything else: only these facts, "
            "only these numbers and tools."
        )
    if selection.flexible:
        return (
            "This is the candidate's CURRENT role and the most flexible part of the resume. "
            "Lead with whatever these facts genuinely offer this posting and describe them in "
            "the posting's vocabulary at a truthful level of abstraction. The work itself does "
            "not change — only which part of it you foreground and the words you tell it in. "
            "If a fact has little to do with this domain, write it plainly rather than "
            "costuming it as domain work."
        )
    return (
        "This is past history and is FIXED. Keep every fact in its own domain — re-angle the "
        "emphasis toward what this posting values, but never restage the work somewhere else."
    )


def _jd_fields(analysis: dict, section: str) -> dict[str, str]:
    placement = (analysis.get("keyword_placement") or {}).get(section) or []
    if not placement:
        placement = (analysis.get("must_have_skills") or [])[:6]
    return {
        "JD_TITLE": str(analysis.get("role_title") or "(not stated)"),
        "JD_DOMAIN": str(analysis.get("domain") or "(not stated)"),
        "JD_PRACTICES": ", ".join(analysis.get("domain_practices") or []) or "(not stated)",
        "JD_TOOLS": ", ".join(analysis.get("tools") or []) or "(none identified)",
        "PLACEMENT_KEYWORDS": ", ".join(placement) or "(use your judgment)",
    }


def _issue_block(bullets: list[str], facts: list[Fact], issues_by_index: dict[int, list[Issue]]) -> str:
    lines = [
        "",
        "## YOUR PREVIOUS ATTEMPT FAILED MECHANICAL CHECKS",
        "Return ALL bullets again in the same order. Keep the ones not listed below "
        "exactly as they are; rewrite only the flagged ones so they pass.",
        "",
    ]
    for idx, issues in sorted(issues_by_index.items()):
        previous = bullets[idx] if idx < len(bullets) else ""
        lines.append(f"BULLET {idx + 1} (fact: {facts[idx].core[:70]}...)")
        lines.append(f'  you wrote: "{previous}"')
        for issue in issues:
            lines.append(f"  PROBLEM [{issue.code}]: {issue.message}")
        lines.append("")
    return "\n".join(lines)


def _write_bullets_for_block(
    selection: Selection,
    analysis: dict,
    lexicon: set,
) -> tuple[list[str], dict]:
    """Write, verify, and repair the bullets for one job or project block."""
    facts = selection.facts
    count = len(facts)
    if not count:
        return [], {"block": selection.owner_label, "status": "no-facts"}

    template = _load_v2_prompt("bullets.txt")
    tenure = ""
    base_prompt = _fill(
        template,
        BLOCK_KIND="Job" if selection.kind == "role" else "Project",
        BLOCK_LABEL=selection.owner_label,
        TENURE=tenure,
        FLEX_RULE=_flex_rule(selection),
        FACTS=_format_facts(facts),
        COUNT=str(count),
        FIX_BLOCK="",
        **_jd_fields(analysis, "experience" if selection.kind == "role" else "projects"),
    )

    bullets: list[str] = []
    attempts = 0
    prompt = base_prompt
    temperature = BULLET_TEMPERATURE
    issues_by_index: dict[int, list[Issue]] = {}

    while attempts <= MAX_REPAIR_ROUNDS:
        attempts += 1
        try:
            raw = call_llm_json(prompt, temperature=temperature, max_tokens=1600, role="writer")
        except Exception as exc:
            _debug_dump(f"v2_{selection.owner_id}_error", str(exc))
            break

        candidate = raw.get("bullets") if isinstance(raw, dict) else None
        if not isinstance(candidate, list):
            _debug_dump(f"v2_{selection.owner_id}_bad_shape", str(raw)[:400])
            break

        candidate = [_latex_text(str(b or "").strip()) for b in candidate][:count]
        while len(candidate) < count:
            candidate.append("")

        issues_by_index = {}
        for i, (bullet, fact) in enumerate(zip(candidate, facts)):
            found = verify_bullet(bullet, fact, lexicon)
            errors = [x for x in found if x.severity == "error"]
            if errors:
                issues_by_index[i] = errors

        bullets = candidate
        if not issues_by_index:
            break

        prompt = base_prompt + _issue_block(candidate, facts, issues_by_index)
        temperature = REPAIR_TEMPERATURE

    # Anything still failing falls back to the plain fact — always true, never pretty.
    fallbacks: list[str] = []
    if not bullets:
        bullets = [""] * count
        issues_by_index = {i: [Issue("no-output", "writer produced nothing")] for i in range(count)}
    for idx in list(issues_by_index.keys()):
        bullets[idx] = _latex_text(facts[idx].core)
        fallbacks.append(facts[idx].id)

    meta = {
        "block": selection.owner_label,
        "attempts": attempts,
        "facts": [f.id for f in facts],
        "fact_scores": selection.scores,
        "fell_back_to_fact": fallbacks,
    }
    _debug_dump(f"v2_{selection.owner_id}_bullets", json.dumps({**meta, "bullets": bullets}, indent=2))
    return bullets, meta


def _write_summary(
    analysis: dict,
    selections: dict[str, Selection],
    rendered_bullets: list[str],
    lexicon: set,
    fallback: str,
) -> tuple[str, dict]:
    facts = [f for sel in selections.values() for f in sel.facts]
    bullet_numbers = set(re.findall(r"\d+(?:\.\d+)?", " ".join(rendered_bullets)))
    template = _load_v2_prompt("summary.txt")
    base_prompt = _fill(
        template,
        SUMMARY_ANGLE=str(analysis.get("ideal_summary_angle") or "(none given)"),
        RESUME_BULLETS="\n".join(f"- {b}" for b in rendered_bullets),
        FACTS=_format_facts(facts),
        FIX_BLOCK="",
        **_jd_fields(analysis, "summary"),
    )

    prompt = base_prompt
    temperature = BULLET_TEMPERATURE
    summary = ""
    issues: list[Issue] = []
    for attempt in range(MAX_REPAIR_ROUNDS + 1):
        try:
            raw = call_llm_json(prompt, temperature=temperature, max_tokens=600, role="writer")
        except Exception as exc:
            _debug_dump("v2_summary_error", str(exc))
            break
        summary = _latex_text(str((raw or {}).get("summary") or "").strip())
        issues = [
            i
            for i in verify_summary(
                summary, facts, lexicon, bullet_numbers, " ".join(rendered_bullets)
            )
            if i.severity == "error"
        ]
        if summary and not issues:
            return summary, {"attempts": attempt + 1, "fell_back": False}
        prompt = base_prompt + (
            "\n\n## YOUR PREVIOUS ATTEMPT FAILED MECHANICAL CHECKS\n"
            f'you wrote: "{summary}"\n'
            + "\n".join(f"  PROBLEM [{i.code}]: {i.message}" for i in issues)
            + "\nRewrite it so every problem is gone."
        )
        temperature = REPAIR_TEMPERATURE

    return fallback, {
        "attempts": MAX_REPAIR_ROUNDS + 1,
        "fell_back": True,
        "issues": [i.message for i in issues],
    }


def _original_summary_text() -> str:
    text = load_template("summary")
    m = re.search(r"\\textit\{([\s\S]*?)\}\s*\n?\s*\\end\{center\}", text)
    return (m.group(1).strip() if m else "").replace("\n", " ")


def measure(
    resume_plain: str,
    analysis: dict,
    selections: dict[str, Selection],
    bullets: list[str],
    skills_lines: list[tuple[str, list[str]]],
) -> dict:
    """Deterministic quality measurement. Same resume in, same numbers out."""
    keywords = jd_keywords_of(analysis)
    coverage = keyword_coverage(resume_plain, keywords)

    with_metric = sum(1 for b in bullets if re.search(r"\d", b))
    metric_density = with_metric / len(bullets) if bullets else 0.0

    profile = theme_profile(analysis)
    wanted = {t for t, w in profile.items() if w >= 2.0}
    covered = {t for sel in selections.values() for f in sel.facts for t in f.themes}
    domain_match = len(wanted & covered) / len(wanted) if wanted else 1.0

    skill_items = [i for _, items in skills_lines for i in items]
    jd_tools = {t.strip().lower() for t in (analysis.get("tools") or []) if t.strip()}
    tools_in_skills = sum(1 for t in jd_tools if any(tool_matches(i, {t}) for i in skill_items))
    skills_coverage = tools_in_skills / len(jd_tools) if jd_tools else 1.0

    score = round(
        45 * coverage["coverage"]
        + 20 * metric_density
        + 20 * domain_match
        + 15 * skills_coverage
    )
    return {
        "score": score,
        "breakdown": {
            "keyword_coverage": {
                "score": round(45 * coverage["coverage"], 1),
                "max": 45,
                "details": f"{len(coverage['matched'])} of {len(keywords)} JD keywords present",
            },
            "metric_density": {
                "score": round(20 * metric_density, 1),
                "max": 20,
                "details": f"{with_metric} of {len(bullets)} bullets carry a real number",
            },
            "domain_match": {
                "score": round(20 * domain_match, 1),
                "max": 20,
                "details": f"{len(wanted & covered)} of {len(wanted)} themes this JD emphasizes are evidenced",
            },
            "skills_coverage": {
                "score": round(15 * skills_coverage, 1),
                "max": 15,
                "details": f"{tools_in_skills} of {len(jd_tools)} JD tools listed in skills",
            },
        },
        "matched_keywords": coverage["matched"],
        "missing_keywords": coverage["missing"],
    }


def build_resume_v2(job_description: str, on_progress=None) -> dict:
    def emit(event: str, payload: dict) -> None:
        if on_progress is None:
            return
        try:
            on_progress(event, payload)
        except Exception:
            pass

    jd = sanitize_jd(job_description)
    bank = load_fact_bank()
    lexicon = tech_lexicon(bank)

    # --- 1. JD analysis (one LLM call) --------------------------------------
    analysis, jd_ok = run_jd_agent(jd)
    emit("jd", {"jd_analysis": analysis, "jd_agent_ok": jd_ok})

    # --- 2. Templates decide the shape --------------------------------------
    templates = {name: load_template(name) for name in ("summary", "experience", "projects", "skills")}
    blocks = {name: parse_section(templates[name]) for name in ("experience", "projects")}
    slots_by_label = {
        block.label: block.bullet_count
        for section_blocks in blocks.values()
        for block in section_blocks
    }

    # --- 3. Select real facts (pure code) -----------------------------------
    selections = select_facts(bank, analysis, slots_by_label)
    missing = [label for label in slots_by_label if label not in selections]
    if missing:
        raise ValueError(
            "These LaTeX blocks have no matching entry in data/facts.yaml: "
            + ", ".join(missing)
            + ". Add a role/project with that exact company or project name."
        )

    # --- 4. Write bullets, one call per block, in parallel ------------------
    order = [block.label for section in ("experience", "projects") for block in blocks[section]]
    with ThreadPoolExecutor(max_workers=max(1, len(order))) as pool:
        futures = {
            label: pool.submit(_write_bullets_for_block, selections[label], analysis, lexicon)
            for label in order
        }
        written = {}
        block_meta = []
        for label in order:
            try:
                bullets, meta = futures[label].result()
            except Exception as exc:
                bullets = [_latex_text(f.core) for f in selections[label].facts]
                meta = {"block": label, "status": f"error: {exc}", "fell_back_to_fact": "all"}
            written[label] = bullets
            block_meta.append(meta)

    # --- 5. Render bullets into the templates -------------------------------
    rendered: dict[str, str] = {}
    for section in ("experience", "projects"):
        bullets_by_block = {
            idx: written.get(block.label, []) for idx, block in enumerate(blocks[section])
        }
        rendered[section] = render_section(templates[section], bullets_by_block)

    # --- 6. Skills: selection from the bank, no LLM -------------------------
    # Constrained by the facts that actually made the page, so the skills line
    # backs up the bullets instead of claiming every technology at once.
    evidenced = {t for sel in selections.values() for f in sel.facts for t in f.tools}
    skills_lines = select_skills(
        bank, analysis, skills_line_count(templates["skills"]), evidenced
    )
    rendered["skills"] = render_skills(templates["skills"], skills_lines)

    # --- 7. Summary, written against the bullets that now exist -------------
    all_bullets = [b for label in order for b in written.get(label, [])]
    summary_text, summary_meta = _write_summary(
        analysis, selections, all_bullets, lexicon, _original_summary_text()
    )
    rendered["summary"] = render_summary(templates["summary"], summary_text)
    emit("sections", {"sections": rendered})

    # --- 8. Bold JD keywords, then assemble ---------------------------------
    keywords = jd_keywords_of(analysis)
    for section in ("experience", "projects"):
        rendered[section] = bold_metrics_in_bullets(rendered[section])
        rendered[section] = bold_keywords_in_bullets(rendered[section], keywords)

    full_latex = assemble_full_resume(
        load_full_template(),
        rendered["summary"],
        rendered["experience"],
        rendered["projects"],
        rendered["skills"],
    )
    resume_plain = latex_to_plain(full_latex)

    # --- 9. Measure (deterministic) -----------------------------------------
    measurement = measure(resume_plain, analysis, selections, all_bullets, skills_lines)

    payload = {
        "latex": full_latex,
        "sections": rendered,
        "jd_analysis": analysis,
        "meta": {
            "architecture": "facts-v3",
            "final": True,
            "jd_agent_ok": jd_ok,
            "llm_calls": 1 + len(order) + summary_meta.get("attempts", 1),
            "final_score": measurement["score"],
            "blocks": block_meta,
            "summary": summary_meta,
            "selected_facts": {
                label: [
                    {"id": f.id, "score": s}
                    for f, s in zip(sel.facts, sel.scores)
                ]
                for label, sel in selections.items()
            },
            "skills_lines": [name for name, _ in skills_lines],
            "summary_words": word_count(latex_to_plain(rendered["summary"])),
        },
        "measurement": measurement,
    }
    emit("final", payload)
    return payload
