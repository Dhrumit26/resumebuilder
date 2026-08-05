"""Fact-grounded pipeline (with optional fabricated current role).

    JD analysis -> select real facts (code) -> write bullets -> verify (code)
    -> targeted repair -> render into the LaTeX templates -> measure

Differences from the v1 pipeline that matter:
- By default, writers never invent. Every bullet is licensed by one fact in
  data/facts.yaml, and verify.py checks that mechanically.
- When a role has fabricated: true (Clerxi), that block invents natively in the
  JD's domain — craft checks only; Intuit and projects stay fact-grounded.
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
    FAB_MAX_WORDS,
    FAB_MIN_WORDS,
    _mentioned_tech,
    _plain,
    keyword_coverage,
    spine_labels_from_analysis,
    tech_lexicon,
    verify_bullet,
    verify_fabricated_block,
    verify_summary,
)

BULLET_TEMPERATURE = 0.35
FABRICATED_TEMPERATURE = 0.2
REPAIR_TEMPERATURE = 0.15
MAX_REPAIR_ROUNDS = 2
MAX_FABRICATED_REPAIR_ROUNDS = 6


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
    if selection.fabricated:
        company = selection.owner_label
        if "intuit" in company.lower():
            return (
                f"FABRICATED MODE for a PAST INTERNSHIP at {company} "
                f"(~{selection.tenure_months or 3} months). Invent a coherent mid-level "
                "intern story in the posting's engineering domain — a DIFFERENT system and "
                "a DIFFERENT facet of the JD than the current role above. Prefer complementary "
                "work: testing/CI, API contracts, platform/refactor, coverage, handoffs — "
                "while the current role owns the primary product spine. Same domain fit, "
                "not a clone. Company name stays real. Size scope to an internship. "
                "Each bullet stands alone; no vague 'cloud technology'."
            )
        return (
            f"FABRICATED MODE for the CURRENT role at {company}: invent the PRIMARY "
            "product story natively in the posting's engineering domain (main spine: "
            "RAG/agents/core stack). Ignore the fact bank for grounding. Company name "
            "stays real; product follows the JD domain, not brand tokens. Do not adopt "
            "the posting's industry. Each bullet stands alone — no mechanical "
            "'that dashboard' chains; name real services; soften coverage→defect claims."
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


def _differentiation_block(
    selection: Selection,
    sibling_fabricated: dict[str, list[str]] | None,
) -> str:
    """Instructions so a second fabricated role complements the first, not clones it."""
    if not selection.fabricated:
        return ""
    siblings = {
        label: bullets
        for label, bullets in (sibling_fabricated or {}).items()
        if label != selection.owner_label and bullets
    }
    if not siblings:
        if "intuit" in selection.owner_label.lower():
            return (
                "\n## STRATEGIC POSITIONING\n"
                "No sibling role text yet. Still: invent an INTERNSHIP-scoped story "
                "(testing, CI, contracts, platform) — leave the deepest product spine "
                "for the current role when both are fabricated.\n"
            )
        return (
            "\n## STRATEGIC POSITIONING\n"
            "You write the CURRENT role first. Own the posting's primary product spine. "
            "A later internship block will cover a complementary facet — do not try to "
            "cover every JD duty here.\n"
        )

    lines = [
        "",
        "## STRATEGIC DIFFERENTIATION — DO NOT CLONE",
        "Another fabricated role on this resume already claimed the work below.",
        "Your block must MATCH the JD strategically but tell a DIFFERENT story:",
        "- Different system / product (not the same dashboard, console, Yocto image,",
        "  MQTT stack, OS app, security protocol suite, or API).",
        "- Different JD facet (if they own the primary product/spine, you own tests/CI/",
        "  contracts/platform/reliability/tooling — or the reverse).",
        "- Different metrics (never reuse the same from→to numbers or % figures).",
        "- Different opening verbs AND sentence shapes — no paraphrased twins.",
        "- FORBIDDEN: mad-lib clones (same bullet with Built→Developed and 45s→50s).",
        "  If a reader can line up your 5 bullets with theirs 1:1, you FAIL.",
        "- Internship tenure: smaller scope, fewer epic claims than a current role.",
        "",
    ]
    for label, bullets in siblings.items():
        lines.append(f"### Already written for {label} — do NOT repeat this story")
        for i, b in enumerate(bullets, 1):
            lines.append(f"{i}. {b}")
        lines.append("")
    return "\n".join(lines)


# Posting titles carry the employer's internal furniture: the team that owns the
# req, the grad-programme label, the location. Mirroring a title is good; echoing
# "AI Agentic Product Dev Team engineer" back at them is not.
_TITLE_NOISE = re.compile(
    r"\b(team|group|org|organi[sz]ation|division|department|dept|"
    r"new\s+(college\s+)?grad(uate)?|university\s+grad(uate)?|early\s+career|"
    r"campus|intern(ship)?\s+program|20\d\d|remote|hybrid|onsite|us|usa|emea)\b",
    re.I,
)


def clean_role_title(title: str) -> str:
    """Strip team names, grad-programme labels and locations from a posting title."""
    text = re.sub(r"\([^)]*\)", " ", title or "")          # "(US)", "(Remote)"
    # Not "/" — "Platform / DevOps Engineer" is one title, not two segments.
    segments = [s.strip() for s in re.split(r"\s*[-–—,|]\s*", text) if s.strip()]
    kept = [s for s in segments if not _TITLE_NOISE.search(s)]
    cleaned = ", ".join(kept) if kept else " ".join(segments)
    return re.sub(r"\s+", " ", cleaned).strip() or (title or "").strip()


def _jd_fields(analysis: dict, section: str, *, fabricated: bool = False) -> dict[str, str]:
    placement = (analysis.get("keyword_placement") or {}).get(section) or []
    if not placement:
        placement = (analysis.get("must_have_skills") or [])[:6]
    practices = [str(p) for p in (analysis.get("domain_practices") or []) if str(p).strip()]
    concepts = [str(c) for c in (analysis.get("concepts") or []) if str(c).strip()]
    tools = [str(t) for t in (analysis.get("tools") or []) if str(t).strip()]

    # Primary language / cloud: first JD-named one, else a safe default.
    lang_order = (
        "Python", "TypeScript", "JavaScript", "Java", "Go", "C++", "C#", "Rust", "Kotlin", "Swift",
    )
    primary_lang = next((t for t in tools if t in lang_order), None) or next(
        (l for l in lang_order if any(l.lower() == t.lower() for t in tools)),
        "Python",
    )
    cloud_order = ("AWS", "GCP", "Azure")
    primary_cloud = next((t for t in tools if t in cloud_order), "AWS")

    spine = spine_labels_from_analysis(analysis)
    if not spine:
        spine = practices[:6] or concepts[:6]

    # Fabricated experience should surface the spine, not the weak "backend/APIs"
    # placement the JD agent often puts under experience.
    if fabricated:
        placement = spine[:8] or placement

    return {
        "JD_TITLE": clean_role_title(str(analysis.get("role_title") or "")) or "(not stated)",
        "JD_DOMAIN": str(analysis.get("domain") or "(not stated)"),
        "JD_PRACTICES": ", ".join(practices) or "(not stated)",
        "JD_TOOLS": ", ".join(tools) or "(none identified)",
        "JD_CONCEPTS": ", ".join(concepts) or "(none identified)",
        "JD_SPINE": ", ".join(spine) or ", ".join(practices) or "(use domain practices)",
        "PRIMARY_LANGUAGE": primary_lang,
        "PRIMARY_CLOUD": primary_cloud,
        "PLACEMENT_KEYWORDS": ", ".join(placement) or "(use your judgment)",
    }


def _issue_block(
    bullets: list[str],
    facts: list[Fact],
    issues_by_index: dict[int, list[Issue]],
    *,
    fabricated: bool = False,
    block_issues: list[Issue] | None = None,
) -> str:
    lines = [
        "",
        "## YOUR PREVIOUS ATTEMPT FAILED MECHANICAL CHECKS",
        "Return ALL bullets again in the same order. Keep the ones not listed below "
        "exactly as they are; rewrite only the flagged ones so they pass.",
        "",
    ]
    if block_issues:
        lines.append("## BLOCK-LEVEL FAILURES — rewrite the WHOLE set to fix these")
        for issue in block_issues:
            lines.append(f"  PROBLEM [{issue.code}]: {issue.message}")
        lines.append("")
    for idx, issues in sorted(issues_by_index.items()):
        # Block-level issues are already listed above; skip duplicating on bullet 0.
        per_bullet = [i for i in issues if not block_issues or i not in block_issues]
        if not per_bullet:
            continue
        previous = bullets[idx] if idx < len(bullets) else ""
        if fabricated:
            lines.append(f"BULLET {idx + 1}")
        else:
            lines.append(f"BULLET {idx + 1} (fact: {facts[idx].core[:70]}...)")
        lines.append(f'  you wrote: "{previous}"')
        for issue in per_bullet:
            lines.append(f"  PROBLEM [{issue.code}]: {issue.message}")
        lines.append("")
    return "\n".join(lines)


def _write_bullets_for_block(
    selection: Selection,
    analysis: dict,
    lexicon: set,
    sibling_fabricated: dict[str, list[str]] | None = None,
) -> tuple[list[str], dict]:
    """Write, verify, and repair the bullets for one job or project block."""
    facts = selection.facts
    count = len(facts)
    if not count:
        return [], {"block": selection.owner_label, "status": "no-facts"}

    tenure = ""
    if selection.tenure_months:
        tenure = f"\nTenure on the resume: about {selection.tenure_months} months"

    jd_kwargs = _jd_fields(
        analysis,
        "experience" if selection.kind == "role" else "projects",
        fabricated=selection.fabricated,
    )

    if selection.fabricated:
        template = _load_v2_prompt("bullets_fabricated.txt")
        base_prompt = _fill(
            template,
            BLOCK_KIND="Job" if selection.kind == "role" else "Project",
            BLOCK_LABEL=selection.owner_label,
            TENURE=tenure,
            FLEX_RULE=_flex_rule(selection),
            DIFFERENTIATION=_differentiation_block(selection, sibling_fabricated),
            COUNT=str(count),
            FIX_BLOCK="",
            **jd_kwargs,
        )
    else:
        template = _load_v2_prompt("bullets.txt")
        base_prompt = _fill(
            template,
            BLOCK_KIND="Job" if selection.kind == "role" else "Project",
            BLOCK_LABEL=selection.owner_label,
            TENURE=tenure,
            FLEX_RULE=_flex_rule(selection),
            FACTS=_format_facts(facts),
            COUNT=str(count),
            FIX_BLOCK="",
            **jd_kwargs,
        )

    bullets: list[str] = []
    attempts = 0
    prompt = base_prompt
    temperature = FABRICATED_TEMPERATURE if selection.fabricated else BULLET_TEMPERATURE
    issues_by_index: dict[int, list[Issue]] = {}
    block_issues: list[Issue] = []
    max_rounds = MAX_FABRICATED_REPAIR_ROUNDS if selection.fabricated else MAX_REPAIR_ROUNDS
    best_bullets: list[str] | None = None
    best_error_count: int | None = None
    best_block_issues: list[Issue] = []
    siblings = [
        b
        for label, blist in (sibling_fabricated or {}).items()
        if label != selection.owner_label
        for b in blist
    ]

    while attempts <= max_rounds:
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
        for i, bullet in enumerate(candidate):
            fact = None if selection.fabricated else facts[i]
            if selection.fabricated:
                found = verify_bullet(
                    bullet,
                    None,
                    lexicon,
                    grounded=False,
                    min_words=FAB_MIN_WORDS,
                    max_words=FAB_MAX_WORDS,
                )
            else:
                found = verify_bullet(bullet, fact, lexicon, grounded=True)
            errors = [x for x in found if x.severity == "error"]
            if errors:
                issues_by_index[i] = errors

        block_issues = []
        if selection.fabricated:
            block_issues = [
                i
                for i in verify_fabricated_block(
                    candidate, analysis, sibling_bullets=siblings
                )
                if i.severity == "error"
            ]
            if block_issues:
                issues_by_index.setdefault(0, []).extend(block_issues)

        bullets = candidate
        error_count = sum(len(v) for v in issues_by_index.values())
        if best_error_count is None or error_count < best_error_count:
            best_error_count = error_count
            best_bullets = list(candidate)
            best_block_issues = list(block_issues)

        if not issues_by_index:
            break

        prompt = base_prompt + _issue_block(
            candidate,
            facts,
            issues_by_index,
            fabricated=selection.fabricated,
            block_issues=block_issues,
        )
        temperature = REPAIR_TEMPERATURE

    if selection.fabricated and best_bullets is not None:
        bullets = best_bullets
        block_issues = best_block_issues

    fallbacks: list[str] = []
    if not bullets:
        bullets = [""] * count
        issues_by_index = {i: [Issue("no-output", "writer produced nothing")] for i in range(count)}

    # Grounded blocks: anything still failing falls back to the plain fact.
    # Fabricated blocks keep the best candidate — invent mode has no fact text to fall to.
    if not selection.fabricated:
        for idx in list(issues_by_index.keys()):
            bullets[idx] = _latex_text(facts[idx].core)
            fallbacks.append(facts[idx].id)

    meta = {
        "block": selection.owner_label,
        "attempts": attempts,
        "facts": [f.id for f in facts],
        "fact_scores": selection.scores,
        "fell_back_to_fact": fallbacks,
        "fabricated": selection.fabricated,
        "block_issues": [i.code for i in block_issues] if selection.fabricated else [],
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
    fabricated_ok = any(sel.fabricated for sel in selections.values())
    bullet_numbers = set(re.findall(r"\d+(?:\.\d+)?", " ".join(rendered_bullets)))
    template = _load_v2_prompt(
        "summary_fabricated.txt" if fabricated_ok else "summary.txt"
    )
    fill_kwargs = dict(
        SUMMARY_ANGLE=str(analysis.get("ideal_summary_angle") or "(none given)"),
        RESUME_BULLETS="\n".join(f"- {b}" for b in rendered_bullets),
        FIX_BLOCK="",
        **_jd_fields(analysis, "summary", fabricated=fabricated_ok),
    )
    if not fabricated_ok:
        fill_kwargs["FACTS"] = _format_facts(facts)
    base_prompt = _fill(template, **fill_kwargs)

    prompt = base_prompt
    temperature = FABRICATED_TEMPERATURE if fabricated_ok else BULLET_TEMPERATURE
    summary = ""
    issues: list[Issue] = []
    max_rounds = MAX_FABRICATED_REPAIR_ROUNDS if fabricated_ok else MAX_REPAIR_ROUNDS
    for attempt in range(max_rounds + 1):
        try:
            raw = call_llm_json(prompt, temperature=temperature, max_tokens=600, role="writer")
        except Exception as exc:
            _debug_dump("v2_summary_error", str(exc))
            break
        summary = _latex_text(str((raw or {}).get("summary") or "").strip())
        issues = [
            i
            for i in verify_summary(
                summary,
                facts,
                lexicon,
                bullet_numbers,
                " ".join(rendered_bullets),
                fabricated_ok=fabricated_ok,
            )
            if i.severity == "error"
        ]
        if summary and not issues:
            return summary, {"attempts": attempt + 1, "fell_back": False, "fabricated": fabricated_ok}
        prompt = base_prompt + (
            "\n\n## YOUR PREVIOUS ATTEMPT FAILED MECHANICAL CHECKS\n"
            f'you wrote: "{summary}"\n'
            + "\n".join(f"  PROBLEM [{i.code}]: {i.message}" for i in issues)
            + "\nRewrite it so every problem is gone."
        )
        temperature = REPAIR_TEMPERATURE

    return fallback, {
        "attempts": max_rounds + 1,
        "fell_back": True,
        "fabricated": fabricated_ok,
        "issues": [i.message for i in issues],
    }


def _original_summary_text() -> str:
    text = load_template("summary")
    m = re.search(r"\\textit\{([\s\S]*?)\}\s*\n?\s*\\end\{center\}", text)
    return (m.group(1).strip() if m else "").replace("\n", " ")


def capability_gaps(bank, analysis: dict) -> tuple[list[str], list[str]]:
    """What this posting wants that NO fact in the bank evidences.

    Returns (missing tools, missing capability themes). These are the questions
    the resume cannot answer — not because the writing is weak, but because the
    fact bank has nothing to draw on.
    """
    bank_tools = {t.lower() for t in bank.all_tools()}
    bank_themes = {t for f in bank.all_facts() for t in f.themes}

    # Match against everything the bank actually says, not just its tool names.
    # "Backend Development" and "Model Inference" are capabilities, so they are
    # evidenced by the wording of the facts ("backend retrieval services",
    # "inference costs") rather than by an entry in a tools list.
    corpus = " ".join(
        [f.core + " " + " ".join(f.angles) + " " + " ".join(f.tools) for f in bank.all_facts()]
        + [i for c in bank.skill_categories for i in c.items]
    )

    wanted: list[str] = []
    for key in ("tools", "must_have_skills"):
        for term in analysis.get(key) or []:
            if term not in wanted:
                wanted.append(term)

    evidenced = set(keyword_coverage(corpus, wanted)["matched"])
    missing_tools = [
        term for term in wanted
        if term not in evidenced and not tool_matches(term, bank_tools)
    ]

    profile = theme_profile(analysis)
    missing_themes = [
        theme for theme, weight in sorted(profile.items(), key=lambda kv: -kv[1])
        if weight >= 2.0 and theme not in bank_themes
    ]
    return missing_tools, missing_themes


_QUESTION_PROMPT = """You help a candidate remember work they actually did, so it can go on
their resume. You are NOT writing resume content and you never assert that they did anything.

This job posting wants experience their fact bank does not currently cover:
  technologies: {TOOLS}
  capabilities: {THEMES}

What they DO have on record:
{FACTS}

Write up to {COUNT} short questions asking whether they did any of this at their current job
({COMPANY}) or on a project. Rules:
- One specific thing per question. Never "do you have backend experience?" — ask
  "did you write or own any service endpoints, and roughly how many?"
- Always ask for the measurable part: how many, how much faster, before/after.
- Ask only about things adjacent to what they already do; skip anything absurd
  for their background.
- A question is not a suggestion that they claim it. If the answer is no, it is no.

Return ONLY: {{"questions": ["...", "..."]}}"""


def verification_questions(bank, analysis: dict, limit: int = 6) -> list[str]:
    """Turn this posting's gaps into questions that could grow the fact bank."""
    missing_tools, missing_themes = capability_gaps(bank, analysis)
    if not missing_tools and not missing_themes:
        return []

    known = "\n".join(f"- {f.core}" for f in bank.all_facts()[:12])
    flexible = next((r.company for r in bank.roles if r.flexible), "their current role")
    prompt = _QUESTION_PROMPT.format(
        TOOLS=", ".join(missing_tools[:10]) or "(none)",
        THEMES=", ".join(t.replace("-", " ") for t in missing_themes[:8]) or "(none)",
        FACTS=known,
        COUNT=limit,
        COMPANY=flexible,
    )
    try:
        raw = call_llm_json(prompt, temperature=0.3, max_tokens=700, role="writer")
        questions = [str(q).strip() for q in (raw.get("questions") or []) if str(q).strip()]
        if questions:
            return questions[:limit]
    except Exception as exc:
        _debug_dump("v2_questions_error", str(exc))

    # Deterministic fallback so a gap is never silently dropped.
    out = [
        f"Did you use {tool} at {flexible} or on a project? If so, for what, and what was the result?"
        for tool in missing_tools[:limit]
    ]
    out += [
        f"Have you done any {theme.replace('-', ' ')} work? What did you change, and by how much?"
        for theme in missing_themes[: max(0, limit - len(out))]
    ]
    return out[:limit]


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
    # Fabricated roles are written natively for the JD — credit the posting's themes.
    if any(sel.fabricated for sel in selections.values()):
        covered |= wanted
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

    # --- 4. Write bullets ----------------------------------------------------
    # Fabricated roles run SEQUENTIALLY so later ones (Intuit) see earlier ones
    # (Clerxi) and can strategically complement instead of cloning.
    # Grounded blocks still run in parallel.
    order = [block.label for section in ("experience", "projects") for block in blocks[section]]
    fab_order = [label for label in order if selections[label].fabricated]
    grounded_order = [label for label in order if not selections[label].fabricated]

    written: dict[str, list[str]] = {}
    block_meta: list[dict] = []
    sibling_fabricated: dict[str, list[str]] = {}

    for label in fab_order:
        try:
            bullets, meta = _write_bullets_for_block(
                selections[label], analysis, lexicon, sibling_fabricated
            )
        except Exception as exc:
            bullets = [_latex_text(f.core) for f in selections[label].facts]
            meta = {"block": label, "status": f"error: {exc}", "fell_back_to_fact": "all"}
        written[label] = bullets
        sibling_fabricated[label] = bullets
        block_meta.append(meta)

    if grounded_order:
        with ThreadPoolExecutor(max_workers=max(1, len(grounded_order))) as pool:
            futures = {
                label: pool.submit(
                    _write_bullets_for_block, selections[label], analysis, lexicon
                )
                for label in grounded_order
            }
            for label in grounded_order:
                try:
                    bullets, meta = futures[label].result()
                except Exception as exc:
                    bullets = [_latex_text(f.core) for f in selections[label].facts]
                    meta = {
                        "block": label,
                        "status": f"error: {exc}",
                        "fell_back_to_fact": "all",
                    }
                written[label] = bullets
                block_meta.append(meta)

    # Keep meta in document order for debugging.
    meta_by_label = {m.get("block"): m for m in block_meta}
    block_meta = [meta_by_label[label] for label in order if label in meta_by_label]

    # --- 5. Render bullets into the templates -------------------------------
    rendered: dict[str, str] = {}
    for section in ("experience", "projects"):
        bullets_by_block = {
            idx: written.get(block.label, []) for idx, block in enumerate(blocks[section])
        }
        rendered[section] = render_section(templates[section], bullets_by_block)

    # --- 6. Skills: selection from the bank, no LLM -------------------------
    # Constrained by what the resume actually shows. When inventing roles,
    # skills must follow fabricated bullets + JD tools on the page — not leftover
    # Playwright/RAG bank filler from a different domain.
    fabricated_any = any(sel.fabricated for sel in selections.values())
    evidenced = {t for sel in selections.values() for f in sel.facts for t in f.tools}
    # Expand lexicon with JD tools so invent stacks (C++, Yocto, MQTT, …) count.
    jd_tool_names = [
        str(t).strip()
        for key in ("tools", "must_have_skills", "exact_keywords_for_ats")
        for t in (analysis.get(key) or [])
        if str(t).strip()
    ]
    scan_lexicon = set(lexicon) | {t.lower() for t in jd_tool_names} | {
        "yocto", "mqtt", "amqp", "jenkins", "armbian", "c++", "c++11", "c++17", "c++20",
    }
    for label, bullets in written.items():
        for bullet in bullets:
            evidenced |= _mentioned_tech(_plain(bullet), scan_lexicon)
        if selections[label].fabricated:
            # Preserve original casing from JD when the bullet used that tool.
            plain = _plain(" ".join(bullets)).lower()
            for tool in jd_tool_names:
                if re.search(
                    r"(?<![A-Za-z0-9_])" + re.escape(tool) + r"(?![A-Za-z0-9_+#])",
                    plain,
                    re.I,
                ):
                    evidenced.add(tool)
    skills_lines = select_skills(
        bank,
        analysis,
        skills_line_count(templates["skills"]),
        evidenced,
        strict_evidence=fabricated_any,
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

    # --- 10. Turn this posting's gaps into questions ------------------------
    # What the resume could not say, asked back as questions. Answer one and it
    # becomes a fact, and every future posting can draw on it.
    questions = verification_questions(bank, analysis)
    missing_tools, missing_themes = capability_gaps(bank, analysis)

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
        "gaps": {
            "questions": questions,
            "missing_tools": missing_tools,
            "missing_capabilities": [t.replace("-", " ") for t in missing_themes],
        },
    }
    emit("final", payload)
    return payload


# ---------------------------------------------------------------------------
# Refine: apply a user suggestion to an already-built resume, then reassemble.
# ---------------------------------------------------------------------------

_BOLD_RE = re.compile(r"\\textbf\{([^}]*)\}")
_SKILLS_PARSE_RE = re.compile(r"\\textbf\{([^}]*)\}\s*\{:\s*([^}]*)\}")


def _strip_tex_markup(text: str) -> str:
    out = _BOLD_RE.sub(r"\1", text or "")
    out = out.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    out = out.replace("\\$", "$").replace("\\#", "#")
    return re.sub(r"\s+", " ", out).strip()


def _extract_summary_plain(summary_tex: str) -> str:
    open_idx = summary_tex.find("\\textit{")
    if open_idx == -1:
        return latex_to_plain(summary_tex)
    brace_idx = summary_tex.index("{", open_idx)
    depth = 0
    i = brace_idx
    while i < len(summary_tex):
        ch = summary_tex[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _strip_tex_markup(summary_tex[brace_idx + 1 : i])
        i += 1
    return latex_to_plain(summary_tex)


def _extract_block_bullets(section_tex: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for block in parse_section(section_tex):
        out[block.label] = [_strip_tex_markup(slot.body) for slot in block.slots]
    return out


def _extract_skills_lines(skills_tex: str) -> list[tuple[str, list[str]]]:
    lines: list[tuple[str, list[str]]] = []
    for m in _SKILLS_PARSE_RE.finditer(skills_tex or ""):
        items = [x.strip() for x in m.group(2).split(",") if x.strip()]
        lines.append((_strip_tex_markup(m.group(1)), items))
    return lines


def _format_block_bullets(blocks: dict[str, list[str]]) -> str:
    parts: list[str] = []
    for label, bullets in blocks.items():
        parts.append(f"### {label} ({len(bullets)} bullets)")
        for i, b in enumerate(bullets, 1):
            parts.append(f"{i}. {b}")
        parts.append("")
    return "\n".join(parts).strip() or "(none)"


def _format_skills_for_prompt(lines: list[tuple[str, list[str]]]) -> str:
    if not lines:
        return "(none)"
    return "\n".join(f"- {name}: {', '.join(items)}" for name, items in lines)


def _pad_bullets(new: list, count: int, fallback: list[str]) -> list[str]:
    cleaned = [str(b).strip() for b in (new or []) if str(b).strip()]
    out = cleaned[:count]
    while len(out) < count:
        out.append(fallback[len(out)] if len(out) < len(fallback) else fallback[-1])
    return [_latex_text(b) for b in out]


def refine_resume_v2(
    job_description: str,
    sections: dict,
    suggestion: str,
    jd_analysis: dict | None = None,
) -> dict:
    """Rewrite an existing v2 resume using one user suggestion."""
    suggestion = (suggestion or "").strip()
    if len(suggestion) < 3:
        raise ValueError("Suggestion is too short — say what you want changed.")
    if not sections or not all(
        sections.get(k) for k in ("summary", "experience", "projects", "skills")
    ):
        raise ValueError("Refine needs the full sections object from the last build.")

    jd = sanitize_jd(job_description)
    bank = load_fact_bank()

    if jd_analysis and isinstance(jd_analysis, dict) and jd_analysis.get("role_title"):
        analysis = jd_analysis
        jd_ok = True
    else:
        analysis, jd_ok = run_jd_agent(jd)

    current_summary = _extract_summary_plain(sections["summary"])
    current_exp = _extract_block_bullets(sections["experience"])
    current_proj = _extract_block_bullets(sections["projects"])
    current_skills = _extract_skills_lines(sections["skills"])

    templates = {name: load_template(name) for name in ("summary", "experience", "projects", "skills")}
    exp_blocks = parse_section(templates["experience"])
    proj_blocks = parse_section(templates["projects"])
    slots_by_label = {
        block.label: block.bullet_count
        for section_blocks in (exp_blocks, proj_blocks)
        for block in section_blocks
    }
    selections = select_facts(bank, analysis, slots_by_label)

    prompt = _fill(
        _load_v2_prompt("refine.txt"),
        SUGGESTION=suggestion,
        JD_TITLE=str(analysis.get("role_title") or ""),
        JD_DOMAIN=str(analysis.get("domain") or ""),
        JD_PRACTICES=", ".join(analysis.get("practices") or []),
        JD_TOOLS=", ".join(analysis.get("tools") or []),
        CURRENT_SUMMARY=current_summary,
        CURRENT_EXPERIENCE=_format_block_bullets(current_exp),
        CURRENT_PROJECTS=_format_block_bullets(current_proj),
        CURRENT_SKILLS=_format_skills_for_prompt(current_skills),
        FIX_BLOCK="",
    )
    raw = call_llm_json(prompt, temperature=0.25, max_tokens=3500)
    if not isinstance(raw, dict):
        raise ValueError("Refine model returned unexpected output. Try again.")

    changed = {str(c).lower() for c in (raw.get("changed") or []) if c}
    note = str(raw.get("note") or "Applied your suggestion.").strip()

    # Start from current rendered sections; overwrite only what the model rewrote.
    rendered = {
        "summary": sections["summary"],
        "experience": sections["experience"],
        "projects": sections["projects"],
        "skills": sections["skills"],
    }

    written: dict[str, list[str]] = {}
    for label, bullets in {**current_exp, **current_proj}.items():
        written[label] = [_latex_text(b) for b in bullets]

    if "summary" in changed and raw.get("summary"):
        summary_text = _latex_text(_strip_tex_markup(str(raw["summary"])))
        rendered["summary"] = render_summary(templates["summary"], summary_text)
    else:
        summary_text = _latex_text(current_summary)
        rendered["summary"] = render_summary(templates["summary"], summary_text)

    if "experience" in changed and isinstance(raw.get("experience"), dict):
        for block in exp_blocks:
            if block.label in raw["experience"]:
                written[block.label] = _pad_bullets(
                    raw["experience"][block.label],
                    block.bullet_count,
                    current_exp.get(block.label, [""] * block.bullet_count),
                )
        bullets_by_block = {
            idx: written.get(block.label, []) for idx, block in enumerate(exp_blocks)
        }
        rendered["experience"] = render_section(templates["experience"], bullets_by_block)

    if "projects" in changed and isinstance(raw.get("projects"), dict):
        for block in proj_blocks:
            if block.label in raw["projects"]:
                written[block.label] = _pad_bullets(
                    raw["projects"][block.label],
                    block.bullet_count,
                    current_proj.get(block.label, [""] * block.bullet_count),
                )
        bullets_by_block = {
            idx: written.get(block.label, []) for idx, block in enumerate(proj_blocks)
        }
        rendered["projects"] = render_section(templates["projects"], bullets_by_block)

    skills_lines = current_skills
    llm_skills_applied = False
    if "skills" in changed and isinstance(raw.get("skills"), list) and raw["skills"]:
        parsed: list[tuple[str, list[str]]] = []
        for row in raw["skills"]:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            name = str(row[0]).strip()
            items = [str(x).strip() for x in (row[1] or []) if str(x).strip()]
            if name and items:
                parsed.append((name, items))
        if parsed:
            skills_lines = parsed
            llm_skills_applied = True

    # When invent bullets change, re-pick skill categories from the page evidence so
    # Yocto/MQTT land under Embedded & Platforms — not leftover Languages/Frontend.
    suggestion_wants_skills = bool(
        re.search(r"\bskills?\b|\bcategor(?:y|ies)\b", suggestion, re.I)
    )
    if not llm_skills_applied and (
        "experience" in changed or "skills" in changed or suggestion_wants_skills
    ):
        fabricated_any = any(sel.fabricated for sel in selections.values())
        lexicon = tech_lexicon(bank)
        jd_tool_names = [
            str(t).strip()
            for key in ("tools", "must_have_skills", "exact_keywords_for_ats")
            for t in (analysis.get(key) or [])
            if str(t).strip()
        ]
        scan_lexicon = set(lexicon) | {t.lower() for t in jd_tool_names} | {
            "yocto", "mqtt", "amqp", "jenkins", "armbian", "ubuntu", "ubuntu core",
            "c++", "c++11", "c++17", "c++20",
        }
        evidenced: set[str] = {
            t for sel in selections.values() for f in sel.facts for t in f.tools
        }
        for label, bullets in written.items():
            for bullet in bullets:
                evidenced |= _mentioned_tech(_plain(bullet), scan_lexicon)
            if selections.get(label) and selections[label].fabricated:
                plain = _plain(" ".join(bullets)).lower()
                for tool in jd_tool_names:
                    if re.search(
                        r"(?<![A-Za-z0-9_])" + re.escape(tool) + r"(?![A-Za-z0-9_+#])",
                        plain,
                        re.I,
                    ):
                        evidenced.add(tool)
        skills_lines = select_skills(
            bank,
            analysis,
            skills_line_count(templates["skills"]),
            evidenced,
            strict_evidence=fabricated_any,
        )
        changed.add("skills")
        if "skills" not in note.lower():
            note = (note.rstrip(".") + "; refreshed skills to match the bullets.").strip()

    rendered["skills"] = render_skills(templates["skills"], skills_lines)

    keywords = jd_keywords_of(analysis)
    for section in ("experience", "projects"):
        if section not in changed:
            continue
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
    order = [b.label for b in exp_blocks] + [b.label for b in proj_blocks]
    all_bullets = [b for label in order for b in written.get(label, [])]
    measurement = measure(resume_plain, analysis, selections, all_bullets, skills_lines)
    questions = verification_questions(bank, analysis)
    missing_tools, missing_themes = capability_gaps(bank, analysis)

    return {
        "latex": full_latex,
        "sections": rendered,
        "jd_analysis": analysis,
        "meta": {
            "architecture": "facts-v3-refine",
            "final": True,
            "jd_agent_ok": jd_ok,
            "llm_calls": 1 if jd_analysis else 2,
            "final_score": measurement["score"],
            "refine_note": note,
            "refine_changed": sorted(changed),
            "suggestion": suggestion,
            "skills_lines": [name for name, _ in skills_lines],
            "summary_words": word_count(latex_to_plain(rendered["summary"])),
        },
        "measurement": measurement,
        "gaps": {
            "questions": questions,
            "missing_tools": missing_tools,
            "missing_capabilities": [t.replace("-", " ") for t in missing_themes],
        },
    }
