"""Offline suite for the fact-grounded pipeline. No API key needed.

Run:  python3 tests/facts_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline_v2 as p2  # noqa: E402
from src.facts import load_fact_bank  # noqa: E402
from src.matching import select_facts, select_skills, theme_profile  # noqa: E402
from src.skeleton import (  # noqa: E402
    latex_safe,
    load_template,
    parse_section,
    render_section,
    render_skills,
    skills_line_count,
)
from src.verify import (  # noqa: E402
    keyword_coverage,
    tech_lexicon,
    verify_bullet,
    verify_summary,
)

BANK = load_fact_bank()
LEXICON = tech_lexicon(BANK)

FRONTEND_JD = {
    "role_title": "Frontend Developer",
    "domain": "frontend web",
    "domain_practices": ["responsive layouts", "form UX", "component state", "code review"],
    "must_have_skills": ["React", "JavaScript", "HTML", "CSS"],
    "tools": ["React", "JavaScript", "TypeScript", "HTML", "CSS"],
    "exact_keywords_for_ats": ["React", "JavaScript", "HTML", "CSS"],
    "concepts": ["accessibility"],
    "keyword_placement": {},
}
BACKEND_JD = {
    "role_title": "Backend Engineer",
    "domain": "backend web services",
    "domain_practices": ["API design", "query optimization", "caching", "observability"],
    "must_have_skills": ["Python", "PostgreSQL", "AWS", "Docker"],
    "tools": ["Python", "FastAPI", "PostgreSQL", "AWS", "Docker", "Redis"],
    "exact_keywords_for_ats": ["Python", "REST APIs", "PostgreSQL", "AWS"],
    "concepts": ["distributed systems"],
    "keyword_placement": {},
}

SLOTS = {"Clerxi AI": 5, "Intuit": 5, "AutoFixee": 2, "Beach Bank": 2}


def _fact(owner: str, fact_id: str):
    role = BANK.role(owner)
    facts = role.facts if role else next(p.facts for p in BANK.projects if p.id == owner)
    return next(f for f in facts if f.id == fact_id)


# --- fact bank ---------------------------------------------------------------

def t01_fact_bank_loads_and_validates():
    assert [r.id for r in BANK.roles] == ["clerxi", "intuit"]
    assert BANK.role("clerxi").flexible is True
    assert BANK.role("intuit").flexible is False
    assert len(BANK.role("clerxi").facts) == 5


def t02_every_latex_block_has_facts():
    """A template block with no fact bank entry would ship empty bullets."""
    for section in ("experience", "projects"):
        for block in parse_section(load_template(section)):
            owner = BANK.role_by_company(block.label) or BANK.project_by_name(block.label)
            assert owner is not None, f"no facts for LaTeX block {block.label!r}"
            assert len(owner.facts) >= block.bullet_count, (
                f"{block.label}: {block.bullet_count} bullet slots but only "
                f"{len(owner.facts)} facts — add more to data/facts.yaml"
            )


# --- skeleton ----------------------------------------------------------------

def t03_templates_define_the_bullet_counts():
    exp = parse_section(load_template("experience"))
    assert [b.bullet_count for b in exp] == [5, 5]
    proj = parse_section(load_template("projects"))
    assert [b.bullet_count for b in proj] == [2, 2]
    assert skills_line_count(load_template("skills")) == 4


def t04_render_keeps_structure_and_drops_extra_bullets():
    template = load_template("experience")
    out = render_section(template, {0: [f"Bullet {i}" for i in range(9)]})
    blocks = parse_section(out)
    assert [b.bullet_count for b in blocks] == [5, 5], "bullet count must not change"
    assert "Bullet 5" not in out, "extra bullets must be dropped, not appended"
    assert out.count("\\resumeSubheading") == template.count("\\resumeSubheading")
    # untouched block keeps the template's own text byte-for-byte
    assert "Cut flaky CI failures by 65\\%" in out


def t05_short_bullet_list_leaves_remaining_slots_untouched():
    template = load_template("projects")
    out = render_section(template, {0: ["Only one"]})
    assert "Only one" in out
    assert "Engineered \\textbf{Python}/FastAPI" in out
    assert [b.bullet_count for b in parse_section(out)] == [2, 2]


def t06_latex_escaping():
    assert latex_safe("Backend & Cloud") == "Backend \\& Cloud"
    assert latex_safe("cut 40%") == "cut 40\\%"
    assert latex_safe("already \\% escaped") == "already \\% escaped"
    assert latex_safe("snake_case") == "snake\\_case"
    assert latex_safe("em — dash") == "em --- dash"


def t07_skills_render_escapes_ampersand():
    out = render_skills(load_template("skills"), [("Backend & Cloud", ["AWS", "Docker"])])
    assert "\\textbf{Backend \\& Cloud}" in out, "unescaped & breaks the LaTeX build"


# --- matching ----------------------------------------------------------------

def t08_domain_changes_fact_priority():
    fe = select_facts(BANK, FRONTEND_JD, SLOTS)["Intuit"].facts
    be = select_facts(BANK, BACKEND_JD, SLOTS)["Intuit"].facts
    assert fe[0].id == "react-form-ux", f"frontend JD should lead React work, got {fe[0].id}"
    assert be[0].id == "api-contracts", f"backend JD should lead API work, got {be[0].id}"


def t09_flexible_role_reorders_for_domain():
    be = select_facts(BANK, BACKEND_JD, SLOTS)["Clerxi AI"]
    assert be.flexible is True
    assert be.facts[0].id in ("retrieval-services", "agent-orchestration-api")
    assert be.scores[0] >= be.scores[-1], "facts must be ordered by relevance"


def t10_lead_fact_goes_first():
    for jd in (FRONTEND_JD, BACKEND_JD):
        sel = select_facts(BANK, jd, SLOTS)
        assert sel["Beach Bank"].facts[0].id == "app", "the intro bullet must lead"
        assert sel["AutoFixee"].facts[0].id == "platform"


def t11_short_tool_names_do_not_fuzzy_match():
    """'C' must not match 'CSS'; that ranked C above JavaScript on frontend JDs."""
    lines = dict(select_skills(BANK, FRONTEND_JD, 4))
    languages = lines["Languages"]
    assert languages[0] in ("TypeScript", "JavaScript")
    if "C" in languages:
        assert languages.index("C") > languages.index("JavaScript")


def t12_skills_are_pruned_to_the_domain():
    """A frontend resume listing Java, C, C++ reads as claiming everything."""
    lines = dict(select_skills(BANK, FRONTEND_JD, 4, evidenced={"React", "Jest", "Python"}))
    assert "Frontend" in lines
    for name, items in lines.items():
        assert len(items) <= 6, f"{name} has {len(items)} items — too many to be credible"
    assert "C++" not in lines["Languages"], "off-domain language should be pruned"


def t13_theme_profile_weights_the_jd_domain_highest():
    profile = theme_profile(FRONTEND_JD)
    assert profile.get("frontend-ui", 0) > profile.get("ai-ml", 0)


# --- verification ------------------------------------------------------------

def t14_invented_number_is_rejected():
    fact = _fact("clerxi", "latency-45")
    issues = verify_bullet(
        "Cut multi-agent query latency by 87\\% by rewriting the retrieval path in Python.",
        fact, LEXICON,
    )
    assert any(i.code == "invented-number" for i in issues)


def t15_invented_tool_is_rejected():
    fact = _fact("clerxi", "latency-45")
    issues = verify_bullet(
        "Cut multi-agent query latency by 45\\% by tuning Kubernetes and Datadog dashboards daily.",
        fact, LEXICON,
    )
    codes = [i.message for i in issues if i.code == "invented-tool"]
    assert any("kubernetes" in m for m in codes)
    assert any("datadog" in m for m in codes)


def t16_frozen_pairing_must_stay_intact():
    fact = _fact("intuit", "ci-migration")
    issues = verify_bullet(
        "Cut flaky CI failures by 65\\% by migrating end-to-end tests from Python to Playwright.",
        fact, LEXICON,
    )
    assert any(i.code in ("broken-pairing", "invented-tool") for i in issues)


def t17_real_bullet_ending_in_a_metric_is_not_truncated():
    """The v1 bug: a bullet ending on '...by 50%.' was read as ending on 'by'."""
    fact = _fact("intuit", "react-form-ux")
    issues = verify_bullet(
        "Built React pre-selection logic that cut form completion time by 35\\% and "
        "dropped configuration-related support tickets by 50\\%.",
        fact, LEXICON,
    )
    assert not issues, [f"{i.code}: {i.message}" for i in issues]


def t18_unanchored_bullet_is_rejected():
    fact = _fact("clerxi", "latency-45")
    issues = verify_bullet(
        "Partnered with stakeholders to align delivery expectations across the wider business unit.",
        fact, LEXICON,
    )
    assert any(i.code == "unanchored" for i in issues)


def t19_summary_may_not_repeat_bullet_numbers():
    facts = [_fact("intuit", "react-form-ux")]
    issues = verify_summary(
        "Frontend engineer building React forms. Cut completion time by 35\\% on shared codebases.",
        facts, LEXICON, bullet_numbers={"35"},
    )
    assert any(i.code == "repeats-bullet" for i in issues)


def t20_summary_may_not_claim_what_no_bullet_proves():
    facts = [_fact("intuit", "react-form-ux")]
    issues = verify_summary(
        "Frontend engineer building accessible React interfaces for large teams. "
        "Ships responsive layouts that hold up across browsers and devices daily.",
        facts, LEXICON, proof_text="Built React pre-selection logic that cut form completion time.",
    )
    assert any(i.code == "unproven-claim" for i in issues)


def t21_vague_summary_filler_is_rejected():
    facts = [_fact("intuit", "react-form-ux")]
    issues = verify_summary(
        "Frontend engineer skilled in React and modern web technologies for the team. "
        "Builds interfaces with a wide range of tools and delivers steady improvements.",
        facts, LEXICON,
    )
    assert any(i.code == "vague" for i in issues)


def t22_keyword_coverage_counts_synonyms():
    result = keyword_coverage(
        "Built responsive pages with HTML, CSS and JavaScript, deployed via GitHub Actions.",
        ["HTML5", "CSS3", "JavaScript", "CI/CD", "Kubernetes"],
    )
    assert result["missing"] == ["Kubernetes"], result


# --- pipeline ----------------------------------------------------------------

class _Stub:
    """Deterministic stand-in for the writer LLM."""

    def __init__(self, mode="good"):
        self.mode = mode
        self.calls = 0

    def __call__(self, prompt, temperature=0.2, max_tokens=4000, retries=1, role="judge"):
        self.calls += 1
        if "two-sentence summary" in prompt:
            return {"summary": "Backend engineer building Python retrieval services on AWS. "
                               "Turns slow query paths into measured wins on production systems."}
        count = prompt.count("WHAT HAPPENED:")
        if self.mode == "fabricate":
            return {"bullets": ["Rebuilt the Kubernetes control plane to serve 900 million "
                                "requests a day for the platform team." for _ in range(count)]}
        if self.mode == "broken-json-escape":
            raise ValueError("Failed to parse LLM JSON")
        # Echo each fact's own sentence back — always passes verification.
        facts = [line.split("WHAT HAPPENED: ", 1)[1]
                 for line in prompt.splitlines() if "WHAT HAPPENED:" in line]
        return {"bullets": facts}


def _run(mode="good", jd=None):
    original_json, original_jd = p2.call_llm_json, p2.run_jd_agent
    stub = _Stub(mode)
    p2.call_llm_json = stub
    p2.run_jd_agent = lambda _jd: (jd or BACKEND_JD, True)
    try:
        return p2.build_resume_v2("A backend engineering role. " * 10), stub
    finally:
        p2.call_llm_json, p2.run_jd_agent = original_json, original_jd


def t23_pipeline_preserves_template_structure():
    result, _ = _run()
    exp = parse_section(result["sections"]["experience"])
    assert [b.bullet_count for b in exp] == [5, 5]
    assert [b.label for b in exp] == ["Clerxi AI", "Intuit"]
    proj = parse_section(result["sections"]["projects"])
    assert [b.bullet_count for b in proj] == [2, 2]


def t24_fabricated_output_never_reaches_the_resume():
    result, _ = _run(mode="fabricate")
    plain = result["latex"].lower()
    assert "kubernetes" not in plain, "invented tool leaked onto the resume"
    assert "900 million" not in plain, "invented scale leaked onto the resume"
    for block in result["meta"]["blocks"]:
        assert block["fell_back_to_fact"], "fabrication should force fact fallback"


def t25_writer_failure_still_ships_a_full_resume():
    result, _ = _run(mode="broken-json-escape")
    exp = parse_section(result["sections"]["experience"])
    assert [b.bullet_count for b in exp] == [5, 5]
    for block in exp:
        for slot in block.slots:
            assert slot.body.strip(), "every bullet slot must be filled"


def t26_measurement_is_deterministic():
    first, _ = _run()
    second, _ = _run()
    assert first["measurement"]["score"] == second["measurement"]["score"]
    assert first["latex"] == second["latex"]


def t27_generated_content_has_no_unescaped_specials():
    """Check the parts WE write — an unescaped & or % there breaks the build."""
    import re

    result, _ = _run()
    bodies = [
        slot.body
        for section in ("experience", "projects")
        for block in parse_section(result["sections"][section])
        for slot in block.slots
    ]
    bodies += re.findall(r"\\textbf\{[^}]*\}\{:[^}]*\}", result["sections"]["skills"])
    for body in bodies:
        stripped = re.sub(r"\\[%&#_$]", "", body)
        assert "&" not in stripped, f"unescaped & in: {body}"
        assert "%" not in stripped, f"unescaped % in: {body}"


def t28_skills_section_reflects_the_bullets():
    result, _ = _run(jd=FRONTEND_JD)
    skills = result["sections"]["skills"]
    assert "Frontend" in skills
    assert "C++" not in skills, "off-domain skills should be pruned"


_TEST_NAME = __import__("re").compile(r"^t\d\d_")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if _TEST_NAME.match(k) and callable(v)]
    passed = failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ok  {test.__name__}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
