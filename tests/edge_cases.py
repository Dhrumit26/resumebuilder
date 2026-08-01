"""Offline edge-case suite for the agent pipeline. No API key needed.

Run:  python3 tests/edge_cases.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline as pl
import src.resume_builder as rb

ORIG = rb.load_original_sections()

TAILORED = {
    "summary": ORIG["summary"].replace("\\textit{", "\\textit{Tailored: ", 1)
    if "\\textit{" in ORIG["summary"] else ORIG["summary"],
    "experience": (
        ORIG["experience"]
        .replace(
            "Built \\textbf{Python} backend retrieval services on AWS for a distributed multi-agent system, applying \\textbf{OOP} and \\textbf{algorithmic design} to architect scalable, service-oriented APIs integrating semantic reranking of technical documents.",
            "Shipped tailored \\textbf{Python} retrieval microservices on AWS that power a multi-agent document platform, using clean APIs and semantic reranking over technical docs.",
        )
        .replace(
            "Cut multi-agent query latency by 45\\% by re-engineering \\textbf{core data structures} and retrieval \\textbf{algorithms}, with \\textbf{systems monitoring} of live production services throughout to preserve accuracy at scale.",
            "Dropped query latency 45\\% on that document platform after rewriting retrieval paths and watching live production accuracy under load.",
        )
        .replace(
            "Reduced infrastructure and inference costs by 40\\% via semantic caching, dynamic top-k sizing, and embedding-dimension tuning across a large-scale distributed vector search backend, improving overall system efficiency.",
            "Lowered infra and inference spend 40\\% on the same platform with semantic caching plus tighter top-k and embedding settings on the vector search backend.",
        )
        .replace(
            "Deployed a containerized API server for stateful agent orchestration using \\textbf{Docker} and DevOps practices, securely exposing internal ERP and project-management systems for end-to-end read/write automation.",
            "Launched a \\textbf{Docker} API server for that platform's agent orchestration that safely wires internal ERP and project systems for automation.",
        )
        .replace(
            "Collaborated with engineering and product teams across the full SDLC, from API contract definition through deployment and production support, shipping reliable, scalable agent infrastructure on time.",
            "Partnered with eng and product from API contracts through production support, shipping the platform's agent infrastructure on schedule.",
        )
        .replace(
            "Cut flaky CI failures by 65\\% and pipeline runtime from 45 to 28 minutes by migrating end-to-end tests from Cypress to Playwright, speeding releases by 2 days per sprint.",
            "Dropped flaky CI failures 65\\% and shrank pipeline time 45 to 28 minutes by moving e2e tests to Playwright, unlocking releases 2 days sooner.",
        )
        .replace(
            "Tripled iteration speed and shortened deployment cycles from 2 weeks to 4 days by refactoring \\textbf{TypeScript} and \\textbf{JavaScript} platform services powering low-code features across teams.",
            "Boosted iteration 3x and cut deploy cycles from 2 weeks to 4 days by refactoring \\textbf{TypeScript}/\\textbf{JavaScript} platform services for low-code teams.",
        )
        .replace(
            "Raised unit and integration \\textbf{test coverage} from 42\\% to 78\\% with \\textbf{Jest} and React Testing Library, reducing production defects by 40\\% and catching regressions earlier in development.",
            "Grew unit and integration coverage 42\\% to 78\\% with \\textbf{Jest} and React Testing Library, cutting production defects 40\\%.",
        )
        .replace(
            "Built React pre-selection logic that cut form completion time by 35\\% and dropped configuration-related support tickets by 50\\%, simplifying onboarding for a large user base.",
            "Shipped React pre-selection that shortened form completion 35\\% and halved config support tickets for onboarding users.",
        )
        .replace(
            "Defined and maintained API contracts across service boundaries, enabling cleaner cross-team handoffs and reducing integration breaks during multi-sprint release cycles.",
            "Wrote and maintained service API contracts that cleaned up cross-team handoffs and cut integration breaks across release cycles.",
        )
    ),
    "projects": (
        ORIG["projects"]
        .replace(
            "Built an AI self-healing CI platform monitoring failed \\textbf{GitHub Actions} and GitLab CI runs in real time, identifying root causes and opening verified fix PRs automatically, cutting manual debug time by 70\\%.",
            "Shipped an AI self-healing CI tool that watches failed \\textbf{GitHub Actions}/GitLab runs, finds root causes, and opens verified fix PRs, cutting debug time 70\\%.",
        )
        .replace(
            "Engineered \\textbf{Python}/FastAPI webhook and REST services backed by \\textbf{PostgreSQL} and AWS, slashing inference costs by 80\\%+ with a three-tier rule-based and multi-agent remediation strategy for \\textbf{scalability}.",
            "Built \\textbf{Python}/FastAPI webhooks on \\textbf{PostgreSQL} and AWS that cut inference cost 80\\%+ using a three-tier rule and multi-agent fix strategy.",
        )
        .replace(
            "Built a \\textbf{full-stack} banking app supporting multi-account linking, real-time balances, transaction history, and secure fund transfers via Plaid and Dwolla REST API integrations.",
            "Shipped a \\textbf{full-stack} banking app with multi-account linking, live balances, history, and secure transfers through Plaid and Dwolla APIs.",
        )
        .replace(
            "Implemented with \\textbf{Next.js} 14, React, \\textbf{TypeScript}, \\textbf{PostgreSQL}, OAuth 2.0, and \\textbf{GitHub Actions} \\textbf{CI/CD}, adding pagination and spending analytics to surface actionable account insights.",
            "Used \\textbf{Next.js} 14, React, \\textbf{TypeScript}, and \\textbf{PostgreSQL} with OAuth 2.0 and \\textbf{GitHub Actions} CI/CD, plus pagination and spending analytics.",
        )
    ),
    "skills": ORIG["skills"],
}

JD = (
    "Senior Backend Engineer. Requirements: Python, FastAPI, PostgreSQL, Redis, AWS, "
    "Docker, Kubernetes, CI/CD, LLM APIs, RAG systems, gRPC, Datadog. Nice: React."
)

JD_ANALYSIS_JSON = {
    "role_title": "Senior Backend Engineer",
    "seniority_level": "senior",
    "company_type": "startup",
    "years_experience_wanted": "3+",
    "must_have_skills": ["Python", "FastAPI", "PostgreSQL", "Kubernetes", "gRPC", "Datadog"],
    "nice_to_have_skills": ["React"],
    "exact_keywords_for_ats": ["Python", "FastAPI", "Kubernetes", "gRPC", "Datadog"],
    "tools": ["Kubernetes", "gRPC", "Datadog"],
    "concepts": ["RAG"],
    "keyword_placement": {"summary": ["Python"], "experience": ["FastAPI"], "projects": [], "skills": ["Kubernetes"]},
    "ideal_summary_angle": "senior backend",
    "competitive_positioning": "strong",
}


def make_scores(overall, fixes=None, missing=None):
    return {
        "ats_scorer": {"overall_score": overall, "breakdown": {}, "verdict": "strong_match"},
        "ats_reviewer": {"ats_score": overall, "recommendations": [], "missing_keywords": missing or []},
        "human_reviewer": {"human_score": overall},
        "section_fixes": fixes or {},
    }


import threading


class MockLLM:
    """Routes pipeline LLM calls by inspecting the prompt text. Thread-safe."""

    def __init__(self, jd_json=None, jd_fail=False, section_behavior=None,
                 reviewer_scores=None, reviewer_fail_at=None):
        self.jd_json = jd_json or JD_ANALYSIS_JSON
        self.jd_fail = jd_fail
        self.section_behavior = section_behavior or {}  # name -> list of outputs/Exceptions
        self.reviewer_scores = list(reviewer_scores or [make_scores(96)])
        self.reviewer_fail_at = reviewer_fail_at  # 0-based reviewer call index
        self.reviewer_calls = 0
        self.section_calls = {n: 0 for n in pl.SECTION_NAMES}
        self._lock = threading.Lock()

    def _section_of(self, prompt):
        for name, tag in (("summary", "Summary Agent"), ("experience", "Experience Agent"),
                          ("projects", "Projects Agent"), ("skills", "Skills Agent")):
            if tag in prompt:
                return name
        return None

    def call_llm(self, prompt, temperature=0.3, max_tokens=4000, role="writer"):
        name = self._section_of(prompt)
        assert name, "unknown call_llm prompt"
        with self._lock:
            idx = self.section_calls[name]
            self.section_calls[name] += 1
        behavior = self.section_behavior.get(name)
        if behavior is None:
            return TAILORED[name]
        out = behavior[min(idx, len(behavior) - 1)]
        if isinstance(out, Exception):
            raise out
        return out

    def call_llm_json(self, prompt, temperature=0.2, max_tokens=4000, retries=1, role="judge"):
        if "JD Analysis Agent" in prompt:
            if self.jd_fail:
                raise ValueError("bad json")
            return dict(self.jd_json)
        if "Reviewer Agent" in prompt:
            with self._lock:
                i = self.reviewer_calls
                self.reviewer_calls += 1
            if self.reviewer_fail_at is not None and i >= self.reviewer_fail_at:
                raise ValueError("reviewer broke")
            return __import__("json").loads(__import__("json").dumps(
                self.reviewer_scores[min(i, len(self.reviewer_scores) - 1)]))
        raise AssertionError("unknown call_llm_json prompt")


def with_mock(mock, best_of_n=1, score_samples=1):
    pl.call_llm = mock.call_llm
    pl.call_llm_json = mock.call_llm_json
    pl.BEST_OF_N = best_of_n
    pl.SCORE_SAMPLES = score_samples
    # Offline tests must not hit Microsoft Learn / Wikipedia.
    pl.build_tech_context = lambda *_a, **_k: ""


PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok  {name}")
    except AssertionError as exc:
        FAIL.append(name)
        print(f"FAIL  {name}: {exc}")


# ---------------------------------------------------------------- scenarios

def t01_happy_path():
    with_mock(MockLLM())
    out = pl.build_tailored_resume(JD)
    assert "Shipped tailored" in out["sections"]["experience"]
    assert out["meta"]["final_score"] == 96
    assert out["meta"]["jd_agent_ok"] and out["meta"]["reviewer_ok"]
    assert out["meta"]["remake_attempts"] == 0  # 96 >= target, no fix rounds
    assert out["meta"]["llm_calls"] == 6  # 1 jd + 4 sections + 1 review
    assert "\\documentclass" in out["latex"]


def t02_section_retry_then_ok():
    m = MockLLM(section_behavior={"experience": ["**markdown junk**", TAILORED["experience"]]})
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["history"][0]["section_statuses"]["experience"] == "retried"
    assert "Shipped tailored" in out["sections"]["experience"]


def t03_section_always_invalid_falls_back():
    m = MockLLM(section_behavior={"projects": ["- markdown", "- markdown again"]})
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["sections"]["projects"].strip() == rb.clean_llm_latex(ORIG["projects"]).strip() or \
        "\\resumeProjectHeading" in out["sections"]["projects"]
    assert out["meta"]["history"][0]["section_statuses"]["projects"] == "invalid"


def t04_section_network_error_falls_back():
    m = MockLLM(section_behavior={"summary": [ConnectionError("boom"), ConnectionError("boom")]})
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["history"][0]["section_statuses"]["summary"] == "error"
    assert "\\textit{" in out["sections"]["summary"]  # original fallback still valid


def t05_jd_agent_failure_degrades():
    with_mock(MockLLM(jd_fail=True))
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["jd_agent_ok"] is False
    assert "Shipped tailored" in out["sections"]["experience"]  # still tailored


def t06_reviewer_failure_pass1():
    with_mock(MockLLM(reviewer_fail_at=0))
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["reviewer_ok"] is False
    assert out["meta"]["final_score"] == 0
    assert out["scores"]["ats_scorer"] == {}
    assert "Shipped tailored" in out["sections"]["experience"]  # resume still delivered


def t07_targeted_fix_only_flagged():
    fixed_summary = ORIG["summary"].replace("\\textit{", "\\textit{FIXEDSUM ", 1)
    m = MockLLM(
        reviewer_scores=[
            make_scores(80, fixes={"summary": ["mirror JD title"]}),
            make_scores(93),
        ],
        section_behavior={"summary": [TAILORED["summary"], fixed_summary]},
    )
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    h = out["meta"]["history"]
    assert h[1]["fixed_sections"] == ["summary"], h
    # experience agent called exactly once (initial) — never for the fix round
    assert m.section_calls["experience"] == 1
    assert "FIXEDSUM" in out["sections"]["summary"]
    assert out["meta"]["final_score"] == 93


def t08_fix_makes_worse_best_kept():
    m = MockLLM(
        reviewer_scores=[
            make_scores(88, fixes={"experience": ["strengthen bullet 2"]}),
            make_scores(70, fixes={"experience": ["strengthen bullet 2"]}),
            make_scores(70, fixes={"experience": ["strengthen bullet 2"]}),
        ],
    )
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["final_score"] == 88
    assert out["meta"]["best_pass"] == 1


def t09_fabricated_tools_reverted():
    """Fixed-history job (Intuit): unevidenced JD tools must be reverted.

    The previous version of this test targeted text that TAILORED had already
    replaced, so the injection silently no-oped and it passed against input that
    never contained the fabrication. The fixture assert below prevents that.
    """
    fabricated = TAILORED["experience"].replace(
        "Wrote and maintained service API contracts",
        "Wrote Kubernetes-backed service API contracts with gRPC and Datadog",
    )
    assert "Kubernetes" in fabricated, "fixture failed to inject the fabrication"
    m = MockLLM(section_behavior={"experience": [fabricated, fabricated]})
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    exp = out["sections"]["experience"].lower()
    assert "kubernetes" not in exp and "grpc" not in exp and "datadog" not in exp, exp


def t24_flexible_company_keeps_jd_tools():
    """Clerxi AI is flexible: it MAY adopt JD tools. Intuit may not, same section."""
    exp = TAILORED["experience"].replace(
        "using clean APIs and semantic reranking",
        "using Kubernetes and semantic reranking",
    ).replace(
        "Wrote and maintained service API contracts",
        "Wrote Datadog-backed service API contracts",
    )
    assert "Kubernetes" in exp and "Datadog" in exp, "fixture failed to inject"
    m = MockLLM(section_behavior={"experience": [exp, exp]})
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    got = out["sections"]["experience"]
    assert "Kubernetes" in got, "Clerxi AI is flexible — JD tool should survive"
    assert "Datadog" not in got, "Intuit is fixed history — JD tool must be reverted"


def t25_flexible_item_indices_splits_by_company():
    idx = rb.flexible_item_indices(ORIG["experience"], ["Clerxi AI"])
    assert idx == {0, 1, 2, 3, 4}, idx  # 5 Clerxi bullets, then 5 Intuit
    assert rb.flexible_item_indices(ORIG["experience"], []) == set()
    assert rb.flexible_item_indices(ORIG["experience"], ["Nonexistent Co"]) == set()


def t10_concepts_stripped_parens_kept():
    bad_skills = ORIG["skills"].replace(
        "FastAPI, Node.js, REST APIs, PostgreSQL, Redis, Docker",
        "LLMs, AWS (Lambda, ECS, S3), RAG systems, FastAPI, Vector Databases, Docker",
    )
    m = MockLLM(section_behavior={"skills": [bad_skills, bad_skills]})
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    plain = rb.latex_to_plain(out["sections"]["skills"])
    assert "LLMs" not in plain and "RAG systems" not in plain and "Vector Databases" not in plain
    assert "AWS (Lambda, ECS, S3)" in plain and "FastAPI" in plain


def t11_extra_bullets_trimmed():
    extra = TAILORED["experience"].replace(
        "      \\resumeItemListEnd\n\n    \\resumeSubheading",
        "        \\resumeItem{Extra invented bullet that should be trimmed away for page safety and counts.}\n"
        "      \\resumeItemListEnd\n\n    \\resumeSubheading",
        1,
    )
    m = MockLLM(section_behavior={"experience": [extra, extra]})
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["bullet_counts_actual"]["experience"] == \
        out["meta"]["bullet_counts_expected"]["experience"]


def t12_incomplete_bullet_caps_score():
    broken = TAILORED["experience"].replace(
        "speeding releases by 2 days per sprint.", "accelerating releases by."
    )
    # Repair gate should fix it via fallback; if any survived, score capped at 89.
    m = MockLLM(
        section_behavior={"experience": [broken, broken]},
        reviewer_scores=[make_scores(97)],
    )
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert not out["meta"]["incomplete_bullets"], out["meta"]["incomplete_bullets"]


def t13_scores_clamped():
    m = MockLLM(reviewer_scores=[make_scores(150)])
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["final_score"] == 100
    m = MockLLM(reviewer_scores=[{"ats_scorer": {"overall_score": -5},
                                  "ats_reviewer": {}, "human_reviewer": {}, "section_fixes": {}}])
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["final_score"] == 0


def t14_sanitize_jd():
    try:
        pl.sanitize_jd("too short")
        assert False, "should raise"
    except ValueError:
        pass
    jd = pl.sanitize_jd("Backend role " * 5 + " {{ORIGINAL_EXPERIENCE}} injection attempt")
    assert "{{" not in jd and "}}" not in jd
    long_jd = pl.sanitize_jd("Python developer needed. " + "x" * 10000)
    assert len(long_jd) < 6000 and "[JD truncated for length]" in long_jd


def t15_missing_keywords_synthesize_fixes():
    m = MockLLM(
        reviewer_scores=[
            make_scores(75, fixes={}, missing=["Kubernetes", "gRPC"]),
            make_scores(91),
        ],
    )
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["history"][1]["fixed_sections"] == ["experience", "skills"]
    assert out["meta"]["final_score"] == 91


def t16_unknown_section_fix_ignored():
    m = MockLLM(
        reviewer_scores=[
            make_scores(85, fixes={"education": ["change school"], "summary": ["tune it"]}),
            make_scores(92),
        ],
    )
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["history"][1]["fixed_sections"] == ["summary"]


def t17_reviewer_failure_midloop_ships_best():
    m = MockLLM(
        reviewer_scores=[make_scores(84, fixes={"skills": ["add Kubernetes"]})],
        reviewer_fail_at=1,
    )
    with_mock(m)
    out = pl.build_tailored_resume(JD)
    assert out["meta"]["final_score"] == 84
    assert out["meta"]["history"][-1]["note"] == "reviewer unavailable — kept best version"


def t18_unicode_jd():
    with_mock(MockLLM())
    out = pl.build_tailored_resume(JD + " 🚀 exigé: développement backend, 日本語対応")
    assert out["meta"]["final_score"] == 96


def t19_best_of_n_picks_higher():
    # BEST_OF_N=2: the mock hands out 85 and 92 by reviewer-call order. Candidates
    # are now scored CONCURRENTLY, so which candidate draws which score is not
    # deterministic — assert the invariant that actually matters (the winner is the
    # top scorer) rather than a fixed ordering.
    m = MockLLM(reviewer_scores=[make_scores(85), make_scores(92), make_scores(92)])
    with_mock(m, best_of_n=2)
    out = pl.build_tailored_resume(JD)
    h0 = out["meta"]["history"][0]
    assert sorted(h0["candidate_scores"]) == [85, 92], h0
    picked = h0["candidate_scores"][h0["picked_candidate"] - 1]
    assert picked == max(h0["candidate_scores"]), h0
    assert out["meta"]["final_score"] >= 92
    # 8 section calls happened (4 per candidate)
    assert all(v >= 2 for v in m.section_calls.values()), m.section_calls


def t21_keyword_bolding():
    sec = (
        "\\resumeItemListStart\n"
        "        \\resumeItem{Built Python retrieval services with FastAPI on AWS, "
        "cutting latency 45\\% via semantic caching in PostgreSQL.}\n"
        "        \\resumeItem{Already bolded \\textbf{Python} plus \\href{https://x.com/Python}{Python link} "
        "and plain Docker deployment for CI/CD pipelines using C++.}\n"
        "\\resumeItemListEnd\n"
    )
    out = rb.bold_keywords_in_bullets(sec, ["Python", "FastAPI", "PostgreSQL", "Docker", "CI/CD", "C++", "AWS"])
    # bullet 1: up to 2 new bolds under the highlight budget
    b1 = rb._extract_resume_items(out)[0]
    assert 1 <= b1.count("\\textbf{") <= 2, b1
    # bullet 2: existing bold Python not double-bolded; href untouched
    b2 = rb._extract_resume_items(out)[1]
    assert "\\textbf{\\textbf{" not in b2
    assert "\\href{https://x.com/Python}{Python link}" in b2
    assert b2.count("\\textbf{") <= 3

    summ = "%-----------Summary-----------\n\\textit{Backend engineer building AI platforms with Python and FastAPI at scale.}\n"
    sout = rb.bold_keywords_in_summary(summ, ["Python", "FastAPI", "AWS"], max_new=1)
    assert sout.count("\\textbf{") == 1, sout
    assert sout.count("{") == sout.count("}")


def t34_story_thin_detects_floating_duty_bullets():
    thin = r"""
\section{Experience}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Software Engineer}{May 2026 -- Present}
      {Clerxi AI}{Huntington Beach, CA}
      \resumeItemListStart
        \resumeItem{Developed Java backend services using Spring Boot and Hibernate.}
        \resumeItem{Optimized SQL queries reducing execution time from 120ms to 80ms.}
        \resumeItem{Implemented unit testing with JUnit increasing coverage to 85\%.}
        \resumeItem{Participated in Agile sprints to deliver REST APIs on schedule.}
        \resumeItem{Integrated Angular front-end components with backend services.}
      \resumeItemListEnd
  \resumeSubHeadingListEnd
"""
    issues = rb.story_thin_in_flexible_bullets(thin, ["Clerxi AI"])
    assert issues, issues

    rich = r"""
\section{Experience}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Software Engineer}{May 2026 -- Present}
      {Clerxi AI}{Huntington Beach, CA}
      \resumeItemListStart
        \resumeItem{Built Spring Boot APIs for an internal Angular console that lets ops review vendor feed errors before customers see them.}
        \resumeItem{Cut the slowest SQL lookups on that console's feed store from 120ms to 80ms with Hibernate fetch fixes.}
        \resumeItem{Covered those API handlers with JUnit from 50\% to 85\% so bad feed fixes stopped regressing in CI.}
        \resumeItem{Shipped the console's Angular screens against the same REST contracts during two-week Agile sprints.}
        \resumeItem{Documented the ingestion API paths on the team wiki so platform owners could deploy without guesswork.}
      \resumeItemListEnd
  \resumeSubHeadingListEnd
"""
    assert rb.story_thin_in_flexible_bullets(rich, ["Clerxi AI"]) == []


def t35_near_copy_fixed_history_detects_intuit_paste():
    orig = ORIG["experience"]
    # Clerxi rewritten, Intuit pasted verbatim → must flag
    pasted = TAILORED["experience"]
    # Force Intuit bullets back to original text
    gen_items = rb._extract_resume_items(pasted)
    orig_items = rb._extract_resume_items(orig)
    assert len(gen_items) >= 10 and len(orig_items) >= 10
    out = pasted
    for i in range(5, 10):
        out = out.replace(gen_items[i], orig_items[i], 1)
    hits = rb.near_copy_fixed_history_bullets(out, orig, ["Clerxi AI"])
    assert hits, hits
    # Fully rewritten tailored fixture should be clean
    assert rb.near_copy_fixed_history_bullets(
        TAILORED["experience"], orig, ["Clerxi AI"]
    ) == []


def t36_made_up_claims_flags_fake_team_and_millions():
    fake = (
        "Software engineer on the Memberships team, developing scalable backend "
        "services using Python and AWS Lambda for millions of concurrent users."
    )
    hits = rb.made_up_claims_in_text(fake, evidence="Python AWS Lambda SQS")
    assert any("Memberships" in h for h in hits), hits
    assert any("millions" in h for h in hits), hits
    ok = (
        "Software engineer building Python services on AWS Lambda that process "
        "membership events through SQS."
    )
    assert rb.made_up_claims_in_text(ok, evidence="Python AWS Lambda SQS") == []


def t23_skills_whitelist_and_go_false_positive():
    # "Go" must NOT match inside "algorithmic"
    evidence = rb.latex_to_plain(
        "\n".join(ORIG[k] for k in ("summary", "experience", "projects", "skills"))
    ).lower()
    banned = rb.unevidenced_tools(["Go", "Rust", "Kubernetes", "Python"], evidence)
    assert "Go" in banned and "Rust" in banned and "Kubernetes" in banned
    assert "Python" not in banned

    invented = ORIG["skills"].replace(
        "Python, SQL, TypeScript, JavaScript, Java, C, C++",
        "Python, Go, Rust, TypeScript, JavaScript, SQL, NoSQL",
    )
    out = rb._whitelist_skills(invented, ORIG["skills"], ["Kubernetes", "Datadog", "gRPC"], max_new=4)
    plain = rb.latex_to_plain(out)
    assert "Go" not in plain and "Rust" not in plain and "NoSQL" not in plain
    assert "Kubernetes" in plain  # JD tool allowed
    assert "Python" in plain and "FastAPI" in plain

    # concept bolds stripped
    bullet = r"\resumeItem{Rewrote \textbf{algorithms} and \textbf{Python} services.}"
    stripped = rb._strip_concept_bolds(bullet)
    assert r"\textbf{algorithms}" not in stripped
    assert r"\textbf{Python}" in stripped


def t22_bolding_in_pipeline_output():
    with_mock(MockLLM())
    out = pl.build_tailored_resume(JD)
    exp = out["sections"]["experience"]
    # JD keyword "Python" appears in bullets and must be bolded somewhere
    assert re.search(r"\\textbf\{[^{}]*Python[^{}]*\}", exp), "Python not bolded"
    assert out["latex"].count("{") == out["latex"].count("}")


def t20_score_samples_averaged_and_fixes_merged():
    # SCORE_SAMPLES=2: first scoring = avg(84, 92) = 88, fixes unioned from both
    m = MockLLM(reviewer_scores=[
        make_scores(84, fixes={"summary": ["tune summary"]}),
        make_scores(92, fixes={"experience": ["strengthen bullet 1"]}),
        make_scores(96), make_scores(96),
    ])
    with_mock(m, score_samples=2)
    out = pl.build_tailored_resume(JD)
    h = out["meta"]["history"]
    assert h[0]["score"] == 88, h[0]
    assert set(h[1]["fixed_sections"]) == {"summary", "experience"}, h[1]
    assert out["meta"]["final_score"] == 96


def t26_language_scatter_detected_in_flexible_bullets():
    scatter = r"""
\section{Experience}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Software Engineer}{May 2026 -- Present}
      {Clerxi AI}{Huntington Beach, CA}
      \resumeItemListStart
        \resumeItem{Shipped 12 production features in Java on AWS that cut task time 30\%.}
        \resumeItem{Cut incident resolution 40\% by debugging C\# services in production.}
        \resumeItem{Raised test coverage to 82\% across core Python and Java modules.}
      \resumeItemListEnd
    \resumeSubheading
      {Software Engineer Intern}{Jun 2025 -- Aug 2025}
      {Intuit}{San Diego, CA}
      \resumeItemListStart
        \resumeItem{Refactored TypeScript and JavaScript platform services for releases.}
      \resumeItemListEnd
  \resumeSubHeadingListEnd
"""
    langs = rb.languages_in_flexible_bullets(scatter, ["Clerxi AI"])
    # Java/C#/Python scatter detected; Intuit's TypeScript/JavaScript don't count
    assert set(langs) == {"Java", "C#", "Python"}, langs

    coherent = scatter.replace("Java on AWS", "C++17 on Linux").replace(
        "C\\# services", "C++ services"
    ).replace("core Python and Java modules", "hot paths with Python benchmark tooling")
    langs = rb.languages_in_flexible_bullets(coherent, ["Clerxi AI"])
    assert set(langs) == {"C++", "Python"}, langs  # primary + tooling language is fine

    # "JavaScript" must never report "Java"
    js_only = scatter.replace("Java on AWS", "JavaScript on AWS").replace(
        "C\\# services", "JavaScript services"
    ).replace("core Python and Java modules", "core JavaScript modules")
    langs = rb.languages_in_flexible_bullets(js_only, ["Clerxi AI"])
    assert langs == ["JavaScript"], langs


def t31_stack_name_overuse_detects_swiftui_spam():
    spam = r"""
\section{Experience}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Software Engineer}{May 2026 -- Present}
      {Clerxi AI}{Huntington Beach, CA}
      \resumeItemListStart
        \resumeItem{Built SwiftUI tools for internal developer workflows.}
        \resumeItem{Cut launch time by optimizing SwiftUI rendering paths.}
        \resumeItem{Added accessibility labels across SwiftUI screens.}
        \resumeItem{Covered regressions with XCTest on the shared suite.}
        \resumeItem{Grew the SwiftUI component library for consistent UI.}
      \resumeItemListEnd
  \resumeSubHeadingListEnd
"""
    hits = rb.stack_name_overuse_in_flexible_bullets(spam, ["Clerxi AI"], max_bullets=2)
    assert any(h.startswith("SwiftUI") for h in hits), hits

    ok = spam.replace("SwiftUI rendering", "list rendering").replace(
        "SwiftUI screens", "detail screens"
    ).replace("SwiftUI component", "shared component")
    # Only first bullet still says SwiftUI
    assert rb.stack_name_overuse_in_flexible_bullets(ok, ["Clerxi AI"]) == []


def t32_stack_family_underuse_wants_uikit_companions():
    starved = r"""
\section{Experience}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Software Engineer}{May 2026 -- Present}
      {Clerxi AI}{Huntington Beach, CA}
      \resumeItemListStart
        \resumeItem{Built a SwiftUI internal tool that cut task time by 30\%.}
        \resumeItem{Improved code review practices and reduced bugs by 25\%.}
        \resumeItem{Standardized design components for UI consistency.}
        \resumeItem{Added accessibility features across existing applications.}
        \resumeItem{Simplified workflows with cross-functional teammates.}
      \resumeItemListEnd
  \resumeSubHeadingListEnd
"""
    jd_blob = "Swift SwiftUI Xcode XCTest"
    under = rb.stack_family_underuse_in_flexible_bullets(
        starved, ["Clerxi AI"], jd_blob
    )
    assert under, under

    rich = starved.replace(
        "Improved code review practices and reduced bugs by 25\\%.",
        "Bridged legacy UIKit screens into the new navigation flow.",
    ).replace(
        "Standardized design components for UI consistency.",
        "Covered regressions with XCTest snapshot suites on critical paths.",
    )
    assert rb.stack_family_underuse_in_flexible_bullets(
        rich, ["Clerxi AI"], jd_blob
    ) == []


def t33_frontend_preferred_tools_and_brand_bleed():
    starved = r"""
\section{Experience}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Software Engineer}{May 2026 -- Present}
      {Clerxi AI}{Huntington Beach, CA}
      \resumeItemListStart
        \resumeItem{Built AI-driven JavaScript interfaces for data visualization tools.}
        \resumeItem{Optimized front-end performance by 30\% with responsive design.}
        \resumeItem{Integrated REST APIs reducing load times by 40\%.}
        \resumeItem{Collaborated on web page development and usability standards.}
        \resumeItem{Led the adoption of modern UI frameworks across projects.}
      \resumeItemListEnd
  \resumeSubHeadingListEnd
"""
    jd_blob = "CSS HTML JavaScript jQuery Bootstrap Foundation HTML5 CSS3"
    under = rb.stack_family_underuse_in_flexible_bullets(
        starved, ["Clerxi AI"], jd_blob
    )
    assert any("preferred" in u.lower() for u in under), under

    bleed = rb.brand_bleed_in_text(
        "Built AI-driven JavaScript interfaces for data visualization tools.",
        "frontend web",
    )
    assert "AI-driven" in bleed, bleed
    assert rb.brand_bleed_in_text("Built AI-driven tools.", "AI/LLM platform") == []

    perc = rb.bare_percent_overuse_in_flexible_bullets(starved, ["Clerxi AI"], max_bare=2)
    # only 2 bare-% bullets (30 and 40) — should be OK at max_bare=2
    assert perc == [], perc
    perc3 = starved.replace(
        "Collaborated on web page development and usability standards.",
        "Raised accessibility scores by 15\\% across client sites.",
    )
    assert rb.bare_percent_overuse_in_flexible_bullets(perc3, ["Clerxi AI"], max_bare=2)

    senior = rb.senior_theater_in_flexible_bullets(starved, ["Clerxi AI"])
    assert any("Led" in s for s in senior), senior

    analysis = {
        "tools": ["CSS", "HTML", "JavaScript"],
        "must_have_skills": ["CSS", "HTML", "JavaScript"],
        "nice_to_have_skills": [],
        "exact_keywords_for_ats": [],
        "research_topics": [],
    }
    jd = (
        "Preferred: jQuery, Front-End UI Frameworks Foundation & Twitter Bootstrap, PHP"
    )
    pl._seed_named_tools_from_jd(analysis, jd)
    lower = {t.lower() for t in analysis["tools"]}
    assert "jquery" in lower and "bootstrap" in lower and "foundation" in lower, analysis["tools"]


def t30_apple_frameworks_placeholder_stripped():
    fog = ORIG["skills"].replace(
        "Frontend}{: React, Next.js}",
        "Apple Platforms}{: SwiftUI, Xcode, Apple frameworks}",
    )
    cleaned = rb._remove_concept_skills(fog)
    plain = rb.latex_to_plain(cleaned)
    assert "Apple frameworks" not in plain, plain
    assert "Foundation" in plain, plain  # seeded when placeholder removed


def t29_ecosystem_companions_expand_swift_tools():
    analysis = {
        "tools": ["Swift", "SwiftUI"],
        "must_have_skills": ["Swift"],
        "exact_keywords_for_ats": [],
        "research_topics": [],
    }
    pl._expand_ecosystem_tools(analysis)
    lower = {t.lower() for t in analysis["tools"]}
    assert "xcode" in lower and "xctest" in lower and "foundation" in lower, analysis["tools"]
    assert "uikit" in lower and "combine" in lower, analysis["tools"]


def t28_web_context_research_topics_and_fabric():
    from src import web_context as wc

    topics = wc.select_research_topics(
        {"research_topics": ["Microsoft Fabric"], "tools": ["Python", "Docker"]},
        "We use Microsoft Fabric for analytics pipelines.",
    )
    assert "Microsoft Fabric" in topics, topics
    assert "Python" not in topics and "Docker" not in topics

    # Stub Microsoft Learn so the test stays offline
    def fake_json(url: str):
        if "learn.microsoft.com/api/search" in url:
            return {
                "results": [{
                    "title": "What is Microsoft Fabric?",
                    "url": "https://learn.microsoft.com/en-us/fabric/",
                    "description": (
                        "Microsoft Fabric is an all-in-one analytics solution for enterprises "
                        "that covers data movement, lakehouse, real-time analytics, and Power BI."
                    ),
                }]
            }
        return None

    orig_json, orig_research = wc._http_json, wc.research_topic
    wc._http_json = fake_json
    try:
        blurb = wc.research_topic("Microsoft Fabric")
        assert blurb and "Fabric" in blurb and "analytics" in blurb.lower(), blurb
        assert "Microsoft Learn" in blurb

        wc.research_topic = lambda topic: (
            "Microsoft Fabric: unified analytics with Lakehouse and OneLake. "
            "(source: Microsoft Learn)"
            if "Fabric" in topic
            else None
        )
        ctx = wc.build_tech_context(
            {"research_topics": ["Microsoft Fabric"], "tools": []},
            "Microsoft Fabric role",
        )
        assert "Lakehouse" in ctx or "Fabric" in ctx, ctx
    finally:
        wc._http_json = orig_json
        wc.research_topic = orig_research


def t27_architecture_fog_detection():
    foggy = (
        "Shipped pipelines that ingest usage events into SQL on Microsoft Azure, "
        "cutting prep for analytics stores."
    )
    hits = rb.architecture_fog_in_text(foggy)
    assert "into SQL" in hits, hits
    assert "analytics store(s)" in hits, hits
    assert "bare cloud platform (no service named)" in hits, hits

    truncated = (
        "Built Python pipelines in Data Factory and wired Git-based CI/CD so every "
        "merge deploys to staging."
    )
    hits = rb.architecture_fog_in_text(truncated)
    assert "Data Factory (missing Azure)" in hits, hits
    assert "Git-based CI/CD" in hits, hits

    clear = (
        "Shipped Python pipelines on Azure Data Factory that load usage events into "
        "Azure SQL Database, with GitHub Actions deploying every merged pull request."
    )
    assert rb.architecture_fog_in_text(clear) == [], rb.architecture_fog_in_text(clear)

    # Product name must not trip the bare "SQL database" fog
    assert "SQL database" not in rb.architecture_fog_in_text(
        "Writes aggregates into Azure SQL Database from Azure Functions."
    )


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("t") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
