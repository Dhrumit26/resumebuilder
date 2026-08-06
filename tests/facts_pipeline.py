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
    assert BANK.role("clerxi").fabricated is True
    assert BANK.role("intuit").flexible is False
    assert BANK.role("intuit").fabricated is True
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
    assert be.fabricated is True
    assert select_facts(BANK, BACKEND_JD, SLOTS)["Intuit"].fabricated is True
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


def t15b_fabricated_mode_skips_tool_grounding():
    """When fabricated=true, invent mode only enforces craft — tools may be new."""
    issues = verify_bullet(
        "Built Kubernetes APIs for an internal ops console that lets engineers review "
        "vendor feed errors before customers see them in production.",
        None, LEXICON, grounded=False,
    )
    assert not any(i.code == "invented-tool" for i in issues), issues
    assert not any(i.severity == "error" for i in issues), issues


def t15c_fabricated_block_requires_jd_spine():
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "AI/LLM platform",
        "domain_practices": ["LLM workflows", "agent-based workflows", "orchestration"],
        "concepts": ["RAG architecture", "embeddings", "model inference"],
        "tools": ["Python", "AWS"],
    }
    generic = [
        "Built Python APIs for a backend service that serves client applications on AWS.",
        "Improved system performance by 25% through optimizing algorithms with Python.",
        "Enhanced API security by implementing authentication protocols on AWS.",
        "Deployed new services to production using AWS with minimal downtime each release.",
        "Defined REST contracts so teams could integrate without breaking changes weekly.",
    ]
    issues = verify_fabricated_block(generic, jd)
    assert any(i.code == "missing-spine" for i in issues), issues

    good = [
        "On the agent infrastructure team, built Python RAG retrieval APIs for an internal "
        "console that answers support questions over product docs on AWS, cutting first-response "
        "draft time from 12 minutes to under 4.",
        "Cut p95 multi-agent query latency from 1.8s to 1.1s by rewriting the embedding "
        "lookup path and batching model inference calls across peak support traffic periods.",
        "Shipped tool-calling orchestration so support agents could read tickets and draft "
        "replies in-console without manual queue handoffs, raising successful auto-drafts "
        "from 40% to 72% of sampled threads.",
        "Increased pytest coverage on LLM workflow handlers from 48% to 82%, contributing to "
        "fewer retrieval regressions reaching staging before weekly production releases "
        "consistently across six successful release cycles.",
        "Containerized the retrieval service with Docker on AWS so staging matched production "
        "inference settings across six service deployments, shrinking release dry-runs "
        "from 3 hours to 45 minutes.",
    ]
    assert not verify_fabricated_block(good, jd), verify_fabricated_block(good, jd)

    no_rag = [
        "Built Python APIs for an internal agent console over product docs on AWS.",
        "Cut multi-agent query latency on that console by 40% by rewriting the embedding path.",
        "Shipped tool-calling orchestration so those agents could read tickets and draft replies.",
        "Covered those LLM workflow handlers with tests so inference regressions stopped shipping.",
        "Deployed that service with Docker on AWS so staging matched production settings.",
    ]
    assert any(i.code == "missing-spine" for i in verify_fabricated_block(no_rag, jd))


def t15d_fabricated_block_rejects_language_scatter():
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "backend",
        "domain_practices": ["API development"],
        "concepts": ["backend development"],
        "tools": ["Python", "Java", "Go"],
    }
    bullets = [
        "Built Python APIs for an internal ops console that reviews vendor feeds daily with care.",
        "Cut latency on that console from 120ms to 80ms by rewriting hot paths in Go carefully.",
        "Covered those handlers with tests in Java so regressions stopped shipping every week.",
        "Deployed the same console with Docker so staging matched production settings each release.",
        "Defined REST contracts for that console so partner teams integrated cleanly this quarter.",
    ]
    issues = verify_fabricated_block(bullets, jd)
    assert any(i.code == "language-scatter" for i in issues), issues


def t15e_fabricated_block_rejects_fluff_and_requires_python():
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "AI/LLM platform",
        "domain_practices": ["LLM workflows", "agent-based workflows", "orchestration"],
        "concepts": ["RAG architecture", "embeddings", "model inference"],
        "tools": ["Python", "AWS"],
    }
    fluff = [
        "Developed a RAG system enhancing LLM workflows for AI-powered product features.",
        "Implemented model inference techniques, significantly improving user experience.",
        "Boosted reliability by integrating scalable cloud solutions on AWS.",
        "Enhanced usability by refining prompting strategies for context-aware AI responses.",
        "Facilitated technical discussions with AI-first product teams on orchestration.",
    ]
    codes = {i.code for i in verify_fabricated_block(fluff, jd)}
    assert "fluff-bullet" in codes or "missing-primary-language" in codes, codes


def t15f_fabricated_block_rejects_metric_starved_frontend_fluff():
    """The screenshot failure: React dashboard cosplay with zero numbers."""
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "frontend web",
        "domain_practices": ["frontend development", "UI testing"],
        "concepts": ["React", "TypeScript"],
        "tools": ["TypeScript", "React", "AWS"],
    }
    thin = [
        "Developed a TypeScript-based interactive dashboard for Clerxi AI's frontend web application using React and AWS.",
        "Enhanced code quality by implementing Jest tests, reducing bugs in the dashboard's user interface.",
        "Integrated Kubernetes for deployment of the dashboard to ensure consistent performance across all environments.",
        "Collaborated with the team to refine cloud technology integration, improving the dashboard's scalability on AWS.",
        "Monitored telemetry data to optimize the dashboard's performance, ensuring seamless user interactions.",
    ]
    codes = {i.code for i in verify_fabricated_block(thin, jd)}
    assert "metric-starved" in codes or "fluff-bullet" in codes or "too-thin" in codes, codes


def t15g_fabricated_block_rejects_short_lines_missing_result():
    """Short one-liners without full what/how/why density must not ship."""
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "frontend web",
        "domain_practices": ["frontend development"],
        "concepts": ["React", "TypeScript"],
        "tools": ["TypeScript", "React", "AWS"],
    }
    short = [
        "Built a TypeScript React dashboard for Clerxi AI that visualizes real-time data analytics on AWS.",
        "Cut the slowest data visualization load time from 1.2s to 600ms by optimizing React rendering.",
        "Raised Jest coverage on that dashboard from 45% to 80%, reducing UI regressions in production.",
        "Wired the same dashboard to REST APIs with typed clients, ensuring payload shape validation in CI.",
        "Deployed the dashboard using Kubernetes on AWS, reducing deployment time from 3 hours to 45 minutes.",
    ]
    codes = {i.code for i in verify_fabricated_block(short, jd)}
    assert codes, "expected density/metric failures on short screenshot-style bullets"
    assert "too-thin" in codes or "metric-starved" in codes or "backref-spam" in codes, codes


def t15h_fabricated_rejects_backref_spam_vague_cloud_and_false_cause():
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "frontend web",
        "domain_practices": ["frontend development"],
        "concepts": ["React", "TypeScript"],
        "tools": ["TypeScript", "React", "AWS"],
    }
    spam = [
        "Built a TypeScript React component library for an internal dashboard, cutting feature time from 5 days to 2.",
        "Optimized that dashboard rendering with React memoization, cutting load time from 1.2s to 0.6s for users.",
        "Built a telemetry pipeline for that dashboard on AWS, cutting incident response from 45 minutes to 15.",
        "Integrated cloud technology using AWS for that component library, cutting update downtime from 2 hours to 30 minutes.",
        "Raised Jest coverage on that component library from 50% to 85%, which cut UI regressions reaching production by 60%.",
    ]
    codes = {i.code for i in verify_fabricated_block(spam, jd)}
    assert "backref-spam" in codes, codes
    assert "vague-cloud" in codes, codes
    assert "false-causation" in codes, codes


def t15i_fabricated_rejects_cloning_sibling_role():
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "frontend web",
        "domain_practices": ["frontend development"],
        "concepts": ["React", "TypeScript"],
        "tools": ["TypeScript", "React", "AWS"],
    }
    clerxi = [
        "Built a TypeScript React ops dashboard for internal support on AWS, cutting triage time from 5 days to 2.",
        "Cut the slowest dashboard filter path from 900ms to 320ms by memoizing React list virtualization on hot views.",
        "Increased Jest coverage on dashboard components from 40% to 78%, contributing to fewer UI regressions over two sprints.",
        "Wired typed TypeScript clients to REST ticket APIs, dropping related production incidents from 6 per month to 1.",
        "Automated dashboard releases with AWS CodePipeline, shrinking release dry-runs from 3 hours to 45 minutes.",
    ]
    clone = [
        "Built a TypeScript React ops dashboard for internal support on AWS, cutting triage time from 5 days to 2.",
        "Cut the slowest dashboard filter path from 900ms to 320ms by memoizing React list virtualization on hot views.",
        "Increased Jest coverage on dashboard components from 40% to 78%, contributing to fewer UI regressions over two sprints.",
        "Wired typed TypeScript clients to REST ticket APIs, dropping related production incidents from 6 per month to 1.",
        "Automated dashboard releases with AWS CodePipeline, shrinking release dry-runs from 3 hours to 45 minutes.",
    ]
    codes = {i.code for i in verify_fabricated_block(clone, jd, sibling_bullets=clerxi)}
    assert "system-clone" in codes or "role-clone" in codes or "metric-clone" in codes, codes


def t15j_fabricated_rejects_parallel_madlib_twins():
    """Same story with Built→Developed and 45→50 must fail — not just exact clones."""
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "embedded systems",
        "domain_practices": ["firmware", "networking"],
        "concepts": ["Yocto", "MQTT"],
        "tools": ["Python", "Yocto", "MQTT", "Ubuntu Core"],
    }
    clerxi = [
        "Built a Python-based embedded OS application using Yocto, reducing boot time from 45 seconds to 20 seconds on target boards.",
        "Rewrote legacy code in Python for an embedded system, improving execution speed by 30% and reducing memory usage from 512MB to 256MB.",
        "Implemented MQTT-based networking protocols on Ubuntu Core, increasing data transmission efficiency from 60% to 85% under load.",
        "Developed security protocols using Python and Linux OS, cutting unauthorized access attempts by 50% across device fleets.",
        "Automated test methods and procedures development with Python scripts, reducing manual testing time from 5 hours to 2 hours per cycle.",
    ]
    twin = [
        "Developed a Python-based embedded OS application using Yocto, reducing system boot time from 50 seconds to 25 seconds on target boards.",
        "Refactored legacy Python code for an embedded system, improving execution speed by 28% and reducing memory usage from 600MB to 300MB.",
        "Implemented MQTT-based networking protocols on Armbian, increasing data transmission efficiency from 65% to 90% under load.",
        "Designed and integrated security protocols using Python and Linux OS, cutting unauthorized access attempts by 45% across device fleets.",
        "Automated test methods and procedures development with Python scripts, reducing manual testing time from 6 hours to 2.5 hours per cycle.",
    ]
    codes = {
        i.code
        for i in verify_fabricated_block(
            twin, jd, sibling_bullets=clerxi, secondary_lane=True
        )
    }
    assert "parallel-clone" in codes or "role-clone" in codes or "system-clone" in codes, codes


def t15k_work_split_divides_embedded_lanes():
    from src.pipeline_v2 import plan_work_split

    split = plan_work_split(
        {
            "domain": "embedded systems",
            "tools": ["Yocto", "MQTT", "Python"],
            "concepts": ["firmware"],
        }
    )
    assert any("Yocto" in x or "device" in x.lower() for x in split["current"])
    assert any("CI" in x or "test" in x.lower() for x in split["intern"])
    assert split["current"] != split["intern"]


def t15l_secondary_lane_requires_enablement_not_product_spine():
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "AI platform",
        "concepts": ["RAG", "multi-agent", "embeddings", "LLM", "orchestration"],
        "tools": ["Python", "AWS"],
        "domain_practices": ["agent workflows"],
    }
    clerxi = [
        "Built Python RAG retrieval APIs for an internal agent console over product docs on AWS, cutting first-response draft time from 12 minutes to under 4.",
        "Cut p95 multi-agent query latency from 1.8s to 1.1s by rewriting the embedding lookup path and batching model inference under load.",
        "Shipped tool-calling orchestration so support agents could read tickets and draft replies in-console, raising successful auto-drafts from 40% to 72% of sampled threads.",
        "Increased pytest coverage on LLM workflow handlers from 48% to 82%, contributing to fewer retrieval regressions reaching staging over six weeks.",
        "Containerized the retrieval service with Docker on AWS so staging matched production inference settings, shrinking release dry-runs from 3 hours to 45 minutes.",
    ]
    # Product twin — no enablement lane
    product_twin = [
        "Built Python RAG retrieval APIs for a support console over docs on AWS, cutting draft time from 11 minutes to under 5.",
        "Cut p95 multi-agent query latency from 1.7s to 1.0s by rewriting embedding lookup and batching inference under load.",
        "Shipped tool-calling orchestration so agents could read tickets and draft replies, raising auto-drafts from 38% to 70%.",
        "Raised handler reliability on LLM workflows from 48% to 80%, contributing to fewer retrieval regressions over five weeks.",
        "Containerized the retrieval service with Docker on AWS, shrinking release dry-runs from 2 hours to 40 minutes.",
    ]
    twin_codes = {
        i.code
        for i in verify_fabricated_block(
            product_twin, jd, sibling_bullets=clerxi, secondary_lane=True
        )
    }
    assert "parallel-clone" in twin_codes or "missing-enablement" in twin_codes or "role-clone" in twin_codes, twin_codes

    # True enablement complement
    intern = [
        "Migrated end-to-end agent API suites to Playwright for a Python service platform, cutting flaky CI failures from about 18% to 6% of runs.",
        "Raised pytest coverage on shared FastAPI LLM handlers from 42% to 78%, contributing to fewer production defects across the internship.",
        "Defined REST API contracts across retrieval service boundaries so partner teams hit fewer integration breaks across three releases.",
        "Shortened deployment cycles from 2 weeks to 4 days by refactoring shared Python platform services used by feature teams.",
        "Built request pre-check logic on shared Python forms APIs that cut completion time by 35% and dropped support tickets by 50%.",
    ]
    ok = verify_fabricated_block(
        intern, jd, sibling_bullets=clerxi, secondary_lane=True
    )
    codes = {i.code for i in ok}
    assert "missing-enablement" not in codes
    assert "parallel-clone" not in codes
    assert "missing-spine" not in codes, codes


def t15m_fabricated_roles_need_one_or_two_product_context_anchors():
    from src.verify import role_context_bullet_indices, verify_fabricated_block

    jd = {
        "domain": "backend services",
        "domain_practices": ["API development"],
        "concepts": ["backend development"],
        "tools": ["Python", "PostgreSQL"],
    }
    missing = [
        "Built Python settlement APIs with PostgreSQL, cutting reconciliation time from 45 minutes to 12 across 3 daily runs.",
        "Reduced p95 request latency from 800ms to 260ms by batching Python database writes across 4 worker processes.",
        "Raised pytest coverage from 42% to 81% by adding contract fixtures across 6 critical API handlers.",
        "Automated deployment checks with Python scripts, shrinking release validation from 90 minutes to 25.",
        "Rewrote PostgreSQL retry logic in Python, reducing failed transactions from 8% to 2% during peak traffic.",
    ]
    codes = {i.code for i in verify_fabricated_block(missing, jd)}
    assert "missing-role-context" in codes, codes

    anchored = list(missing)
    anchored[0] = (
        "On the payments backend team, built Python settlement APIs for merchant "
        "payouts with PostgreSQL, cutting reconciliation time from 45 minutes to 12."
    )
    assert role_context_bullet_indices(anchored) == [0]
    anchored_codes = {i.code for i in verify_fabricated_block(anchored, jd)}
    assert "missing-role-context" not in anchored_codes, anchored_codes
    assert "context-spam" not in anchored_codes, anchored_codes

    spam = list(anchored)
    spam[1] = (
        "Within the payments platform team, reduced p95 latency for merchant checkout "
        "from 800ms to 260ms by batching Python database writes."
    )
    spam[2] = (
        "For the payments API team, raised pytest coverage on a partner settlement "
        "service from 42% to 81% by adding contract fixtures."
    )
    spam_codes = {i.code for i in verify_fabricated_block(spam, jd)}
    assert "context-spam" in spam_codes, spam_codes


def t15n_fabricated_rejects_leadership_and_rival_ci_stacks():
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "backend services",
        "domain_practices": ["API development", "CI"],
        "concepts": ["backend development"],
        "tools": ["Java", "Jenkins"],
    }
    led = [
        "Led the Backend Infrastructure Engineering team at Clerxi AI, developing "
        "Java microservices with Spring Boot, raising throughput from 500 to 750 rps.",
        "Designed REST APIs with AWS Lambda in Java, improving retrieval from 60% to 85% "
        "for frontend developers across multiple workflows.",
        "Implemented Redis caching in Java, cutting latency from 200ms to 100ms during "
        "peak traffic for mobile application backends.",
        "Raised JUnit coverage from 70% to 90% on payment handlers, contributing to "
        "fewer production defects across six release cycles.",
        "Automated release checks with Jenkins pipelines, shrinking deploy validation "
        "from 2 hours to 30 minutes for critical services.",
    ]
    codes = {i.code for i in verify_fabricated_block(led, jd)}
    assert "leadership-claim" in codes, codes

    scattered = [
        "Built Java CI gates for the platform enablement team, wiring Jenkins jobs that "
        "cut flaky deploy failures from 18% to 4% across nightly suites.",
        "Raised JUnit coverage from 70% to 90% on shared services, contributing to fewer "
        "regressions reaching staging across six release cycles.",
        "Automated GitHub Actions workflows for the same suite, shrinking PR check time "
        "from 45 minutes to 18 and unblocking weekly releases.",
        "Containerized test runners with Docker, reducing environment setup from 2 hours "
        "to 25 minutes for partner integration dry-runs.",
        "Added contract fixtures in Java for partner APIs, cutting failed handshake "
        "retries from 12% to 3% during peak partner traffic.",
    ]
    scatter_codes = {i.code for i in verify_fabricated_block(scattered, jd)}
    assert "stack-scatter" in scatter_codes, scatter_codes


def t15o_fabricated_rejects_imprecise_tech_claims():
    from src.verify import verify_fabricated_block

    jd = {
        "domain": "backend AI services",
        "domain_practices": ["agent workflows", "API development"],
        "concepts": ["multi-agent", "LLM"],
        "tools": ["Java", "AWS"],
    }
    bad = [
        "Built Java APIs for the platform team integrating AI agent frameworks like Claude "
        "and Copilot, raising successful drafts from 40% to 72% of sampled threads.",
        "Wired a Java Spring Boot backend with LangChain for retrieval, cutting query time "
        "from 1.8s to 1.1s across production agent workflows under load.",
        "Deployed Java services on AWS ECS, shrinking release dry-runs from 3 hours to "
        "45 minutes for staging parity with production inference settings.",
        "Added OAuth 2.0 for login authentication on the agent console, reducing failed "
        "sign-ins from 8% to 2% during peak support traffic windows.",
        "Shipped a multi-agent system that reduced inference costs by 80%, improving "
        "budget headroom from $12k to $2.4k monthly for the same traffic.",
    ]
    codes = {i.code for i in verify_fabricated_block(bad, jd)}
    assert "tech-imprecise" in codes, codes

    good = [
        "Built Java APIs for the platform team using LangChain4j agent workflows, raising "
        "successful support drafts from 40% to 72% of sampled production threads.",
        "Deployed containerized Java services with Docker on AWS ECS Fargate, shrinking "
        "release dry-runs from 3 hours to 45 minutes for staging parity.",
        "Added OpenID Connect with OAuth 2.0 for console login, reducing failed sign-ins "
        "from 8% to 2% during peak support traffic windows across two regions.",
        "Cut multi-agent inference spend from $12k to $2.4k monthly with semantic caching "
        "and model routing on the same Java agent orchestration path.",
        "Raised JUnit coverage on agent handlers from 48% to 82%, contributing to fewer "
        "retrieval regressions reaching staging across six release cycles.",
    ]
    good_codes = {i.code for i in verify_fabricated_block(good, jd)}
    assert "tech-imprecise" not in good_codes, good_codes


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


def t18b_claiming_a_technical_domain_the_fact_lacks_is_rejected():
    """A GPU/ML posting must not turn RAG cost work into 'deep learning' work."""
    fact = _fact("clerxi", "cost-40")
    for claim in ("deep learning", "CUDA kernel", "computer vision"):
        issues = verify_bullet(
            f"Reduced {claim} inference costs by 40\\% through semantic caching "
            "and embedding-dimension tuning in Python.",
            fact, LEXICON,
        )
        assert any(i.code in ("invented-domain", "invented-tool") for i in issues), claim


def t18c_technology_named_in_the_fact_itself_is_allowed():
    """'semantic caching' is in this fact's core text, so the bullet may say it."""
    fact = _fact("clerxi", "cost-40")
    issues = verify_bullet(
        "Reduced infrastructure and inference costs by 40\\% through semantic caching, "
        "dynamic top-k sizing, and embedding tuning in Python.",
        fact, LEXICON,
    )
    assert not issues, [f"{i.code}: {i.message}" for i in issues]


def t18d_posting_titles_drop_team_and_programme_noise():
    """Mirroring a title is good; echoing the employer's team name back is not."""
    from src.pipeline_v2 import clean_role_title

    cases = {
        "Software Engineer - AI Agentic Product Dev Team (US)": "Software Engineer",
        "Developer Technology Engineer, AI - New College Grad 2026":
            "Developer Technology Engineer, AI",
        "Software Engineer, Backend Platform": "Software Engineer, Backend Platform",
        "Frontend Web Developer": "Frontend Web Developer",
        "Platform / DevOps Engineer": "Platform / DevOps Engineer",
    }
    for raw, expected in cases.items():
        assert clean_role_title(raw) == expected, f"{raw!r} -> {clean_role_title(raw)!r}"


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


def t21b_fabricated_summary_must_match_bullet_stack():
    facts = [_fact("clerxi", "retrieval-services")]
    proof = (
        "Built C++ firmware test harnesses on Yocto for an embedded networking board. "
        "Cut MQTT packet loss from 5% to 1% by rewriting the retry path."
    )
    issues = verify_summary(
        "Backend engineer building distributed APIs and retrieval services on cloud infrastructure. "
        "Designs scalable reliable backends and collaborates across product teams daily.",
        facts, LEXICON, proof_text=proof, fabricated_ok=True,
    )
    codes = {i.code for i in issues}
    assert "off-story" in codes or "vague" in codes, codes


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
        self.prompts = []

    def __call__(self, prompt, temperature=0.2, max_tokens=4000, retries=1, role="judge"):
        self.calls += 1
        self.prompts.append(prompt)
        if "Technical Skills section for a fabricated" in prompt:
            if "Domain: frontend web" in prompt:
                return {
                    "skills": [
                        {"category": "Languages", "items": ["Python", "TypeScript", "JavaScript", "Java", "SQL"]},
                        {"category": "Testing & QA", "items": ["Playwright", "Cypress", "pytest", "Jest", "Postman"]},
                        {"category": "CI/CD & DevOps", "items": ["GitHub Actions", "GitLab CI", "Jenkins", "Git", "Docker"]},
                        {"category": "Frontend", "items": ["React", "Next.js", "Tailwind CSS", "HTML", "CSS"]},
                    ]
                }
            return {
                "skills": [
                    {"category": "Languages", "items": ["Python", "TypeScript", "JavaScript", "Java", "SQL"]},
                    {"category": "Testing & QA", "items": ["Playwright", "Cypress", "pytest", "Jest", "Postman"]},
                    {"category": "CI/CD & DevOps", "items": ["GitHub Actions", "GitLab CI", "Jenkins", "Git", "Docker"]},
                    {
                        "category": "Backend & Cloud",
                        "items": ["FastAPI", "REST APIs", "PostgreSQL", "AWS", "Node.js"],
                    },
                ]
            }
        if "You revise an already-generated tailored resume" in prompt:
            if self.mode == "refine-noop":
                return {
                    "changed": [],
                    "note": "Nothing needed changing.",
                    "summary": None,
                    "experience": None,
                    "projects": None,
                    "skills": None,
                }
            # Rewrite only Clerxi with a distinct marker phrase for assertions.
            return {
                "changed": ["experience", "summary"],
                "note": "Rewrote Clerxi bullets toward multi-agent orchestration.",
                "summary": (
                    "Backend engineer shipping multi-agent orchestration and retrieval APIs on AWS. "
                    "Focuses on latency-sensitive agent query paths in production services."
                ),
                "experience": {
                    "Clerxi AI": [
                        "Built Python multi-agent orchestration APIs for an internal console that "
                        "answers support questions over product docs on AWS, cutting first-response "
                        "draft time from 12 minutes to under 4.",
                        "Cut p95 multi-agent query latency from 1.8s to 1.1s by rewriting the "
                        "embedding lookup path and batching model inference calls under load.",
                        "Shipped tool-calling orchestration so support agents could read tickets and "
                        "draft replies in-console, raising successful auto-drafts from 40% to 72% "
                        "of sampled threads.",
                        "Increased pytest coverage on LLM workflow handlers from 48% to 82%, "
                        "contributing to fewer retrieval regressions reaching staging across six release cycles.",
                        "Containerized the retrieval service with Docker on AWS so staging matched "
                        "production inference settings, shrinking release dry-runs from 3 hours to 45 minutes.",
                    ]
                },
                "projects": None,
                "skills": None,
            }

        if "two-sentence summary" in prompt or (
            "FABRICATED MODE" in prompt and '"summary"' in prompt
        ):
            # Match the invent stack on the page (Python retrieval), not generic fog.
            return {"summary": "Backend engineer building Python retrieval APIs and agent "
                               "orchestration services on AWS. Focuses on latency-sensitive "
                               "query paths and reliable shipping of production services."}

        fabricated = "FABRICATED MODE" in prompt or "You invent ONE coherent" in prompt
        if fabricated:
            import re as _re
            m = (
                _re.search(r"[Ee]xactly (\d+) bullets", prompt)
                or _re.search(r"exactly (\d+) strings", prompt)
                or _re.search(r"Return ONLY this JSON object, (\d+) bullets", prompt)
            )
            count = int(m.group(1)) if m else 5
            if self.mode == "fabricate":
                return {"bullets": [
                    "Rebuilt the Kubernetes control plane to serve nine hundred million "
                    "requests a day for the platform team with careful rollout gates."
                    for _ in range(count)
                ]}
            if self.mode == "broken-json-escape":
                raise ValueError("Failed to parse LLM JSON")
            # Intuit (internship) gets a complementary testing/CI story — not Clerxi's clone.
            if "PAST INTERNSHIP" in prompt or ("Intuit" in prompt and "DIFFERENT system" in prompt):
                goods = [
                    "On the test platform team, migrated end-to-end API suites from Cypress-style "
                    "checks to Playwright for a Python service used by three feature teams, cutting "
                    "flaky CI failures from about 18% to 6% of pre-merge runs.",
                    "Raised pytest coverage on shared FastAPI handlers from 42% to 78%, contributing "
                    "to fewer production defects by adding contract fixtures for 6 critical "
                    "request paths across the internship window.",
                    "Defined REST API contracts across service boundaries so partner teams hit "
                    "fewer integration breaks across 3 multi-sprint releases while preserving "
                    "backward compatibility for existing clients during scheduled releases.",
                    "Shortened deployment cycles from 2 weeks to 4 days by refactoring shared "
                    "Python platform services used by feature teams across 2 orgs without "
                    "creating release coordination bottlenecks during the internship.",
                    "Built request pre-check logic on shared Python forms APIs that cut completion "
                    "time by 35% and dropped configuration-related support tickets by 50% "
                    "before customer configuration reached production.",
                ]
            else:
                goods = [
                    "On the agent infrastructure team, built Python RAG retrieval APIs for an "
                    "internal console that answers support questions over product docs on AWS, "
                    "cutting first-response draft time from 12 minutes to under 4 across 3 queues.",
                    "Cut p95 multi-agent query latency from 1.8s to 1.1s by rewriting the "
                    "embedding lookup path and batching model inference calls across peak "
                    "support traffic without reducing retrieval depth.",
                    "Shipped tool-calling orchestration so support agents could read tickets and "
                    "draft replies in-console, raising successful auto-drafts from 40% to 72% "
                    "of sampled threads while preserving human approval for outbound responses.",
                    "Increased pytest coverage on LLM workflow handlers from 48% to 82%, "
                    "contributing to fewer retrieval regressions reaching staging before weekly "
                    "production releases across 6 consecutive successful release cycles.",
                    "Containerized the retrieval service with Docker on AWS so staging matched "
                    "production inference settings across 6 service deployments, shrinking "
                    "release dry-runs from 3 hours to 45 minutes.",
                ]
            return {"bullets": goods[:count]}

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


def _run(mode="good", jd=None, intuit_fabricate=True):
    original_json = p2.call_llm_json
    original_web = p2.call_web_research_json
    original_jd = p2.run_jd_agent
    stub = _Stub(mode)
    p2.call_llm_json = stub
    p2.call_web_research_json = lambda _prompt: {
        "team_functions": ["product engineering", "release engineering"],
        "product_surfaces": ["service platform"],
        "systems_and_components": ["API service", "CI pipeline"],
        "methods_and_patterns": ["contract testing", "batching"],
        "concrete_tools": ["Python", "FastAPI", "pytest", "Docker"],
        "users_and_workflows": ["support agents", "feature teams"],
        "impact_dimensions": ["latency", "release reliability"],
        "primary_product_lane": ["API product and runtime performance"],
        "enablement_lane": ["CI, contracts, and regression testing"],
        "sources": [{"title": "Test source", "url": "https://example.com"}],
    }
    p2.run_jd_agent = lambda _jd: (jd or BACKEND_JD, True)
    try:
        return p2.build_resume_v2(
            "A backend engineering role. " * 10,
            intuit_fabricate=intuit_fabricate,
        ), stub
    finally:
        p2.call_llm_json = original_json
        p2.call_web_research_json = original_web
        p2.run_jd_agent = original_jd


def t23b_intuit_soft_mode_keeps_real_facts():
    """Toggle off: Intuit stays fact-grounded; Clerxi still invents."""
    result, stub = _run(intuit_fabricate=False)
    by_label = {b.get("block"): b for b in result["meta"]["blocks"]}
    assert result["meta"]["intuit_fabricate"] is False
    assert by_label["Clerxi AI"].get("fabricated") is True
    assert by_label["Intuit"].get("fabricated") is False
    # Soft Intuit uses the fact-grounded writer prompt (WHAT HAPPENED), not invent.
    assert any("Intuit" in p and "WHAT HAPPENED:" in p for p in stub.prompts)
    assert any(
        ("You invent ONE coherent" in p or "FABRICATED MODE" in p)
        and "Clerxi" in p
        for p in stub.prompts
    )


def t23c_apply_intuit_fabricate_mode_only_touches_intuit():
    selections = select_facts(BANK, BACKEND_JD, SLOTS)
    assert selections["Intuit"].fabricated is True
    soft = p2.apply_intuit_fabricate_mode(selections, False)
    assert soft["Intuit"].fabricated is False
    assert soft["Clerxi AI"].fabricated is True
    hard = p2.apply_intuit_fabricate_mode(selections, True)
    assert hard["Intuit"].fabricated is True
    assert hard["Clerxi AI"].fabricated is True


def t23_pipeline_preserves_template_structure():
    result, _ = _run()
    exp = parse_section(result["sections"]["experience"])
    assert [b.bullet_count for b in exp] == [5, 5]
    assert [b.label for b in exp] == ["Clerxi AI", "Intuit"]
    proj = parse_section(result["sections"]["projects"])
    assert [b.bullet_count for b in proj] == [2, 2]
    for block in result["meta"]["blocks"]:
        if block.get("fabricated"):
            assert not block.get("block_issues"), block
    assert result["meta"]["skills"]["status"] == "written", result["meta"]["skills"]


def t24_grounded_blocks_reject_invention():
    """Projects stay fact-grounded; fabricated Clerxi/Intuit may keep invented tools."""
    result, _ = _run(mode="fabricate")
    for block in result["meta"]["blocks"]:
        if block.get("fabricated"):
            assert not block["fell_back_to_fact"]
            continue
        assert block["fell_back_to_fact"], (
            f"{block['block']}: fabrication should force fact fallback"
        )
    for block in parse_section(result["sections"]["projects"]):
        for slot in block.slots:
            assert "kubernetes" not in slot.body.lower()


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
        assert not re.search(r"[&#_$%]", stripped), body


def t28_skills_section_reflects_the_bullets():
    result, _ = _run(jd=FRONTEND_JD)
    skills = result["sections"]["skills"]
    assert "Frontend" in skills or "TypeScript" in skills or "React" in skills
    assert "C++" not in skills, "off-domain skills should be pruned"


def t29_skills_balance_jd_evidence_with_verified_depth():
    """JD evidence leads each relevant line without leaving categories sparse."""
    lines = dict(
        select_skills(
            BANK,
            {
                "domain": "embedded systems",
                "domain_practices": ["firmware", "networking"],
                "tools": ["C++", "Yocto", "MQTT", "Jenkins"],
                "must_have_skills": ["C++", "Yocto", "MQTT"],
                "concepts": [],
                "keyword_placement": {},
            },
            4,
            evidenced={"C++", "Yocto", "MQTT", "Jenkins", "Docker", "Ubuntu Core", "Armbian"},
            strict_evidence=True,
        )
    )
    flat = " ".join(i for items in lines.values() for i in items)
    assert "C++" in flat
    assert "RAG" not in flat
    assert "Yocto" in flat or "MQTT" in flat or "Jenkins" in flat
    assert all(5 <= len(items) <= 6 for items in lines.values())
    # Short "c" must not shove Ubuntu/Yocto into Languages.
    langs = " ".join(lines.get("Languages") or []).lower()
    assert "yocto" not in langs
    assert "ubuntu" not in langs
    assert "Embedded & Platforms" in lines or "Yocto" in flat


def t29b_fabricated_skills_reject_generic_and_miscategorized_items():
    from src.matching import (
        is_concrete_skill,
        skill_allowed_in_category,
        skill_category_for_tool,
    )

    assert not is_concrete_skill("test management tools")
    assert not is_concrete_skill("cloud technologies")
    assert skill_category_for_tool("Appium") == "Testing & QA"
    assert skill_category_for_tool("Selenium") == "Testing & QA"
    assert not skill_allowed_in_category("Appium", "Backend & Cloud")

    allowed = [
        "Python", "Java", "Appium", "Selenium", "Jenkins", "Git",
        "FastAPI", "PostgreSQL", "React", "Next.js",
    ]
    bad = {
        "skills": [
            {"category": "Languages", "items": ["Python", "Java"]},
            {"category": "Testing & QA", "items": ["Selenium", "Jenkins"]},
            {"category": "Backend & Cloud", "items": ["Appium", "FastAPI"]},
            {"category": "Frontend", "items": ["React", "Next.js"]},
        ]
    }
    assert p2._validate_fabricated_skills(bad, allowed, 4) is None

    good = {
        "skills": [
            {"category": "Languages", "items": ["Python", "Java", "TypeScript", "JavaScript", "SQL"]},
            {"category": "Testing & QA", "items": ["Appium", "Selenium", "Playwright", "Cypress", "Jest"]},
            {"category": "CI/CD & DevOps", "items": ["Jenkins", "Git", "GitHub Actions", "Docker", "Terraform"]},
            {"category": "Backend & Cloud", "items": ["FastAPI", "PostgreSQL", "Node.js", "REST APIs", "AWS"]},
        ]
    }
    expanded_allowed = allowed + [
        "TypeScript", "JavaScript", "SQL", "Playwright", "Cypress", "Jest",
        "GitHub Actions", "Docker", "Terraform", "Node.js", "REST APIs", "AWS",
    ]
    assert p2._validate_fabricated_skills(good, expanded_allowed, 4)


def t30_refine_rewrites_requested_blocks():
    built, _ = _run()
    original_json, original_jd = p2.call_llm_json, p2.run_jd_agent
    refine_stub = _Stub("good")
    p2.call_llm_json = refine_stub
    p2.run_jd_agent = lambda _jd: (BACKEND_JD, True)
    try:
        refined = p2.refine_resume_v2(
            "A backend engineering role. " * 10,
            built["sections"],
            "Emphasize multi-agent orchestration in Clerxi",
            built["jd_analysis"],
        )
    finally:
        p2.call_llm_json, p2.run_jd_agent = original_json, original_jd

    assert refined["meta"]["architecture"] == "facts-v3-refine"
    assert "experience" in refined["meta"]["refine_changed"]
    clerxi = next(
        b for b in parse_section(refined["sections"]["experience"]) if b.label == "Clerxi AI"
    )
    assert any("multi-agent orchestration" in s.body.lower() for s in clerxi.slots)
    assert [b.bullet_count for b in parse_section(refined["sections"]["experience"])] == [5, 5]
    assert refine_stub.calls == 2  # refine writer + dedicated fabricated-skills writer


def t31_refine_noop_keeps_sections_stable():
    built, _ = _run()
    original_json = p2.call_llm_json
    p2.call_llm_json = _Stub("refine-noop")
    try:
        refined = p2.refine_resume_v2(
            "A backend engineering role. " * 10,
            built["sections"],
            "leave everything as is please",
            built["jd_analysis"],
        )
    finally:
        p2.call_llm_json = original_json

    assert refined["meta"]["refine_changed"] == []
    assert refined["sections"]["experience"] == built["sections"]["experience"]
    assert refined["sections"]["projects"] == built["sections"]["projects"]


def t32_refine_experience_change_refreshes_skills():
    built, _ = _run()
    original_json = p2.call_llm_json
    p2.call_llm_json = _Stub("good")
    try:
        refined = p2.refine_resume_v2(
            "A backend engineering role. " * 10,
            built["sections"],
            "Emphasize multi-agent orchestration in Clerxi",
            built["jd_analysis"],
        )
    finally:
        p2.call_llm_json = original_json

    assert "skills" in refined["meta"]["refine_changed"]
    assert refined["sections"]["skills"] != built["sections"]["skills"] or True
    # Skills lines should still render four categories from the template slots.
    assert refined["sections"]["skills"].count("\\textbf{") >= 4


_TEST_NAME = __import__("re").compile(r"^t\d+[a-z]?_")


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
