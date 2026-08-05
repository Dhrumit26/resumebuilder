"""JD -> fact matching. Deterministic, no LLM.

Which real facts fill the resume's fixed bullet slots, and in what order, is a
ranking problem over the fact bank — not something a model should guess at.
This module turns a JD analysis into a weighted theme profile, scores every
fact against it, and returns the selection the writers must work from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .facts import Fact, FactBank, Project, Role, SkillCategory

# --- JD vocabulary -> engineering themes -------------------------------------
# Matched as substrings against the JD's domain, practices, concepts and tools.
_DOMAIN_THEMES: list[tuple[str, list[str]]] = [
    ("frontend", ["frontend-ui", "react", "typescript-js", "ux-performance", "full-stack"]),
    ("front-end", ["frontend-ui", "react", "typescript-js", "ux-performance", "full-stack"]),
    ("ui", ["frontend-ui", "ux-performance"]),
    ("web", ["frontend-ui", "backend-services", "api-design", "full-stack"]),
    ("full stack", ["full-stack", "frontend-ui", "backend-services", "api-design"]),
    ("full-stack", ["full-stack", "frontend-ui", "backend-services", "api-design"]),
    ("backend", ["backend-services", "api-design", "databases", "distributed-systems"]),
    ("back-end", ["backend-services", "api-design", "databases", "distributed-systems"]),
    ("server", ["backend-services", "api-design"]),
    ("api", ["api-design", "api-contracts", "backend-services"]),
    ("microservice", ["backend-services", "distributed-systems", "api-design"]),
    ("distributed", ["distributed-systems", "backend-services", "performance-optimization"]),
    ("systems programming", ["performance-optimization", "algorithms-data-structures"]),
    ("embedded", ["performance-optimization", "algorithms-data-structures"]),
    ("firmware", ["performance-optimization", "algorithms-data-structures"]),
    ("ai", ["ai-ml", "search-retrieval"]),
    ("llm", ["ai-ml", "search-retrieval"]),
    ("ml", ["ai-ml"]),
    ("machine learning", ["ai-ml"]),
    ("search", ["search-retrieval", "performance-optimization"]),
    ("data engineering", ["data-pipeline", "databases", "distributed-systems"]),
    ("data platform", ["data-pipeline", "databases", "cloud-aws"]),
    ("analytics", ["data-pipeline", "databases"]),
    ("devops", ["containers-devops", "ci-cd", "cloud-aws", "monitoring-observability"]),
    ("platform", ["containers-devops", "ci-cd", "cloud-aws", "developer-experience", "backend-services"]),
    ("infrastructure", ["containers-devops", "cloud-aws", "monitoring-observability"]),
    ("sre", ["monitoring-observability", "production-support", "containers-devops"]),
    ("reliability", ["monitoring-observability", "production-support", "quality"]),
    ("cloud", ["cloud-aws", "containers-devops", "backend-services"]),
    ("test", ["testing", "test-automation", "ci-cd", "quality"]),
    ("qa", ["testing", "test-automation", "quality"]),
    ("quality", ["testing", "quality"]),
    ("mobile", ["mobile", "frontend-ui", "ux-performance"]),
    ("ios", ["mobile", "frontend-ui"]),
    ("android", ["mobile", "frontend-ui"]),
    ("security", ["security-auth"]),
    ("automation", ["automation-tooling", "ci-cd"]),
    ("developer tool", ["developer-experience", "automation-tooling"]),
    ("integration", ["integrations", "api-design"]),
]

_PRACTICE_THEMES: list[tuple[str, list[str]]] = [
    ("api design", ["api-design", "api-contracts"]),
    ("api contract", ["api-contracts", "api-design"]),
    ("rest", ["api-design", "backend-services"]),
    ("graphql", ["api-design", "backend-services"]),
    ("caching", ["caching", "performance-optimization"]),
    ("cache", ["caching", "performance-optimization"]),
    ("scaling", ["distributed-systems", "performance-optimization"]),
    ("scalab", ["distributed-systems", "performance-optimization"]),
    ("latency", ["performance-optimization"]),
    ("throughput", ["performance-optimization"]),
    ("profil", ["performance-optimization", "algorithms-data-structures"]),
    ("benchmark", ["performance-optimization"]),
    ("optimiz", ["performance-optimization", "cost-optimization"]),
    ("cost", ["cost-optimization"]),
    ("query optimization", ["databases", "performance-optimization"]),
    ("database", ["databases"]),
    ("sql", ["databases"]),
    ("schema", ["databases"]),
    ("data structure", ["algorithms-data-structures"]),
    ("algorithm", ["algorithms-data-structures"]),
    ("object-oriented", ["oop-design"]),
    ("oop", ["oop-design"]),
    ("observability", ["monitoring-observability"]),
    ("monitoring", ["monitoring-observability"]),
    ("logging", ["monitoring-observability"]),
    ("debug", ["production-support", "monitoring-observability"]),
    ("on-call", ["production-support"]),
    ("incident", ["production-support", "monitoring-observability"]),
    ("production support", ["production-support"]),
    ("ci/cd", ["ci-cd", "test-automation"]),
    ("continuous integration", ["ci-cd", "test-automation"]),
    ("continuous delivery", ["ci-cd"]),
    ("pipeline", ["ci-cd", "data-pipeline"]),
    ("deployment", ["ci-cd", "containers-devops"]),
    ("container", ["containers-devops"]),
    ("docker", ["containers-devops"]),
    ("kubernetes", ["containers-devops", "distributed-systems"]),
    ("unit test", ["testing", "test-automation"]),
    ("integration test", ["testing", "test-automation"]),
    ("end-to-end", ["testing", "test-automation"]),
    ("coverage", ["testing", "quality"]),
    ("code review", ["collaboration-process", "quality"]),
    ("refactor", ["refactoring", "quality"]),
    ("legacy", ["refactoring"]),
    ("agile", ["collaboration-process"]),
    ("scrum", ["collaboration-process"]),
    ("sdlc", ["collaboration-process", "ci-cd"]),
    ("cross-functional", ["collaboration-process", "product-engineering"]),
    ("collaborat", ["collaboration-process"]),
    ("stakeholder", ["collaboration-process", "product-engineering"]),
    ("product", ["product-engineering"]),
    ("customer", ["product-engineering", "ux-performance"]),
    ("responsive", ["frontend-ui", "ux-performance"]),
    ("accessib", ["frontend-ui", "ux-performance"]),
    ("component", ["frontend-ui", "react"]),
    ("state management", ["frontend-ui", "react"]),
    ("authentication", ["security-auth"]),
    ("authorization", ["security-auth"]),
    ("oauth", ["security-auth"]),
    ("encrypt", ["security-auth"]),
    ("etl", ["data-pipeline"]),
    ("ingestion", ["data-pipeline", "backend-services"]),
    ("streaming", ["data-pipeline", "distributed-systems"]),
    ("vector", ["search-retrieval", "ai-ml"]),
    ("embedding", ["search-retrieval", "ai-ml"]),
    ("rag", ["search-retrieval", "ai-ml"]),
    ("prompt", ["ai-ml"]),
    ("agent", ["ai-ml", "automation-tooling"]),
]

_TOOL_THEMES: dict[str, list[str]] = {
    "react": ["react", "frontend-ui", "typescript-js"],
    "next.js": ["react", "frontend-ui", "full-stack", "typescript-js"],
    "vue": ["frontend-ui", "typescript-js"],
    "angular": ["frontend-ui", "typescript-js"],
    "svelte": ["frontend-ui", "typescript-js"],
    "jquery": ["frontend-ui", "typescript-js"],
    "bootstrap": ["frontend-ui"],
    "tailwind": ["frontend-ui"],
    "html": ["frontend-ui"],
    "css": ["frontend-ui"],
    "html5": ["frontend-ui"],
    "css3": ["frontend-ui"],
    "javascript": ["typescript-js", "frontend-ui"],
    "typescript": ["typescript-js", "frontend-ui"],
    "node.js": ["backend-services", "typescript-js"],
    "python": ["backend-services"],
    "java": ["backend-services", "oop-design"],
    "go": ["backend-services"],
    "c++": ["performance-optimization", "algorithms-data-structures"],
    "c#": ["backend-services", "oop-design"],
    "rust": ["performance-optimization"],
    "swift": ["mobile", "frontend-ui"],
    "kotlin": ["mobile"],
    "fastapi": ["backend-services", "api-design"],
    "django": ["backend-services", "api-design"],
    "flask": ["backend-services", "api-design"],
    "spring": ["backend-services", "api-design", "oop-design"],
    "express": ["backend-services", "api-design"],
    "graphql": ["api-design", "backend-services"],
    "grpc": ["api-design", "distributed-systems"],
    "rest apis": ["api-design", "backend-services"],
    "postgresql": ["databases"],
    "mysql": ["databases"],
    "mongodb": ["databases"],
    "redis": ["caching", "databases"],
    "elasticsearch": ["search-retrieval", "databases"],
    "snowflake": ["data-pipeline", "databases"],
    "kafka": ["data-pipeline", "distributed-systems"],
    "airflow": ["data-pipeline"],
    "spark": ["data-pipeline"],
    "docker": ["containers-devops"],
    "kubernetes": ["containers-devops", "distributed-systems"],
    "terraform": ["containers-devops", "cloud-aws"],
    "aws": ["cloud-aws", "containers-devops"],
    "azure": ["cloud-aws", "containers-devops"],
    "gcp": ["cloud-aws", "containers-devops"],
    "lambda": ["cloud-aws", "backend-services"],
    "datadog": ["monitoring-observability"],
    "prometheus": ["monitoring-observability"],
    "grafana": ["monitoring-observability"],
    "jest": ["testing", "test-automation", "frontend-ui"],
    "playwright": ["testing", "test-automation"],
    "cypress": ["testing", "test-automation"],
    "selenium": ["testing", "test-automation"],
    "pytest": ["testing", "test-automation"],
    "junit": ["testing", "test-automation"],
    "react testing library": ["testing", "frontend-ui", "react"],
    "github actions": ["ci-cd", "automation-tooling"],
    "gitlab ci": ["ci-cd", "automation-tooling"],
    "jenkins": ["ci-cd", "automation-tooling"],
    "circleci": ["ci-cd"],
    "git": ["collaboration-process"],
    "oauth 2.0": ["security-auth"],
    "oauth": ["security-auth"],
}


@dataclass
class Selection:
    """The facts chosen for one block, in the order they should be written."""
    owner_id: str
    owner_label: str
    kind: str                    # "role" | "project"
    flexible: bool
    facts: list[Fact]
    scores: list[float]


def _texts_from_analysis(analysis: dict) -> tuple[str, list[str]]:
    """(domain blob, tool list) pulled out of a JD analysis dict."""
    parts = [
        str(analysis.get("domain") or ""),
        str(analysis.get("role_title") or ""),
        " ".join(str(p) for p in (analysis.get("domain_practices") or [])),
        " ".join(str(c) for c in (analysis.get("concepts") or [])),
        str(analysis.get("ideal_summary_angle") or ""),
        str(analysis.get("competitive_positioning") or ""),
    ]
    tools: list[str] = []
    for key in ("tools", "must_have_skills", "nice_to_have_skills", "exact_keywords_for_ats"):
        tools.extend(str(t) for t in (analysis.get(key) or []))
    return " ".join(parts).lower(), tools


def theme_profile(analysis: dict) -> dict[str, float]:
    """Weighted themes the JD cares about. Higher weight = more important."""
    blob, tools = _texts_from_analysis(analysis)
    profile: dict[str, float] = {}

    def add(themes: list[str], weight: float) -> None:
        for theme in themes:
            profile[theme] = profile.get(theme, 0.0) + weight

    domain = str(analysis.get("domain") or "").lower()
    for needle, themes in _DOMAIN_THEMES:
        if needle in domain:
            add(themes, 3.0)          # the JD's own domain label dominates
        elif needle in blob:
            add(themes, 0.75)

    for needle, themes in _PRACTICE_THEMES:
        if needle in blob:
            add(themes, 1.25)

    for tool in tools:
        key = tool.strip().lower()
        themes = _TOOL_THEMES.get(key)
        if themes is None:
            for tool_name, tool_themes in _TOOL_THEMES.items():
                if tool_name in key or key in tool_name:
                    themes = tool_themes
                    break
        if themes:
            add(themes, 1.0)

    return profile


def _jd_tool_set(analysis: dict) -> set[str]:
    _, tools = _texts_from_analysis(analysis)
    return {t.strip().lower() for t in tools if t.strip()}


# Same technology, different spelling. Used both for scoring a JD keyword as
# covered and for deciding a skill is JD-relevant — a posting that says "HTML5"
# must not cause the candidate's real "HTML" skill to be pruned off the page.
TECH_SYNONYMS: dict[str, list[str]] = {
    "html5": ["html"],
    "css3": ["css"],
    "js": ["javascript"],
    "ts": ["typescript"],
    "postgres": ["postgresql"],
    "postgresql": ["postgres"],
    "node": ["node.js"],
    "node.js": ["node"],
    "next": ["next.js"],
    "next.js": ["next"],
    "k8s": ["kubernetes"],
    "kubernetes": ["k8s"],
    "rest": ["rest apis", "rest api"],
    "rest api": ["rest apis", "rest"],
    "rest apis": ["rest api", "rest"],
    "ci/cd": ["github actions", "gitlab ci", "jenkins"],
    "continuous integration": ["github actions", "gitlab ci", "jenkins"],
    "unit testing": ["jest", "pytest", "junit", "react testing library"],
    "automated testing": ["jest", "playwright", "cypress", "react testing library"],
    "version control": ["git", "github"],
    "git": ["github actions", "gitlab ci", "github"],
    "relational database": ["postgresql", "mysql", "sql"],
    "aws lambda": ["lambda", "aws"],
    "containerization": ["docker"],
    "containers": ["docker"],
    # Capability phrasings. A posting asks for "Distributed Systems"; the resume
    # says "distributed vector search backend". Same evidence, different noun —
    # scoring it as a gap understates the match and invites keyword stuffing.
    "distributed systems": ["distributed"],
    "backend development": ["backend"],
    "backend services": ["backend"],
    "full-stack development": ["full-stack"],
    "model inference": ["inference"],
    "agentic systems": ["multi-agent", "agent orchestration", "agent infrastructure"],
    "agent-based workflows": ["multi-agent", "agent orchestration"],
    "orchestration frameworks": ["agent orchestration", "orchestration"],
    "rag": ["retrieval", "semantic reranking"],
    "rag architecture": ["rag", "retrieval"],
    "retrieval augmented generation": ["rag", "retrieval"],
    "software testing": ["test coverage", "jest", "playwright", "cypress"],
    "system design": ["service-oriented", "api design", "api contracts"],
    "embeddings": ["embedding"],
    "apis": ["api", "rest apis"],
    "performance optimization": ["latency", "performance tuning", "optimizing"],
    "ai algorithms": ["retrieval algorithms"],
    "code review": ["code review", "cross-team"],
}


def token_pattern(term: str) -> str:
    """Regex matching `term` as a whole technology token.

    Two things have to hold at once. "C" must not match inside "CSS", so letters
    and digits are boundaries. But "Node.js" must stay one token while
    "GitHub Actions." at the end of a sentence still matches — so a dot counts
    as part of the name only when a letter or digit follows it.
    """
    return (
        r"(?<![a-z0-9+#])(?<![a-z0-9]\.)"
        + re.escape(term)
        + r"(?![a-z0-9+#])(?!\.[a-z0-9])"
    )


def _token_bounded(needle: str, haystack: str) -> bool:
    """True when `needle` appears in `haystack` as a whole token."""
    return re.search(token_pattern(needle), haystack) is not None


def tool_matches(item: str, jd_tools: set[str]) -> bool:
    """Does the JD ask for this technology, by name or by a known synonym?"""
    key = (item or "").strip().lower()
    if not key:
        return False
    if key in jd_tools:
        return True
    # "HTML5" in the posting must count as asking for the candidate's "HTML".
    if any(alt in jd_tools for alt in TECH_SYNONYMS.get(key, [])):
        return True
    for jd_tool in jd_tools:
        if key in TECH_SYNONYMS.get(jd_tool, []):
            return True
    # Short names ("C", "Go", "R") are too ambiguous for fuzzy matching.
    if len(key) < 3:
        return False
    for jd_tool in jd_tools:
        if len(jd_tool) < 3:
            continue
        if _token_bounded(key, jd_tool) or _token_bounded(jd_tool, key):
            return True
    return False


def score_fact(fact: Fact, profile: dict[str, float], jd_tools: set[str]) -> float:
    """How well one real fact answers this JD."""
    score = 0.0
    for theme in fact.themes:
        score += profile.get(theme, 0.0)

    # A tool the JD literally names, evidenced by this fact, is worth a lot:
    # it is a keyword the ATS will match against real work.
    for tool in fact.tools:
        if tool_matches(tool, jd_tools):
            score += 2.5

    if fact.metrics:
        score += 1.0                    # measurable outcomes beat unmeasured ones

    # A transferable angle that speaks to the JD keeps distant-domain facts alive.
    if fact.angles and profile:
        angle_blob = " ".join(fact.angles).lower()
        for needle, themes in _PRACTICE_THEMES:
            if needle in angle_blob and any(profile.get(t, 0) > 0 for t in themes):
                score += 0.4
                break
    return score


def _cohesion_bonus(fact: Fact, anchor: Fact) -> float:
    """Reward facts that belong to the same body of work as the top-ranked one.

    A role whose bullets share a stack and a subject reads as one engineer who
    owns a system. Five unrelated highlights read as a list of everything the
    candidate has ever touched — the "knows everything, expert in nothing" tell.
    """
    if fact is anchor:
        return 0.0
    shared_tools = len({t.lower() for t in fact.tools} & {t.lower() for t in anchor.tools})
    shared_themes = len(set(fact.themes) & set(anchor.themes))
    return 1.5 * shared_tools + 0.75 * shared_themes


def select_for_block(
    facts: list[Fact], slots: int, profile: dict[str, float], jd_tools: set[str]
) -> tuple[list[Fact], list[float]]:
    """Pick and order the `slots` best facts. Ties keep fact-bank order."""
    scored = [(score_fact(f, profile, jd_tools), -i, f) for i, f in enumerate(facts)]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    # Re-rank everything after the best-matching fact by how well it coheres with
    # it, so the block tells one story instead of listing unrelated highlights.
    if len(scored) > slots and scored:
        anchor = scored[0][2]
        rest = [
            (score + _cohesion_bonus(fact, anchor), tie, fact)
            for score, tie, fact in scored[1:]
        ]
        rest.sort(key=lambda t: (t[0], t[1]), reverse=True)
        scored = [scored[0]] + rest

    chosen = scored[:slots]
    # A `lead` fact introduces the system the other bullets talk about, so it is
    # ranked for SELECTION but always written FIRST.
    chosen.sort(key=lambda t: 0 if t[2].lead else 1)
    return [t[2] for t in chosen], [round(t[0], 2) for t in chosen]


def select_facts(
    bank: FactBank,
    analysis: dict,
    slots_by_label: dict[str, int],
) -> dict[str, Selection]:
    """Choose facts for every role and project block.

    slots_by_label maps a block label (company or project name, as parsed from
    the LaTeX template) to that block's bullet count.
    """
    profile = theme_profile(analysis)
    jd_tools = _jd_tool_set(analysis)
    out: dict[str, Selection] = {}

    owners: list[tuple[str, Role | Project]] = [("role", r) for r in bank.roles]
    owners += [("project", p) for p in bank.projects]

    for kind, owner in owners:
        label = owner.company if isinstance(owner, Role) else owner.name
        slots = slots_by_label.get(label)
        if slots is None:
            continue
        facts, scores = select_for_block(owner.facts, slots, profile, jd_tools)
        out[label] = Selection(
            owner_id=owner.id,
            owner_label=label,
            kind=kind,
            flexible=isinstance(owner, Role) and owner.flexible,
            facts=facts,
            scores=scores,
        )
    return out


# --- skills ------------------------------------------------------------------

MAX_SKILLS_PER_LINE = 6


def select_skills(
    bank: FactBank,
    analysis: dict,
    line_count: int,
    evidenced: set[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Pick which skill lines to show, what goes on them, and in what order.

    Only skills already in the fact bank are used — nothing is added to make a JD
    happy. Lines are also PRUNED: a skills section listing every language the
    candidate has ever touched reads as someone who claims everything and
    specialises in nothing. Priority is JD-named first, then technologies the
    resume's own bullets actually demonstrate, then the rest as filler.

    evidenced: tools named by the facts selected for this resume.
    """
    profile = theme_profile(analysis)
    jd_tools = _jd_tool_set(analysis)
    evidenced_lower = {t.lower() for t in (evidenced or set())}

    def category_score(cat: SkillCategory) -> float:
        score = sum(profile.get(t, 0.0) for t in cat.themes if t != "any")
        for item in cat.items:
            if tool_matches(item, jd_tools):
                score += 3.0
        return score

    always = [c for c in bank.skill_categories if c.always]
    rest = [c for c in bank.skill_categories if not c.always]
    rest.sort(key=lambda c: (category_score(c), -bank.skill_categories.index(c)), reverse=True)

    chosen = (always + rest)[:line_count]

    lines: list[tuple[str, list[str]]] = []
    for cat in chosen:
        def item_rank(item: str, category: SkillCategory = cat) -> tuple[int, int]:
            if tool_matches(item, jd_tools):
                tier = 0                                    # the JD asked for it
            elif item.lower() in evidenced_lower:
                tier = 1                                    # a bullet proves it
            else:
                tier = 2                                    # true, but off-topic here
            return (tier, category.items.index(item))

        ranked = sorted(cat.items, key=item_rank)
        keep = [i for i in ranked if item_rank(i)[0] < 2][:MAX_SKILLS_PER_LINE]
        # Never ship an empty line; top up with the category's own ordering.
        if len(keep) < 3:
            for item in ranked:
                if item not in keep:
                    keep.append(item)
                if len(keep) >= 3:
                    break
        lines.append((cat.name, keep))
    return lines
