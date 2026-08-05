"""Fact bank: the only source of truth for what may appear on the resume.

Every bullet the pipeline writes traces back to exactly one Fact here. Writers
select, order, and rephrase facts; they never invent systems, tools, or numbers.
Verification in verify.py checks generated bullets back against these objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import BASE_DIR

FACTS_PATH = BASE_DIR / "data" / "facts.yaml"

# Theme vocabulary shared by facts.yaml and matching.py. A theme is an
# ENGINEERING activity, never an industry or a job title.
THEMES = {
    "backend-services",
    "api-design",
    "api-contracts",
    "distributed-systems",
    "performance-optimization",
    "cost-optimization",
    "caching",
    "algorithms-data-structures",
    "oop-design",
    "ai-ml",
    "search-retrieval",
    "cloud-aws",
    "containers-devops",
    "ci-cd",
    "testing",
    "test-automation",
    "quality",
    "monitoring-observability",
    "production-support",
    "frontend-ui",
    "react",
    "typescript-js",
    "ux-performance",
    "full-stack",
    "databases",
    "data-pipeline",
    "integrations",
    "automation-tooling",
    "security-auth",
    "refactoring",
    "developer-experience",
    "collaboration-process",
    "product-engineering",
    "mobile",
    "any",
}


@dataclass
class Fact:
    id: str
    core: str
    owner_id: str = ""          # role/project id this fact belongs to
    owner_label: str = ""       # human label, e.g. "Clerxi AI"
    tools: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)
    angles: list[str] = field(default_factory=list)
    # Introduces what the role/project IS. When selected it is placed first, so
    # later bullets never refer back to something the reader has not met yet.
    lead: bool = False

    @property
    def key(self) -> str:
        return f"{self.owner_id}.{self.id}"

    def numbers(self) -> set[str]:
        """Every numeric token this fact licenses a bullet to contain."""
        out: set[str] = set()
        for metric in self.metrics:
            out.update(_numeric_tokens(metric))
        out.update(_numeric_tokens(self.core))
        return out


@dataclass
class Role:
    id: str
    company: str
    title: str
    location: str
    dates: str
    facts: list[Fact]
    flexible: bool = False
    tenure_months: int | None = None


@dataclass
class Project:
    id: str
    name: str
    tagline: str
    url: str
    date: str
    facts: list[Fact]


@dataclass
class SkillCategory:
    name: str
    items: list[str]
    themes: list[str] = field(default_factory=list)
    always: bool = False


@dataclass
class FactBank:
    profile: dict
    education: list[dict]
    roles: list[Role]
    projects: list[Project]
    skill_categories: list[SkillCategory]

    def role(self, role_id: str) -> Role | None:
        return next((r for r in self.roles if r.id == role_id), None)

    def role_by_company(self, company: str) -> Role | None:
        target = _norm(company)
        return next((r for r in self.roles if _norm(r.company) == target), None)

    def project_by_name(self, name: str) -> Project | None:
        target = _norm(name)
        return next((p for p in self.projects if _norm(p.name) == target), None)

    def all_facts(self) -> list[Fact]:
        out: list[Fact] = []
        for role in self.roles:
            out.extend(role.facts)
        for project in self.projects:
            out.extend(project.facts)
        return out

    def all_tools(self) -> set[str]:
        tools = {t for f in self.all_facts() for t in f.tools}
        for cat in self.skill_categories:
            tools.update(cat.items)
        return tools


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numeric_tokens(text: str) -> set[str]:
    return set(_NUM_RE.findall(text or ""))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _parse_facts(raw_facts, owner_id: str, owner_label: str, errors: list[str]) -> list[Fact]:
    facts: list[Fact] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_facts or []):
        if not isinstance(raw, dict):
            errors.append(f"{owner_id}: fact #{i + 1} is not a mapping")
            continue
        fact_id = str(raw.get("id") or f"fact{i + 1}").strip()
        if fact_id in seen:
            errors.append(f"{owner_id}: duplicate fact id '{fact_id}'")
            continue
        seen.add(fact_id)
        core = _clean(str(raw.get("core") or ""))
        if not core:
            errors.append(f"{owner_id}.{fact_id}: missing 'core'")
            continue
        themes = [str(t).strip() for t in (raw.get("themes") or [])]
        for theme in themes:
            if theme not in THEMES:
                errors.append(
                    f"{owner_id}.{fact_id}: unknown theme '{theme}' "
                    f"(add it to THEMES in src/facts.py or fix the spelling)"
                )
        facts.append(
            Fact(
                id=fact_id,
                core=core,
                owner_id=owner_id,
                owner_label=owner_label,
                tools=[str(t).strip() for t in (raw.get("tools") or []) if str(t).strip()],
                themes=themes,
                metrics=[_clean(str(m)) for m in (raw.get("metrics") or []) if str(m).strip()],
                frozen=[_clean(str(m)) for m in (raw.get("frozen") or []) if str(m).strip()],
                angles=[_clean(str(a)) for a in (raw.get("angles") or []) if str(a).strip()],
                lead=bool(raw.get("lead")),
            )
        )
    return facts


def load_fact_bank(path: Path | None = None) -> FactBank:
    """Load and validate the fact bank. Raises ValueError on a malformed file."""
    path = path or FACTS_PATH
    if not path.exists():
        raise ValueError(f"Fact bank not found at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Fact bank must be a YAML mapping")

    errors: list[str] = []

    roles: list[Role] = []
    for raw in data.get("roles") or []:
        role_id = str(raw.get("id") or "").strip()
        company = _clean(str(raw.get("company") or ""))
        if not role_id or not company:
            errors.append("a role is missing 'id' or 'company'")
            continue
        roles.append(
            Role(
                id=role_id,
                company=company,
                title=_clean(str(raw.get("title") or "")),
                location=_clean(str(raw.get("location") or "")),
                dates=_clean(str(raw.get("dates") or "")),
                flexible=bool(raw.get("flexible")),
                tenure_months=raw.get("tenure_months"),
                facts=_parse_facts(raw.get("facts"), role_id, company, errors),
            )
        )

    projects: list[Project] = []
    for raw in data.get("projects") or []:
        project_id = str(raw.get("id") or "").strip()
        name = _clean(str(raw.get("name") or ""))
        if not project_id or not name:
            errors.append("a project is missing 'id' or 'name'")
            continue
        projects.append(
            Project(
                id=project_id,
                name=name,
                tagline=_clean(str(raw.get("tagline") or "")),
                url=_clean(str(raw.get("url") or "")),
                date=_clean(str(raw.get("date") or "")),
                facts=_parse_facts(raw.get("facts"), project_id, name, errors),
            )
        )

    skill_categories: list[SkillCategory] = []
    for raw in (data.get("skills") or {}).get("categories") or []:
        name = _clean(str(raw.get("name") or ""))
        items = [str(i).strip() for i in (raw.get("items") or []) if str(i).strip()]
        if not name or not items:
            errors.append(f"skill category '{name or '?'}' has no name or no items")
            continue
        skill_categories.append(
            SkillCategory(
                name=name,
                items=items,
                themes=[str(t).strip() for t in (raw.get("themes") or [])],
                always=bool(raw.get("always")),
            )
        )

    if not roles:
        errors.append("fact bank has no roles")
    if errors:
        raise ValueError("Invalid fact bank:\n  - " + "\n  - ".join(errors))

    return FactBank(
        profile=data.get("profile") or {},
        education=data.get("education") or [],
        roles=roles,
        projects=projects,
        skill_categories=skill_categories,
    )
