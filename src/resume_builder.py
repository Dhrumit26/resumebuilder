from __future__ import annotations

import json
import re
from pathlib import Path

from .config import FLEXIBLE_EXPERIENCE_COMPANIES, LATEX_DIR, PROMPTS_DIR

# Jake's Resume template section markers (comment markers in the .tex files)
SECTION_MARKERS = {
    "summary": ("%-----------Summary-----------", "%-----------EDUCATION-----------"),
    "experience": ("%-----------EXPERIENCE-----------", "%-----------PROJECTS-----------"),
    "projects": ("%-----------PROJECTS-----------", "%-----------PROGRAMMING SKILLS-----------"),
    "skills": ("%-----------PROGRAMMING SKILLS-----------", "%-------------------------------------------"),
}

MAX_JD_CHARS = 4500
# Jake \small: ~2 lines ≈ 22–26 words. Allow up to ~30 before forcing repair.
TARGET_BULLET_WORDS = 30
MAX_BULLET_WORDS = 38
MAX_SUMMARY_WORDS = 38

# Words that genuinely cannot end a declarative sentence: function words plus
# transitive participles that require an object.
#
# Adjectives and nouns are deliberately NOT listed. The old set included
# "features", "own", "scalable", "reliable", "accurate", "collaborative" and
# friends, which are all valid sentence endings — "Shipped three customer-facing
# features." was classified as truncated, then silently reverted to the
# untailored original bullet AND capped the resume score at 89, forcing another
# fix round. ("features" was simultaneously whitelisted as a valid ending 300
# lines below, so the two rules contradicted each other.)
#
# Real mid-sentence truncation is now caught at the source by the
# finish_reason=="length" check in llm.py, which is where it actually happens.
INCOMPLETE_LAST_WORDS = {
    "by", "and", "to", "of", "for", "with", "from", "the", "a", "an",
    "cutting", "reducing", "using", "via", "into", "onto", "as", "or",
    "on", "in", "at", "that", "which", "while", "when", "where",
    "preserve",  # bare transitive verb — "...monitoring to preserve." is cut
}

# Concept/methodology terms that don't belong in a Technical Skills section
# (they live in bullets/summary instead). Matched as full comma-separated entries.
SKILL_CONCEPT_PATTERNS = [
    r"LLMs?",
    r"Large Language Models?",
    r"Multi[- ]?Agent( Systems?)?",
    r"Agent Orchestration",
    r"Vector Search",
    r"Semantic (Caching|Reranking|Search)",
    r"Retrieval(-Augmented Generation)?",
    r"RAG( Systems?| Pipelines?)?",
    r"Vector (Databases?|Stores?)",
    r"Evaluation Frameworks?",
    r"LLM APIs?",
    r"Evals?",
    r"Guardrails?",
    r"Distributed Systems?",
    r"Microservices?",
    r"AI/?ML( Prototyping)?",
    r"Machine Learning",
    r"Prompt Engineering",
    r"Latency Optimization",
    r"Productioni[sz]ation",
    r"Scalability",
    r"System Design",
    r"OOP",
    r"Data Structures( and Algorithms)?",
    r"Algorithms?",
    r"Agile( /? ?Scrum)?",
    r"SDLC",
    # Vague category placeholders — not real installable tools
    r"Apple Frameworks?",
    r"iOS Frameworks?",
    r"Apple SDKs?",
    r"Native Frameworks?",
    r"Cloud (Services?|Platforms?|Tools?)",
    r"Web Frameworks?",
    r"Backend Frameworks?",
    r"Frontend Frameworks?",
    r"Testing Frameworks?",
    r"CI/?CD Tools?",
    r"DevOps Tools?",
]

# Only full parser tags — never bare words like "Summary" or "\end{...}"
DELIMITER_ARTIFACT_RE = re.compile(
    r"(?:<<<|»»»|¡¡¡|\{\{\{)"
    r"(?:JD_ANALYSIS|SUMMARY|EXPERIENCE|PROJECTS|SKILLS|END)"
    r"(?:>>>|«««|¿¿¿|\}\}\})",
    re.IGNORECASE,
)


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_prompt(name: str) -> str:
    return load_file(PROMPTS_DIR / name)


def fill_prompt(template: str, **kwargs: str) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


def load_original_sections() -> dict[str, str]:
    return {
        "summary": load_file(LATEX_DIR / "summary" / "original.tex"),
        "experience": load_file(LATEX_DIR / "experience" / "original.tex"),
        "projects": load_file(LATEX_DIR / "projects" / "original.tex"),
        "skills": load_file(LATEX_DIR / "skills" / "original.tex"),
    }


def load_full_template() -> str:
    original = LATEX_DIR / "full" / "original.tex"
    if original.exists() and original.stat().st_size > 100:
        return load_file(original)
    return load_file(LATEX_DIR / "full" / "empty.tex")


def clean_llm_latex(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:latex|tex|json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return strip_delimiter_artifacts(text)


def strip_delimiter_artifacts(text: str) -> str:
    """Remove parser delimiters that sometimes leak into section LaTeX / PDF."""
    if not text:
        return text
    lines = []
    for line in text.splitlines():
        cleaned = DELIMITER_ARTIFACT_RE.sub("", line)
        # Drop lines that became empty/whitespace-only after stripping tags
        if cleaned.strip() == "" and DELIMITER_ARTIFACT_RE.search(line):
            continue
        lines.append(cleaned.rstrip())
    text = "\n".join(lines)
    # Catch any leftover inline tags
    text = DELIMITER_ARTIFACT_RE.sub("", text)
    return text.strip()


def truncate_jd(job_description: str) -> str:
    jd = job_description.strip()
    if len(jd) <= MAX_JD_CHARS:
        return jd
    return jd[:MAX_JD_CHARS] + "\n\n[JD truncated for length]"


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'+%-]+\b", text))


def _extract_resume_items(block: str) -> list[str]:
    """Extract full \\resumeItem{...} commands, supporting one level of nested braces."""
    items = []
    i = 0
    needle = "\\resumeItem{"
    while True:
        start = block.find(needle, i)
        if start == -1:
            break
        j = start + len(needle)
        depth = 1
        while j < len(block) and depth:
            ch = block[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth == 0:
            items.append(block[start:j])
        i = start + len(needle)
    return items


def count_resume_items_per_block(section: str) -> list[int]:
    blocks = re.findall(
        r"\\resumeItemListStart(.*?)\\resumeItemListEnd", section, re.DOTALL
    )
    return [len(_extract_resume_items(block)) for block in blocks]


def latex_to_plain(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"(?<!\\)%.*$", "", line)
        if stripped.strip():
            lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\underline\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = text.replace("\\%", "%").replace("\\|", "|")
    text = text.replace("\\#", "#").replace("\\&", "&").replace("\\_", "_").replace("\\$", "$")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def replace_section_block(resume: str, start_marker: str, end_marker: str, content: str) -> str:
    start_idx = resume.find(start_marker)
    end_idx = resume.find(end_marker)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError(
            f"Could not find section markers in template: {start_marker!r} ... {end_marker!r}"
        )

    cleaned = clean_llm_latex(content).rstrip() + "\n\n"
    block = cleaned if start_marker in cleaned else start_marker + "\n" + cleaned
    return resume[:start_idx] + block + resume[end_idx:]


def assemble_full_resume(
    template: str,
    summary: str,
    experience: str,
    projects: str,
    skills: str,
) -> str:
    resume = template
    sections = {
        "summary": summary,
        "experience": experience,
        "projects": projects,
        "skills": skills,
    }
    for name, content in sections.items():
        start, end = SECTION_MARKERS[name]
        resume = replace_section_block(resume, start, end, content)
    # Final sweep so no delimiter leaks into the full document
    resume = strip_delimiter_artifacts(resume)
    # Never allow markdown bold into compiled resume
    resume = resume.replace("**", "")
    return resume


def _trim_resume_items_to_counts(section: str, target_counts: list[int]) -> str:
    """If model added EXTRA bullets, trim each block down to the original count.

    Never pads/invents bullets. Does not reduce below original targets unless
    the model already produced fewer (in which case we keep what we have).
    When trimming extras, drop from the end so original content is preferred.
    """
    if not target_counts:
        return section

    blocks = list(
        re.finditer(r"(\\resumeItemListStart)(.*?)(\\resumeItemListEnd)", section, re.DOTALL)
    )
    if not blocks:
        return section

    out = section
    for idx, match in reversed(list(enumerate(blocks))):
        if idx >= len(target_counts):
            continue
        limit = target_counts[idx]
        body = match.group(2)
        items = _extract_resume_items(body)
        if len(items) <= limit:
            continue
        # Drop extras from the end (keeps original bullets if model prepended junk)
        kept = "".join(f"        {item}\n" for item in items[:limit])
        new_body = "\n" + kept
        out = out[: match.start(2)] + new_body + out[match.end(2) :]
    return out


def _visible_bullet_text(item_cmd: str) -> str:
    """Plain text inside \\resumeItem{...} for length/incomplete checks (safe with %)."""
    if not item_cmd.startswith("\\resumeItem{"):
        return item_cmd
    inner = item_cmd[len("\\resumeItem{") : -1] if item_cmd.endswith("}") else item_cmd
    # Protect percent signs before any comment-style stripping
    inner = inner.replace("\\%", "PERCENT").replace("%", "PERCENT")
    text = inner
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = text.replace("PERCENT", "%")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _shorten_resume_item(item_cmd: str, max_words: int = MAX_BULLET_WORDS) -> str:
    """Shorten oversized bullets to a complete clause; never leave fragments."""
    if not item_cmd.startswith("\\resumeItem{"):
        return item_cmd
    plain = _visible_bullet_text(item_cmd)
    words = plain.split()
    if len(words) <= max_words and not is_incomplete_plain(plain):
        return item_cmd

    inner = item_cmd[len("\\resumeItem{") : -1] if item_cmd.endswith("}") else item_cmd
    parts = re.findall(r"\\[a-zA-Z]+\{[^{}]*\}|\\%|\\[^a-zA-Z]|[^\s\\]+|\s+", inner)
    kept = []
    visible = 0
    last_complete = None
    for part in parts:
        if part.isspace():
            if kept:
                kept.append(part)
            continue
        add_words = len(_visible_bullet_text("\\resumeItem{" + part + "}").split()) if "{" in part else len(part.replace("\\%", "%").split())
        if add_words == 0:
            add_words = 1 if part.strip() else 0
        if visible + add_words > max_words and visible > 0:
            break
        kept.append(part)
        visible += add_words
        candidate = "".join(kept).strip().rstrip(",.;:") + "."
        cand_plain = _visible_bullet_text("\\resumeItem{" + candidate.rstrip(".") + "}")
        if not is_incomplete_plain(cand_plain) and len(cand_plain.split()) >= 12:
            last_complete = candidate

    if last_complete:
        return "\\resumeItem{" + last_complete.rstrip(".") + ".}"
    # Could not safely shorten
    return item_cmd


def is_incomplete_plain(plain: str) -> bool:
    """True if bullet looks cut mid-thought."""
    text = (plain or "").strip()
    if not text:
        return True
    core = text.rstrip(".")
    words = core.split()
    if not words:
        return True
    last = words[-1].lower().rstrip(",;:")
    if last in INCOMPLETE_LAST_WORDS:
        return True
    # NOTE: a "dangling adjective" rule used to live here, flagging any bullet
    # ending in -powered/-minded/-driven/-oriented/-focused. Those are ordinary
    # sentence endings ("...making the pipeline self-healing and event-driven."),
    # so it fired constantly on good bullets and reverted them. Removed — buzzword
    # adjectives are handled by scrub_ai_fluff, and true truncation is detected by
    # finish_reason in llm.py.
    # bare number that should be a percent: "... by 70."
    if re.fullmatch(r"\d+", last):
        return True
    # ends with preposition/conjunction phrase
    if re.search(r"(?i)\b(by|and|to|of|for|with|from|using|via|into)\.?$", core):
        return True
    return False


def find_incomplete_bullets(section: str) -> list[str]:
    bad = []
    for item in _extract_resume_items(section):
        plain = _visible_bullet_text(item)
        if is_incomplete_plain(plain):
            bad.append(plain)
    return bad


def bullet_rewrite_ratio(generated: str, original: str) -> float:
    """Fraction of bullets that meaningfully differ from the original (0–1)."""
    gen_items = [_visible_bullet_text(i) for i in _extract_resume_items(generated)]
    orig_items = [_visible_bullet_text(i) for i in _extract_resume_items(original)]
    if not gen_items or not orig_items:
        return 0.0
    changed = 0
    for i, g in enumerate(gen_items):
        o = orig_items[i] if i < len(orig_items) else ""
        gw, ow = set(g.lower().split()), set(o.lower().split())
        if not ow:
            changed += 1
            continue
        overlap = len(gw & ow) / max(len(gw | ow), 1)
        # require clear rewrite: Jaccard overlap under 0.72
        if overlap < 0.72:
            changed += 1
    return changed / len(gen_items)


def _repair_bullets_with_original(generated_section: str, original_section: str) -> str:
    """Fix ONLY broken (incomplete) bullets. Keep tailored content whenever possible.

    Order of preference per broken bullet:
    1. Shorten tailored bullet to its last complete clause (keeps tailoring).
    2. If that fails, swap in the original bullet at the same index.
    Long-but-complete bullets are left alone here (length handled separately).
    """
    gen_items = _extract_resume_items(generated_section)
    orig_items = _extract_resume_items(original_section)
    if not gen_items:
        return generated_section

    out = generated_section
    for i in range(len(gen_items) - 1, -1, -1):
        gen = gen_items[i]
        plain = _visible_bullet_text(gen)
        if not is_incomplete_plain(plain):
            continue

        # Try to salvage the tailored bullet by trimming to a complete clause
        salvaged = _shorten_resume_item(gen, MAX_BULLET_WORDS)
        if salvaged != gen and not is_incomplete_plain(_visible_bullet_text(salvaged)):
            replacement = salvaged
        elif i < len(orig_items):
            replacement = orig_items[i]
        else:
            continue

        idx = out.rfind(gen)
        if idx != -1:
            out = out[:idx] + replacement + out[idx + len(gen) :]
    return out


def _split_skill_entries(body: str) -> list[str]:
    """Split a skills category on commas, but never inside parentheses.

    "AWS (Lambda, ECS, S3), Docker" -> ["AWS (Lambda, ECS, S3)", "Docker"]
    """
    entries: list[str] = []
    depth = 0
    current = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            entry = "".join(current).strip()
            if entry:
                entries.append(entry)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        entries.append(tail)
    return entries


# Matches "\textbf{Category}{: items}" with optional flexible spacing around the colon
SKILL_CATEGORY_RE = re.compile(r"(\\textbf\{[^}]*\}\{\s*:\s*)([^}]*?)(\})")


def _compact_skills(skills: str) -> str:
    """Keep skills dense but prevent any category from wrapping into a mini-essay."""

    def trim_category(m: re.Match) -> str:
        parts = _split_skill_entries(m.group(2))
        if len(parts) > 12:
            parts = parts[:12]
        return f"{m.group(1)}{', '.join(parts)}{m.group(3)}"

    return SKILL_CATEGORY_RE.sub(trim_category, skills)


def _enforce_bullet_length(section: str) -> str:
    """Only touch clearly oversized bullets; never leave incomplete fragments."""
    items = _extract_resume_items(section)
    if not items:
        return section
    out = section
    for item in reversed(items):
        plain = _visible_bullet_text(item)
        if len(plain.split()) <= MAX_BULLET_WORDS and not is_incomplete_plain(plain):
            continue
        if len(plain.split()) <= MAX_BULLET_WORDS:
            # Incomplete but not long — leave for repair-with-original pass
            continue
        shortened = _shorten_resume_item(item, TARGET_BULLET_WORDS)
        if is_incomplete_plain(_visible_bullet_text(shortened)):
            continue  # refuse unsafe trim
        idx = out.rfind(item)
        if idx != -1:
            out = out[:idx] + shortened + out[idx + len(item) :]
    return out


def _matches_concept(term: str) -> bool:
    return any(re.fullmatch(pat, term.strip(), re.IGNORECASE) for pat in SKILL_CONCEPT_PATTERNS)


# Role/job-title words: never tools, always allowed (summary SHOULD mirror the JD title)
_ROLE_TITLE_RE = re.compile(
    r"(?i)\b(engineer|engineering|developer|architect|scientist|manager|analyst|intern|lead|consultant)s?\b"
)


def unevidenced_tools(jd_keywords: list[str], evidence_lower: str) -> list[str]:
    """JD TOOL keywords (not concepts or job titles) missing from the user's original resume."""
    out = []
    for kw in jd_keywords:
        kw = (kw or "").strip()
        if not kw or len(kw) < 2 or _matches_concept(kw) or _ROLE_TITLE_RE.search(kw):
            continue
        # Word-boundary match so "Go" does not hit "algorithmic"
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(kw.lower()) + r"(?![A-Za-z0-9])", evidence_lower):
            continue
        out.append(kw)
    return out


def _skill_entries_from_section(skills: str) -> set[str]:
    """All concrete skill tokens currently listed in a skills section."""
    found: set[str] = set()
    for m in SKILL_CATEGORY_RE.finditer(skills):
        for entry in _split_skill_entries(m.group(2)):
            found.add(entry.strip().lower())
            # also index bare base before parentheses: "AWS (Lambda, ECS)" -> aws
            base = re.sub(r"\s*\(.*\)$", "", entry).strip().lower()
            if base:
                found.add(base)
    return found


def _whitelist_skills(skills: str, original_skills: str, jd_tools: list[str], max_new: int = 4) -> str:
    """Keep ORIGINAL skills + up to max_new JD tools. Drop invented tools (Go/Rust/etc.)."""
    allowed = _skill_entries_from_section(original_skills)
    to_inject: list[str] = []
    for tool in jd_tools or []:
        tool = (tool or "").strip()
        if not tool or _matches_concept(tool) or _ROLE_TITLE_RE.search(tool):
            continue
        key = tool.lower()
        if key in allowed:
            continue
        if len(to_inject) >= max_new:
            break
        allowed.add(key)
        to_inject.append(tool)

    emptied: list[str] = []

    def filter_category(m: re.Match) -> str:
        kept = []
        for entry in _split_skill_entries(m.group(2)):
            base = re.sub(r"\s*\(.*\)$", "", entry).strip()
            if entry.strip().lower() in allowed or base.lower() in allowed:
                kept.append(entry.strip())
        if not kept:
            # Nothing in this category survived the whitelist, i.e. the model
            # replaced it wholesale with tools the candidate never used. The old
            # code kept the first 3 entries UNFILTERED here, smuggling the
            # inventions through in exactly the case the whitelist exists to stop.
            emptied.append(m.group(0))
            return m.group(0)
        return m.group(1) + ", ".join(kept) + m.group(3)

    out = SKILL_CATEGORY_RE.sub(filter_category, skills)
    if emptied:
        # Trust the candidate's real skills section; JD tools still get injected below.
        out = original_skills

    # Inject missing JD tools into the category they belong to: languages go to the
    # Languages line (a C# filed under "Backend & Cloud" reads as not knowing what
    # C# is), everything else to the Backend/Cloud/DevOps category.
    still_missing = [t for t in to_inject if t.lower() not in latex_to_plain(out).lower()]
    if still_missing:
        lang_names = {l.lower() for l in _BULLET_LANGUAGES} | {"c", "f#", "sql"}
        lang_missing = [t for t in still_missing if t.lower() in lang_names]
        other_missing = [t for t in still_missing if t.lower() not in lang_names]

        def make_inject(tools: list[str]):
            def inject(m: re.Match) -> str:
                entries = _split_skill_entries(m.group(2))
                for t in tools:
                    if t not in entries:
                        entries.append(t)
                return m.group(1) + ", ".join(entries[:12]) + m.group(3)
            return inject

        if lang_missing:
            lang_pattern = re.compile(
                r"(\\textbf\{Languages[^}]*\}\{\s*:\s*)([^}]*?)(\})", re.IGNORECASE
            )
            if lang_pattern.search(out):
                out = lang_pattern.sub(make_inject(lang_missing), out, count=1)
            else:
                other_missing = lang_missing + other_missing
        if other_missing:
            pattern = re.compile(
            r"(\\textbf\{(?:Backend[^}]*|Cloud[^}]*|DevOps[^}]*|Apple[^}]*|"
            r"iOS[^}]*|macOS[^}]*|Mobile[^}]*|Frontend[^}]*)\}\{\s*:\s*)([^}]*?)(\})",
            re.IGNORECASE,
        )
            if pattern.search(out):
                out = pattern.sub(make_inject(other_missing), out, count=1)
            else:
                # fallback: first category
                out = SKILL_CATEGORY_RE.sub(make_inject(other_missing), out, count=1)
    return out


_TEXTBF_RE = re.compile(r"\\textbf\{([^{}]*)\}")


def strip_bold(text: str) -> str:
    """Unwrap every \\textbf{...}, keeping the inner text. Loops to handle nesting."""
    out = text or ""
    prev = None
    while prev != out:
        prev = out
        out = _TEXTBF_RE.sub(r"\1", out)
    return out


def _strip_concept_bolds(section: str) -> str:
    """Un-bold concept/methodology terms the model wrapped in \\textbf{...}."""

    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        if _matches_concept(inner):
            return inner
        return m.group(0)

    return re.sub(r"\\textbf\{([^{}]+)\}", repl, section)


_SUBHEADING_RE = re.compile(r"\\resumeSubheading\s*((?:\{[^{}]*\}\s*){4})", re.DOTALL)


def flexible_item_indices(section: str, companies: list[str]) -> set[int]:
    """Global \\resumeItem indices belonging to an exempt company's block.

    Indices are counted across the whole section so they line up with
    _extract_resume_items, which the revert/repair passes index into.
    """
    wanted = {c.strip().lower() for c in (companies or []) if c.strip()}
    if not wanted:
        return set()
    items = _extract_resume_items(section)
    if not items:
        return set()

    positions = []
    cursor = 0
    for item in items:
        at = section.find(item, cursor)
        if at == -1:
            at = section.find(item)
        positions.append(at)
        cursor = (at if at != -1 else cursor) + len(item)

    # (offset, is_flexible) for each job block, in document order
    blocks: list[tuple[int, bool]] = []
    for m in _SUBHEADING_RE.finditer(section):
        args = re.findall(r"\{([^{}]*)\}", m.group(1))
        company = args[2].strip().lower() if len(args) >= 3 else ""
        flexible = bool(company) and any(
            w == company or w in company or company in w for w in wanted
        )
        blocks.append((m.start(), flexible))
    if not blocks:
        return set()

    out: set[int] = set()
    for idx, pos in enumerate(positions):
        if pos == -1:
            continue
        current = False
        for start, flexible in blocks:
            if start < pos:
                current = flexible
            else:
                break
        if current:
            out.add(idx)
    return out


# Case-sensitive on purpose: language names are proper nouns in resume prose, and
# case-insensitive matching would hit words like "go" and "swift".
_BULLET_LANGUAGES = (
    "Python", "Java", "C#", "C++", "Go", "Rust", "TypeScript", "JavaScript",
    "Ruby", "PHP", "Kotlin", "Swift", "Scala",
)

# Framework/product names that get stuffed into every bullet the same way.
_BULLET_STACK_TERMS = _BULLET_LANGUAGES + (
    "SwiftUI", "UIKit", "AppKit", "React", "Next.js", "Angular", "Vue",
    "FastAPI", "Django", "Flask", "Spring", "Node.js",
    "PostgreSQL", "MongoDB", "Redis", "Kubernetes", "Docker",
    "Azure", "AWS", "GCP",
)


def languages_in_flexible_bullets(section: str, companies: list[str]) -> list[str]:
    """Distinct programming languages named across the flexible company's bullets.

    More than two is the language-scatter tell of a fake resume (a Java bullet,
    a C# bullet, a Python bullet inside one job) — the writer agent gets one
    retry to rebuild the role around a single primary language.
    """
    idx = flexible_item_indices(section, companies)
    if not idx:
        return []
    items = _extract_resume_items(section)
    text = latex_to_plain(" ".join(items[i] for i in sorted(idx) if i < len(items)))
    found = []
    for lang in _BULLET_LANGUAGES:
        # Trailing digits stay allowed so "C++17" and "Python3" still count.
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(lang) + r"(?![A-Za-z_+#])", text):
            found.append(lang)
    return found


def stack_name_overuse_in_flexible_bullets(
    section: str, companies: list[str], max_bullets: int = 2
) -> list[str]:
    """Stack terms that appear in too many bullets of the flexible role.

    "SwiftUI" in 4 of 5 bullets is keyword stuffing — the opposite of language
    scatter, and just as fake. Returns labels like "SwiftUI in 4 bullets".
    """
    idx = flexible_item_indices(section, companies)
    if not idx:
        return []
    items = _extract_resume_items(section)
    flex_items = [latex_to_plain(items[i]) for i in sorted(idx) if i < len(items)]
    if not flex_items:
        return []

    # Longer names first so SwiftUI is matched before Swift
    terms = sorted(set(_BULLET_STACK_TERMS), key=len, reverse=True)
    offenders: list[str] = []
    for term in terms:
        if term == "Go":
            pat = re.compile(r"(?<![A-Za-z0-9_])Go(?![A-Za-z0-9_+#])")
        elif term == "Swift":
            # Do not count the "Swift" inside "SwiftUI"
            pat = re.compile(r"(?<![A-Za-z0-9_])Swift(?!UI)")
        else:
            pat = re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_+#])",
                re.IGNORECASE,
            )
        hit_bullets = sum(1 for text in flex_items if pat.search(text))
        if hit_bullets > max_bullets:
            offenders.append(f"{term} in {hit_bullets} bullets")
    return offenders


# When the JD is clearly in a stack family, flexible bullets must name enough
# distinct companions — not one SwiftUI line and four tech-empty process bullets.
_STACK_FAMILIES = (
    {
        "name": "Apple/Swift",
        "triggers": ("swift", "swiftui", "uikit", "xcode"),
        "terms": (
            "SwiftUI", "UIKit", "AppKit", "Combine", "XCTest", "Xcode",
            "Instruments", "Foundation", "Swift",
        ),
        "min_distinct": 3,
    },
)


def stack_family_underuse_in_flexible_bullets(
    section: str,
    companies: list[str],
    jd_tools_blob: str,
    min_distinct: int | None = None,
) -> list[str]:
    """Return underuse labels when an Apple (etc.) JD has too few named companions.

    Example: JD is Swift/SwiftUI but Clerxi bullets only say SwiftUI once — missing
    UIKit / Combine / XCTest variety that a real Apple role would show.
    """
    blob = (jd_tools_blob or "").lower()
    if not blob:
        return []
    idx = flexible_item_indices(section, companies)
    if not idx:
        return []
    items = _extract_resume_items(section)
    flex_text = latex_to_plain(
        " ".join(items[i] for i in sorted(idx) if i < len(items))
    )
    if not flex_text:
        return []

    under: list[str] = []
    for family in _STACK_FAMILIES:
        if not any(t in blob for t in family["triggers"]):
            continue
        need = min_distinct if min_distinct is not None else family["min_distinct"]
        found: list[str] = []
        # Longer first so SwiftUI before Swift
        for term in sorted(family["terms"], key=len, reverse=True):
            if term == "Swift":
                pat = re.compile(r"(?<![A-Za-z0-9_])Swift(?!UI)")
            else:
                pat = re.compile(
                    r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_+#])",
                    re.IGNORECASE,
                )
            if pat.search(flex_text) and term not in found:
                found.append(term)
        if len(found) < need:
            under.append(
                f"{family['name']} only {len(found)} companion(s) "
                f"({', '.join(found) or 'none'}; need ≥{need})"
            )
    return under


# Architecture fog: SQL-as-destination, vague "stores", bare/truncated products.
# Do NOT match product names like "Azure SQL Database" or "SQL Server".
_ARCH_FOG_PATTERNS = [
    (re.compile(r"\binto\s+SQL\b", re.I), "into SQL"),
    (re.compile(r"\bbacked\s+by\s+SQL\b", re.I), "backed by SQL"),
    (re.compile(r"\bSQL\s+store\b", re.I), "SQL store"),
    # Bare "SQL database" — but not "Azure SQL Database"
    (re.compile(r"(?<!Azure\s)(?<!azure\s)\bSQL\s+database\b", re.I), "SQL database"),
    (re.compile(r"\banalytics\s+stores?\b", re.I), "analytics store(s)"),
    (re.compile(r"\bdata\s+stores?\b", re.I), "data store(s)"),
    (re.compile(r"\bcloud\s+infrastructure\b", re.I), "cloud infrastructure"),
    # Truncated Azure product names (must say Azure Data Factory, etc.)
    (re.compile(r"(?<!Azure\s)(?<!azure\s)\bData\s+Factory\b", re.I), "Data Factory (missing Azure)"),
    (re.compile(r"(?<!Azure\s)(?<!azure\s)\bBlob\s+Storage\b", re.I), "Blob Storage (missing Azure)"),
    # Git is source control, not CI
    (re.compile(r"\bGit[- ]based\s+CI(?:/?CD)?\b", re.I), "Git-based CI/CD"),
    (re.compile(r"\bGit\s+CI(?:/?CD)?\b", re.I), "Git CI/CD"),
]

_CLOUD_PLATFORM_RE = re.compile(
    r"\b(?:Microsoft\s+)?Azure\b|\bAWS\b|\bAmazon\s+Web\s+Services\b|\bGCP\b|\bGoogle\s+Cloud\b",
    re.I,
)
# Concrete services that make a cloud claim real. Keep short; false negatives retry.
_CLOUD_SERVICE_RE = re.compile(
    r"\b(?:Azure\s+Data\s+Factory|Azure\s+Functions?|Azure\s+Blob\s+Storage|"
    r"Azure\s+SQL\s+Database|Azure\s+DevOps|Cosmos\s+DB|Synapse|"
    r"Event\s+Hubs?|Service\s+Bus|App\s+Service|AKS|Key\s+Vault|"
    r"Lambda|S3|RDS|ECS|EKS|EC2|DynamoDB|SQS|SNS|CloudWatch|CloudFormation|"
    r"BigQuery|Cloud\s+Run|Cloud\s+Functions?|Pub/?Sub|Cloud\s+Storage|"
    r"Kubernetes|Terraform|Docker|GitHub\s+Actions|GitLab\s+CI|Jenkins)\b",
    re.I,
)


def architecture_fog_in_text(text: str) -> list[str]:
    """Return fog phrases that make storage/cloud claims look fake or wrong."""
    plain = latex_to_plain(text) if "\\" in text else text
    hits = [label for pat, label in _ARCH_FOG_PATTERNS if pat.search(plain)]
    # Bare Azure/AWS/GCP with no concrete service named nearby in the same text.
    if _CLOUD_PLATFORM_RE.search(plain) and not _CLOUD_SERVICE_RE.search(plain):
        hits.append("bare cloud platform (no service named)")
    return hits


def architecture_fog_in_flexible_bullets(section: str, companies: list[str]) -> list[str]:
    idx = flexible_item_indices(section, companies)
    if not idx:
        return []
    items = _extract_resume_items(section)
    text = " ".join(items[i] for i in sorted(idx) if i < len(items))
    return architecture_fog_in_text(text)


def _revert_bullets_with_unevidenced_tools(
    section: str,
    fallback_section: str,
    jd_keywords: list[str],
    evidence_lower: str,
    flexible_companies: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Truthfulness gate for bullets: a JD TOOL keyword that never appears in the
    user's original resume must not appear in an experience/project bullet.
    Tainted bullets revert to the fallback bullet at the same index.
    (The Skills section is exempt — new tools are allowed there by user choice.)

    Returns (section, offending terms found) so the fix loop can teach the agent.
    """
    gen_items = _extract_resume_items(section)
    fb_items = _extract_resume_items(fallback_section)
    if not gen_items:
        return section, []

    banned = unevidenced_tools(jd_keywords, evidence_lower)
    if not banned:
        return section, []

    patterns = {
        kw: re.compile(r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        for kw in banned
    }

    # Bullets under a flexible company are allowed to adopt JD tools.
    exempt = flexible_item_indices(section, flexible_companies or [])

    out = section
    offenders: list[str] = []
    for i in range(len(gen_items) - 1, -1, -1):
        if i in exempt:
            continue
        item = gen_items[i]
        hits = [kw for kw, p in patterns.items() if p.search(item)]
        if not hits:
            continue
        offenders.extend(h for h in hits if h not in offenders)
        if i >= len(fb_items):
            continue
        idx = out.rfind(item)
        if idx != -1:
            out = out[:idx] + fb_items[i] + out[idx + len(item):]
    return out, offenders


def _remove_concept_skills(skills: str) -> str:
    """Skills lines must list concrete tools only — strip concept/methodology terms.

    Vague placeholders like "Apple frameworks" are removed. If an Apple Platforms
    line loses its only non-concrete entry, seed Foundation so the line stays useful.
    """

    def clean_category(m: re.Match) -> str:
        header = m.group(1)
        entries = _split_skill_entries(m.group(2))
        kept = [
            e for e in entries
            if e and not any(re.fullmatch(pat, e, re.IGNORECASE) for pat in SKILL_CONCEPT_PATTERNS)
        ]
        if not kept:
            kept = entries
        # Apple line with only SwiftUI/Xcode and no named framework — add Foundation
        if re.search(r"Apple|iOS|macOS|Mobile", header, re.I):
            lower = {e.lower() for e in kept}
            if "foundation" not in lower and any(
                re.fullmatch(pat, e, re.IGNORECASE)
                for e in entries
                for pat in (r"Apple Frameworks?", r"iOS Frameworks?", r"Apple SDKs?")
            ):
                kept.append("Foundation")
        return m.group(1) + ", ".join(kept) + m.group(3)

    return SKILL_CATEGORY_RE.sub(clean_category, skills)


# Spans we must never bold inside: existing bold/href/any LaTeX command token
_PROTECTED_SPAN_RE = re.compile(r"\\textbf\{[^{}]*\}|\\href\{[^{}]*\}\{[^{}]*\}|\\[a-zA-Z]+\*?")

MAX_BOLD_PER_BULLET = 2      # total \textbf per bullet (existing + new) — less bold = less "AI resume"
MAX_NEW_BOLD_PER_BULLET = 1  # new bolds this pass may add to one bullet
MAX_BOLD_PER_KEYWORD = 2     # times one keyword may be bolded across a section


def _boldable_keywords(keywords: list[str]) -> list[str]:
    """Drop role titles, concepts, and noise — bold concrete tools only."""
    out = []
    seen = set()
    for kw in keywords:
        kw = (kw or "").strip()
        if not kw or len(kw) < 2:
            continue
        if _ROLE_TITLE_RE.search(kw) or _matches_concept(kw):
            continue
        # Skip soft / abstract phrases that look silly when bolded
        if " " in kw and not any(ch in kw for ch in "/.+#"):
            # allow multi-word tools like "GitHub Actions", "React Testing Library"
            if not re.search(
                r"(?i)\b(actions?|library|testing|studio|cloud|code|sql|db|api|js|ts)\b",
                kw,
            ) and len(kw.split()) > 2:
                continue
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(kw)
    return sorted(out, key=len, reverse=True)


def _bold_first_occurrence(text: str, keyword: str) -> tuple[str, bool]:
    """Bold the first free occurrence of keyword in text (case preserved)."""
    spans = [(m.start(), m.end()) for m in _PROTECTED_SPAN_RE.finditer(text)]
    pattern = re.compile(
        r"(?<![A-Za-z0-9\\{])" + re.escape(keyword) + r"(?![A-Za-z0-9}])", re.IGNORECASE
    )
    for m in pattern.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        return text[: m.start()] + "\\textbf{" + m.group(0) + "}" + text[m.end():], True
    return text, False


def bold_keywords_in_bullets(section: str, keywords: list[str]) -> str:
    """Guarantee JD keywords appearing in \\resumeItem bullets are bolded.

    Caps keep the page readable: max 2 bolds per bullet, max 1 new per bullet,
    each keyword bolded at most twice per section. Longest keywords first so
    "React Testing Library" wins over "React".
    """
    ordered = _boldable_keywords(keywords)
    if not ordered:
        return section

    out = section
    keyword_uses: dict[str, int] = {}
    for item in _extract_resume_items(section):
        existing = item.count("\\textbf{")
        budget = min(MAX_BOLD_PER_BULLET - existing, MAX_NEW_BOLD_PER_BULLET)
        if budget <= 0:
            continue
        new_item = item
        for kw in ordered:
            if budget <= 0:
                break
            if keyword_uses.get(kw.lower(), 0) >= MAX_BOLD_PER_KEYWORD:
                continue
            # skip if this keyword is already bolded anywhere in the bullet
            if re.search(r"\\textbf\{[^{}]*" + re.escape(kw) + r"[^{}]*\}", new_item, re.IGNORECASE):
                continue
            new_item, done = _bold_first_occurrence(new_item, kw)
            if done:
                budget -= 1
                keyword_uses[kw.lower()] = keyword_uses.get(kw.lower(), 0) + 1
        if new_item != item:
            idx = out.find(item)
            if idx != -1:
                out = out[:idx] + new_item + out[idx + len(item):]
    return out


def bold_keywords_in_summary(summary: str, keywords: list[str], max_new: int = 1) -> str:
    """Bold up to max_new JD keywords inside the summary's \\textit{...} text."""
    ordered = _boldable_keywords(keywords)
    if not ordered:
        return summary
    m = re.search(r"\\textit\{([^{}]*(?:\\textbf\{[^{}]*\}[^{}]*)*)\}", summary)
    if not m:
        return summary
    inner = m.group(1)
    budget = max(0, max_new - inner.count("\\textbf{"))
    for kw in ordered:
        if budget <= 0:
            break
        if re.search(r"\\textbf\{[^{}]*" + re.escape(kw) + r"[^{}]*\}", inner, re.IGNORECASE):
            continue
        inner, done = _bold_first_occurrence(inner, kw)
        if done:
            budget -= 1
    return summary[: m.start(1)] + inner + summary[m.end(1):]


# Soft-skill / AI-resume slogans that make human reviewers distrust the resume.
# Removed as whole phrases when they appear as filler (not as part of a concrete story).
AI_FLUFF_PHRASES = [
    (r"(?i)\bcustomer[-\s]?minded\b[, ]*", ""),
    (r"(?i)\bimpact[-\s]?driven\b[, ]*", ""),
    (r"(?i)\bhigh[-\s]?agency\b[, ]*", ""),
    (r"(?i)\bresults[-\s]?driven\b[, ]*", ""),
    (r"(?i)\bdetail[-\s]?oriented\b[, ]*", ""),
    (r"(?i)\bholistic(?:ally)?\b[, ]*", ""),
    (r"(?i)\bnavigate ambiguity\b[, ]*", ""),
    (r"(?i)\bproven track record(?: of [^,.]+)?\b[, ]*", ""),
    (r"(?i)\bcutting[-\s]?edge\b[, ]*", ""),
    (r"(?i)\bleverag(?:e|es|ed|ing)\b", "using"),
    (r"(?i)\butiliz(?:e|es|ed|ing)\b", "using"),
    (r"(?i)\bspearhead(?:ed|s|ing)?\b", "led"),
    (r"(?i)\borchestrat(?:e|es|ed|ing)\b", "ran"),
    (r"(?i)\brobust\b[, ]*", ""),
    (r"(?i)\bseamless(?:ly)?\b[, ]*", ""),
]


def scrub_ai_fluff(text: str) -> str:
    """Replace common AI-resume tell phrases with plainer language."""
    if not text:
        return text
    out = text
    for pat, repl in AI_FLUFF_PHRASES:
        out = re.sub(pat, repl, out)
    # tidy leftover double spaces / awkward ", ,"
    out = re.sub(r" ,", ",", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out


WEAK_SUMMARY_PHRASES = [
    (r"(?i)\bearly[-\s]?career\s+", ""),
    (r"(?i)\bentry[-\s]?level\s+", ""),
    (r"(?i)\baspiring\s+", ""),
    (r"(?i)\brecent\s+graduate\s+", ""),
    (r"(?i)\bjunior\s+(?=ai\b|software\b|backend\b|frontend\b|full[-\s]?stack\b|engineer\b|developer\b)", ""),
]


def _scrub_weak_summary_language(summary: str) -> str:
    """Remove seniority-diminishing phrases the model sometimes inserts."""
    def repl(m: re.Match) -> str:
        text = m.group(1)
        for pattern, replacement in WEAK_SUMMARY_PHRASES:
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        # Capitalize first letter after scrubbing
        if text:
            text = text[0].upper() + text[1:]
        return "\\textit{" + text + "}"

    return re.sub(r"\\textit\{([^{}]*)\}", repl, summary, count=1)


def looks_like_markdown(text: str) -> bool:
    """True only when output is REAL markdown (no Jake macros to salvage)."""
    if not text:
        return True
    # If Jake macros are present, it's LaTeX with maybe stray tokens — salvageable
    if "\\resumeItem{" in text or "\\resumeSubheading" in text or "\\textit{" in text or "\\textbf{" in text:
        return False
    if "**" in text:
        return True
    if re.search(r"(?m)^\s*[-*]\s+\S", text):
        return True
    if re.search(r"(?m)^\s*#{1,3}\s+\S", text):
        return True
    return False


def is_valid_jake_summary(text: str) -> bool:
    return "\\textit{" in text or "\\begin{center}" in text


def is_valid_jake_experience(text: str) -> bool:
    return (
        "\\resumeSubheading" in text
        and "\\resumeItem{" in text
        and "\\resumeItemListStart" in text
        and bool(count_resume_items_per_block(text))
    )


def is_valid_jake_projects(text: str) -> bool:
    return (
        "\\resumeProjectHeading" in text
        and "\\resumeItem{" in text
        and "\\resumeItemListStart" in text
    )


def is_valid_jake_skills(text: str) -> bool:
    return "\\textbf{" in text and ("Technical Skills" in text or "Languages" in text or "\\item{" in text)


def strip_markdown_artifacts(text: str) -> str:
    """Remove stray markdown tokens mixed into otherwise-LaTeX output."""
    text = text.replace("**", "")
    # Only strip markdown list dashes at line start when NOT a LaTeX line
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*[-*]\s+\S", line) and "\\" not in line:
            line = re.sub(r"^\s*[-*]\s+", "", line)
        lines.append(line)
    return "\n".join(lines)


def clean_generated_sections(
    summary: str,
    experience: str,
    projects: str,
    skills: str,
    original_sections: dict[str, str],
    fallback_sections: dict[str, str] | None = None,
    jd_keywords: list[str] | None = None,
) -> dict[str, str]:
    """Sanitize first, keep tailored content; fall back ONLY if truly unusable.

    fallback_sections: what to substitute when a section is invalid. During remakes
    this is the previous BEST tailored version — falling back to the untailored
    original would silently throw away all tailoring.
    jd_keywords: JD tool keywords used for the bullet truthfulness gate.
    """
    fallback = fallback_sections or original_sections
    fallback_used: dict[str, bool] = {k: False for k in ("summary", "experience", "projects", "skills")}

    # 1) Sanitize markdown artifacts BEFORE validating, so tailored LaTeX is kept
    summary = strip_markdown_artifacts(clean_llm_latex(summary))
    experience = strip_markdown_artifacts(clean_llm_latex(experience))
    projects = strip_markdown_artifacts(clean_llm_latex(projects))
    skills = strip_markdown_artifacts(clean_llm_latex(skills))

    # 2) Fall back only when the section has no usable Jake macros at all
    if not is_valid_jake_summary(summary):
        summary = fallback["summary"]
        fallback_used["summary"] = True
    if not is_valid_jake_experience(experience):
        experience = fallback["experience"]
        fallback_used["experience"] = True
    if not is_valid_jake_projects(projects):
        projects = fallback["projects"]
        fallback_used["projects"] = True
    if not is_valid_jake_skills(skills):
        skills = fallback["skills"]
        fallback_used["skills"] = True

    # Bullet COUNTS always come from the true original resume
    exp_targets = count_resume_items_per_block(original_sections["experience"])
    proj_targets = count_resume_items_per_block(original_sections["projects"])

    experience = _trim_resume_items_to_counts(experience, exp_targets)
    projects = _trim_resume_items_to_counts(projects, proj_targets)

    # Truthfulness gate: bullets/summary may not claim JD tools absent from the original resume
    gate_reverts: dict[str, list[str]] = {"summary": [], "experience": [], "projects": []}
    if jd_keywords:
        evidence_lower = latex_to_plain(
            "\n".join(original_sections[k] for k in ("summary", "experience", "projects", "skills"))
        ).lower()
        experience, gate_reverts["experience"] = _revert_bullets_with_unevidenced_tools(
            experience,
            fallback["experience"],
            jd_keywords,
            evidence_lower,
            flexible_companies=FLEXIBLE_EXPERIENCE_COMPANIES,
        )
        projects, gate_reverts["projects"] = _revert_bullets_with_unevidenced_tools(
            projects, fallback["projects"], jd_keywords, evidence_lower
        )
        banned = unevidenced_tools(jd_keywords, evidence_lower)
        # A JD tool the flexible current role legitimately adopted (it survived the
        # bullet gate above) is claimable in the summary too — the summary describes
        # the current role, and reverting it would contradict the experience section.
        flex_idx = flexible_item_indices(experience, FLEXIBLE_EXPERIENCE_COMPANIES)
        if flex_idx:
            items = _extract_resume_items(experience)
            flex_text = latex_to_plain(
                " ".join(items[i] for i in flex_idx if i < len(items))
            )
            banned = [
                kw for kw in banned
                if not re.search(
                    r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])",
                    flex_text,
                    re.IGNORECASE,
                )
            ]
        summary_plain = latex_to_plain(summary)
        hit = [
            kw for kw in banned
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])", summary_plain, re.IGNORECASE)
        ]
        if hit:
            summary = fallback["summary"]
            fallback_used["summary"] = True
            gate_reverts["summary"] = hit

    # ORDER MATTERS: every text-mutating pass runs BEFORE the final repair pass.
    # scrub_ai_fluff deletes whole words, so it can turn a complete bullet into a
    # fragment ("...built for robust." -> "...built for."). It used to run *after*
    # the last repair, so those self-inflicted fragments were never fixed — they
    # just surfaced as incomplete_bullets, capped the score at 89 and bought
    # another fix round.
    experience = scrub_ai_fluff(experience)
    projects = scrub_ai_fluff(projects)
    experience = _strip_concept_bolds(experience)
    projects = _strip_concept_bolds(projects)
    experience = _enforce_bullet_length(experience)
    projects = _enforce_bullet_length(projects)
    experience = _repair_bullets_with_original(experience, fallback["experience"])
    projects = _repair_bullets_with_original(projects, fallback["projects"])

    def trim_summary(text: str) -> str:
        def repl(m: re.Match) -> str:
            words = m.group(1).split()
            if len(words) <= MAX_SUMMARY_WORDS:
                return m.group(0)
            return "\\textit{" + " ".join(words[:MAX_SUMMARY_WORDS]).rstrip(",.;:") + ".}"

        return re.sub(r"\\textit\{([^{}]*)\}", repl, text, count=1)

    if is_valid_jake_summary(summary):
        # No bold in the summary, by choice. Doing this FIRST also un-nests the
        # braces, which is the only reason trim_summary and the weak-language
        # scrub can match at all — both are anchored on \textit{[^{}]*}, so a
        # \textbf inside \textit used to make them silently no-op.
        summary = strip_bold(summary)
        summary = trim_summary(summary)
        summary = _scrub_weak_summary_language(summary)
        summary = scrub_ai_fluff(summary)
        if "\\textit{" in summary:
            m = re.search(r"\\textit\{([^{}]*)\}", summary)
            if m and is_incomplete_plain(m.group(1)):
                summary = fallback["summary"]
                fallback_used["summary"] = True
    else:
        summary = fallback["summary"]
        fallback_used["summary"] = True

    summary = _strip_concept_bolds(summary)

    if is_valid_jake_skills(skills):
        skills = _remove_concept_skills(skills)
        skills = _whitelist_skills(
            skills,
            original_sections["skills"],
            jd_keywords or [],
            max_new=4,
        )
        skills = _compact_skills(skills)
    else:
        skills = fallback["skills"]
        fallback_used["skills"] = True

    # Final sanity: if experience still isn't Jake LaTeX, force fallback
    if not is_valid_jake_experience(experience):
        experience = fallback["experience"]
        fallback_used["experience"] = True
    if not is_valid_jake_projects(projects):
        projects = fallback["projects"]
        fallback_used["projects"] = True

    return {
        "summary": summary,
        "experience": experience,
        "projects": projects,
        "skills": skills,
        "expected_experience_bullets": exp_targets,
        "expected_project_bullets": proj_targets,
        "actual_experience_bullets": count_resume_items_per_block(experience),
        "actual_project_bullets": count_resume_items_per_block(projects),
        "fallback_used": fallback_used,
        "gate_reverts": gate_reverts,
    }


DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"


def _debug_dump(name: str, content: str) -> None:
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        (DEBUG_DIR / f"{name}.txt").write_text(content or "", encoding="utf-8")
    except Exception:
        pass
