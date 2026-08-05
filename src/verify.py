"""Verification: check written bullets back against the facts that licensed them.

The old pipeline asked the model not to make things up and then ran regex gates
looking for specific lies it had told before (brand bleed, invented teams,
hyperscale claims...). Each gate was a patch for one observed symptom.

This works the other way round: a bullet is valid only if every number and every
technology in it is licensed by its own fact. Fabrication fails by construction,
so there is nothing left for symptom-specific gates to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .facts import Fact, FactBank
from .matching import _TOOL_THEMES, TECH_SYNONYMS, token_pattern

MIN_WORDS = 14
MAX_WORDS = 34
# Fabricated Clerxi bullets must match Intuit density on the page: >1 line, ≤2 lines.
FAB_MIN_WORDS = 22
FAB_MAX_WORDS = 36
FAB_MIN_CHARS = 115
FAB_MAX_CHARS = 210
SUMMARY_MIN_WORDS = 24
SUMMARY_MAX_WORDS = 40

# Words that look like technologies to a reader. Used to catch a bullet naming a
# tool its fact does not license — the "Kubernetes/Datadog appeared out of
# nowhere" failure. Built from the JD tool map plus tools models reach for.
_EXTRA_TECH = {
    "kubernetes", "datadog", "terraform", "kafka", "graphql", "grpc", "mongodb",
    "elasticsearch", "prometheus", "grafana", "spark", "airflow", "snowflake",
    "databricks", "azure", "gcp", "vue", "angular", "svelte", "bootstrap",
    "tailwind", "django", "flask", "spring", "express", "selenium", "junit",
    "pytest", "circleci", "rabbitmq", "nginx", "kibana", "splunk", "helm",
    "ansible", "openshift", "dynamodb", "firebase", "supabase", "vercel",
    "sqlalchemy", "celery", "webpack", "vite", "babel", "eslint", "storybook",
    "cypress", "playwright", "jest", "redis", "postgresql", "mysql", "docker",
    "aws", "lambda", "s3", "ec2", "ecs", "eks", "rds", "sqs", "sns",
    "yocto", "mqtt", "amqp", "jenkins", "armbian", "ubuntu", "zephyr",
    "c++", "c++11", "c++17", "c++20",
}

_BANNED_PHRASES = {
    "leverage", "leveraging", "leveraged", "leverages",
    "utilize", "utilized", "utilizes", "utilizing",
    "spearheaded", "orchestrated", "championed", "robust", "seamless", "seamlessly",
    "cutting-edge", "holistic",
    "results-driven", "impact-driven", "detail-oriented", "proven track record",
    "responsible for", "helped with", "worked on", "state-of-the-art",
    "best-in-class", "mission-critical", "enterprise-grade", "at scale",
    "end-to-end automation", "cross-functional synergy", "value-add",
}

# Vague filler that survives every other check because it says nothing at all.
_VAGUE_SUMMARY = {
    "modern web technologies", "modern technologies", "modern frameworks",
    "various technologies", "a wide range", "a variety of", "significantly",
    "a range of", "numerous", "cutting edge", "state of the art",
    "industry best practices", "best practices", "latest technologies",
    "specializing in", "specialising in", "skilled in", "proficient in",
    "adept at", "expertise in", "experienced in", "well-versed",
    "passionate about", "seeking to", "a strong background in",
    # Generic invent-summary fog that survives tool swaps.
    "cloud infrastructure", "real customer problems", "customer problems",
    "scalable, reliable", "reliable backends", "designs scalable",
    "collaborates across", "collaborate across", "ships production services",
}

# Practice claims a summary may only make when the resume actually shows them.
# The summary positions; the bullets prove. A claim with no proof below it is
# the exact move that makes a resume read invented.
_PRACTICE_CLAIMS = (
    "accessib", "responsive", "cross-browser", "real-time",
    "scalab", "distributed", "secure", "mobile", "microservice", "observab",
    # Whole technical domains. A posting's headline term is the thing a model
    # reaches for hardest, and claiming one you have never worked in is the
    # single most damaging line on a resume — an interviewer opens there.
    "machine learning", "deep learning", "neural network", "computer vision",
    "natural language processing", "reinforcement learning", "recommender",
    "cuda", "gpu", "parallel programming", "parallelization", "kernel",
    "computer architecture", "low-level", "embedded", "compiler",
    "cryptograph", "blockchain", "quantum",
)


_TRAILING_WORDS = {
    "by", "to", "with", "using", "and", "for", "of", "in", "on", "the", "a", "an",
    "that", "which", "from", "into", "as", "at", "via", "through",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "by", "from", "at", "as", "that", "this", "it", "its", "into", "through",
    "across", "over", "under", "was", "were", "is", "are", "be", "been", "built",
    "made", "used", "using", "new", "one", "two", "up", "out", "so", "than",
    "their", "them", "they", "when", "while", "who", "whom", "each", "every",
}

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# Tokens include numbers on purpose. Counting only letter-initial tokens made
# "...support tickets by 50%." look like it ended on the word "by", so every
# bullet finishing on a metric was falsely reported as cut off mid-sentence.
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.%\-]*")
_ALPHA_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")


@dataclass
class Issue:
    code: str
    message: str
    severity: str = "error"     # "error" forces a rewrite; "warn" is advisory


def _plain(text: str) -> str:
    """Strip LaTeX markup down to readable prose."""
    out = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text or "")
    out = re.sub(r"\\textit\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\resumeItem\{|\}", " ", out)
    out = out.replace("\\%", "%").replace("\\&", "&").replace("\\$", "$")
    out = out.replace("\\#", "#").replace("\\_", "_")
    out = re.sub(r"\\[a-zA-Z]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def _words(text: str) -> list[str]:
    """Every token that occupies space on the printed line, numbers included."""
    return _WORD_RE.findall(text)


def _content_words(text: str) -> set[str]:
    return {
        w.lower()
        for w in _ALPHA_RE.findall(text)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }


def _final_word(plain: str) -> str:
    """The last real token of a sentence, ignoring trailing punctuation."""
    stripped = plain.rstrip(" .,;:!?)’'\"")
    tokens = _WORD_RE.findall(stripped)
    return tokens[-1].lower() if tokens else ""


def tech_lexicon(bank: FactBank) -> set[str]:
    known = {t.lower() for t in bank.all_tools()}
    return known | set(_TOOL_THEMES.keys()) | _EXTRA_TECH


def _mentioned_tech(plain: str, lexicon: set[str]) -> set[str]:
    """Technologies from the lexicon that appear in the text as whole tokens."""
    found: set[str] = set()
    low = plain.lower()
    for tech in lexicon:
        if re.search(token_pattern(tech), low):
            found.add(tech)
    return found


def verify_bullet(
    bullet: str,
    fact: Fact | None,
    lexicon: set[str],
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
    grounded: bool = True,
) -> list[Issue]:
    """Check one written bullet.

    grounded=True (default): every number/tool must be licensed by `fact`.
    grounded=False: invent mode — only craft checks apply (fabricated roles).
    """
    issues: list[Issue] = []
    plain = _plain(bullet)
    if not plain:
        return [Issue("empty", "bullet is empty")]

    words = _words(plain)

    # --- fabrication checks (the load-bearing ones) --------------------------
    if grounded:
        if fact is None:
            return [Issue("no-fact", "grounded bullet requires a licensing fact")]
        allowed_numbers = fact.numbers()
        for number in _NUM_RE.findall(plain):
            if number not in allowed_numbers:
                issues.append(
                    Issue(
                        "invented-number",
                        f"'{number}' is not a number this fact records "
                        f"(allowed: {sorted(allowed_numbers) or 'none'}) — "
                        f"remove it or use a real one",
                    )
                )

        # A technology the fact's own sentence names is licensed even when it is not
        # in `tools` — "semantic caching" appears in the core text of the fact that
        # describes it, and the bullet is allowed to repeat what the fact says.
        allowed_tech = {t.lower() for t in fact.tools}
        allowed_tech |= _mentioned_tech(fact.core + " " + " ".join(fact.angles), lexicon)
        for tech in sorted(_mentioned_tech(plain, lexicon)):
            if tech in allowed_tech:
                continue
            if any(tech in a or a in tech for a in allowed_tech):
                continue
            issues.append(
                Issue(
                    "invented-tool",
                    f"'{tech}' was not used for this work "
                    f"(allowed: {sorted(fact.tools) or 'none'}) — drop it",
                )
            )

        for pairing in fact.frozen:
            members = [m.strip().lower() for m in re.split(r"\bto\b|->|→", pairing) if m.strip()]
            low = plain.lower()
            if any(m in low for m in members) and not all(m in low for m in members):
                issues.append(
                    Issue(
                        "broken-pairing",
                        f"'{pairing}' is a fixed pairing — name both sides or neither",
                    )
                )

        anchors = _content_words(fact.core) | {t.lower() for t in fact.tools}
        anchors |= _content_words(" ".join(fact.angles))
        if len(_content_words(plain) & anchors) < 2:
            issues.append(
                Issue(
                    "unanchored",
                    "bullet shares almost nothing with its fact — it must describe "
                    f"this work: {fact.core}",
                )
            )

        # Claiming a technical domain the fact never mentions — the model reaching
        # for the posting's headline term (a RAG bullet becoming "deep learning").
        fact_corpus = (
            fact.core + " " + " ".join(fact.angles) + " " + " ".join(fact.tools)
        ).lower()
        low_plain = plain.lower()
        for claim in _PRACTICE_CLAIMS:
            if claim in low_plain and claim not in fact_corpus:
                issues.append(
                    Issue(
                        "invented-domain",
                        f"'{claim}...' is not what this work was — the fact says: {fact.core}",
                    )
                )

    # --- craft checks --------------------------------------------------------
    if len(words) < min_words:
        issues.append(Issue("too-short", f"{len(words)} words; needs at least {min_words}"))
    elif len(words) > max_words:
        issues.append(Issue("too-long", f"{len(words)} words; keep it under {max_words}"))

    last = _final_word(plain)
    if last in _TRAILING_WORDS:
        issues.append(Issue("truncated", f"ends on '{last}' — finish the sentence"))
    if plain and not plain[0].isupper():
        issues.append(Issue("lowercase-start", "starts lowercase — likely cut off"))

    low = plain.lower()
    for phrase in _BANNED_PHRASES:
        if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low):
            issues.append(Issue("banned-phrase", f"remove '{phrase}' — recruiters read it as AI filler"))

    if re.search(r"(?<!\\)%", bullet):
        issues.append(Issue("bare-percent", "escape percent signs as \\% or the PDF breaks"))
    if "\\textbf" in bullet:
        issues.append(Issue("manual-bold", "do not bold — bolding is applied automatically"))

    # Compression grammar: "cut latency 38%" is missing its preposition.
    if re.search(r"\b(cut|reduced|dropped|raised|increased|improved|lowered)\s+[\w\s]{0,30}?\d+%", low):
        if not re.search(r"\b(by|from|to)\b[\w\s]{0,20}?\d+%", low):
            issues.append(Issue("dropped-preposition", "needs 'by'/'from...to' before the number"))

    return issues


def verify_summary(
    summary: str,
    facts: list[Fact],
    lexicon: set[str],
    bullet_numbers: set[str] | None = None,
    proof_text: str = "",
    fabricated_ok: bool = False,
) -> list[Issue]:
    """The summary may draw on every selected fact, but still invents nothing.

    bullet_numbers: figures the bullets already headline. The summary sits two
    inches above them, so repeating one wastes the page's most valuable lines.
    proof_text: the bullets as written, used to check that positioning claims
    have something on the page backing them up.
    fabricated_ok: when a role was invented for the JD, technologies named in
    proof_text are also licensed (they came from that invented block).
    """
    issues: list[Issue] = []
    plain = _plain(summary)
    if not plain:
        return [Issue("empty", "summary is empty")]

    allowed_numbers: set[str] = set()
    allowed_tech: set[str] = set()
    for fact in facts:
        allowed_numbers |= fact.numbers()
        allowed_tech |= {t.lower() for t in fact.tools}
        allowed_tech |= _mentioned_tech(fact.core + " " + " ".join(fact.angles), lexicon)
    if fabricated_ok and proof_text:
        allowed_tech |= _mentioned_tech(proof_text, lexicon)
        # Numbers that already appear in fabricated bullets are real for this page;
        # the summary still should not invent fresh ones.
        allowed_numbers |= set(_NUM_RE.findall(proof_text))

    used_numbers = _NUM_RE.findall(plain)
    for number in used_numbers:
        if number not in allowed_numbers:
            issues.append(Issue("invented-number", f"'{number}' appears in no fact — drop it"))
        elif bullet_numbers and number in bullet_numbers:
            issues.append(
                Issue(
                    "repeats-bullet",
                    f"'{number}' is already headlined by a bullet below — the summary "
                    f"must say something the bullets do not",
                )
            )
    if len(used_numbers) > 1:
        issues.append(
            Issue("too-many-numbers", f"{len(used_numbers)} numbers; a summary carries at most one")
        )

    for tech in sorted(_mentioned_tech(plain, lexicon)):
        if tech in allowed_tech or any(tech in a or a in tech for a in allowed_tech):
            continue
        issues.append(Issue("invented-tool", f"'{tech}' is not in the candidate's evidence — drop it"))

    # Fabricated pages: summary must name at least one tech the bullets already show.
    if fabricated_ok and proof_text:
        proof_tech = _mentioned_tech(proof_text, lexicon)
        summary_tech = _mentioned_tech(plain, lexicon)
        if proof_tech and not (summary_tech & proof_tech):
            issues.append(
                Issue(
                    "off-story",
                    "summary names none of the technologies in the bullets below — "
                    "mirror the invent stack on this page (same domain, same tools)",
                )
            )

    words = _words(plain)
    if len(words) < SUMMARY_MIN_WORDS:
        issues.append(Issue("too-short", f"{len(words)} words; needs {SUMMARY_MIN_WORDS}-{SUMMARY_MAX_WORDS}"))
    elif len(words) > SUMMARY_MAX_WORDS:
        issues.append(Issue("too-long", f"{len(words)} words; keep to {SUMMARY_MAX_WORDS}"))

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", plain) if s.strip()]
    if len(sentences) > 3:
        issues.append(Issue("too-many-sentences", f"{len(sentences)} sentences; use 2"))

    low = plain.lower()
    for phrase in _BANNED_PHRASES | {"owns features end to end", "full sdlc", "passionate", "aspiring"}:
        if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low):
            issues.append(Issue("banned-phrase", f"remove '{phrase}'"))

    for phrase in _VAGUE_SUMMARY:
        if phrase in low:
            issues.append(
                Issue("vague", f"'{phrase}' says nothing — name the actual work instead")
            )

    # A positioning claim the bullets do not demonstrate is an unsupported claim.
    proof = " ".join([f.core for f in facts] + [" ".join(f.angles) for f in facts]).lower()
    if proof_text:
        proof += " " + proof_text.lower()
    for claim in _PRACTICE_CLAIMS:
        if claim in low and claim not in proof:
            issues.append(
                Issue(
                    "unproven-claim",
                    f"the summary claims '{claim}...' but no bullet on this resume "
                    f"shows it — drop the claim or make a claim the bullets prove",
                )
            )

    if re.search(r"\bI\b|\bmy\b", plain):
        issues.append(Issue("first-person", "no first person in a resume summary"))
    if re.search(r"(?<!\\)%", summary):
        issues.append(Issue("bare-percent", "escape percent signs as \\%"))

    return issues


# --- fabricated-role block checks --------------------------------------------

_FAB_LANGUAGES = (
    "Python", "Java", "C#", "C++", "Go", "Rust", "TypeScript", "JavaScript",
    "Ruby", "PHP", "Kotlin", "Swift", "Scala",
)
_FAB_CLOUDS = ("AWS", "GCP", "Azure")

# Practices that every backend JD mentions — not enough to prove domain fit.
_GENERIC_SPINE = {
    "api development", "apis", "api", "backend services", "backend development",
    "backend", "full-stack", "full stack", "software testing", "testing",
    "debugging", "system design", "collaboration", "code review", "sdlc",
    "product mindset", "distributed systems",
}

# Distinctive domain tokens → searchable needles (substring in plain lower text).
_SPINE_NEEDLES: list[tuple[str, tuple[str, ...]]] = [
    ("RAG / retrieval", ("rag", "retrieval-augmented", "retrieval augmented", "retrieval")),
    ("agent workflows", ("agentic", "agent-based", "multi-agent", "agent workflow",
                         "agent behavior", "agent loop", "tool-calling", "tool calling",
                         "tool execution")),
    ("embeddings", ("embedding",)),
    ("model inference", ("inference",)),
    ("LLM", ("llm", "large language")),
    ("NLP", ("nlp", "natural language")),
    ("orchestration", ("orchestration",)),
    ("machine learning", ("machine learning", " ml ")),
]


def spine_labels_from_analysis(analysis: dict) -> list[str]:
    """Distinctive practice/concept labels the fabricated block must cover."""
    seen: set[str] = set()
    labels: list[str] = []
    for key in ("domain_practices", "concepts"):
        for raw in analysis.get(key) or []:
            text = str(raw or "").strip()
            if not text:
                continue
            low = text.lower()
            key_id = re.sub(r"\s+", " ", low)
            if key_id in seen:
                continue
            # Keep if it matches a distinctive needle family, or is not generic.
            distinctive = any(
                any(n.strip() in low for n in needles)
                for _, needles in _SPINE_NEEDLES
            )
            generic = any(g == low or g in low for g in _GENERIC_SPINE) and not distinctive
            if generic:
                continue
            seen.add(key_id)
            labels.append(text)
    return labels


def _spine_hits(plain_low: str, analysis: dict) -> list[str]:
    """Which distinctive spine families appear in the fabricated block text."""
    # Prefer families that the JD actually asked for.
    wanted_blob = " ".join(
        str(x).lower()
        for key in ("domain_practices", "concepts", "tools", "must_have_skills",
                    "exact_keywords_for_ats", "domain")
        for x in ([analysis.get(key)] if isinstance(analysis.get(key), str)
                  else (analysis.get(key) or []))
    )
    hits: list[str] = []
    for label, needles in _SPINE_NEEDLES:
        # Only require families the JD mentioned (or always check if JD is silent).
        jd_cares = any(n.strip() in wanted_blob for n in needles) or not wanted_blob.strip()
        if not jd_cares:
            continue
        if any(n in plain_low for n in needles):
            hits.append(label)
    return hits


def _named_in_text(plain: str, names: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for name in names:
        if name == "Go":
            pat = re.compile(r"(?<![A-Za-z])Go(?![A-Za-z])")
        elif name == "C++":
            pat = re.compile(r"(?<![A-Za-z])C\+\+(?![A-Za-z])")
        elif name == "C#":
            pat = re.compile(r"(?<![A-Za-z])C#(?![A-Za-z])")
        else:
            pat = re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_+#])"
            )
        if pat.search(plain) and name not in found:
            found.append(name)
    return found


def verify_fabricated_block(
    bullets: list[str],
    analysis: dict,
    sibling_bullets: list[str] | None = None,
    *,
    secondary_lane: bool = False,
) -> list[Issue]:
    """Block-level checks for invent-mode roles: spine, stack, story, no fluff theater.

    secondary_lane: internship / enablement role that must complement a sibling
    primary product block — spine already covered by the sibling is waived, and
    enablement work (tests/CI/contracts) is required instead.
    """
    issues: list[Issue] = []
    plains = [_plain(b) for b in bullets]
    plain = " ".join(plains)
    low = plain.lower()

    hits = _spine_hits(low, analysis)
    jd_spine = spine_labels_from_analysis(analysis)
    wanted_blob = " ".join(s.lower() for s in jd_spine) + " " + str(
        analysis.get("domain") or ""
    ).lower()
    for key in ("tools", "must_have_skills", "exact_keywords_for_ats", "concepts",
                "domain_practices"):
        for x in analysis.get(key) or []:
            wanted_blob += " " + str(x).lower()

    wanted_families = [
        label for label, needles in _SPINE_NEEDLES
        if any(n.strip() in wanted_blob for n in needles)
    ]
    _must_if_jd = (
        "RAG / retrieval",
        "agent workflows",
        "embeddings",
        "LLM",
        "model inference",
        "orchestration",
    )
    sibling_hits: set[str] = set()
    if secondary_lane and sibling_bullets:
        sibling_hits = _spine_hits(_plain(" ".join(sibling_bullets)).lower(), analysis)

    for label in _must_if_jd:
        if label not in wanted_families:
            continue
        if label in sibling_hits:
            continue  # primary role already owns this spine piece
        if label not in hits:
            issues.append(
                Issue(
                    "missing-spine",
                    f"JD requires '{label}' — none of the bullets show it as real work. "
                    f"Plant it on the system (see the GOLD example shape).",
                )
            )

    if secondary_lane:
        # Internship owns enablement — do not demand full product-spine density.
        enablement = re.search(
            r"\b(?:pytest|jest|coverage|ci\b|jenkins|github actions|gitlab|"
            r"contract|conformance|regression|flak(?:y|e)|test suite|e2e|"
            r"playwright|cypress|hardware-in-the-loop|hil)\b",
            low,
        )
        if not enablement:
            issues.append(
                Issue(
                    "missing-enablement",
                    "internship lane must show enablement work (tests, CI, coverage, "
                    "contracts, conformance) — not a twin of the current product story",
                )
            )
    else:
        need = min(3, len(wanted_families)) if wanted_families else 0
        # Credit spine the sibling will not cover; secondary lane credits sibling hits.
        effective_hits = hits
        if need and len(effective_hits) < need:
            missing = [f for f in wanted_families if f not in effective_hits]
            issues.append(
                Issue(
                    "missing-spine",
                    f"block only covers {effective_hits or 'none'} of the JD spine — need ≥{need}: "
                    f"{', '.join(wanted_families)}. Missing: {', '.join(missing)}",
                )
            )

    # Primary language from the JD must appear; no scatter.
    tools = [str(t) for t in (analysis.get("tools") or [])]
    lang_order = (
        "Python", "TypeScript", "JavaScript", "Java", "Go", "C++", "C#", "Rust",
        "Kotlin", "Swift",
    )
    primary = next((t for t in tools if t in lang_order), "Python")
    langs = _named_in_text(plain, _FAB_LANGUAGES)
    if primary not in langs:
        issues.append(
            Issue(
                "missing-primary-language",
                f"primary language '{primary}' never appears — name it in bullet 1 or 2",
            )
        )
    # One primary language only. A second language is the language-scatter tell
    # (Python RAG bullets next to C++/Yocto bullets in the same job).
    if len(langs) > 1:
        issues.append(
            Issue(
                "language-scatter",
                f"named {', '.join(langs)} across one job — pick only '{primary}' "
                f"and rebuild every bullet around that stack",
            )
        )

    clouds = _named_in_text(plain, _FAB_CLOUDS)
    if len(clouds) > 1:
        issues.append(
            Issue(
                "cloud-scatter",
                f"named {', '.join(clouds)} — pick ONE cloud and drop the others",
            )
        )

    # Product setting in bullet 1 only — later bullets stand alone (no "that X" spam).
    setting_re = re.compile(
        r"\b(?:platform|service|console|dashboard|portal|application|pipeline|"
        r"library|framework|suite|tool|website|site|module|feed|API|apis)\b",
        re.I,
    )
    if plains and not setting_re.search(plains[0]):
        issues.append(
            Issue(
                "story-thin",
                "bullet 1 must name the product setting "
                "(e.g. 'internal dashboard', 'agent console', 'component library')",
            )
        )

    backref_re = re.compile(
        r"\b(?:that|those|the same)\s+"
        r"(?:platform|service|console|dashboard|portal|application|app|API|apis|"
        r"pipeline|library|tool|website|site|module|feed|handlers?|endpoints?|"
        r"agents?|retrieval|RAG|system|workflows?|component library)\b",
        re.I,
    )
    backref_hits = sum(1 for t in plains if backref_re.search(t))
    if backref_hits > 1:
        issues.append(
            Issue(
                "backref-spam",
                f"{backref_hits} bullets use mechanical 'that/those/the same …' backrefs — "
                f"keep at most one; each bullet should stand alone "
                f"(e.g. 'Built a TypeScript/React component library for an internal dashboard…')",
            )
        )

    vague_cloud = re.compile(
        r"(?i)\b(?:cloud technology|cloud technologies|cloud integration|"
        r"cloud services(?!\s+with\b)|integrate(?:d)?\s+cloud)\b"
    )
    for i, text in enumerate(plains):
        if vague_cloud.search(text):
            issues.append(
                Issue(
                    "vague-cloud",
                    f"bullet {i + 1} says vague 'cloud technology/integration' — name the "
                    f"real approach (CodePipeline, S3, Docker, ECS, Lambda, …) or drop it",
                )
            )

    # Coverage/tests → defect claims: ban "which cut/reduced" causation theater.
    false_cause = re.compile(
        r"(?i)\b(?:coverage|tests?|jest|pytest|testing library)\b.{0,80}\bwhich\s+"
        r"(?:cut|reduced|decreased|dropped|eliminated)\b"
    )
    for i, text in enumerate(plains):
        if false_cause.search(text):
            issues.append(
                Issue(
                    "false-causation",
                    f"bullet {i + 1} overclaims causation (coverage 'which cut' regressions) — "
                    f"use 'contributing to' / 'alongside' instead of 'which cut'",
                )
            )

    # Fluff / meeting theater / empty JD cosplay.
    fluff_open = re.compile(
        r"(?i)^(facilitated|collaborated|conducted|participated|helped|ensured|"
        r"boosted|monitored|enhanced(?:\s+code\s+quality)?)\b"
    )
    fluff_anywhere = re.compile(
        r"(?i)\b(?:collaborated with|facilitated|participated in|"
        r"ensuring seamless|ensuring consistent|ensuring scalable|ensuring data|"
        r"seamless user|refine cloud technology|"
        r"improving the dashboard.?s scalability|"
        r"enhancing user experience|significantly|"
        r"reducing bugs(?!\s+by\b)(?!\s+from\b))\b"
    )
    fluff_phrase = re.compile(
        r"(?i)\b(?:ai-powered product features|user experience|user satisfaction|"
        r"real-world impact|"
        r"scalable cloud solutions|growing user demands|ai-first product teams|"
        r"context-aware ai responses|technical discussions?|"
        r"significantly improving|ensuring efficient operation|"
        r"consistent and reliable updates|"
        r"consistent performance across|seamless user interactions|"
        r"scalable and resilient infrastructure)\b"
    )
    for i, text in enumerate(plains):
        if fluff_open.search(text.strip()) or fluff_anywhere.search(text):
            issues.append(
                Issue(
                    "fluff-bullet",
                    f"bullet {i + 1} is meeting/process/empty theater — rewrite as "
                    f"Built/Shipped/Cut/Raised work with a tool and a measured change",
                )
            )
        if fluff_phrase.search(text):
            issues.append(
                Issue(
                    "fluff-bullet",
                    f"bullet {i + 1} uses empty recruiter phrases — replace with concrete "
                    f"engineering (tool + change + system back-reference)",
                )
            )

    # Metrics + length: every invent bullet must carry WHAT/HOW/WHY density.
    for i, text in enumerate(plains):
        words = len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text))
        chars = len(text)
        if words < FAB_MIN_WORDS or chars < FAB_MIN_CHARS:
            issues.append(
                Issue(
                    "too-thin",
                    f"bullet {i + 1} is too short ({words} words / {chars} chars) — "
                    f"need ≥{FAB_MIN_WORDS} words and ≥{FAB_MIN_CHARS} chars so it fills "
                    f"more than one resume line with what/how/result",
                )
            )
        if words > FAB_MAX_WORDS or chars > FAB_MAX_CHARS:
            issues.append(
                Issue(
                    "too-long",
                    f"bullet {i + 1} is too long ({words} words / {chars} chars) — "
                    f"keep ≤{FAB_MAX_WORDS} words and ≤{FAB_MAX_CHARS} chars (under two lines)",
                )
            )
        if not re.search(r"\d", text):
            issues.append(
                Issue(
                    "metric-starved",
                    f"bullet {i + 1} has no number — every invent bullet needs a metric "
                    f"or counted result (what changed and by how much)",
                )
            )

    with_from_to = sum(
        1 for t in plains if re.search(r"\bfrom\b.+\bto\b", t.lower())
    )
    if len(plains) >= 4 and with_from_to < 2:
        issues.append(
            Issue(
                "metric-starved",
                "need at least two from→to metrics across the block "
                "(e.g. 'from 1.8s to 1.1s', 'from 40% to 78%')",
            )
        )

    # Bare percent theater: "by 25%" with no from→to baseline.
    bare_pct = 0
    for text in plains:
        if re.search(r"\bby\s+\d+%", text.lower()) and not re.search(
            r"\bfrom\b.+\bto\b", text.lower()
        ):
            bare_pct += 1
    if bare_pct > 1:
        issues.append(
            Issue(
                "metric-theater",
                f"{bare_pct} bullets use bare 'by N%' with no from→to — keep at most one, "
                f"or rewrite as 'from X to Y'",
            )
        )

    # Anti-clone: another fabricated role already claimed overlapping story/metrics.
    if sibling_bullets:
        sib_plain = _plain(" ".join(sibling_bullets)).lower()
        sib_words = _content_words(sib_plain)
        mine_words = _content_words(low)
        if sib_words and mine_words:
            overlap = sib_words & mine_words
            # High content-word overlap → paraphrased twin of the other role.
            ratio = len(overlap) / max(1, len(mine_words))
            if ratio >= 0.38 and len(overlap) >= 10:
                issues.append(
                    Issue(
                        "role-clone",
                        f"this block overlaps ~{int(ratio * 100)}% of content words with "
                        f"another fabricated role — invent a different system and JD facet",
                    )
                )

        sib_metrics = set(re.findall(r"\d+(?:\.\d+)?%?", sib_plain))
        mine_metrics = set(re.findall(r"\d+(?:\.\d+)?%?", low))
        shared_metrics = {
            m for m in (sib_metrics & mine_metrics)
            if m not in {"1", "2", "3", "4", "5"}  # trivial counts
        }
        if len(shared_metrics) >= 3:
            issues.append(
                Issue(
                    "metric-clone",
                    f"reuses metrics already on another fabricated role "
                    f"({', '.join(sorted(shared_metrics)[:6])}) — invent different numbers",
                )
            )

        product_re = re.compile(
            r"\b(?:dashboard|console|component library|retrieval service|"
            r"agent console|ops dashboard|embedded os application|"
            r"security protocols?|mqtt(?:-based)?(?:\s+networking)?|"
            r"yocto(?:\s+(?:image|build|layer))?|ubuntu core|armbian|"
            r"test methods? and procedures?)\b",
            re.I,
        )
        sib_products = {m.group(0).lower() for m in product_re.finditer(sib_plain)}
        mine_products = {m.group(0).lower() for m in product_re.finditer(low)}
        shared_products = sib_products & mine_products
        if shared_products:
            issues.append(
                Issue(
                    "system-clone",
                    f"same product noun(s) as another fabricated role "
                    f"({', '.join(sorted(shared_products))}) — pick a different system",
                )
            )

        # Parallel mad-lib twin: same bullet shapes with only verbs/numbers swapped.
        twin_hits = _parallel_twin_count(plains, sibling_bullets)
        if twin_hits >= 3:
            issues.append(
                Issue(
                    "parallel-clone",
                    f"{twin_hits} bullets are near-twins of the other fabricated role "
                    f"(same story, swapped numbers/verbs) — rewrite this internship as a "
                    f"DIFFERENT facet (tests/CI/contracts/platform), not a mad-lib of the "
                    f"current role",
                )
            )

    return issues


_NUM_STRIP_RE = re.compile(r"\d+(?:\.\d+)?%?")


def _parallel_twin_count(mine_bullets: list[str], sibling_bullets: list[str]) -> int:
    """How many of my bullets are structural twins of a sibling bullet."""
    sib_plains = [_plain(b) for b in sibling_bullets]
    hits = 0
    used_sib: set[int] = set()
    for mine in mine_bullets:
        mine_p = _plain(mine)
        best_i, best_score = -1, 0.0
        for i, sib in enumerate(sib_plains):
            if i in used_sib:
                continue
            score = _structure_similarity(mine_p, sib)
            if score > best_score:
                best_i, best_score = i, score
        if best_i >= 0 and best_score >= 0.52:
            used_sib.add(best_i)
            hits += 1
    return hits


def _structure_similarity(a: str, b: str) -> float:
    """Jaccard of content words after stripping numbers — catches mad-lib twins."""
    wa = _content_words(_NUM_STRIP_RE.sub(" ", a))
    wb = _content_words(_NUM_STRIP_RE.sub(" ", b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# --- coverage: measured in code, not guessed by a model ----------------------

def keyword_coverage(resume_plain: str, jd_keywords: list[str]) -> dict:
    """Which JD keywords the finished resume actually contains.

    This is string matching. The old pipeline paid an LLM to estimate it and got
    a number that swung 26 points between identical runs.
    """
    low = resume_plain.lower()

    def present(term: str) -> bool:
        if re.search(token_pattern(term), low):
            return True
        # "APIs" vs "API", "embeddings" vs "embedding" — a plural is not a gap.
        variant = term[:-1] if term.endswith("s") else term + "s"
        return re.search(token_pattern(variant), low) is not None

    matched: list[str] = []
    missing: list[str] = []
    for keyword in jd_keywords:
        key = keyword.strip().lower()
        if not key:
            continue
        hit = present(key) or any(present(alt) for alt in TECH_SYNONYMS.get(key, []))
        (matched if hit else missing).append(keyword)
    total = len(matched) + len(missing)
    return {
        "matched": matched,
        "missing": missing,
        "coverage": round(len(matched) / total, 3) if total else 0.0,
    }
