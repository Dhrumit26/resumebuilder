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
}

_BANNED_PHRASES = {
    "leverage", "leveraging", "leveraged", "utilize", "utilized", "spearheaded",
    "orchestrated", "championed", "robust", "seamless", "cutting-edge", "holistic",
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
    # Job-board voice. The prompt bans these; models reach for them anyway, so
    # the check has to be mechanical. An engineer says what they build, not what
    # they "specialize in".
    "specializing in", "specialising in", "skilled in", "proficient in",
    "adept at", "expertise in", "experienced in", "well-versed",
    "passionate about", "seeking to", "a strong background in",
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
    fact: Fact,
    lexicon: set[str],
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> list[Issue]:
    """Check one written bullet against the single fact that licensed it."""
    issues: list[Issue] = []
    plain = _plain(bullet)
    if not plain:
        return [Issue("empty", "bullet is empty")]

    words = _words(plain)

    # --- fabrication checks (the load-bearing ones) --------------------------
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
    fact_corpus = (fact.core + " " + " ".join(fact.angles) + " " + " ".join(fact.tools)).lower()
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
) -> list[Issue]:
    """The summary may draw on every selected fact, but still invents nothing.

    bullet_numbers: figures the bullets already headline. The summary sits two
    inches above them, so repeating one wastes the page's most valuable lines.
    proof_text: the bullets as written, used to check that positioning claims
    have something on the page backing them up.
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
