"""Fetch short, factual web blurbs for specialized JD tools/products.

LLMs often invent the wrong architecture for products they barely know
(e.g. treating Microsoft Fabric as generic Azure SQL). Writers get a compact
TECH_CONTEXT block built from Microsoft Learn / Wikipedia so bullets can name
real components (OneLake, Lakehouse, Data Factory pipelines inside Fabric, …).
"""

from __future__ import annotations

import html as htmlmod
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from .resume_builder import _debug_dump

USER_AGENT = "ResumeBuilder/1.0 (https://github.com/Dhrumit26/resumebuilder; educational)"
REQUEST_TIMEOUT = 8
MAX_TOPICS = 5
MAX_BLURB_CHARS = 520

# Everyday tools the model already knows — researching them wastes time.
_COMMON_TOOLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "react", "angular", "vue", "next.js", "node.js", "nodejs",
    "fastapi", "django", "flask", "spring", "express",
    "postgresql", "mysql", "mongodb", "redis", "sqlite",
    "docker", "kubernetes", "jenkins", "github actions", "gitlab ci",
    "git", "linux", "aws", "azure", "gcp", "terraform", "ansible",
    "jest", "playwright", "cypress", "pytest", "junit",
    "rest", "rest apis", "graphql", "html", "css",
}

_MS_HINT = re.compile(
    r"\b(microsoft|azure|fabric|power\s*bi|dynamics|synapse|cosmos|"
    r"data\s*factory|devops|onedrive|sharepoint|\.net|nuget)\b",
    re.I,
)


def _http_json(url: str) -> dict | list | None:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def _http_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return ""


def _clean_html(raw: str) -> str:
    raw = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", raw)
    text = re.sub(r"(?is)<[^>]+>", " ", raw)
    text = htmlmod.unescape(re.sub(r"\s+", " ", text)).strip()
    return text


def _trim(text: str, n: int = MAX_BLURB_CHARS) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= n:
        return text
    cut = text[: n - 1].rsplit(" ", 1)[0]
    return cut + "…"


def select_research_topics(jd_analysis: dict, jd_text: str = "") -> list[str]:
    """Pick specialized products/tools worth looking up on the web."""
    seen: list[str] = []
    for key in ("research_topics", "tools", "must_have_skills", "exact_keywords_for_ats"):
        for item in jd_analysis.get(key) or []:
            term = str(item).strip()
            if not term or len(term) < 3:
                continue
            if term.lower() in _COMMON_TOOLS:
                continue
            # Skip soft skills / vague phrases
            if " " in term and not re.search(
                r"[A-Z]|Fabric|Azure|AWS|GCP|SQL|BI|\.NET|Spark|Kafka|Snowflake|Databricks",
                term,
            ):
                # allow multi-word product names with capitals or known brands
                if not _MS_HINT.search(term) and not re.search(
                    r"\b(Snowflake|Databricks|Salesforce|SAP|Oracle|Kafka|Spark|"
                    r"Terraform|Helm|Argo|Airflow|dbt|Looker|Tableau)\b",
                    term,
                    re.I,
                ):
                    continue
            if term not in seen:
                seen.append(term)
            if len(seen) >= MAX_TOPICS:
                return seen

    # Catch product names that appear in the JD but the analyzer missed
    for pat in (
        r"\bMicrosoft\s+Fabric\b",
        r"\bAzure\s+Data\s+Factory\b",
        r"\bPower\s*BI\b",
        r"\bSnowflake\b",
        r"\bDatabricks\b",
        r"\bSalesforce\b",
        r"\bApache\s+Spark\b",
    ):
        m = re.search(pat, jd_text or "", re.I)
        if m:
            term = m.group(0)
            # normalize casing for known brands
            if re.search(r"fabric", term, re.I):
                term = "Microsoft Fabric"
            if term not in seen and term.lower() not in _COMMON_TOOLS:
                seen.append(term)
        if len(seen) >= MAX_TOPICS:
            break
    return seen[:MAX_TOPICS]


_LEARN_NOISE = re.compile(
    r"Summarize this article for me\.?|In this article\.?|"
    r"Upgrade to Microsoft Edge[^.]*\.|Download Microsoft Edge[^.]*\.|"
    r"This browser is no longer supported[^.]*\.",
    re.I,
)


def _clean_learn_blurb(desc: str) -> str:
    desc = _LEARN_NOISE.sub("", desc or "")
    desc = htmlmod.unescape(desc)
    desc = re.sub(r"&nbsp;?", " ", desc)
    desc = re.sub(
        r"^(Microsoft Fabric documentation|Microsoft Fabric)\s+", "", desc, flags=re.I
    )
    desc = re.sub(
        r"\b(Overview|Get started|Try Fabric for free|Tutorial|Deploy|Concept|"
        r"What's new|Fabric roadmap|At a glance|Level Intermediate|"
        r"Skill|Product|Role|Subject)\b",
        "",
        desc,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", desc).strip(" .-")


def _research_microsoft_learn(topic: str) -> str | None:
    q = urllib.parse.quote(topic)
    data = _http_json(
        f"https://learn.microsoft.com/api/search?search={q}&locale=en-us&$top=6"
    )
    if not isinstance(data, dict):
        return None
    results = data.get("results") or []
    if not results:
        return None

    def score(r: dict) -> tuple:
        title = r.get("title") or ""
        desc = r.get("description") or ""
        blob = f"{title} {desc}".lower()
        # Higher is better — prefer product definition over admin/security docs
        points = 0
        if re.search(r"all-in-one analytics|unified (platform|analytics)", blob):
            points += 8
        if re.search(r"lakehouse end-to-end|data movement to data science", blob):
            points += 6
        if re.search(r"\blakehouse\b", title, re.I):
            points += 4
        if re.search(r"what is microsoft fabric|fabric documentation", title, re.I):
            points += 5
        if re.search(r"architecture|overview", title, re.I):
            points += 2
        if re.search(r"workspace roles|fabric iq|data agent|training|license|capacity", title, re.I):
            points -= 6
        if re.search(r"manage who can|secured independently|roles let you", blob):
            points -= 4
        return (-points, -len(_clean_learn_blurb(desc)))

    ranked = sorted(results, key=score)
    desc = _clean_learn_blurb(ranked[0].get("description") or "")
    # Append a second blurb when it adds concrete components the first lacked
    extras = []
    for r in ranked[1:4]:
        alt = _clean_learn_blurb(r.get("description") or "")
        if len(alt) < 60:
            continue
        if re.search(r"workspace roles|manage who can|license|capacity", alt, re.I):
            continue
        if re.search(
            r"all-in-one analytics|lakehouse|data movement|data warehouse|spark",
            alt,
            re.I,
        ):
            if not re.search(r"lakehouse|data movement|all-in-one", desc, re.I):
                extras.append(alt)
                break
    if extras:
        desc = f"{desc} Key pieces: {extras[0]}"
    # Final scrub of nav crumbs that survive search blurbs
    desc = re.sub(
        r"\b(What is Microsoft Fabric\.?|Develop with end-to-end tutorials|Security)\b",
        "",
        desc,
        flags=re.I,
    )
    desc = re.sub(r"\s+", " ", desc).strip(" .-")
    if len(desc) < 40:
        return None
    return _trim(f"{topic}: {desc} (source: Microsoft Learn)")


_WIKI_REJECT = re.compile(
    r"\b(censorship|tor network|proxy nodes|textile|clothing|fabric softener|"
    r"disambiguation)\b",
    re.I,
)


def _research_wikipedia(topic: str) -> str | None:
    # Disambiguate common collisions (Snowflake the data cloud vs Tor Snowflake)
    query = topic
    if re.search(r"snowflake", topic, re.I):
        query = "Snowflake Inc. cloud data platform"
    elif re.search(r"^fabric$", topic, re.I):
        query = "Microsoft Fabric analytics"
    else:
        query = f"{topic} software OR platform OR database"
    q = urllib.parse.quote(query)
    data = _http_json(
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={q}&srlimit=6&format=json"
    )
    if not isinstance(data, dict):
        return None
    hits = (data.get("query") or {}).get("search") or []
    for hit in hits:
        title = hit.get("title") or ""
        snippet = re.sub(r"<[^>]+>", "", hit.get("snippet") or "")
        if not title or "disambiguation" in title.lower():
            continue
        if title.lower() in {"fabric", "fabric (disambiguation)", "snowflake"}:
            continue
        if _WIKI_REJECT.search(title) or _WIKI_REJECT.search(snippet):
            continue
        slug = urllib.parse.quote(title.replace(" ", "_"))
        summary = _http_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        )
        if not isinstance(summary, dict):
            continue
        if summary.get("type") == "disambiguation":
            continue
        extract = (summary.get("extract") or "").strip()
        if len(extract) < 40 or _WIKI_REJECT.search(extract):
            continue
        return _trim(f"{topic}: {extract} (source: Wikipedia — {title})")
    return None


def research_topic(topic: str) -> str | None:
    """Return one short factual blurb for a topic, or None."""
    topic = (topic or "").strip()
    if not topic:
        return None
    if _MS_HINT.search(topic) or topic.lower() in {"fabric", "power bi", "powerbi"}:
        # Prefer the full Microsoft product name for Fabric
        query = "Microsoft Fabric" if topic.lower() in {"fabric", "microsoft fabric"} else topic
        blurb = _research_microsoft_learn(query)
        if blurb:
            return blurb
    blurb = _research_wikipedia(topic)
    if blurb:
        return blurb
    # Last resort: Microsoft Learn even for non-obvious MS terms
    return _research_microsoft_learn(topic)


def build_tech_context(jd_analysis: dict, jd_text: str = "") -> str:
    """Research specialized JD tools in parallel; return a prompt-ready block."""
    topics = select_research_topics(jd_analysis, jd_text)
    if not topics:
        _debug_dump("web_context", "no research topics selected")
        return ""

    blurbs: list[str] = []
    with ThreadPoolExecutor(max_workers=min(4, len(topics))) as pool:
        futures = {pool.submit(research_topic, t): t for t in topics}
        for fut in as_completed(futures):
            topic = futures[fut]
            try:
                blurb = fut.result()
            except Exception as exc:
                _debug_dump("web_context_error", f"{topic}: {exc}")
                blurb = None
            if blurb:
                blurbs.append(blurb)

    # Keep topic order stable for the prompt
    ordered = []
    for t in topics:
        for b in blurbs:
            if b.lower().startswith(t.lower()) or t.lower() in b.lower()[:80]:
                if b not in ordered:
                    ordered.append(b)
                break
    for b in blurbs:
        if b not in ordered:
            ordered.append(b)

    block = "\n".join(f"- {b}" for b in ordered)
    _debug_dump("web_context", json.dumps({"topics": topics, "blurbs": ordered}, indent=2))
    return block
