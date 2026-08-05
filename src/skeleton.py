"""LaTeX skeleton: the templates in latex/ define the layout, not the model.

The pipeline never asks an LLM for LaTeX. It parses the shipped templates to
learn the exact structure — which jobs and projects exist, and how many bullet
slots each one has — then substitutes bullet TEXT back into those same slots.
The surrounding LaTeX comes out byte-identical to the template, so the resume
keeps its layout and bullet counts no matter what the model returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import LATEX_DIR

_ITEM_CMD = "\\resumeItem"
_SUBHEADING = "\\resumeSubheading"
_PROJECT_HEADING = "\\resumeProjectHeading"


@dataclass
class Slot:
    """One \\resumeItem{...} in the template."""
    start: int          # index of the "\" in \resumeItem
    body_start: int     # index just after the opening brace
    body_end: int       # index of the closing brace
    body: str


@dataclass
class Block:
    """A job or a project: a heading plus its bullet slots."""
    kind: str           # "role" | "project"
    label: str          # company name, or project name
    header_start: int
    slots: list[Slot] = field(default_factory=list)

    @property
    def bullet_count(self) -> int:
        return len(self.slots)


def _match_brace(text: str, open_idx: int) -> int:
    """Index of the brace closing the one at open_idx, or -1."""
    depth = 0
    i = open_idx
    while i < len(text):
        ch = text[i]
        if ch == "\\":          # skip an escaped char, e.g. \% or \{
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _brace_groups(text: str, start: int, count: int) -> list[str]:
    """Read `count` consecutive {...} groups starting at/after `start`."""
    groups: list[str] = []
    i = start
    while len(groups) < count:
        open_idx = text.find("{", i)
        if open_idx == -1:
            break
        close_idx = _match_brace(text, open_idx)
        if close_idx == -1:
            break
        groups.append(text[open_idx + 1 : close_idx])
        i = close_idx + 1
    return groups


def _find_slots(text: str) -> list[Slot]:
    slots: list[Slot] = []
    for m in re.finditer(re.escape(_ITEM_CMD) + r"\s*\{", text):
        open_idx = text.index("{", m.start() + len(_ITEM_CMD) - 1)
        close_idx = _match_brace(text, open_idx)
        if close_idx == -1:
            continue
        slots.append(
            Slot(
                start=m.start(),
                body_start=open_idx + 1,
                body_end=close_idx,
                body=text[open_idx + 1 : close_idx],
            )
        )
    return slots


def _project_label(group: str) -> str:
    m = re.search(r"\\textbf\{([^}]*)\}", group)
    return (m.group(1) if m else group).strip()


def parse_section(text: str) -> list[Block]:
    """Split a section template into heading blocks with their bullet slots."""
    heads: list[tuple[int, str]] = []
    for m in re.finditer(re.escape(_SUBHEADING) + r"(?![a-zA-Z])", text):
        heads.append((m.start(), "role"))
    for m in re.finditer(re.escape(_PROJECT_HEADING) + r"(?![a-zA-Z])", text):
        heads.append((m.start(), "project"))
    heads.sort()

    blocks: list[Block] = []
    for i, (pos, kind) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        chunk = text[pos:end]
        if kind == "role":
            groups = _brace_groups(text, pos, 4)
            # \resumeSubheading{title}{dates}{company}{location}
            label = groups[2].strip() if len(groups) >= 3 else ""
        else:
            groups = _brace_groups(text, pos, 2)
            label = _project_label(groups[0]) if groups else ""
        block = Block(kind=kind, label=label, header_start=pos)
        for slot in _find_slots(chunk):
            block.slots.append(
                Slot(
                    start=slot.start + pos,
                    body_start=slot.body_start + pos,
                    body_end=slot.body_end + pos,
                    body=slot.body,
                )
            )
        blocks.append(block)
    return blocks


def render_section(text: str, bullets_by_block: dict[int, list[str]]) -> str:
    """Substitute bullet text into a template's slots, keeping all other bytes.

    bullets_by_block maps block index -> bullet strings. A block that is absent,
    or a slot with no replacement, keeps the template's original text. Extra
    bullets beyond a block's slot count are dropped: the template owns the count.
    """
    blocks = parse_section(text)
    replacements: list[tuple[int, int, str]] = []
    for block_idx, block in enumerate(blocks):
        bullets = bullets_by_block.get(block_idx) or []
        for slot_idx, slot in enumerate(block.slots):
            if slot_idx >= len(bullets):
                continue
            new_body = (bullets[slot_idx] or "").strip()
            if not new_body:
                continue
            replacements.append((slot.body_start, slot.body_end, new_body))

    out = text
    for body_start, body_end, new_body in sorted(replacements, reverse=True):
        out = out[:body_start] + new_body + out[body_end:]
    return out


_UNICODE_FIXES = [
    ("—", "---"), ("–", "--"),          # em/en dash
    ("‘", "`"), ("’", "'"),             # smart single quotes
    ("“", "``"), ("”", "''"),           # smart double quotes
    ("→", "to"), ("…", "..."),          # arrow, ellipsis
    (" ", " "),                              # non-breaking space
]

_ESCAPE_RE = re.compile(r"(?<!\\)([&#_$%])")


def latex_safe(text: str) -> str:
    """Make model-written prose safe to drop into a LaTeX template.

    Escapes the characters that silently break a build — a bare & is an
    alignment tab and a bare % comments out the rest of the line — and folds
    unicode punctuation the model likes to emit into LaTeX equivalents.
    Already-escaped sequences (\\%) are left alone.
    """
    out = text or ""
    for needle, replacement in _UNICODE_FIXES:
        out = out.replace(needle, replacement)
    return _ESCAPE_RE.sub(r"\\\1", out)


def load_template(section: str, variant: str = "original") -> str:
    path = LATEX_DIR / section / f"{variant}.tex"
    if not path.exists():
        raise ValueError(f"LaTeX template not found: {path}")
    return path.read_text(encoding="utf-8")


def section_blocks(section: str) -> list[Block]:
    return parse_section(load_template(section))


# ---------------------------------------------------------------------------
# Skills: the template has a fixed number of "\textbf{Category}{: items}" lines.
# ---------------------------------------------------------------------------

_SKILL_LINE_RE = re.compile(r"\\textbf\{([^}]*)\}\s*\{:\s*([^}]*)\}")


def skills_line_count(text: str) -> int:
    return len(_SKILL_LINE_RE.findall(text))


def render_skills(text: str, lines: list[tuple[str, list[str]]]) -> str:
    """Replace each skills line with (category, items), keeping the template shape."""
    matches = list(_SKILL_LINE_RE.finditer(text))
    out = text
    for i in range(len(matches) - 1, -1, -1):
        if i >= len(lines):
            continue
        name, items = lines[i]
        m = matches[i]
        replacement = (
            "\\textbf{" + latex_safe(name) + "}{: "
            + ", ".join(latex_safe(item) for item in items) + "}"
        )
        out = out[: m.start()] + replacement + out[m.end() :]
    return out


# ---------------------------------------------------------------------------
# Summary: one \textit{...} inside a center environment.
# ---------------------------------------------------------------------------

def render_summary(text: str, summary: str) -> str:
    open_idx = text.find("\\textit{")
    if open_idx == -1:
        return text
    brace_idx = text.index("{", open_idx)
    close_idx = _match_brace(text, brace_idx)
    if close_idx == -1:
        return text
    return text[: brace_idx + 1] + summary.strip() + text[close_idx:]
