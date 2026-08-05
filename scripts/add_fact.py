#!/usr/bin/env python3
"""Add a fact to data/facts.yaml from the command line.

The builder asks verification questions when a posting wants something the fact
bank cannot answer. This is how an answer becomes a fact, so the next build --
and every build after it -- can use it.

    python3 scripts/add_fact.py clerxi \
        --id eval-harness \
        --core "Built an evaluation harness measuring retrieval precision@k and
                p95 latency across releases, cutting regression triage to minutes." \
        --tools Python pytest \
        --themes testing quality search-retrieval \
        --metrics "precision@k" "p95"

Themes must come from THEMES in src/facts.py; the script validates before writing
and refuses to corrupt the file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.facts import FACTS_PATH, THEMES, load_fact_bank  # noqa: E402


def _wrap(text: str, indent: str = "          ") -> str:
    words = re.sub(r"\s+", " ", text.strip()).split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > 78:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return ("\n" + indent).join(lines)


def _yaml_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"


def _quoted_list(values: list[str]) -> str:
    return "[" + ", ".join('"' + v.replace('"', '\\"') + '"' for v in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Add a fact to the fact bank")
    parser.add_argument("owner", help="role or project id, e.g. clerxi / intuit / autofixee")
    parser.add_argument("--id", required=True, help="short slug for this fact")
    parser.add_argument("--core", required=True, help="what actually happened")
    parser.add_argument("--tools", nargs="*", default=[], help="technologies genuinely used")
    parser.add_argument("--themes", nargs="*", default=[], help="see THEMES in src/facts.py")
    parser.add_argument("--metrics", nargs="*", default=[], help='numbers, e.g. "35%%" "2 hours to 10 minutes"')
    parser.add_argument("--angles", nargs="*", default=[], help="truthful alternative framings")
    parser.add_argument("--lead", action="store_true", help="this fact introduces the role/project")
    args = parser.parse_args()

    bank = load_fact_bank()
    owners = {r.id: r.company for r in bank.roles}
    owners.update({p.id: p.name for p in bank.projects})
    if args.owner not in owners:
        print(f"error: unknown owner '{args.owner}'. Known: {', '.join(sorted(owners))}")
        return 1

    bad = [t for t in args.themes if t not in THEMES]
    if bad:
        print(f"error: unknown theme(s): {', '.join(bad)}")
        print(f"valid: {', '.join(sorted(THEMES))}")
        return 1

    existing = {f.id for f in bank.all_facts() if f.owner_id == args.owner}
    if args.id in existing:
        print(f"error: '{args.owner}' already has a fact with id '{args.id}'")
        return 1

    text = FACTS_PATH.read_text(encoding="utf-8")

    # Anchor on the owner's block, then find where its fact list ends: the next
    # line at or below the owner's own indentation that starts a new key/item.
    anchor = re.search(rf"^(\s*)- id: {re.escape(args.owner)}\s*$", text, re.M)
    if not anchor:
        print(f"error: could not locate '{args.owner}' in {FACTS_PATH}")
        return 1
    owner_indent = len(anchor.group(1))

    lines = text.splitlines()
    start = text[: anchor.start()].count("\n")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= owner_indent and line.lstrip().startswith(("- ", "roles:", "projects:", "skills:")):
            end = i
            break

    block = [f"      - id: {args.id}"]
    if args.lead:
        block.append("        lead: true")
    block.append("        core: >-")
    block.append("          " + _wrap(args.core))
    block.append(f"        tools: {_yaml_list(args.tools)}")
    block.append(f"        themes: {_yaml_list(args.themes)}")
    block.append(f"        metrics: {_quoted_list(args.metrics)}")
    if args.angles:
        block.append("        angles:")
        block.extend(f"          - {a}" for a in args.angles)
    block.append("")

    # Insert after the owner's last fact, before any trailing comments.
    insert_at = end
    while insert_at > start and (
        not lines[insert_at - 1].strip() or lines[insert_at - 1].lstrip().startswith("#")
    ):
        insert_at -= 1

    updated = lines[:insert_at] + block + lines[insert_at:]
    new_text = "\n".join(updated) + ("\n" if text.endswith("\n") else "")

    backup = FACTS_PATH.with_suffix(".yaml.bak")
    backup.write_text(text, encoding="utf-8")
    FACTS_PATH.write_text(new_text, encoding="utf-8")

    try:
        reloaded = load_fact_bank()
    except Exception as exc:
        FACTS_PATH.write_text(text, encoding="utf-8")
        print(f"error: the edit produced an invalid fact bank, reverted.\n  {exc}")
        return 1

    count = len([f for f in reloaded.all_facts() if f.owner_id == args.owner])
    print(f"added {args.owner}.{args.id}  ({owners[args.owner]} now has {count} facts)")
    print(f"backup: {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
