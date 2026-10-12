#!/usr/bin/env python3
"""Align the pipe tables in Markdown files so the raw text is as tidy as the rendered page.

    python tools/format_md_tables.py README.md USAGE.md        # rewrite in place
    python tools/format_md_tables.py --check README.md USAGE.md   # exit 1 if anything would change

Rules: every cell in a column is padded to the same width, the separator row keeps its alignment
markers (``:---`` left, ``---:`` right, ``:---:`` centre; default left), tables inside fenced code
blocks are left alone, and a literal pipe inside a cell must be written ``\\|``.
A table whose rows have differing numbers of cells is reported as an error.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SEP_CELL = re.compile(r"^:?-{1,}:?$")


def split_row(line: str) -> list[str]:
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return [c.strip() for c in re.split(r"(?<!\\)\|", body)]


def format_table(lines: list[str]) -> list[str]:
    rows = [split_row(l) for l in lines]
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("inconsistent number of cells in table starting with: " + lines[0][:70])
    if len(rows) < 2 or not all(SEP_CELL.match(c) for c in rows[1]):
        return lines                      # not a real table (no separator row)
    aligns = ["right" if c.endswith(":") and not c.startswith(":") else
              "center" if c.startswith(":") and c.endswith(":") else "left" for c in rows[1]]
    widths = [max(3, max(len(r[j]) for i, r in enumerate(rows) if i != 1)) for j in range(n)]

    def sep(j):
        w = widths[j]
        return {"left": ":" + "-" * (w - 1), "right": "-" * (w - 1) + ":", "center": ":" + "-" * (w - 2) + ":"}[aligns[j]]

    def cell(j, t):
        return t.rjust(widths[j]) if aligns[j] == "right" else t.center(widths[j]) if aligns[j] == "center" else t.ljust(widths[j])

    out = []
    for i, r in enumerate(rows):
        out.append("| " + " | ".join(sep(j) if i == 1 else cell(j, t) for j, t in enumerate(r)) + " |")
    return out


def format_text(text: str) -> str:
    out, block, fence = [], [], False
    def flush():
        nonlocal block
        if block:
            out.extend(format_table(block))
            block = []
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            flush(); fence = not fence; out.append(line); continue
        if not fence and line.startswith("|"):
            block.append(line)
        else:
            flush(); out.append(line)
    flush()
    return "\n".join(out)


def main(argv: list[str]) -> int:
    check = "--check" in argv
    files = [Path(a) for a in argv if not a.startswith("--")]
    changed = []
    for f in files:
        old = f.read_text(encoding="utf-8")
        new = format_text(old)
        if new != old:
            changed.append(f)
            if not check:
                f.write_text(new, encoding="utf-8")
    for f in changed:
        print(("would reformat " if check else "reformatted ") + str(f))
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
