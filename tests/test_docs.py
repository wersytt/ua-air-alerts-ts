"""Documentation hygiene: Markdown tables stay aligned and well-formed."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import format_md_tables as fmt  # noqa: E402

DOCS = ["README.md", "README-UA.md", "USAGE.md"]


def test_doc_tables_are_aligned_and_consistent():
    for name in DOCS:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert fmt.format_text(text) == text, f"{name}: run `python tools/format_md_tables.py {name}`"


def test_formatter_handles_alignment_markers_escapes_and_code_fences():
    src = "| a | b |\n|:--|--:|\n| x \\| y | 1 |\n| longer cell | 22 |\n\n```\n| not | a table |\n```\n"
    out = fmt.format_text(src).split("\n")
    assert len({len(l) for l in out[:4]}) == 1                 # constant width
    assert "x \\| y" in out[2] and out[1].split("|")[2].strip().endswith(":")
    assert "| not | a table |" in out                          # fenced block untouched
