"""One table renderer for everything (console, report.md, validation report).

Why this exists: hand-formatted tables drifted out of alignment (ragged columns, numbers
left-aligned, 1 vs 2 vs 3 decimals in the same column, "12.4 [3]" strings mixing a value and
a rank). Here every column is padded to a common width, numbers are right-aligned with a fixed
number of decimals per column, and text is left-aligned - in the raw text as well as in the
rendered Markdown.
"""
from __future__ import annotations

from numbers import Number

import numpy as np
import pandas as pd

MISSING = "-"


def _is_num(v) -> bool:
    return isinstance(v, Number) and not isinstance(v, (bool, np.bool_))


def _fmt_cell(v, decimals: int, signed: bool, pct: bool, is_int_col: bool) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return MISSING
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if _is_num(v):
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        if pct:
            return f"{v:{'+' if signed else ''}.{decimals}%}"
        if is_int_col:
            return f"{int(v):{'+' if signed else ''},d}"
        return f"{v:{'+' if signed else ''},.{decimals}f}"
    return str(v)


def render(
    df: pd.DataFrame,
    kind: str = "text",
    index_name: str | None = "",
    decimals: int | dict[str, int] = 2,
    signed: tuple[str, ...] = (),
    pct: tuple[str, ...] = (),
    include_index: bool = True,
) -> str:
    """Render ``df`` as an aligned table. ``kind`` is ``"text"`` or ``"md"``."""
    cols = list(df.columns)
    header = ([index_name if index_name is not None else (df.index.name or "")] if include_index else []) \
        + [str(c) for c in cols]
    numeric = [False] * len(header)
    off = 1 if include_index else 0

    col_cells: dict[str, list[str]] = {}
    for c in cols:
        vals = df[c].tolist()
        non_null = [v for v in vals if not (isinstance(v, float) and np.isnan(v)) and v is not None]
        is_num = bool(non_null) and all(_is_num(v) for v in non_null)
        is_int = is_num and pd.api.types.is_integer_dtype(df[c].dtype)
        d = decimals.get(c, 2) if isinstance(decimals, dict) else decimals
        col_cells[c] = [_fmt_cell(v, d, c in signed, c in pct, is_int) for v in vals]
        numeric[off + cols.index(c)] = is_num
    body = [([str(idx)] if include_index else []) + [col_cells[c][i] for c in cols]
            for i, idx in enumerate(df.index)]

    widths = [max([len(header[j])] + [len(r[j]) for r in body]) for j in range(len(header))]
    if kind == "md":
        widths = [max(w, 3) for w in widths]

    def cell(j, text):
        return text.rjust(widths[j]) if numeric[j] else text.ljust(widths[j])

    lines = []
    if kind == "md":
        lines.append("| " + " | ".join(cell(j, h) for j, h in enumerate(header)) + " |")
        seps = [("-" * (widths[j] - 1) + ":") if numeric[j] else (":" + "-" * (widths[j] - 1))
                for j in range(len(header))]
        lines.append("| " + " | ".join(seps) + " |")
        lines += ["| " + " | ".join(cell(j, t) for j, t in enumerate(r)) + " |" for r in body]
    else:
        lines.append("  ".join(cell(j, h) for j, h in enumerate(header)).rstrip())
        lines.append("  ".join("-" * w for w in widths))
        lines += ["  ".join(cell(j, t) for j, t in enumerate(r)).rstrip() for r in body]
    return "\n".join(lines)


def md(df: pd.DataFrame, **kw) -> str:
    return render(df, kind="md", **kw)


def text(df: pd.DataFrame, **kw) -> str:
    return render(df, kind="text", **kw)
