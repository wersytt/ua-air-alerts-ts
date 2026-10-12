import numpy as np
import pandas as pd

from ua_alerts import tables


def _df():
    return pd.DataFrame({"name": ["a", "longer name"], "n": [1, 12345], "x": [0.5, -12.345], "ok": [True, False],
                         "gap": [np.nan, 2.0]}, index=["r1", "row two"])


def test_every_line_has_the_same_width_in_text_and_markdown():
    for kind in ("text", "md"):
        lines = tables.render(_df(), kind=kind, index_name="idx").split("\n")
        assert len({len(l) for l in lines}) == 1, lines


def test_numbers_right_aligned_with_fixed_decimals_and_text_left_aligned():
    t = tables.md(_df(), index_name="idx", signed=("x",)).split("\n")
    assert "12,345" in t[3] and "-12.35" in t[3] and "+0.50" in t[2]
    assert t[1].split("|")[2].strip().startswith(":")          # text column -> left-aligned marker
    assert t[1].split("|")[3].strip().endswith(":")            # numeric column -> right-aligned marker
    assert " - " in t[2] or "| -" in t[2]                      # NaN shown as '-'
