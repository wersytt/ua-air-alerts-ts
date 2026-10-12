import json

import pandas as pd
import pytest

from ua_alerts import sources

GOOD = "region,started_at,finished_at,naive\nKyiv City,2024-01-01 10:00:00+00:00,2024-01-01 11:00:00+00:00,False\n" \
       "Kyiv City,2024-01-02 10:00:00+00:00,2024-01-02 11:00:00+00:00,False\nLvivska oblast,2024-01-02 10:00:00+00:00,2024-01-02 10:30:00+00:00,True\n"


@pytest.fixture
def mirrors(tmp_path, monkeypatch):
    html = tmp_path / "err.html"; html.write_text("<html>404 Not Found</html>")
    good = tmp_path / "good.csv"; good.write_text(GOOD)
    spec = dict(sources.SOURCES["volunteer"]); spec["min_rows"] = 2
    monkeypatch.setitem(sources.SOURCES, "volunteer", spec)
    return tmp_path, html.as_uri(), good.as_uri(), (tmp_path / "missing.csv").as_uri(), monkeypatch, spec


def test_fetch_skips_broken_mirrors_and_rejects_garbage(mirrors):
    tmp, html, good, missing, mp, spec = mirrors
    spec["urls"] = [missing, html, good]
    res = sources.fetch("volunteer", tmp / "cache", refresh=True)
    assert not res.from_cache and res.origin == good
    assert any("rejected" in n for n in res.notes) and any("failed" in n for n in res.notes)
    assert res.path.read_text() == GOOD          # cache holds the *good* file, not the HTML


def test_fetch_falls_back_to_stale_cache_and_never_overwrites_it_with_garbage(mirrors):
    tmp, html, good, missing, mp, spec = mirrors
    spec["urls"] = [good]
    sources.fetch("volunteer", tmp / "cache", refresh=True)
    spec["urls"] = [missing, html]
    res = sources.fetch("volunteer", tmp / "cache", refresh=True)
    assert res.from_cache and "using cached copy" in " ".join(res.notes)
    assert res.path.read_text() == GOOD


def test_fetch_raises_clear_error_without_cache(mirrors):
    tmp, html, good, missing, mp, spec = mirrors
    spec["urls"] = [missing]
    with pytest.raises(RuntimeError, match="no local cache"):
        sources.fetch("volunteer", tmp / "empty", refresh=True)


def test_env_override_is_tried_first(mirrors, monkeypatch):
    tmp, html, good, missing, mp, spec = mirrors
    spec["urls"] = [missing]
    monkeypatch.setenv(spec["env"], good)
    assert sources.fetch("volunteer", tmp / "c2", refresh=True).origin == good


# TODO (disabled with the Kyiv JSON parser; see README, To-do)
# @pytest.mark.parametrize("payload", [
#     [{"start_time": "2024-03-01T10:00:00", "end_time": "2024-03-01T11:30:00", "type": "air"}],
#     {"data": [{"alarmStart": "2024-03-01 10:00", "alarmEnd": "2024-03-01 11:30"}]},
#     [{"початок_тривоги": "2024-03-01T10:00:00", "кінець_тривоги": "2024-03-01T11:30:00"}],
# ])
# def test_parse_kyiv_official_tolerates_several_schemas(payload):
#     df = sources.parse_kyiv_official(json.dumps(payload))
#     assert len(df) == 1 and df.loc[0, "region"] == "Kyiv City"
#     # naive timestamps are Kyiv local time (UTC+2 in March) -> 08:00 UTC, 1.5 h long
#     assert df.loc[0, "started_at"] == pd.Timestamp("2024-03-01 08:00", tz="UTC")
#     assert df.loc[0, "finished_at"] - df.loc[0, "started_at"] == pd.Timedelta("90min")
#
#
# def test_parse_kyiv_official_explains_failure_and_accepts_explicit_keys():
#     bad = [{"a": "2024-03-01T10:00:00", "b": "2024-03-01T11:00:00"}]
#     with pytest.raises(ValueError, match="Columns found"):
#         sources.parse_kyiv_official(bad)
#     assert len(sources.parse_kyiv_official(bad, start_key="a", end_key="b")) == 1


def test_quality_report_flags_stale_data_and_volume_jump():
    now = pd.Timestamp("2026-10-12 12:00", tz="UTC")
    as_of = now - pd.Timedelta(hours=100)
    rows = []
    for d in range(120):
        t = as_of.floor("D") - pd.Timedelta(days=d)
        n = 1 if d > 30 else 6
        for k in range(n):
            rows.append(("Kyiv City", t + pd.Timedelta(hours=k), t + pd.Timedelta(hours=k, minutes=20), False))
    a = pd.DataFrame(rows, columns=["region", "started_at", "finished_at", "naive"])
    msgs = " | ".join(sources.quality_report(a, as_of, now))
    assert "100 h old" in msgs and "x vs the previous 30 days" in msgs


def test_ukrainian_name_matching():
    assert sources.ukrainian_to_region("м. Київ") == "Kyiv City"
    assert sources.ukrainian_to_region("Київська область") == "Kyivska oblast"
    assert sources.ukrainian_to_region("Харківська область") == "Kharkivska oblast"
    # TODO (disabled with the alerts.in.ua client; see README, To-do): group_live_alerts test
