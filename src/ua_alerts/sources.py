"""Where the data comes from, and how much to trust it.

Design goals (all driven by "what if the upstream repo disappears or the data is wrong?"):

1. **Mirrors + cache.** Each dataset has an ordered list of URLs (GitHub raw, then a CDN
   mirror, then anything added via an environment variable). A download only replaces the local cache if it
   *looks like the right file* (columns present, plausible row count), so a broken, empty or
   HTML-error response can never poison the cache. If every URL fails, the last good cached
   copy is used and a note says how old it is.
2. **Two datasets** ("volunteer" and "official") share one loader, so any analysis can be
   re-run on either one (``--source``).
3. **Quality checks** that turn silent data problems (stale, gaps, sudden volume changes,
   absurd durations) into explicit warnings.
4. **Disabled extras (README, To-do):** a parser for the Kyiv City open-data JSON (data.gov.ua) and a live-status
   call to alerts.in.ua. Their code is kept below as comments until it can be tested against the real services.

Nothing here needs an API key.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_RAW = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/"
_CDN = "https://cdn.jsdelivr.net/gh/Vadimkin/ukrainian-air-raid-sirens-dataset@main/datasets/"

SOURCES: dict[str, dict] = {
    "volunteer": dict(
        filename="volunteer_data_en.csv",
        urls=[_RAW + "volunteer_data_en.csv", _CDN + "volunteer_data_en.csv"],
        required={"region", "started_at", "finished_at", "naive"},
        min_rows=50_000,
        env="ALERTS_VOLUNTEER_URL",
    ),
    "official": dict(
        filename="official_data_en.csv",
        urls=[_RAW + "official_data_en.csv", _CDN + "official_data_en.csv"],
        required={"oblast", "level", "started_at", "finished_at"},
        min_rows=100_000,
        env="ALERTS_OFFICIAL_URL",
    ),
}

# TODO (disabled): Kyiv City open-data JSON URL (see README, To-do).
# # The Kyiv City open-data resource (Kyiv only; refreshed weekly upstream).
# KYIV_OFFICIAL_URL = (
#     "https://data.gov.ua/dataset/83eda433-b4d9-47ac-89a0-5868dd4d5ce6/resource/"
#     "08559e6a-863f-4d9b-86c1-d23e8b8b4630/download/airalerts.json"
# )

USER_AGENT = "ua-air-alerts-ts/1.0 (open-data research)"


@dataclass
class FetchResult:
    path: Path
    origin: str                      # URL used, or "cache"
    from_cache: bool
    age_hours: float                 # age of the local file
    notes: list[str] = field(default_factory=list)


def _looks_valid(path: Path, required: set[str], min_rows: int) -> str | None:
    """Return None if the file is plausible, else a human-readable reason it is not."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().strip().split(",")
            if not required.issubset({h.strip() for h in header}):
                return f"missing columns {sorted(required - set(header))} (got {header[:8]})"
            rows = sum(1 for _ in fh)
        if rows < min_rows:
            return f"only {rows:,} rows (< {min_rows:,}); truncated or wrong file"
    except OSError as exc:
        return str(exc)
    return None


def _download(url: str, dest: Path, timeout: float = 90) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)


def fetch(
    name: str,
    cache_dir: str | Path,
    refresh: bool = False,
    max_age_hours: float = 12,
    extra_urls: list[str] | None = None,
) -> FetchResult:
    """Get the CSV for source ``name``: fresh cache, else try each mirror, else stale cache."""
    spec = SOURCES[name]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / spec["filename"]
    notes: list[str] = []

    def age() -> float:
        return (time.time() - target.stat().st_mtime) / 3600 if target.exists() else float("inf")

    if target.exists() and not refresh and age() <= max_age_hours:
        return FetchResult(target, "cache", True, age(), notes)

    urls = list(extra_urls or [])
    if os.environ.get(spec["env"]):
        urls.insert(0, os.environ[spec["env"]])
    urls += spec["urls"]

    for url in urls:
        tmp = target.with_suffix(".part")
        try:
            _download(url, tmp)
            problem = _looks_valid(tmp, spec["required"], spec["min_rows"])
            if problem:
                notes.append(f"rejected {url}: {problem}")
                tmp.unlink(missing_ok=True)
                continue
            os.replace(tmp, target)
            return FetchResult(target, url, False, 0.0, notes)
        except Exception as exc:  # network errors, 404 (repo private/removed), timeouts ...
            notes.append(f"failed {url}: {type(exc).__name__}: {exc}")
            tmp.unlink(missing_ok=True)

    if target.exists():
        notes.append(f"all mirrors failed; using cached copy that is {age():.0f} h old")
        return FetchResult(target, "cache", True, age(), notes)
    raise RuntimeError(
        f"Could not obtain the '{name}' dataset and there is no local cache.\n  "
        + "\n  ".join(notes)
        + f"\nTip: download it manually and place it at {target}, or set {spec['env']} to a mirror/fork URL."
    )


# ------------------------------------------------------------------- quality checks
def quality_report(alerts: pd.DataFrame, as_of: pd.Timestamp, now: pd.Timestamp | None = None,
                   check_long_alerts: bool = True) -> list[str]:
    """Return a list of human-readable warnings (empty list = nothing suspicious)."""
    now = now or pd.Timestamp.now(tz="UTC")
    w: list[str] = []
    lag_h = (now - as_of).total_seconds() / 3600
    if lag_h > 48:
        w.append(f"Data is {lag_h:.0f} h old (latest record {as_of:%Y-%m-%d %H:%M} UTC); the upstream may have stopped updating.")
    if as_of > now + pd.Timedelta(hours=1):
        w.append("Latest record is in the future relative to this machine's clock; check the system time.")

    last = alerts[alerts["started_at"] >= as_of - pd.Timedelta(days=90)]
    daily = last.groupby(last["started_at"].dt.floor("D")).size()
    days = pd.date_range(as_of.floor("D") - pd.Timedelta(days=90), as_of.floor("D") - pd.Timedelta(days=1), tz="UTC")
    empty = int((daily.reindex(days).fillna(0) == 0).sum())
    if empty:
        w.append(f"{empty} day(s) in the last 90 with zero alerts nationwide - likely a collection gap.")

    recent = (alerts["started_at"] >= as_of - pd.Timedelta(days=30)).sum()
    before = ((alerts["started_at"] >= as_of - pd.Timedelta(days=60))
              & (alerts["started_at"] < as_of - pd.Timedelta(days=30))).sum()
    if before and not (0.5 <= recent / before <= 2.0):
        w.append(f"Alert volume changed {recent / before:.1f}x vs the previous 30 days "
                 "(real escalation OR a change in how alerts are recorded - verify with `validate.py`).")

    dur_h = (alerts["finished_at"] - alerts["started_at"]).dt.total_seconds() / 3600
    if check_long_alerts and (dur_h > 72).any():
        w.append(f"{int((dur_h > 72).sum())} alert(s) longer than 72 h (probably missing 'all clear' signals).")
    return w


# TODO (disabled): Kyiv City open-data JSON parser. Disabled until it can be tested against the real file (see README, To-do).
# # ---------------------------------------------------------- Kyiv City official JSON
# _START_HINTS = ("start", "begin", "from", "почат", "початок")
# _END_HINTS = ("end", "finish", "stop", "until", "кінец", "кінц", "закінч", "відбій")
#
#
# def _pick_key(keys, hints, exclude=()):
#     for k in keys:
#         lk = str(k).lower()
#         if any(h in lk for h in hints) and not any(e in lk for e in exclude):
#             return k
#     return None
#
#
# def parse_kyiv_official(payload: bytes | str | list | dict, start_key: str | None = None,
#                         end_key: str | None = None) -> pd.DataFrame:
#     """Parse the data.gov.ua Kyiv City alert history into ``region/started_at/finished_at/naive``.
#
#     The schema of this file could not be verified when the parser was written (the resource was
#     not retrievable as text), so the parser is deliberately tolerant: it accepts a JSON list (or a dict wrapping one), finds the
#     start / end columns by name hints, and treats timezone-less timestamps as Europe/Kyiv local
#     time. If auto-detection fails, an error lists the keys found; pass them explicitly.
#     """
#     obj = json.loads(payload) if isinstance(payload, (bytes, str)) else payload
#     if isinstance(obj, dict):  # e.g. {"data": [...]} / {"alerts": [...]}
#         lists = [v for v in obj.values() if isinstance(v, list)]
#         if not lists:
#             raise ValueError(f"No list of records found in JSON object with keys {list(obj)[:10]}")
#         obj = max(lists, key=len)
#     df = pd.json_normalize(obj)
#     if df.empty:
#         raise ValueError("Kyiv official JSON contains no records")
#     sk = start_key or _pick_key(df.columns, _START_HINTS)
#     ek = end_key or _pick_key(df.columns, _END_HINTS, exclude=("start", "begin"))
#     if sk is None or ek is None or sk not in df.columns or ek not in df.columns:
#         raise ValueError(f"Could not detect start/end columns. Columns found: {list(df.columns)}. "
#                          "Pass start_key=... and end_key=... (validate.py --start-key/--end-key).")
#
#     def to_utc(s: pd.Series) -> pd.Series:
#         t = pd.to_datetime(s, errors="coerce", utc=False, format="mixed")
#         if t.dt.tz is None:
#             t = t.dt.tz_localize("Europe/Kyiv", ambiguous="NaT", nonexistent="shift_forward")
#         return t.dt.tz_convert("UTC")
#
#     out = pd.DataFrame({"region": "Kyiv City", "started_at": to_utc(df[sk]), "finished_at": to_utc(df[ek])})
#     out = out.dropna()
#     out = out[out["finished_at"] > out["started_at"]].reset_index(drop=True)
#     out["naive"] = False
#     return out
#
#
# def fetch_kyiv_official(cache_dir: str | Path, url: str = KYIV_OFFICIAL_URL, refresh: bool = False,
#                         max_age_hours: float = 24 * 7) -> Path:
#     """Download the Kyiv JSON (weekly refresh upstream, so a week of cache is fine)."""
#     dest = Path(cache_dir) / "kyiv_airalerts.json"
#     dest.parent.mkdir(parents=True, exist_ok=True)
#     fresh = dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_hours * 3600
#     if fresh and not refresh:
#         return dest
#     tmp = dest.with_suffix(".part")
#     try:
#         _download(url, tmp)
#         json.loads(tmp.read_bytes())  # must at least be valid JSON
#         os.replace(tmp, dest)
#     except Exception as exc:
#         tmp.unlink(missing_ok=True)
#         if dest.exists():
#             print(f"  warning: could not refresh Kyiv JSON ({exc}); using cached copy")
#             return dest
#         raise RuntimeError(f"Could not download Kyiv official JSON: {exc}") from exc
#     return dest


# ------------------------------------------------------------ Ukrainian oblast names
_UA_STEMS = {  # Ukrainian name stems -> dataset region names (matches alerts.in.ua output)
    "вінниц": "Vinnytska oblast", "волин": "Volynska oblast", "дніпропетров": "Dnipropetrovska oblast",
    "донец": "Donetska oblast", "житомир": "Zhytomyrska oblast", "закарпат": "Zakarpatska oblast",
    "запор": "Zaporizka oblast", "івано": "Ivano-Frankivska oblast", "київськ": "Kyivska oblast",
    "кіровоград": "Kirovohradska oblast", "луган": "Luhanska oblast", "львів": "Lvivska oblast",
    "миколаїв": "Mykolaivska oblast", "одес": "Odeska oblast", "полтав": "Poltavska oblast",
    "рівнен": "Rivnenska oblast", "сумськ": "Sumska oblast", "тернопіл": "Ternopilska oblast",
    "харків": "Kharkivska oblast", "херсон": "Khersonska oblast", "хмельниц": "Khmelnytska oblast",
    "черкас": "Cherkaska oblast", "чернівец": "Chernivetska oblast", "чернігів": "Chernihivska oblast",
}


def ukrainian_to_region(text: str) -> str | None:
    t = text.lower().strip()
    if re.fullmatch(r"(м\.?\s*|місто\s+)?київ", t):
        return "Kyiv City"
    for stem, region in _UA_STEMS.items():
        if stem in t:
            return region
    return None


# TODO (disabled): alerts.in.ua live-status client. Disabled until it can be tested with a real API token (see README, To-do).
# def live_active_alerts(token: str, timeout: float = 10) -> dict[str, list[dict]]:
#     """OPTIONAL. Currently active air-raid alerts per dataset-region via the alerts.in.ua API.
#
#     Needs a personal token (https://alerts.in.ua/api-request). Endpoint and field names follow the
#     public docs at https://devs.alerts.in.ua/ (``/v1/alerts/active.json`` with a Bearer token) -
#     This function has not been tested against the live service with a real token. Returns {} on any failure so callers can carry on without live data.
#     """
#     try:
#         req = urllib.request.Request(
#             "https://api.alerts.in.ua/v1/alerts/active.json",
#             headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT},
#         )
#         with urllib.request.urlopen(req, timeout=timeout) as resp:
#             payload = json.loads(resp.read())
#     except Exception as exc:
#         print(f"  warning: live status unavailable ({type(exc).__name__}: {exc})")
#         return {}
#     return group_live_alerts(payload)
#
#
# def group_live_alerts(payload: dict) -> dict[str, list[dict]]:
#     out: dict[str, list[dict]] = {}
#     for a in payload.get("alerts", []):
#         if a.get("alert_type") != "air_raid":
#             continue
#         region = (ukrainian_to_region(str(a.get("location_oblast", "")))
#                   or ukrainian_to_region(str(a.get("location_title", ""))))
#         if region:
#             out.setdefault(region, []).append(a)
#     return out
