# extraction.py
"""
Fast extractor for the new RTLS CSV schema (positions-first).

CSV columns (case-insensitive):
  trackable, trackable_uid, mac, ts (timestamp), x, y, z

Speed features
- Reads only needed columns with C engine, memory_map, and no NA coercion.
- Vectorized MAC→(name, uid) mapping via /mnt/data/trackable_objects.json.
- Timestamp "zero-parse" path (strict_ts=False): build ts_iso/ts_short with
  vectorized slice/regex instead of pandas.to_datetime (orders-of-magnitude faster).
- Optional row_limit and chunked preview for huge files.

Outputs
- rows: strings-only records safe for table rendering (no large dtype conversions here).
- audit: ingestion metadata.

Public API
  extract_tracks(csv_path: str,
                 mac_map_path: str="/mnt/data/trackable_objects.json",
                 row_limit: int | None = None,
                 strict_ts: bool = False) -> dict
"""

from __future__ import annotations
import io
import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# ---------------------------- Config ----------------------------
ENCODING_TRIES = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "utf-16", "utf-16le"]
CANON_COLS = ["trackable", "trackable_uid", "mac", "ts", "x", "y", "z"]
USECOLS = CANON_COLS  # keep exact order; we will rename/fill if missing
READ_OPTS = dict(
    sep=",",
    header=0,
    dtype=str,
    engine="c",
    quotechar='"',
    keep_default_na=False,
    na_filter=False,
    low_memory=True,
    memory_map=True,
)

ISO_Z_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[ T](?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2})(?:\.\d{1,6})?)?(Z|[+-]\d{2}:?\d{2})?$"
)
DATE_FALLBACK_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2}).*?(\d{2}):(\d{2})")

# ---------------------------- IO helpers ----------------------------
def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _decode_bytes(b: bytes) -> Tuple[str, str]:
    for enc in ENCODING_TRIES:
        try:
            return b.decode(enc), enc
        except Exception:
            continue
    return b.decode("latin-1", errors="replace"), "latin-1"

def _preamble_len(text: str, scan: int = 300) -> int:
    pre = 0
    for line in text.splitlines()[:scan]:
        s = line.lstrip()
        if s.startswith("#") or s == "":
            pre += 1
        else:
            break
    return pre

def _read_csv_fast(path: str, row_limit: Optional[int]) -> Tuple[pd.DataFrame, str, int, str]:
    raw = _read_bytes(path)
    text, encoding = _decode_bytes(raw)
    pre = _preamble_len(text)

    # usecols restricts memory & speed; if csv misses some, pandas will raise.
    # We’ll relax by reading all, then subset+create missing.
    engine = "c"
    try:
        df = pd.read_csv(io.StringIO(text), skiprows=pre, usecols=None, **READ_OPTS)
    except Exception:
        engine = "python"
        df = pd.read_csv(io.StringIO(text), skiprows=pre, usecols=None, engine="python",
                         **{k:v for k,v in READ_OPTS.items() if k not in ("engine","memory_map")})

    # Subset to canonical columns (and create blanks for missing)
    low = {c.lower().strip(): c for c in df.columns}
    rename_map = {}
    for want in CANON_COLS:
        if want in low:
            rename_map[low[want]] = want
        else:
            # fuzzy: allow minor variants
            for c in df.columns:
                lc = c.lower().strip()
                if want == "trackable_uid" and lc in ("trackable_uid","trackableuid","uid","id"):
                    rename_map[c] = "trackable_uid"; break
                if want == "mac" and lc in ("mac","mac_address","tag_mac","device_mac","node_mac","bt_mac"):
                    rename_map[c] = "mac"; break
                if want == "ts" and lc in ("ts","time","timestamp","date_time","datetime","occurred_at"):
                    rename_map[c] = "ts"; break
    if rename_map:
        df = df.rename(columns=rename_map)

    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = ""

    df = df[CANON_COLS]

    # Row limiting (for huge files / previews)
    if row_limit is not None and len(df) > row_limit:
        df = df.iloc[:row_limit].copy()

    return df, encoding, pre, engine

# ---------------------------- MAC mapping ----------------------------
def _normalize_mac(s: str) -> str:
    return re.sub(r"[^0-9a-f]", "", str(s or "").lower())

def _load_mac_map(path: str = "/mnt/data/trackable_objects.json") -> Dict[str, Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        arr = data.get("trackable_objects", [])
        out: Dict[str, Dict[str, str]] = {}
        for obj in arr:
            mac = (obj.get("mac_address")
                   or (obj.get("node") or {}).get("mac_address")
                   or "")
            k = _normalize_mac(mac)
            if not k: continue
            out[k] = {
                "name": (obj.get("name") or obj.get("display_name") or "").strip(),
                "uid":  (obj.get("uid") or (obj.get("node") or {}).get("uid") or obj.get("assoc_uid") or "").strip()
            }
        return out
    except Exception:
        return {}

# ---------------------------- Trade from final trackable ----------------------------
TRADE_PATTERNS = [
    (re.compile(r"\bcarpentr(y|y_\d+)\b|\bcarpenter\b", re.I), "carpentry"),
    (re.compile(r"\bpainter\b|\bpainting\b", re.I), "painter"),
    (re.compile(r"\bsteel\b", re.I), "steel"),
    (re.compile(r"\bmechanical\b|\bhvac\b", re.I), "mechanical"),
    (re.compile(r"\bclean(ing|er)?\b", re.I), "cleaning"),
    (re.compile(r"\brefrig(eration)?\b|\brefrig\b", re.I), "refrigeration"),
    (re.compile(r"\bstrip(ing|e)?\b", re.I), "striping"),
    (re.compile(r"\broof(ing)?\b", re.I), "roofing"),
    (re.compile(r"\bfloor(ing)?\b", re.I), "flooring"),
    (re.compile(r"\bconcrete\b", re.I), "concrete"),
    (re.compile(r"\b(building\s*signage|signage)\b", re.I), "building_signage"),
    (re.compile(r"\bfire\s*sprink(ler)?\b|\bfire\s*spk\b", re.I), "fire_sprinkler"),
    (re.compile(r"\bems\b", re.I), "ems"),
    (re.compile(r"\bplumb(ing|er)?\b", re.I), "plumbing"),
    (re.compile(r"\belectric(ians?|ian)?\b|\belec\b", re.I), "electrician"),
]
def _infer_trade_from_trackable(trackable: str) -> str:
    s = str(trackable or "").strip().lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    for pat, canon in TRADE_PATTERNS:
        if pat.search(s): return canon
    return ""

# ---------------------------- Timestamp helpers (fast) ----------------------------
def _ts_iso_fast(series: pd.Series) -> pd.Series:
    # If matches ISO-ish, normalize to YYYY-MM-DDTHH:MM:SSZ by slicing where possible.
    s = series.astype(str)
    # vectorized slice for the common form 'YYYY-MM-DDTHH:MM'
    base = s.str.slice(0, 16)
    # append ':00Z' when seconds/zone missing
    out = base.where(base.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"), other="")
    out = out.mask(out == "", s)  # keep original when slice didn't match
    # try regex normalize to ensure Z suffix
    def norm(v: str) -> str:
        m = ISO_Z_RE.match(v)
        if not m: return ""
        return f"{m.group('y')}-{m.group('m')}-{m.group('d')}T{m.group('H')}:{m.group('M')}:{m.group('S') or '00'}Z"
    return out.map(norm)

def _ts_short_fast(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    # Try vector slice MM-DD + '\n' + HH:MM
    short = s.str.slice(5, 10) + "\n" + s.str.slice(11, 16)
    # Validate: if month/day not digits, fallback regex
    ok = short.str.match(r"^\d{2}-\d{2}\n\d{2}:\d{2}$")
    if ok.all(): return short
    def fb(v: str) -> str:
        m = DATE_FALLBACK_RE.search(v)
        if m: return f"{m.group(2)}-{m.group(3)}\n{m.group(4)}:{m.group(5)}"
        return ""
    return s.map(fb)

def _ts_iso_strict(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True, infer_datetime_format=True)
    return dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")

def _ts_short_strict(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True, infer_datetime_format=True)
    return (dt.dt.strftime("%m-%d\n%H:%M")).fillna("")

# ---------------------------- Public API ----------------------------
def extract_tracks(csv_path: str,
                   mac_map_path: str = "/mnt/data/trackable_objects.json",
                   row_limit: Optional[int] = None,
                   strict_ts: bool = False) -> Dict[str, Any]:
    """
    Fast ingestion for (trackable, trackable_uid, mac, ts, x, y, z).
    - row_limit: read only the first N rows (for previews/very large files).
    - strict_ts: True → pandas.to_datetime parsing; False (default) → slice/regex fast path.

    Returns: {"columns":[...], "rows":[...], "audit":{...}}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df, enc, pre, engine = _read_csv_fast(csv_path, row_limit)

    # MAC mapping
    df["mac_norm"] = df["mac"].map(lambda m: re.sub(r"[^0-9a-f]", "", str(m or "").lower()))
    mac_map = _load_mac_map(mac_map_path)
    df["mac_name"] = df["mac_norm"].map(lambda k: mac_map.get(k, {}).get("name", ""))
    df["mac_uid"]  = df["mac_norm"].map(lambda k: mac_map.get(k, {}).get("uid", ""))

    # Prefer CSV values; fill from MAC map if empty
    def _coalesce(a: str, b: str) -> str:
        a = str(a or "").strip(); b = str(b or "").strip()
        return a if a else b

    df["trackable"]     = [ _coalesce(a, b) for a, b in zip(df["trackable"], df["mac_name"]) ]
    df["trackable_uid"] = [ _coalesce(a, b) for a, b in zip(df["trackable_uid"], df["mac_uid"]) ]

    # Trade from final trackable
    df["trade"] = df["trackable"].map(_infer_trade_from_trackable)

    # Timestamps (fast or strict)
    if strict_ts:
        df["ts_iso"]   = _ts_iso_strict(df["ts"])
        df["ts_short"] = _ts_short_strict(df["ts"])
    else:
        df["ts_iso"]   = _ts_iso_fast(df["ts"])
        df["ts_short"] = _ts_short_fast(df["ts"])

    # NOTE: Keep x,y,z as strings (faster). Downstream code can cast for math.
    # Build strings-only rows
    out_cols = ["trackable","trackable_uid","trade","mac","ts","ts_iso","ts_short","x","y","z"]
    rows: List[Dict[str, str]] = []
    view = df[out_cols] if set(out_cols).issubset(df.columns) else df.assign(**{c:"" for c in out_cols})[out_cols]
    for _, r in view.iterrows():
        row = {c: ("" if pd.isna(r[c]) else str(r[c])) for c in out_cols}
        rows.append(row)

    audit = {
        "encoding_used": enc,
        "engine_used": engine,
        "delimiter_used": ",",
        "preamble_lines_skipped": pre,
        "rows_returned": len(rows),
        "columns_detected": out_cols,
        "mac_map_loaded": bool(mac_map),
        "strict_ts": bool(strict_ts),
        "row_limit": row_limit if row_limit is not None else "all"
    }
    return {"columns": out_cols, "rows": rows, "audit": audit}
