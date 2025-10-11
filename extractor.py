# ------------------------------------------------------------------------------
# extractor.py
#
# Local, MAC/UID-aware extractor for Walmart RTLS positions (x, y, z).
# Populates:
#   - trackable / trackable_uid via local trackable_objects.json (MAC and UID)
#   - trade from FINAL trackable (regex/prefix canonicalization)
#   - ts_utc (ISO 'Z'), ts_iso (alias), ts_short ("MM-DDHH:MM")
#
# Returns: {"rows": List[dict], "audit": dict}
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# =========================== ROOT resolution (LOCAL) ===========================

def _resolve_root() -> Path:
    root_env = os.environ.get("INFOZONE_ROOT", "").strip()
    root = Path(root_env).resolve() if root_env else None
    if not root or not root.exists():
        root = Path(__file__).resolve().parent
    if not root or not root.exists():
        root = Path.cwd().resolve()
    return root


ROOT = _resolve_root()
EXTRACTOR_SIGNATURE = "extractor/v4-mac+uid-safe-name"


# ============================== MAC/UID map (LOCAL) ============================

def _norm_mac(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^0-9a-f]", "", str(s).lower())


def _load_maps(mac_map_path: Optional[str | os.PathLike] = None) -> tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Returns (mac_map, uid_map) where keys are normalized MAC or raw UID.
    Accepts either:
      { "trackable_objects": [ {mac_address,name,uid}, ... ] }  OR
      { mac_or_uid: { name, uid }, ... }    # keyed dict
    """
    candidates: List[Path] = []
    if mac_map_path:
        candidates.append(Path(mac_map_path))
    candidates.extend([
        ROOT / "trackable_objects.json",
        Path.cwd() / "trackable_objects.json",
    ])

    mac_map: Dict[str, Dict[str, str]] = {}
    uid_map: Dict[str, Dict[str, str]] = {}

    for p in candidates:
        try:
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            items = data.get("trackable_objects") or data.get("objects") or data

            if isinstance(items, list):
                for it in items:
                    mac = _norm_mac(it.get("mac") or it.get("mac_address"))
                    uid = str(it.get("uid", "") or it.get("id", "") or "")
                    name = str(it.get("name", "") or it.get("display_name", "") or "")
                    if mac:
                        mac_map[mac] = {"name": name, "uid": uid}
                    if uid:
                        uid_map[uid] = {"name": name, "uid": uid}
            elif isinstance(items, dict):
                for k, v in items.items():
                    if not isinstance(v, dict):
                        continue
                    mac = _norm_mac(k)
                    uid = str(v.get("uid", "") or v.get("id", "") or "")
                    name = str(v.get("name", "") or v.get("display_name", "") or "")
                    if mac:
                        mac_map[mac] = {"name": name, "uid": uid}
                    if uid:
                        uid_map[uid] = {"name": name, "uid": uid}
            # Stop at the first file that gave us data
            if mac_map or uid_map:
                break
        except Exception:
            continue

    return mac_map, uid_map


# ============================== Trade inference ===============================

_TRADE_REGEX = [
    (re.compile(r"\bcarp(enter|entry|entry_\d+)?\b|carpentr(y)?", re.I), "carpentry"),
    (re.compile(r"\belectric(ians?|ian)?\b|\belec\b|\belectric\b", re.I), "electrician"),
    (re.compile(r"\bplumb(ing|er)?\b", re.I), "plumbing"),
    (re.compile(r"\bpainter\b|\bpainting\b|\bpaint\b", re.I), "painter"),
    (re.compile(r"\bmechanical\b|\bhvac\b|\bmech\b", re.I), "mechanical"),
    (re.compile(r"\bclean(ing|er)?\b|\bclean\b", re.I), "cleaning"),
    (re.compile(r"\brefrig(eration)?\b|\brefrig\b", re.I), "refrigeration"),
    (re.compile(r"\bstrip(ing|e)?\b|\bstripe\b", re.I), "striping"),
    (re.compile(r"\broof(ing)?\b|\broof\b", re.I), "roofing"),
    (re.compile(r"\bfloor(ing)?\b|\bfloor\b", re.I), "flooring"),
    (re.compile(r"\bconcrete\b", re.I), "concrete"),
    (re.compile(r"\b(building\s*signage|blgd\s*signage|signage)\b|\bbuilding\b", re.I), "building_signage"),
    (re.compile(r"\bfire\s*sprink(ler)?\b|\bfire\s*spk\b|\bfire\b|\bsprinkler\b", re.I), "fire_sprinkler"),
    (re.compile(r"\bems\b", re.I), "ems"),
    (re.compile(r"\bsteel\b", re.I), "steel"),
]

_TRADE_PREFIX_MAP = {
    "carpentry":"carpentry","carpenter":"carpentry","carp":"carpentry",
    "electrician":"electrician","electric":"electrician","elec":"electrician",
    "plumbing":"plumbing","plumber":"plumbing","plumb":"plumbing",
    "painter":"painter","painting":"painter","paint":"painter",
    "mechanical":"mechanical","mech":"mechanical","hvac":"mechanical",
    "cleaning":"cleaning","cleaner":"cleaning","clean":"cleaning",
    "refrigeration":"refrigeration","refrig":"refrigeration",
    "striping":"striping","stripe":"striping",
    "roofing":"roofing","roof":"roofing",
    "flooring":"flooring","floor":"flooring",
    "concrete":"concrete",
    "signage":"building_signage","building":"building_signage","building_signage":"building_signage","blgd":"building_signage",
    "fire":"fire_sprinkler","sprinkler":"fire_sprinkler","fire_sprinkler":"fire_sprinkler","fire_spk":"fire_sprinkler",
    "ems":"ems",
    "steel":"steel",
}

def _infer_trade_from_trackable(label: str) -> str:
    if not label:
        return ""
    s = str(label).replace("_", " ").replace("-", " ").strip()
    prefix = re.split(r"\s+", s, maxsplit=1)[0].lower()
    if prefix in _TRADE_PREFIX_MAP:
        return _TRADE_PREFIX_MAP[prefix]
    for pat, canon in _TRADE_REGEX:
        if pat.search(s):
            return canon
    return ""


# ============================ CSV reading helpers =============================

def _read_csv_comma(csv_path: str, row_limit: Optional[int]) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                csv_path,
                sep=",",
                header=0,
                dtype=str,
                engine="python",
                on_bad_lines="skip",
                comment="#",
                encoding=enc,
                na_filter=False,
            )
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(csv_path, sep=None, header=0, dtype=str, engine="python",
                         on_bad_lines="skip", comment="#", na_filter=False)

    if row_limit and len(df) > row_limit:
        df = df.iloc[:row_limit, :].copy()

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = lower.get(name.lower())
        if c:
            return c
    for c in df.columns:
        lc = c.lower()
        for name in candidates:
            if name.lower() in lc:
                return c
    return None


def _choose_trackable_source(cols: List[str], zone_present: bool) -> Optional[str]:
    """
    Choose a safe *name* column.
    - Prefer exact name fields.
    - Never pick columns containing 'zone', 'area', 'region', 'id', 'uid'.
    - Only accept bare 'name' if no zone-like column exists.
    """
    lower = {c.lower(): c for c in cols}
    exact_pref = [
        "trackable", "trackable_name", "device_name", "tag_name",
        "asset_name", "display_name"
    ]
    for nm in exact_pref:
        c = lower.get(nm)
        if c:
            return c
    if not zone_present:
        c = lower.get("name")
        if c:
            return c
    for c in cols:
        lc = c.lower()
        if lc.endswith("_name") and not any(b in lc for b in ("zone", "area", "region", "id", "uid")):
            return c
    return None


# ================================ Public API ==================================

def extract_tracks(csv_path: str,
                   mac_map_path: Optional[str | os.PathLike] = None,
                   row_limit: Optional[int] = None,
                   **kwargs) -> dict:
    """
    Read a positions CSV and return {"rows": [...], "audit": {...}}.
    """
    df = _read_csv_comma(csv_path, row_limit=row_limit)

    # Ensure string dtype
    for c in df.columns:
        if not pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str)

    # Identify columns
    col_zone = _find_col(df, ["zone_name", "zone", "area", "region"])
    col_mac  = _find_col(df, ["mac", "mac_address", "macaddr", "bt_mac"])
    col_uidc = _find_col(df, ["trackable_uid", "uid", "user_id", "device_id", "tag_id"])
    col_ts   = _find_col(df, ["ts_utc", "ts", "timestamp", "time", "datetime", "date_time"])

    # Canonical outputs
    if col_uidc is None:
        df["trackable_uid"] = ""
        col_uidc = "trackable_uid"

    col_trade = _find_col(df, ["trade", "role", "craft", "discipline"])
    if col_trade is None:
        df["trade"] = ""
        col_trade = "trade"

    # Timestamps
    df["ts_utc"] = ""
    df["ts_iso"] = ""
    df["ts_short"] = ""
    if col_ts:
        ts_parsed = pd.to_datetime(df[col_ts], utc=True, errors="coerce")
        mask = ts_parsed.notna()
        if mask.any():
            iso = ts_parsed[mask].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            df.loc[mask, "ts_utc"] = iso
            df.loc[mask, "ts_iso"] = iso
            df.loc[mask, "ts_short"] = ts_parsed[mask].dt.strftime("%m-%d|%H:%M")

    # Load maps (LOCAL)
    mac_map, uid_map = _load_maps(mac_map_path)

    # Normalized keys
    mac_norm = df[col_mac].map(_norm_mac) if col_mac else pd.Series([""] * len(df), index=df.index)
    uid_vals = df[col_uidc].astype(str) if col_uidc else pd.Series([""] * len(df), index=df.index)

    # Safe source column for names (never zone/area/region/id/uid)
    col_trackable_src = _choose_trackable_source(list(df.columns), zone_present=(col_zone is not None))

    # Build mapped name/uid from MAC and UID
    def _map_from_mac(uid_series=False) -> pd.Series:
        vals = []
        for mac in mac_norm.tolist():
            m = mac_map.get(mac) if mac else None
            vals.append((m.get("uid") if uid_series else m.get("name")) if m else "")
        return pd.Series(vals, index=df.index, dtype="object")

    def _map_from_uid(uid_series=False) -> pd.Series:
        vals = []
        for u in uid_vals.tolist():
            m = uid_map.get(u) if u else None
            vals.append((m.get("uid") if uid_series else m.get("name")) if m else "")
        return pd.Series(vals, index=df.index, dtype="object")

    mapped_name_mac = _map_from_mac(uid_series=False)
    mapped_uid_mac  = _map_from_mac(uid_series=True)
    mapped_name_uid = _map_from_uid(uid_series=False)
    mapped_uid_uid  = _map_from_uid(uid_series=True)

    # Final canonical 'trackable':
    # priority = safe source column (non-blank) → MAC map → UID map
    if col_trackable_src and not any(k in col_trackable_src.lower() for k in ("zone", "area", "region", "id", "uid")):
        cur = df[col_trackable_src].astype(str).str.strip()
    else:
        cur = pd.Series([""] * len(df), index=df.index, dtype="object")

    candidate = cur.where(cur.ne(""), mapped_name_mac)
    candidate = candidate.where(candidate.ne(""), mapped_name_uid)
    df["trackable"] = candidate.fillna("")

    # Ensure UID column: prefer existing, else fill from MAC→uid, then UID map
    curu = df[col_uidc].astype(str).str.strip()
    fillu = curu.where(curu.ne(""), mapped_uid_mac)
    fillu = fillu.where(fillu.ne(""), mapped_uid_uid)
    df[col_uidc] = fillu.fillna("")

    # Trade inference for blank/unknown
    tcur = df[col_trade].astype(str).str.strip()
    need = tcur.eq("") | tcur.str.lower().eq("unknown")
    if need.any():
        df.loc[need, col_trade] = df.loc[need, "trackable"].map(_infer_trade_from_trackable).fillna("")

    # Rows + audit
    out_rows = df.to_dict(orient="records")
    macs_seen = int((mac_norm != "").sum())
    mac_hits = int(sum(1 for m in mac_norm if m and m in mac_map))
    uids_seen = int((uid_vals.astype(bool)).sum())
    uid_hits = int(sum(1 for u in uid_vals if u and u in uid_map))
    trade_nonempty_rate = float((df[col_trade].astype(str).str.strip() != "").mean()) if len(df) else 0.0

    audit = {
        "source_file": str(csv_path),
        "root": str(ROOT),
        "extractor_signature": EXTRACTOR_SIGNATURE,
        "mac_map_loaded": bool(mac_map),
        "uid_map_loaded": bool(uid_map),
        "macs_seen": macs_seen,
        "mac_hits": mac_hits,
        "uids_seen": uids_seen,
        "uid_hits": uid_hits,
        "rows_total": int(len(df)),
        "columns_detected": list(map(str, df.columns.tolist())),
        "trade_nonempty_rate": round(trade_nonempty_rate, 4),
        "notes": "trackable from safe source or MAC/UID maps; trade inferred from final trackable; ts_utc is canonical.",
    }

    return {"rows": out_rows, "audit": audit}
