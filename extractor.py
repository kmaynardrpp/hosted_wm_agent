# ------------------------------------------------------------------------------
# extractor.py
#
# Local, MAC/UID-aware extractor for Walmart RTLS positions (x, y, z).
# Populates:
#   - trackable / trackable_uid via local trackable_objects.json (MAC and UID)
#   - trade from FINAL trackable (regex/prefix canonicalization)
#   - ts_utc (ISO 'Z'), ts_iso (alias), ts_short ("MM-DD HH:MM")
#   - zone_name normalization to canonical labels ("Sales Floor", "Breakroom", "Receiving", "FET", ...),
#     with polygon-based classification when only UIDs are present; then filter Trailer rows.
#
# Returns: {"rows": List[dict], "audit": dict}
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================== ROOT resolution (LOCAL) ===========================
def _resolve_root() -> Path:
    v = os.environ.get("INFOZONE_ROOT", "").strip()
    if v:
        p = Path(v).resolve()
        if p.exists():
            return p
    p = Path(__file__).resolve().parent
    return p if p.exists() else Path.cwd().resolve()

ROOT = _resolve_root()
EXTRACTOR_SIGNATURE = "extractor/v8-mac-uid+zone-canon+polyclass-no-trailer"

# ============================== MAC/UID map (LOCAL) ============================
def _norm_mac(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^0-9a-f]", "", str(s).lower())

def _load_maps(mac_map_path: Optional[str | os.PathLike] = None) -> tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    candidates: List[Path] = []
    if mac_map_path:
        candidates.append(Path(mac_map_path))
    candidates.extend([ROOT / "trackable_objects.json", Path.cwd() / "trackable_objects.json"])

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

# ============================ Zone normalization (STRICT) =====================
# Canonical string mapping (contains checks, case-insensitive)
_ZONE_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(100\s+gr\s+)?sales\s+floor\b", re.I), "Sales Floor"),
    (re.compile(r"\bvestibule\b", re.I), "Vestibule"),
    (re.compile(r"\bfet\b", re.I), "FET"),
    (re.compile(r"\bpersonnel\b", re.I), "Personnel"),
    (re.compile(r"\bpharmacy\b", re.I), "Pharmacy"),
    (re.compile(r"\b(restroom|customer\s+restroom)s?\b", re.I), "Restroom"),
    (re.compile(r"\bbreak\s*room\b|\bbreakroom\b", re.I), "Breakroom"),
    (re.compile(r"\bpickup\b", re.I), "Pickup"),
    (re.compile(r"\bdeli\b", re.I), "Deli"),
    (re.compile(r"\breceiving\b", re.I), "Receiving"),
    (re.compile(r"\btrailer\b", re.I), "Trailer"),
]
_ZONE_FALLBACK_STOP = {
    "zone","area","aisle","hall","hallway","bay","dock","office",
    "department","dept","section","room","floor","gm","gr","store",
    "backroom","front","front-end","pickup","storage","information","active","inactive"
}
def _desired_zone_display(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    # direct canonical match
    for pat, display in _ZONE_MAP:
        if pat.search(s):
            return display
    # strip "Zone <num> - " and numeric noise
    tail = re.sub(r"^\s*zone\s+\d+(\.\d+)?\s*-\s*", "", s, flags=re.I)
    tail = re.sub(r"^\s*(\d{1,4}(\.\d+)?(\s+[A-Z]{2,})?\s*)+", "", tail).strip()
    if not tail:
        return ""
    joined = " ".join(re.findall(r"[A-Za-z]+", tail)).title()
    if "Sales" in joined and "Floor" in joined:
        return "Sales Floor"
    for target in ["Vestibule","Personnel","Pharmacy","Breakroom","Receiving","Deli","Restroom","Pickup","FET"]:
        if re.search(fr"\b{target}\b", joined, flags=re.I):
            return target
    words = [w for w in re.findall(r"[A-Za-z]+", tail) if w.lower() not in _ZONE_FALLBACK_STOP]
    return (words[-1].title() if words else tail.title())

# UID→name lookup from zones.json (if UIDs match)
def _load_zone_name_lookup(zones_path: Optional[str | os.PathLike] = None) -> Dict[str, str]:
    candidates: List[Path] = []
    if zones_path:
        candidates.append(Path(zones_path))
    candidates.extend([ROOT / "zones.json", Path.cwd() / "zones.json"])
    for p in candidates:
        try:
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            zones = data.get("zones") or []
            mp: Dict[str, str] = {}
            for z in zones:
                uid = str(z.get("uid", "") or "")
                name = str(z.get("name", "") or "")
                if uid and name:
                    mp[uid] = name
            if mp:
                return mp
        except Exception:
            continue
    return {}

# Polygon-based classification fallback (when only UIDs or nothing is present)
def _classify_by_polygon(df: pd.DataFrame,
                         zones_path: Optional[str | os.PathLike] = None) -> pd.Series:
    """
    Return a Series of canonical zone display names by classifying (x,y)
    into polygons from zones.json. Empty string if no hit.
    """
    # Lazy import; zones_process is already in repo
    try:
        from zones_process import load_zones
        import numpy as _np
        try:
            from matplotlib.path import Path as _MPPath  # fast point-in-poly
            have_path = True
        except Exception:
            have_path = False
    except Exception:
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    zones = load_zones(zones_path or None, only_active=True)
    if not zones:
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    # Prepare points
    xs = pd.to_numeric(df.get("x", ""), errors="coerce")
    ys = pd.to_numeric(df.get("y", ""), errors="coerce")
    mask = xs.notna() & ys.notna()
    out = pd.Series([""] * len(df), index=df.index, dtype="object")
    if not mask.any():
        return out

    pts = _np.c_[xs[mask].to_numpy(), ys[mask].to_numpy()]

    if have_path:
        # Iterate zones; first hit wins
        for z in zones:
            poly = z.get("polygon")
            if poly is None or len(poly) < 3:
                continue
            path = _MPPath(poly)
            inside = path.contains_points(pts)
            if inside.any():
                name = _desired_zone_display(z.get("name", ""))
                out.loc[mask.index[mask].to_numpy()[inside]] = name
    else:
        # Fallback: simple ray-casting (slower)
        def _pip(point, poly):
            x, y = point
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
                    inside = not inside
            return inside
        for idx, (xv, yv) in zip(mask.index[mask], pts):
            for z in zones:
                poly = z.get("polygon")
                if poly is None or len(poly) < 3:
                    continue
                if _pip((xv, yv), poly):
                    out.loc[idx] = _desired_zone_display(z.get("name", ""))
                    break

    return out

# ============================ CSV reading helpers =============================
def _read_csv_comma(csv_path: str, row_limit: Optional[int]) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                csv_path, sep=",", header=0, dtype=str, engine="python",
                on_bad_lines="skip", comment="#", encoding=enc, na_filter=False,
            ); break
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

_MAC_SHAPED = re.compile(r"^(?:[0-9A-Fa-f]{2}([:-]?)){5}[0-9A-Fa-f]{2}$|^[0-9A-Fa-f]{12}$")

def _guess_mac_col_by_pattern(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        s = df[c].astype(str)
        s = s[s.str.len() > 0].head(4000)
        if s.empty:
            continue
        hit = (s.str.match(_MAC_SHAPED)).mean()
        if hit >= 0.05:
            return c
    return None

# ================================ Public API ==================================
def extract_tracks(csv_path: str,
                   mac_map_path: Optional[str | os.PathLike] = None,
                   row_limit: Optional[int] = None,
                   **kwargs) -> dict:
    df = _read_csv_comma(csv_path, row_limit=row_limit)

    # Ensure string dtype
    for c in df.columns:
        if not pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str)

    # Identify columns
    mac_aliases = ["mac","mac_address","macaddr","bt_mac","ble_mac","bluetooth_mac","bdaddr","btaddr","addr","address","bleaddress","ble_address"]
    col_mac  = _find_col(df, mac_aliases) or _guess_mac_col_by_pattern(df)

    uid_aliases = ["trackable_uid","uid","user_id","device_id","tag_id","entity_id","badge_id","tag_uid"]
    col_uidc = _find_col(df, uid_aliases)
    ts_aliases  = ["ts_utc","ts","timestamp","time","datetime","date_time"]
    col_ts   = _find_col(df, ts_aliases)
    # Zone-like source (textual or UID-ish)
    col_zone_src = _find_col(df, ["zone_name","zone","area","area_name","region","location","area_uid"])

    # Canonical outputs
    if col_uidc is None:
        df["trackable_uid"] = ""
        col_uidc = "trackable_uid"

    col_trade = _find_col(df, ["trade","role","craft","discipline"]) or "trade"
    if col_trade not in df.columns:
        df[col_trade] = ""

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
            df.loc[mask, "ts_short"] = ts_parsed[mask].dt.strftime("%m-%d %H:%M")

    # Load maps (LOCAL)
    mac_map, uid_map = _load_maps(mac_map_path)

    # Normalized keys
    mac_norm = df[col_mac].map(_norm_mac) if col_mac else pd.Series([""] * len(df), index=df.index)
    uid_vals = df[col_uidc].astype(str) if col_uidc else pd.Series([""] * len(df), index=df.index)

    # Choose a safe trackable name source
    def _choose_trackable_source(cols: List[str], zone_present: bool) -> Optional[str]:
        lower = {c.lower(): c for c in cols}
        exact_pref = ["trackable","trackable_name","device_name","tag_name","asset_name","display_name"]
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
            if lc.endswith("_name") and not any(b in lc for b in ("zone","area","region","id","uid")):
                return c
        return None

    col_trackable_src = _choose_trackable_source(list(df.columns), zone_present=(col_zone_src is not None))

    # Pull trackable name/uid from maps if missing
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

    if col_trackable_src and not any(k in col_trackable_src.lower() for k in ("zone","area","region","id","uid")):
        cur = df[col_trackable_src].astype(str).str.strip()
    else:
        cur = pd.Series([""] * len(df), index=df.index, dtype="object")
    candidate = cur.where(cur.ne(""), mapped_name_mac)
    candidate = candidate.where(candidate.ne(""), mapped_name_uid)
    df["trackable"] = candidate.fillna("")

    curu = df[col_uidc].astype(str).str.strip()
    fillu = curu.where(curu.ne(""), mapped_uid_mac)
    fillu = fillu.where(fillu.ne(""), mapped_uid_uid)
    df[col_uidc] = fillu.fillna("")

    # Trade inference for blank/unknown
    tcur = df[col_trade].astype(str).str.strip()
    need = tcur.eq("") | tcur.str.lower().eq("unknown")
    if need.any():
        df.loc[need, col_trade] = df.loc[need, "trackable"].map(_infer_trade_from_trackable).fillna("")

    # ---- Zone normalization with UID + polygon fallback; then drop Trailer ----
    removed_trailer = 0
    df["zone_name"] = ""  # canonical output field
    if col_zone_src:
        raw_zone = df[col_zone_src].astype(str).fillna("")
        uid2name = _load_zone_name_lookup()  # may be empty if uids differ
        UIDISH = re.compile(r"^[A-Za-z0-9_\-]{16,}$")  # looks like a UID (no spaces, long)

        # First pass: direct text normalize or UID→name normalize if UID matches zones.json
        def _first_pass(s: str) -> str:
            if not s:
                return ""
            if s in uid2name:
                return _desired_zone_display(uid2name[s])
            if UIDISH.match(s) and (" " not in s):
                # looks like a UID we don't know → leave blank for polygon classification
                return ""
            return _desired_zone_display(s)

        zone1 = raw_zone.map(_first_pass)
        df["zone_name"] = zone1

        # Second pass: polygon classification for anything still blank
        still_blank = df["zone_name"].eq("")
        if still_blank.any():
            zone_poly = _classify_by_polygon(df.loc[still_blank, ["x","y"]])
            df.loc[still_blank, "zone_name"] = zone_poly

        # Final: drop Trailer rows
        mask_trailer = df["zone_name"].str.contains(r"\bTrailer\b", case=False, na=False)
        removed_trailer = int(mask_trailer.sum())
        if removed_trailer:
            df = df.loc[~mask_trailer].copy()

    # Rows → list-of-dicts
    out_rows = df.to_dict(orient="records")

    # Audit
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
        "rows_removed_trailer": removed_trailer,
        "columns_detected": list(map(str, df.columns.tolist())),
        "mac_col_selected": col_mac or "",
        "trade_nonempty_rate": round(trade_nonempty_rate, 4),
        "notes": "zone_name normalized (UID-aware, polygon fallback); Trailer rows removed; trackable from safe/MAC/UID; ts_utc canonical.",
    }

    return {"rows": out_rows, "audit": audit}
