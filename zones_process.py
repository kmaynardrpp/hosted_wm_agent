# zones_process.py
# ------------------------------------------------------------------------------
# Zone processing utilities with robust handling for ad-hoc polygons, duplicate
# timestamps, and noisy data — **without any downsampling**.
#
# LOCAL-ONLY CHANGES (aligned with extractor.py patch):
# - Remove all implicit /mnt/data references. Resolve files from the local
#   project ROOT (INFOZONE_ROOT → this file’s parent → CWD).
# - load_zones() searches LOCAL paths by default (ROOT/zones.json, CWD/zones.json)
#   unless an explicit path is provided.
# - compute_zone_intervals() defaults to ts_col="ts_utc" to match the pipeline.
# - Duplicate timestamps per id are handled by **keeping the last row** at each
#   timestamp before classification.
# - Vectorized point-in-polygon using matplotlib.path.Path when available, with
#   a bbox prefilter for speed.
# - resample_sec is accepted for backward compatibility but **ignored**.
#
# APIs
# - load_zones(zones_path: str|Path|None=None, only_active=True) -> List[zone]
# - compute_zone_intervals(df, zones, id_col="trackable_uid", ts_col="ts_utc",
#                          x_col="x", y_col="y", resample_sec=None) -> List[dict]
# - summarize_zones(intervals) -> {"zone_totals":[...], "per_tag":[...]}
# - occupancy_over_time(intervals, freq="1T") -> List[dict]
# - make_polygon(name, points) -> zone dict (sanitized polygon)
# - dwell_in_polygon(df, points, name="AdHoc Area", **kwargs)
#   -> {"intervals":[...], "summary":{...}}
#
# Expected df columns (minimum): ts_utc (UTC string or datetime-like), x, y (mm),
# and an id_col (default "trackable_uid").
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

# ---------- ROOT resolution (LOCAL; no /mnt/data) ----------
def _resolve_root() -> Path:
    root = Path(os.environ.get("INFOZONE_ROOT", "")).resolve()
    if not root or not root.exists():
        root = Path(__file__).resolve().parent
    if not root or not root.exists():
        root = Path.cwd().resolve()
    return root

ROOT = _resolve_root()

# Fast, C-backed point-in-polygon
try:
    from matplotlib.path import Path as _Path
    _HAVE_MPL_PATH = True
except Exception:  # pragma: no cover
    _HAVE_MPL_PATH = False


# ---------------------------- Zones IO ----------------------------
def _coerce_list_or_first(obj: Any) -> Any:
    """Return obj if dict-like, else first element if list-like, else obj."""
    if isinstance(obj, list):
        return obj[0] if obj else None
    return obj

def _to_float_xy_list(positions: Any) -> np.ndarray:
    """Convert JSON positions [{x,y}, ...] or [[x,y], ...] into float ndarray Nx2."""
    out: List[List[float]] = []
    if isinstance(positions, list):
        for p in positions:
            try:
                if isinstance(p, dict):
                    out.append([float(p["x"]), float(p["y"])])
                else:
                    out.append([float(p[0]), float(p[1])])
            except Exception:
                continue
    return np.asarray(out, dtype=float) if out else np.empty((0, 2), dtype=float)

def load_zones(zones_path: str | Path | None = None,
               only_active: bool = True) -> List[Dict[str, Any]]:
    """
    Load polygons from a local zones.json.

    Search order when zones_path is None:
      1) ROOT / "zones.json"
      2) CWD  / "zones.json"

    Expected JSON shapes:
      { "zones": [ { "name":..., "uid":..., "active":bool, "zone_geometry": { "positions":[{x,y},...] } }, ... ] }
      or a compatible variant.

    Returns a list of dicts: {name, uid, polygon (Nx2 ndarray), active}
    """
    candidates: List[Path] = []
    if zones_path:
        candidates.append(Path(zones_path))
    candidates.extend([ROOT / "zones.json", Path.cwd() / "zones.json"])

    src: Optional[Path] = None
    for cand in candidates:
        if cand.exists():
            src = cand
            break
    if not src:
        return []

    try:
        data = json.loads(src.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []

    zones_list = data.get("zones")
    if not isinstance(zones_list, list):
        return []

    out: List[Dict[str, Any]] = []
    for z in zones_list:
        if only_active and not z.get("active", False):
            continue
        geom = z.get("zone_geometry") or {}
        pos = geom.get("positions") or []
        poly = _to_float_xy_list(pos)
        if poly.shape[0] < 3:
            continue
        out.append({
            "name": str(z.get("name", "") or ""),
            "uid":  str(z.get("uid", "") or ""),
            "polygon": poly,
            "active": bool(z.get("active", False)),
        })
    return out


# ---------------------------- Polygon sanitation ----------------------------
def _to_xy_tuple(p: Any) -> Optional[Tuple[float, float]]:
    """Accept (x,y) tuple/list or {'x':..,'y':..} dict; return (float, float) or None."""
    try:
        if isinstance(p, dict):
            return (float(p["x"]), float(p["y"]))
        return (float(p[0]), float(p[1]))  # type: ignore[index]
    except Exception:
        return None

def sanitize_polygon_points(points: Iterable[Any]) -> np.ndarray:
    """
    Convert point specs to float tuples, remove consecutive duplicates and
    a closing duplicate (last==first). Require ≥ 3 unique points.
    """
    raw: List[Tuple[float, float]] = []
    for p in points or []:
        xy = _to_xy_tuple(p)
        if xy is not None:
            raw.append(xy)
    if not raw:
        return np.empty((0, 2), dtype=float)

    uniq: List[Tuple[float, float]] = []
    prev = None
    for xy in raw:
        if prev is None or xy != prev:
            uniq.append(xy)
        prev = xy
    if len(uniq) >= 2 and uniq[0] == uniq[-1]:
        uniq = uniq[:-1]

    uniq2: List[Tuple[float, float]] = []
    seen = set()
    for xy in uniq:
        if xy not in seen:
            uniq2.append(xy)
            seen.add(xy)

    if len(uniq2) < 3:
        return np.empty((0, 2), dtype=float)
    return np.array(uniq2, dtype=float)

def make_polygon(name: str, points: Iterable[Any]) -> Optional[Dict[str, Any]]:
    """Create a zone-like dict from arbitrary point specs; None if invalid."""
    poly = sanitize_polygon_points(points)
    if poly.size == 0:
        return None
    return {"name": str(name or "AdHoc Area"), "uid": "", "polygon": poly, "active": True}


# ---------------------------- Vectorized point-in-polygon ----------------------------
def _poly_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    return float(np.min(poly[:, 0])), float(np.max(poly[:, 0])), float(np.min(poly[:, 1])), float(np.max(poly[:, 1]))

def _bbox_mask(xs: np.ndarray, ys: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bbox
    return (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

def _contains_points(poly: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorized contains-points with bbox prefilter."""
    pts = np.column_stack((xs, ys))
    if _HAVE_MPL_PATH:
        return _Path(poly).contains_points(pts)
    # Fallback vectorized ray casting
    n = poly.shape[0]
    x = pts[:, 0]; y = pts[:, 1]
    inside = np.zeros(x.shape, dtype=bool)
    xj, yj = poly[-1, 0], poly[-1, 1]
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        xj, yj = xi, yi
    return inside


# ---------------------------- Intervals (NO DOWNSAMPLING) ----------------------------
def compute_zone_intervals(df: pd.DataFrame,
                           zones: List[Dict[str, Any]],
                           id_col: str = "trackable_uid",
                           ts_col: str = "ts_utc",
                           x_col: str = "x",
                           y_col: str = "y",
                           resample_sec: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Build per-tag intervals inside any of the polygons in `zones`.

    No downsampling is performed. `resample_sec` is accepted for backward
    compatibility but **ignored**.

    Steps:
      - Parse timestamps to UTC; drop rows with invalid ts/x/y.
      - Sort by (id, ts).
      - Within each id, drop duplicate timestamps (keep last).
      - Vectorized polygon classification with bbox prefilter.
      - Compress consecutive equal-zone runs to intervals.
    """
    if df.empty or not zones:
        return []

    use = df.copy()

    # Parse timestamps and clean coordinates
    use[ts_col] = pd.to_datetime(use[ts_col], errors="coerce", utc=True)
    use = use.dropna(subset=[ts_col])
    use[x_col] = pd.to_numeric(use[x_col], errors="coerce")
    use[y_col] = pd.to_numeric(use[y_col], errors="coerce")
    use = use.dropna(subset=[x_col, y_col])

    # Stable order
    use = use.sort_values([id_col, ts_col])

    # Prepare polygons (with bbox) once
    poly_info: List[Tuple[str, np.ndarray, Tuple[float, float, float, float]]] = []
    for z in zones:
        poly = z.get("polygon")
        if poly is None:
            continue
        poly = np.asarray(poly, dtype=float)
        if poly.shape[0] < 3:
            continue
        poly_info.append((z.get("name") or "", poly, _poly_bbox(poly)))
    if not poly_info:
        return []

    intervals: List[Dict[str, Any]] = []

    for tag, g0 in use.groupby(id_col, sort=False):
        # Drop exact duplicate timestamps (keep last)
        g = g0[~g0[ts_col].duplicated(keep="last")].copy()
        if g.empty:
            continue

        xs = g[x_col].to_numpy(dtype=float, copy=False)
        ys = g[y_col].to_numpy(dtype=float, copy=False)

        # First matching polygon wins
        labels = np.array([""] * len(g), dtype=object)
        for zname, poly, bbox in poly_info:
            mb = _bbox_mask(xs, ys, bbox)
            if not mb.any():
                continue
            inside = np.zeros(len(g), dtype=bool)
            inside[mb] = _contains_points(poly, xs[mb], ys[mb])
            labels = np.where((labels == "") & inside, zname, labels)

        g = g.assign(_zone=labels)

        # Compress consecutive runs
        prev = None
        enter = None
        tname = ""
        if "trackable" in g.columns:
            vals = g["trackable"].astype(str)
            tname = next((v for v in vals if v.strip()), "")

        times = g[ts_col].to_list()
        zlist = g["_zone"].to_list()
        for t, zname in zip(times, zlist):
            if prev is None:
                prev, enter = zname, t
                continue
            if zname != prev:
                if prev:
                    dur = (t - enter).total_seconds()
                    intervals.append({
                        "trackable_uid": tag,
                        "trackable": tname,
                        "zone_name": prev,
                        "enter_ts": enter.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "leave_ts": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "duration_sec": max(0, int(dur))
                    })
                prev, enter = zname, t

        # close tail
        if prev and enter is not None and len(times) > 0:
            last_t = times[-1]
            dur = (last_t - enter).total_seconds()
            intervals.append({
                "trackable_uid": tag,
                "trackable": tname,
                "zone_name": prev,
                "enter_ts": enter.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "leave_ts": last_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_sec": max(0, int(dur))
            })

    return intervals


# ---------------------------- Summaries ----------------------------
def summarize_zones(intervals: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not intervals:
        return {"zone_totals": [], "per_tag": []}
    df = pd.DataFrame(intervals)
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0).astype(int)
    zsum = (df.groupby("zone_name")["duration_sec"].sum()
              .sort_values(ascending=False).reset_index())
    uniq = df.groupby("zone_name")["trackable_uid"].nunique().reset_index(name="unique_tags")
    ztot = pd.merge(zsum, uniq, on="zone_name", how="left")
    per_tag = (df.groupby(["trackable_uid","trackable","zone_name"])["duration_sec"].sum()
                 .reset_index().sort_values(["trackable_uid","duration_sec"], ascending=[True, False]))
    return {"zone_totals": ztot.to_dict(orient="records"),
            "per_tag": per_tag.to_dict(orient="records")}

def occupancy_over_time(intervals: List[Dict[str, Any]], freq: str = "1T") -> List[Dict[str, Any]]:
    if not intervals:
        return []
    df = pd.DataFrame(intervals)
    df["enter_ts"] = pd.to_datetime(df["enter_ts"], utc=True)
    df["leave_ts"] = pd.to_datetime(df["leave_ts"], utc=True)

    rows = []
    for _, r in df.iterrows():
        rows.append({"ts": r["enter_ts"], "zone_name": r["zone_name"], "delta": 1})
        rows.append({"ts": r["leave_ts"], "zone_name": r["zone_name"], "delta": -1})
    tl = pd.DataFrame(rows).sort_values("ts")

    out: List[Dict[str, Any]] = []
    for z, g in tl.groupby("zone_name"):
        s = g.set_index("ts")["delta"].resample(freq).sum().fillna(0).cumsum().clip(lower=0)
        for t, v in s.items():
            out.append({"ts": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "zone_name": z, "count": int(v)})
    return out


# ---------------------------- Ad-hoc polygon dwell (NO DOWNSAMPLING) ----------------------------
def dwell_in_polygon(df: pd.DataFrame,
                     points: Iterable[Any],
                     name: str = "AdHoc Area",
                     id_col: str = "trackable_uid",
                     ts_col: str = "ts_utc",
                     x_col: str = "x",
                     y_col: str = "y",
                     resample_sec: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute dwell inside a user-specified polygon **without downsampling**.
    `resample_sec` is accepted for compatibility but **ignored**.
    """
    zone = make_polygon(name, points)
    if zone is None:
        return {"intervals": [], "summary": {"zone_totals": [], "per_tag": []}}
    intervals = compute_zone_intervals(df, [zone], id_col=id_col, ts_col=ts_col,
                                       x_col=x_col, y_col=y_col, resample_sec=None)
    return {"intervals": intervals, "summary": summarize_zones(intervals)}
