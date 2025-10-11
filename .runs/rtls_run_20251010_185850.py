#!/usr/bin/env python3
import sys, os, re, json, math, traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA; import numpy as _np; _FCA.tostring_rgb = getattr(_FCA,"tostring_rgb", lambda self: _np.asarray(self.buffer_rgba())[..., :3].tobytes())

# ------------------------ ROOT resolution & local imports ------------------------
ROOT = Path(os.environ.get("INFOZONE_ROOT", ""))
if not ROOT or not (ROOT / "guidelines.txt").exists():
    script_dir = Path(__file__).resolve().parent
    ROOT = script_dir if (script_dir / "guidelines.txt").exists() else script_dir.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GUIDELINES = ROOT / "guidelines.txt"
CONTEXT    = ROOT / "context.txt"
FLOORJSON  = ROOT / "floorplans.json"
LOGO       = ROOT / "redpoint_logo.png"
CONFIG     = ROOT / "report_config.json"
LIMITS_PY  = ROOT / "report_limits.py"
ZONES_JSON = ROOT / "zones.json"

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

# ------------------------ Helpers imports ------------------------
try:
    from extractor import extract_tracks
except Exception as e:
    print("Error Report:")
    print(f"Missing local helper extractor: {e}")
    raise SystemExit(1)

try:
    from zones_process import load_zones, compute_zone_intervals, sanitize_polygon_points, make_polygon
except Exception as e:
    print("Error Report:")
    print(f"Missing local helper zones_process: {e}")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception as e:
    print("Error Report:")
    print(f"Missing local helper pdf_creation_script: {e}")
    raise SystemExit(1)

try:
    from report_limits import apply_budgets, make_lite
except Exception:
    # Minimal fallbacks
    def apply_budgets(report, caps=None): return report
    def make_lite(report): 
        # drop figures after first 2 and trim tables
        out = dict(report); secs = []
        for s in report.get("sections", []):
            s2 = dict(s)
            if s2.get("type") == "charts":
                figs = s2.get("figures") or []
                s2["figures"] = figs[:2]
            if s2.get("type") == "table":
                data = s2.get("data") or []
                s2["data"] = data[:60]
            secs.append(s2)
        out["sections"] = secs
        return out

# ------------------------ CLI args ------------------------
if len(sys.argv) < 3:
    print("Error Report:")
    print("Usage: python generated.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
    raise SystemExit(1)

user_prompt = sys.argv[1]
csv_paths = [Path(p) for p in sys.argv[2:] if p]
first_csv = csv_paths[0]
out_dir = first_csv.parent.resolve()
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------ Load config & contexts ------------------------
config = {}
try:
    if CONFIG.exists():
        config = json.loads(read_text(CONFIG)) or {}
except Exception:
    config = {}
# apply some defaults if missing
cfg_overlay_subsample = int(config.get("overlay_subsample", 20000))
cfg_figsize_overlay = tuple(config.get("figsize_overlay", [9, 7]))
cfg_figsize_bar = tuple(config.get("figsize_bar", [7, 5]))
cfg_top_n = int(config.get("top_n", 10))
cfg_draw_zones = bool(config.get("draw_zones", True))
cfg_max_figures = int(config.get("max_figures", 12))

# ------------------------ Floorplan helpers ------------------------
def load_floorplan_world():
    try:
        if not FLOORJSON.exists():
            return None
        data = json.loads(read_text(FLOORJSON))
        fp = (data.get("floorplans") or data.get("plans") or data or [None])
        if isinstance(fp, list):
            fp = fp[0]
        if not isinstance(fp, dict):
            return None
        width  = float(fp.get("width", 0))
        height = float(fp.get("height", 0))
        x_c    = float(fp.get("image_offset_x", 0))
        y_c    = float(fp.get("image_offset_y", 0))
        image_scale = float(fp.get("image_scale", 0))
        scale = image_scale * 100.0  # mm/pixel
        x_min = (x_c - width/2.0)  * scale
        x_max = (x_c + width/2.0)  * scale
        y_min = (y_c - height/2.0) * scale
        y_max = (y_c + height/2.0) * scale
        # image file: prefer floorplan.jpeg (provided)
        img_path = None
        for name in ["floorplan.jpeg", "floorplan.jpg", "floorplan.png"]:
            p = ROOT / name
            if p.exists():
                img_path = p
                break
        if not img_path:
            return None
        img = plt.imread(str(img_path))
        return {"extent": (x_min, x_max, y_min, y_max), "image": img, "path": str(img_path)}
    except Exception:
        return None

def points_inside_polygon(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    try:
        from matplotlib.path import Path as _Path
        path = _Path(poly)
        pts = np.column_stack([xs, ys])
        return path.contains_points(pts)
    except Exception:
        # Fallback ray casting
        n = poly.shape[0]
        inside = np.zeros_like(xs, dtype=bool)
        xj, yj = poly[-1, 0], poly[-1, 1]
        for i in range(n):
            xi, yi = poly[i, 0], poly[i, 1]
            cond = ((yi > ys) != (yj > ys)) & (xs < (xj - xi) * (ys - yi) / (yj - yi + 1e-12) + xi)
            inside ^= cond
            xj, yj = xi, yi
        return inside

# ------------------------ Polygon parsing from user prompt ------------------------
def parse_user_polygon(prompt: str) -> Optional[np.ndarray]:
    # Find a bracketed list like [(x,y),(x,y),...]
    m = re.search(r"\[\s*\(?\s*\d", prompt)
    if m:
        start = m.start()
        text = prompt[start:]
        # Try to extract up to the matching closing bracket
        # Simple heuristic: take last ']' in string
        end = text.rfind("]")
        if end != -1:
            raw = text[: end + 1]
            try:
                import ast
                obj = ast.literal_eval(raw)
                # obj should be list-like of tuples/lists/dicts
                pts: List[Tuple[float,float]] = []
                if isinstance(obj, (list, tuple)):
                    for p in obj:
                        if isinstance(p, dict) and "x" in p and "y" in p:
                            pts.append((float(p["x"]), float(p["y"])))
                        elif isinstance(p, (list, tuple)) and len(p) >= 2:
                            pts.append((float(p[0]), float(p[1])))
                if pts:
                    poly = sanitize_polygon_points(pts)
                    if isinstance(poly, np.ndarray) and poly.shape[0] >= 3:
                        return poly
            except Exception:
                return None
    return None

user_poly = parse_user_polygon(user_prompt)
if user_poly is None or not isinstance(user_poly, np.ndarray) or user_poly.shape[0] < 3:
    print("Error Report:")
    print("Invalid or missing polygon points in user prompt.")
    raise SystemExit(1)

ad_hoc_zone_name = "User Area"

# ------------------------ Zones load (requested by user) ------------------------
zones_list = load_zones(ZONES_JSON, only_active=True) if ZONES_JSON.exists() else load_zones(None, only_active=True)
if not zones_list:
    print("Error Report:")
    print("Zones requested but no valid zones.json polygons were found.")
    raise SystemExit(1)

# ------------------------ Aggregation state ------------------------
agg_dwell_trade_zone: Dict[Tuple[str, str], float] = {}  # (trade, zone_name) -> seconds
ad_hoc_dwell_by_trade: Dict[str, float] = {}  # trade -> seconds
trades_set: set = set()
total_rows = 0
tmin: Optional[pd.Timestamp] = None
tmax: Optional[pd.Timestamp] = None
evidence_rows: List[Dict[str, str]] = []
overlay_points: List[Tuple[float, float, str]] = []  # (x,y,trade)
overlay_cap = max(1000, cfg_overlay_subsample)

# ------------------------ Processing loop (per-file) ------------------------
first_df_columns: Optional[List[str]] = None
for idx, csv_path in enumerate(csv_paths):
    try:
        raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
    except Exception as e:
        print("Error Report:")
        print(f"Ingestion failed: {e.__class__.__name__}: {e}")
        raise SystemExit(1)
    audit = raw.get("audit", {}) or {}
    if not audit.get("mac_map_loaded", False):
        print("Error Report:")
        print("Local trackable_objects.json not loaded; MAC→name mapping is required.")
        raise SystemExit(1)

    df = pd.DataFrame(raw.get("rows", []))
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Timestamp canon
    ts_src = df["ts_iso"] if "ts_iso" in df.columns else df.get("ts")
    df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

    if idx == 0:
        first_df_columns = list(df.columns)
        # Schema validation
        cols = set(c.lower() for c in df.columns.astype(str))
        identity_ok = ("trackable" in cols) or ("trackable_uid" in cols)
        trade_ok = ("trade" in cols)
        xy_ok = ("x" in cols and "y" in cols)
        if not (identity_ok and trade_ok and xy_ok):
            print("Error Report:")
            print("Missing required columns for analysis.")
            print(f"Columns detected: {','.join(df.columns.astype(str))}")
            raise SystemExit(1)

        # Collect evidence sample rows
        try:
            cols_e = ["trackable", "trade", "ts_short", "x", "y", "z"]
            cols_e = [c for c in cols_e if c in df.columns]
            if cols_e:
                e_rows = df[cols_e].head(50).fillna("").astype(str).to_dict(orient="records")
                evidence_rows.extend(e_rows)
        except Exception:
            pass

    # Update totals and time range
    total_rows += len(df)
    if "ts_utc" in df.columns:
        tnon = df["ts_utc"].dropna()
        if not tnon.empty:
            tmin = tnon.min() if tmin is None else min(tmin, tnon.min())
            tmax = tnon.max() if tmax is None else max(tmax, tnon.max())

    # Valid rows for zones/overlay
    need_cols = ["trackable_uid", "trade", "x", "y", "ts_utc"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan if c in ("x","y") else ""
    # cast numeric for x,y
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    valid = df.loc[df["ts_utc"].notna() & x.notna() & y.notna(), ["trackable_uid", "trade", "x", "y", "ts_utc", "trackable"]].copy()
    if valid.empty:
        continue

    # Map id -> dominant trade
    try:
        # prefer non-empty trade; use the most frequent per id
        valid["trade"] = valid["trade"].astype(str)
        id_trade = (valid[valid["trade"].str.len() > 0]
                    .groupby("trackable_uid")["trade"]
                    .agg(lambda s: s.value_counts().idxmax()))
        id_trade_map: Dict[str, str] = id_trade.to_dict()
    except Exception:
        id_trade_map = {}

    # Compute official zone intervals (skip 'TRAILER' zones)
    try:
        intervals = compute_zone_intervals(valid, zones_list, id_col="trackable_uid", ts_col="ts_utc", x_col="x", y_col="y")
    except Exception as e:
        print("Error Report:")
        print(f"Zone interval computation failed: {e.__class__.__name__}: {e}")
        raise SystemExit(1)

    for iv in intervals or []:
        zname = str(iv.get("zone_name", "") or iv.get("zone") or "")
        if not zname:
            continue
        # IGNORE TRAILER zones (unless explicitly requested, not the case here)
        if "trailer" in zname.lower():
            continue
        dur = iv.get("duration_sec", 0) or 0
        if dur and dur > 0:
            uid = str(iv.get("trackable_uid", "") or "")
            trade = id_trade_map.get(uid, "")
            if trade is None:
                trade = ""
            trades_set.add(trade or "unknown")
            key = (trade or "unknown", zname)
            agg_dwell_trade_zone[key] = agg_dwell_trade_zone.get(key, 0.0) + float(dur)

    # Ad-hoc polygon: compute intervals by creating a temporary zone and reusing engine
    try:
        adhoc_zone = make_polygon(ad_hoc_zone_name, user_poly.tolist())
    except Exception:
        adhoc_zone = None
    if adhoc_zone:
        try:
            iv2 = compute_zone_intervals(valid, [adhoc_zone], id_col="trackable_uid", ts_col="ts_utc", x_col="x", y_col="y")
        except Exception:
            iv2 = []
        for iv in iv2 or []:
            dur = iv.get("duration_sec", 0) or 0
            if dur and dur > 0:
                uid = str(iv.get("trackable_uid", "") or "")
                trade = id_trade_map.get(uid, "")
                if trade is None:
                    trade = ""
                ad_hoc_dwell_by_trade[trade or "unknown"] = ad_hoc_dwell_by_trade.get(trade or "unknown", 0.0) + float(dur)

    # Overlay reservoir: points inside ad-hoc polygon
    try:
        xs = pd.to_numeric(valid["x"], errors="coerce").to_numpy()
        ys = pd.to_numeric(valid["y"], errors="coerce").to_numpy()
        mask = points_inside_polygon(xs, ys, user_poly)
        if mask.any():
            inside = valid.loc[mask, ["x", "y", "trade"]].copy()
            # append to reservoir
            for _, r in inside.iterrows():
                overlay_points.append((float(r["x"]), float(r["y"]), str(r["trade"])))
            # Cap the reservoir by thinning if needed
            if len(overlay_points) > overlay_cap:
                # thin by step
                step = max(1, len(overlay_points) // overlay_cap)
                overlay_points = overlay_points[::step][:overlay_cap]
    except Exception:
        pass

    # Free memory early
    del df

# ------------------------ Build aggregates for charts ------------------------
# Per-trade zone dwell hours
per_trade_zone_hours: Dict[str, List[Tuple[str, float]]] = {}
for (trade, zone_name), sec in agg_dwell_trade_zone.items():
    hrs = float(sec) / 3600.0
    if hrs <= 0:
        continue
    lst = per_trade_zone_hours.get(trade, [])
    lst.append((zone_name, hrs))
    per_trade_zone_hours[trade] = lst

# Active trades set align with aggregates
active_trades = sorted([t for t in per_trade_zone_hours.keys() if any(h > 0 for _, h in per_trade_zone_hours[t])])
# Ensure ad-hoc dwell map has defaults
for t in active_trades:
    ad_hoc_dwell_by_trade.setdefault(t, 0.0)

# ------------------------ Figures ------------------------
figs: List[plt.Figure] = []
png_paths: List[Path] = []

# 1) Per-trade bar charts: top zones by dwell hours
for trade in active_trades:
    data = per_trade_zone_hours.get(trade, [])
    if not data:
        continue
    # sort desc and cap top N
    data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
    data_top = data_sorted[:cfg_top_n]
    if not data_top:
        continue
    zones = [z for z, _ in data_top]
    hours = [h for _, h in data_top]
    fig = plt.figure(figsize=cfg_figsize_bar)
    ax = fig.add_subplot(111)
    ax.barh(range(len(zones)), hours, color="#4E79A7")
    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels(zones, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Dwell (hours)")
    ax.set_title(f"Top Zones by Dwell — {trade}")
    fig.tight_layout()
    figs.append(fig)
    if len(figs) >= cfg_max_figures:
        break

# 2) Floorplan overlay: plot all points inside the ad-hoc polygon, colored by trade
fp = load_floorplan_world()
if fp is not None and overlay_points:
    fig2 = plt.figure(figsize=cfg_figsize_overlay)
    ax2 = fig2.add_subplot(111)
    x_min, x_max, y_min, y_max = fp["extent"]  # type: ignore
    ax2.imshow(fp["image"], extent=[x_min, x_max, y_min, y_max], origin="upper")
    # draw polygon
    ax2.add_patch(Polygon(user_poly, closed=True, facecolor=(1,0,0,0.12), edgecolor=(1,0,0,0.75), linewidth=1.2))
    # scatter points by trade
    trades_in_overlay = list(dict.fromkeys([t for _, _, t in overlay_points]))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = {t: cmap(i % 10) for i, t in enumerate(trades_in_overlay)}
    # thin if too many points
    pts = overlay_points
    if len(pts) > overlay_cap:
        step = max(1, len(pts) // overlay_cap)
        pts = pts[::step]
    for t in trades_in_overlay:
        xs = [x for (x, y, tt) in pts if tt == t]
        ys = [y for (x, y, tt) in pts if tt == t]
        if not xs:
            continue
        ax2.scatter(xs, ys, s=8, alpha=0.8, color=colors.get(t), label=str(t))
    # legend if suitable
    handles, labels = ax2.get_legend_handles_labels()
    if len(labels) <= 12 and len(labels) > 0:
        ax2.legend(loc="upper left", fontsize=8, frameon=True)
    # axes
    mx = 0.10
    xr = (x_max - x_min); yr = (y_max - y_min)
    ax2.set_xlim(x_min - mx * xr, x_max + mx * xr)
    ax2.set_ylim(y_min - mx * yr, y_max + mx * yr)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X (mm)"); ax2.set_ylabel("Y (mm)")
    ax2.set_title("Floorplan — Points Inside User Area")
    fig2.tight_layout()
    figs.append(fig2)

# ------------------------ Tables ------------------------
sections: List[Dict[str, Any]] = []

# Summary section with at least one bullet
date_range = ""
if tmin is not None and tmax is not None:
    # keep Z out of duplicated suffix, only UTC ISO
    date_range = f"{tmin.strftime('%Y-%m-%d %H:%M UTC')} → {tmax.strftime('%Y-%m-%d %H:%M UTC')}"
else:
    date_range = "No valid timestamps"

bullets = [
    f"Analyzed {len(csv_paths)} file(s), {total_rows} samples across {date_range}.",
    f"Computed busiest zones by trade over the week (hours).",
    f"Ad-hoc area analyzed with polygon of {user_poly.shape[0]} points.",
]
sections.append({"type": "summary", "title": "Summary", "bullets": bullets})

# Evidence table (cap 50 rows)
if evidence_rows:
    cols_e = list(evidence_rows[0].keys())
    sections.append({"type":"table","title":"Evidence (first 50 samples from first file)","data":evidence_rows,"headers":cols_e,"rows_per_page":24})

# Ad-hoc area dwell by trade table (hours)
adhoc_rows: List[Dict[str, Any]] = []
for t in sorted(ad_hoc_dwell_by_trade.keys()):
    hrs = float(ad_hoc_dwell_by_trade.get(t, 0.0) / 3600.0)
    adhoc_rows.append({"trade": t, "hours_in_area": f"{hrs:.2f}"})
if adhoc_rows:
    sections.append({"type":"table","title":f"Time Spent in User Area ({ad_hoc_zone_name}) — Hours by Trade","data":adhoc_rows,"headers":["trade","hours_in_area"],"rows_per_page":24})

# ------------------------ Save figures to PNGs ------------------------
report_date = (tmin.strftime("%Y%m%d") + "_" + tmax.strftime("%Y%m%d")) if (tmin is not None and tmax is not None) else "undated"
png_paths = []
for i, fig in enumerate([f for f in figs if getattr(f, "savefig", None)], start=1):
    png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
    try:
        fig.savefig(str(png), dpi=120)
        png_paths.append(png)
    except Exception:
        # continue without adding path
        pass

# Add charts section if we have figures
live_figs = [f for f in figs if getattr(f, "savefig", None)]
if live_figs:
    sections.append({"type": "charts", "title": "Figures", "figures": live_figs})

# ------------------------ Build report dict ------------------------
title = "Walmart Renovation — Zones by Trade (Weekly, Hours)"
meta_lines = []
meta_lines.append(f"CSV files: {len(csv_paths)}")
if date_range:
    meta_lines.append(f"Window: {date_range}")
meta_lines.append(f"Zones file: {ZONES_JSON.name if ZONES_JSON.exists() else 'none'}")
meta = " | ".join(meta_lines)

report: Dict[str, Any] = {
    "title": title,
    "meta": meta,
    "sections": sections
}

# Apply budgets
try:
    report = apply_budgets(report)
except Exception:
    pass

# Ensure out_dir exists
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"

# ------------------------ Build PDF with fallback ------------------------
try:
    safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
except Exception as e:
    print("Error Report:")
    print(f"PDF build failed: {e.__class__.__name__}: {e}")
    traceback.print_exc(limit=2)
    try:
        report_lite = make_lite(report)
        safe_build_pdf(report_lite, str(pdf_path), logo_path=str(LOGO))
    except Exception as e2:
        print("Error Report:")
        print(f"Lite PDF failed: {e2.__class__.__name__}: {e2}")
        traceback.print_exc(limit=2)
        raise SystemExit(1)
finally:
    # Only close figures after PDF build attempt
    try:
        plt.close("all")
    except Exception:
        pass

# ------------------------ Print links (success only) ------------------------
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

print(f"[Download the PDF]({file_uri(pdf_path)})")
for i, pth in enumerate(png_paths, 1):
    print(f"[Download Plot {i}]({file_uri(pth)})")