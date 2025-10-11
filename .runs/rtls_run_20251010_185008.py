```python
import sys, os
from pathlib import Path

# ========================== ROOT resolution & local imports ==========================
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

# =========================== Matplotlib 3.9+ tostring_rgb shim =======================
try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
    import numpy as _np
    _FCA.tostring_rgb = getattr(_FCA,"tostring_rgb", lambda self: _np.asarray(self.buffer_rgba())[..., :3].tobytes())
except Exception:
    pass

# ====================================== Imports =====================================
import json
import math
import traceback
import argparse
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from extractor import extract_tracks
from zones_process import load_zones, compute_zone_intervals, dwell_in_polygon, sanitize_polygon_points, make_polygon
from pdf_creation_script import safe_build_pdf
from report_limits import apply_budgets, make_lite

# =============================== Helpers & Config Load ===============================
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def load_config():
    cfg = {
        "prefer_floorplan": True,
        "floorplan_margin": 0.10,
        "overlay_point_size": 8,
        "overlay_alpha": 0.85,
        "overlay_color_by": "trade",
        "overlay_subsample": 20000,
        "draw_trails": False,
        "trail_seconds": 900,
        "draw_zones": True,
        "zone_face_alpha": 0.20,
        "zone_edge_alpha": 0.65,
        "top_n": 10,
        "max_figures": 12,
        "figsize_overlay": (9, 7),
        "figsize_bar": (7, 5)
    }
    try:
        if CONFIG.exists():
            data = json.loads(read_text(CONFIG) or "{}")
            for k, v in data.items():
                cfg[k] = v
    except Exception:
        pass
    return cfg

def parse_args():
    # The CLI will call: python generated.py "<USER_PROMPT>" /abs/csv1 [/abs/csv2 ...]
    if len(sys.argv) < 3:
        print("Error Report:")
        print("Expected: python script.py \"<USER_PROMPT>\" <csv1> [<csv2> ...]")
        raise SystemExit(1)
    user_prompt = sys.argv[1]
    csv_paths = [Path(p) for p in sys.argv[2:]]
    return user_prompt, csv_paths

def get_out_dir(csv_paths):
    first_dir = csv_paths[0].parent
    out_dir = first_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def load_floorplan_assets():
    # Load extent from floorplans.json and image from ROOT/floorplan.jpeg (or .jpg/.png fallback)
    extent = None
    img = None
    try:
        if FLOORJSON.exists():
            data = json.loads(read_text(FLOORJSON) or "{}")
            fp = (data.get("floorplans") or data.get("plans") or data or [None])
            if isinstance(fp, list):
                fp = fp[0]
            if isinstance(fp, dict):
                width  = float(fp.get("width", 0.0))
                height = float(fp.get("height", 0.0))
                x_c    = float(fp.get("image_offset_x", 0.0))
                y_c    = float(fp.get("image_offset_y", 0.0))
                image_scale = float(fp.get("image_scale", 0.0))  # meters/pixel
                scale = image_scale * 100.0  # mm/pixel
                x_min = (x_c - width/2.0)  * scale
                x_max = (x_c + width/2.0)  * scale
                y_min = (y_c - height/2.0) * scale
                y_max = (y_c + height/2.0) * scale
                extent = (x_min, x_max, y_min, y_max)
        # image lookup
        for name in ["floorplan.jpeg", "floorplan.jpg", "floorplan.png"]:
            p = ROOT / name
            if p.exists():
                try:
                    img = plt.imread(str(p))
                    break
                except Exception:
                    img = None
    except Exception:
        extent, img = None, None
    return extent, img

def ts_range_str(min_ts: pd.Timestamp | None, max_ts: pd.Timestamp | None) -> str:
    def fmt(t):
        if t is None or pd.isna(t):
            return "n/a"
        # Ensure UTC ISO without double Z
        try:
            return t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            try:
                if getattr(t, "tzinfo", None) is not None:
                    return t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                pass
            try:
                return pd.to_datetime(t, utc=True).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                return str(t)
    return f"{fmt(min_ts)} to {fmt(max_ts)}"

def sanitize_user_polygon_from_prompt(prompt: str) -> list:
    # Extract the first bracketed list and try literal_eval
    import ast, re
    points = []
    try:
        m = re.search(r"\[.*\]", prompt, re.S)
        if m:
            pts = ast.literal_eval(m.group(0))
            # normalize
            def _to_xy(p):
                if isinstance(p, dict):
                    return (float(p.get("x")), float(p.get("y")))
                x, y = p
                return (float(x), float(y))
            points = [_to_xy(p) for p in pts]
    except Exception:
        points = []
    return points

def validate_schema(df: pd.DataFrame, zones_requested: bool, zones_available: bool) -> bool:
    cols = set(df.columns.astype(str).str.lower())
    identity_ok = ("trackable" in cols) or ("trackable_uid" in cols)
    trade_ok = ("trade" in cols)
    pos_ok = ("x" in cols) and ("y" in cols)
    zone_ok = True
    if zones_requested:
        zone_ok = ("zone_name" in cols) or zones_available
    if not (identity_ok and trade_ok and pos_ok and zone_ok):
        print("Error Report:")
        print("Missing required columns for analysis.")
        print(f"Columns detected: {','.join(df.columns.astype(str))}")
        return False
    return True

def legend_if_labeled(ax):
    handles, labels = ax.get_legend_handles_labels()
    if len([l for l in labels if l]) <= 12 and len([h for h in handles if h]) > 0:
        ax.legend(loc="best", frameon=True, fontsize=8)

# =============================== Main Processing =====================================
def main():
    user_prompt, csv_paths = parse_args()
    out_dir = get_out_dir(csv_paths)
    cfg = load_config()
    overlay_cap = int(cfg.get("overlay_subsample", 20000) or 20000)

    # Zones requested? (user said: "zones" and polygon dwell)
    zones_requested = True

    # Load zones.json polygons (active only)
    zones_list = load_zones(str(ZONES_JSON), only_active=True) if ZONES_JSON.exists() else load_zones(None, only_active=True)

    # Prepare accumulators
    dwell_by_trade_zone_hours: dict[str, dict[str, float]] = {}  # trade -> {zone_name: hours}
    ad_hoc_dwell_by_trade_hours: dict[str, float] = {}           # for the polygon area
    polygon_points: list[tuple[float, float, str]] = []          # (x,y,trade)
    polygon_point_count = 0

    # Parse ad-hoc polygon points from prompt
    user_poly_raw = sanitize_user_polygon_from_prompt(user_prompt)
    poly_np = sanitize_polygon_points(user_poly_raw)
    ad_hoc_polygon = [(float(x), float(y)) for x, y in (poly_np.tolist() if poly_np.size else [])]

    # Validate polygon if required by user; if invalid, we will still proceed with zone charts but skip polygon-based analytics gracefully
    polygon_valid = len(ad_hoc_polygon) >= 3

    # First file schema validation, and mac map check
    first_checked = False
    any_rows = False
    global_min_ts = None
    global_max_ts = None

    # For bars, we will need zone display names; from zones_list we have names
    # Build mapping for quick trailer detection
    def is_trailer_zone(name: str) -> bool:
        return "trailer" in str(name or "").lower()

    # Per-file processing (large-data mode)
    for csv_path in csv_paths:
        try:
            raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
        except Exception as e:
            print("Error Report:")
            print(f"Failed to read: {csv_path}")
            raise SystemExit(1)

        audit = raw.get("audit", {}) if isinstance(raw, dict) else {}
        if not audit or not audit.get("mac_map_loaded", False):
            print("Error Report:")
            print("MAC→Trackable map not loaded; cannot proceed.")
            raise SystemExit(1)

        rows = raw.get("rows", []) if isinstance(raw, dict) else []
        df = pd.DataFrame(rows)
        # Duplicate-name guard
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # If no rows, continue
        if df.empty:
            continue
        any_rows = True

        # Timestamp canon
        src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
        df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

        # Global min/max
        try:
            mn = df["ts_utc"].min()
            mx = df["ts_utc"].max()
            if pd.notna(mn):
                global_min_ts = mn if global_min_ts is None else min(global_min_ts, mn)
            if pd.notna(mx):
                global_max_ts = mx if global_max_ts is None else max(global_max_ts, mx)
        except Exception:
            pass

        # Early schema validation (after first file only)
        if not first_checked:
            zones_available = len(zones_list) > 0
            if not validate_schema(df, zones_requested=zones_requested, zones_available=zones_available or ("zone_name" in df.columns)):
                raise SystemExit(1)
            first_checked = True

        # Build id -> trade mapping for this file
        trade_map = {}
        if "trackable_uid" in df.columns and "trade" in df.columns:
            for uid, s in df.groupby("trackable_uid")["trade"]:
                # choose first non-empty trade or blank
                tvals = [str(v) for v in s.values if str(v).strip() != ""]
                trade_map[str(uid)] = (tvals[0].strip().lower() if tvals else "unknown")
        elif "trackable" in df.columns and "trade" in df.columns:
            for name, s in df.groupby("trackable")["trade"]:
                tvals = [str(v) for v in s.values if str(v).strip() != ""]
                trade_map[str(name)] = (tvals[0].strip().lower() if tvals else "unknown")

        # Normalize x,y numeric for any spatial work
        if "x" in df.columns and "y" in df.columns:
            df["x"] = pd.to_numeric(df["x"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")

        # -------------------- ZONES DWELL (by trade) --------------------
        # If df has zone_name, approximate dwell by time deltas within same zone; else compute via polygons
        if "zone_name" in df.columns:
            # Keep only rows with valid ts and id
            id_col = "trackable_uid" if "trackable_uid" in df.columns else ("trackable" if "trackable" in df.columns else None)
            if id_col:
                use = df.loc[df["ts_utc"].notna() & df[id_col].notna(), [id_col, "ts_utc", "zone_name"]].copy()
                use["zone_name"] = use["zone_name"].astype(str)
                # Exclude trailer zones
                use = use.loc[~use["zone_name"].str.lower().str.contains("trailer"), :]
                if not use.empty:
                    use = use.sort_values([id_col, "ts_utc"])
                    next_ts = use.groupby(id_col)["ts_utc"].shift(-1)
                    dur = (next_ts - use["ts_utc"]).dt.total_seconds().fillna(0)
                    dur = dur.clip(lower=0, upper=3600*4)  # cap large gaps at 4h
                    use["dur_sec"] = dur
                    # Attach trade
                    def map_trade(v):
                        return trade_map.get(str(v), "unknown")
                    use["trade"] = use[id_col].map(map_trade).fillna("unknown").astype(str)
                    # Aggregate per trade-zone
                    grp = use.groupby(["trade", "zone_name"], as_index=False)["dur_sec"].sum()
                    for _, row in grp.iterrows():
                        trade = str(row["trade"] or "unknown").lower()
                        zone = str(row["zone_name"] or "")
                        hrs = float(row["dur_sec"]) / 3600.0
                        if hrs <= 0:
                            continue
                        if trade not in dwell_by_trade_zone_hours:
                            dwell_by_trade_zone_hours[trade] = {}
                        dwell_by_trade_zone_hours[trade][zone] = dwell_by_trade_zone_hours[trade].get(zone, 0.0) + hrs
        else:
            # Compute intervals with zones_process (no downsampling)
            try:
                valid_idx = df["ts_utc"].notna() & df["x"].notna() & df["y"].notna()
                use = df.loc[valid_idx, ["trackable_uid", "trackable", "ts_utc", "x", "y"]].copy()
                # Ensure id_col present
                if "trackable_uid" not in use.columns:
                    # Without trackable_uid, we cannot compute according to contract
                    print("Error Report:")
                    print("Missing required columns for analysis.")
                    print(f"Columns detected: {','.join(df.columns.astype(str))}")
                    raise SystemExit(1)
                if not use.empty and len(zones_list) > 0:
                    intervals = compute_zone_intervals(use, zones_list, id_col="trackable_uid", ts_col="ts_utc", x_col="x", y_col="y")
                    # intervals: list of dicts with keys: trackable_uid, trackable, zone_name, enter_ts, leave_ts, duration_sec
                    if intervals:
                        # Build into DataFrame for grouping
                        iz = pd.DataFrame(intervals)
                        if not iz.empty:
                            iz["zone_name"] = iz["zone_name"].astype(str)
                            # Exclude trailer zones
                            iz = iz.loc[~iz["zone_name"].str.lower().str.contains("trailer"), :]
                            iz["duration_sec"] = pd.to_numeric(iz["duration_sec"], errors="coerce").fillna(0.0)
                            # Attach trade via trade_map
                            def map_trade_uid(v):
                                return trade_map.get(str(v), "unknown")
                            iz["trade"] = iz["trackable_uid"].map(map_trade_uid).fillna("unknown").astype(str)
                            grp = iz.groupby(["trade", "zone_name"], as_index=False)["duration_sec"].sum()
                            for _, row in grp.iterrows():
                                trade = str(row["trade"] or "unknown").lower()
                                zone = str(row["zone_name"] or "")
                                hrs = float(row["duration_sec"]) / 3600.0
                                if hrs <= 0:
                                    continue
                                if trade not in dwell_by_trade_zone_hours:
                                    dwell_by_trade_zone_hours[trade] = {}
                                dwell_by_trade_zone_hours[trade][zone] = dwell_by_trade_zone_hours[trade].get(zone, 0.0) + hrs
            except Exception:
                # If zones requested and we cannot compute, error out per contract
                print("Error Report:")
                print("Zones polygons missing/invalid or compute_zone_intervals failed.")
                print(f"Columns detected: {','.join(df.columns.astype(str))}")
                raise SystemExit(1)

        # -------------------- Ad-hoc polygon dwell + points inside --------------------
        if polygon_valid:
            try:
                valid_idx = df["ts_utc"].notna() & df["x"].notna() & df["y"].notna()
                use = df.loc[valid_idx, ["trackable_uid", "ts_utc", "x", "y", "trade"]].copy()
                if not use.empty:
                    # Dwell via helper
                    adhoc = dwell_in_polygon(use.rename(columns={"trackable_uid": "trackable_uid"}), ad_hoc_polygon, name="User Area", id_col="trackable_uid", ts_col="ts_utc", x_col="x", y_col="y")
                    intervals = adhoc.get("intervals", [])
                    if intervals:
                        ia = pd.DataFrame(intervals)
                        if not ia.empty:
                            ia["duration_sec"] = pd.to_numeric(ia["duration_sec"], errors="coerce").fillna(0.0)
                            # Attach trade via trade_map
                            def map_trade_uid(v):
                                return trade_map.get(str(v), "unknown")
                            ia["trade"] = ia["trackable_uid"].map(map_trade_uid).fillna("unknown").astype(str)
                            grp = ia.groupby("trade", as_index=False)["duration_sec"].sum()
                            for _, row in grp.iterrows():
                                trade = str(row["trade"] or "unknown").lower()
                                hrs = float(row["duration_sec"]) / 3600.0
                                if hrs <= 0:
                                    continue
                                ad_hoc_dwell_by_trade_hours[trade] = ad_hoc_dwell_by_trade_hours.get(trade, 0.0) + hrs

                    # Points inside polygon for overlay (bounded reservoir)
                    xs = use["x"].to_numpy(dtype=float)
                    ys = use["y"].to_numpy(dtype=float)
                    # Bbox filter
                    poly_arr = np.array(ad_hoc_polygon, dtype=float)
                    xmin, xmax = float(np.min(poly_arr[:, 0])), float(np.max(poly_arr[:, 0]))
                    ymin, ymax = float(np.min(poly_arr[:, 1])), float(np.max(poly_arr[:, 1]))
                    bbox_mask = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
                    if np.any(bbox_mask):
                        try:
                            from matplotlib.path import Path as _Path
                            pts = np.column_stack((xs[bbox_mask], ys[bbox_mask]))
                            inside_mask = _Path(poly_arr).contains_points(pts)
                            if inside_mask.any():
                                inside_idx = np.nonzero(bbox_mask)[0][inside_mask]
                                # reservoir cap
                                for idx in inside_idx:
                                    if polygon_point_count < overlay_cap:
                                        polygon_points.append((float(use.iloc[idx]["x"]), float(use.iloc[idx]["y"]), str(use.iloc[idx]["trade"] or "unknown").lower()))
                                        polygon_point_count += 1
                                    else:
                                        # simple decimate by skipping when over cap
                                        break
                        except Exception:
                            pass
            except Exception:
                # If polygon invalid we skip silently (not a fatal error if zones overall are processed)
                pass

        # free per-file
        del df
        plt.close('all')

    # ========================= Build Figures & Report =========================
    report_date = ""
    try:
        if global_min_ts is not None and global_max_ts is not None:
            report_date = f"{pd.to_datetime(global_min_ts).strftime('%Y%m%d')}-{pd.to_datetime(global_max_ts).strftime('%Y%m%d')}"
        else:
            report_date = dt.datetime.utcnow().strftime("%Y%m%d")
    except Exception:
        report_date = dt.datetime.utcnow().strftime("%Y%m%d")

    figures: list = []
    png_paths: list[Path] = []

    # Bar graphs for every active trade: busiest zones by hours (top_n)
    top_n = int(cfg.get("top_n", 10) or 10)
    figsize_bar = tuple(cfg.get("figsize_bar", (7, 5)))
    # sort trades by total hours descending
    trade_totals = []
    for tr, zmap in dwell_by_trade_zone_hours.items():
        tot = sum(v for v in zmap.values())
        trade_totals.append((tr, tot))
    trade_totals.sort(key=lambda x: x[1], reverse=True)

    for tr, tot in trade_totals:
        if tot <= 0:
            continue
        zmap = dwell_by_trade_zone_hours.get(tr, {})
        if not zmap:
            continue
        zones_sorted = sorted(zmap.items(), key=lambda kv: kv[1], reverse=True)
        zones_sorted = zones_sorted[:top_n]
        labels = [z for z, _ in zones_sorted]
        hours  = [float(h) for _, h in zones_sorted]

        fig = plt.figure(figsize=figsize_bar)
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, hours, color="#4C78A8", alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Hours")
        ax.set_title(f"Busiest Zones by Trade (Hours) — {tr}")
        for i, v in enumerate(hours):
            ax.text(v + max(hours) * 0.01, i, f"{v:.1f}", va="center", fontsize=8)
        fig.tight_layout()

        # Save PNG first
        p = out_dir / f"info_zone_report_{report_date}_plot{len(png_paths)+1:02d}.png"
        try:
            fig.savefig(str(p), dpi=120)
            png_paths.append(p)
            figures.append(fig)
        except Exception:
            plt.close(fig)

    # Floorplan overlay for points inside ad-hoc polygon
    extent, floor_img = load_floorplan_assets()
    if polygon_valid and polygon_points and extent is not None and floor_img is not None:
        fig = plt.figure(figsize=tuple(cfg.get("figsize_overlay", (9, 7))))
        ax = fig.add_subplot(111)
        x_min, x_max, y_min, y_max = extent
        ax.imshow(floor_img, extent=[x_min, x_max, y_min, y_max], origin="upper")
        # draw polygon
        poly_patch = MplPolygon(np.array(ad_hoc_polygon, dtype=float), closed=True,
                                facecolor=(1,0,0,cfg.get("zone_face_alpha",0.20)),
                                edgecolor=(1,0,0,cfg.get("zone_edge_alpha",0.65)), linewidth=1.5)
        ax.add_patch(poly_patch)
        # scatter points colored by trade
        trades_list = [t for (_, _, t) in polygon_points]
        uniq_trades = list(dict.fromkeys(trades_list))
        cmap = matplotlib.colormaps.get_cmap("tab10")
        colors = {t: cmap(i % 10) for i, t in enumerate(uniq_trades)}
        xs = [x for (x, _, _) in polygon_points]
        ys = [y for (_, y, _) in polygon_points]
        ts = [t for (_, _, t) in polygon_points]
        for t in uniq_trades:
            mask = [tt == t for tt in ts]
            ax.scatter(np.array(xs)[mask], np.array(ys)[mask],
                       s=float(cfg.get("overlay_point_size", 8)),
                       alpha=float(cfg.get("overlay_alpha", 0.85)),
                       color=colors[t], label=t)
        legend_if_labeled(ax)
        mx = float(cfg.get("floorplan_margin", 0.10))
        xr = (x_max - x_min); yr = (y_max - y_min)
        ax.set_xlim(x_min - mx * xr, x_max + mx * xr)
        ax.set_ylim(y_min - mx * yr, y_max + mx * yr)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Points Inside User-Defined Area (on Floorplan)")
        fig.tight_layout()

        p = out_dir / f"info_zone_report_{report_date}_plot{len(png_paths)+1:02d}.png"
        try:
            fig.savefig(str(p), dpi=120)
            png_paths.append(p)
            figures.append(fig)
        except Exception:
            plt.close(fig)

    # ----------------------------- Build Report Dict -----------------------------
    title = "InfoZone — Busiest Zones by Trade (Hours)"
    meta = f"Files: {len(csv_paths)} | Window: {ts_range_str(global_min_ts, global_max_ts)}"
    sections = []

    # Summary (at least one bullet)
    bullets = []
    if any_rows:
        n_trades = len([t for t, tot in trade_totals if tot > 0])
        top_entry = trade_totals[0] if trade_totals else ("n/a", 0.0)
        bullets.append(f"Analyzed {len(csv_paths)} file(s) over {ts_range_str(global_min_ts, global_max_ts)}.")
        bullets.append(f"Active trades: {n_trades}. Top by dwell: {top_entry[0]} ({top_entry[1]:.1f} hours).")
        if polygon_valid:
            tot_poly_hours = sum(ad_hoc_dwell_by_trade_hours.values())
            bullets.append(f"User area dwell total: {tot_poly_hours:.1f} hours across {len(ad_hoc_dwell_by_trade_hours)} trade(s).")
        else:
            bullets.append("User area polygon was invalid or not provided; skipped area dwell.")
    else:
        bullets.append("No data rows available in the provided CSV files.")

    sections.append({"type": "summary", "title": "Summary", "bullets": bullets})

    # Table: Ad-hoc polygon dwell per trade (hours)
    if polygon_valid and ad_hoc_dwell_by_trade_hours:
        rows = []
        for t, h in sorted(ad_hoc_dwell_by_trade_hours.items(), key=lambda kv: kv[1], reverse=True):
            rows.append({"trade": t, "hours": f"{h:.2f}"})
        sections.append({"type": "table", "title": "Time Spent in User-Defined Area (Hours by Trade)",
                         "data": rows, "headers": ["trade", "hours"], "rows_per_page": 24})

    # Evidence table (first file's small sample if any)
    try:
        # Re-open the first CSV minimally for evidence if needed
        raw0 = extract_tracks(str(csv_paths[0]), mac_map_path=str(ROOT / "trackable_objects.json"))
        d0 = pd.DataFrame(raw0.get("rows", []))
        if d0.columns.duplicated().any():
            d0 = d0.loc[:, ~d0.columns.duplicated()]
        if not d0.empty:
            src0 = d0["ts_iso"] if "ts_iso" in d0.columns else d0["ts"]
            d0["ts_utc"] = pd.to_datetime(src0, utc=True, errors="coerce")
            cols = [c for c in ["trackable","trade","ts_short","x","y","z"] if c in d0.columns]
            if cols:
                rows = d0[cols].head(50).fillna("").astype(str).to_dict(orient="records")
                sections.append({"type":"table","title":"Evidence (sample of first file)","data":rows,"headers":cols,"rows_per_page":24})
    except Exception:
        pass

    # Charts section (only if we have figures)
    live_figs = [f for f in figures if getattr(f, "savefig", None)]
    if live_figs:
        sections.append({"type": "charts", "title": "Figures", "figures": live_figs})

    report = {"title": title, "meta": meta, "sections": sections}

    # Apply budgets (cap figures, rows, etc.)
    report = apply_budgets(report, None)

    # Output paths
    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"

    # Build PDF
    try:
        # Save PNGs already done; pass live figures to PDF builder
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
    except Exception as e:
        # Fallback: lite
        try:
            lite = make_lite(report)
            safe_build_pdf(lite, str(pdf_path), logo_path=str(LOGO))
        except Exception as e2:
            print("Error Report:")
            print(f"PDF build failed: {e2.__class__.__name__}: {e2}")
            raise SystemExit(1)

    # Success: print links
    print(f"[Download the PDF]({file_uri(pdf_path)})")
    for i, pth in enumerate(png_paths, 1):
        print(f"[Download Plot {i}]({file_uri(pth)})")

    # Close figures after PDF built
    try:
        for f in figures:
            plt.close(f)
    except Exception:
        pass