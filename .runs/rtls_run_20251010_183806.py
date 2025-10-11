import sys, os
from pathlib import Path

# -------------------- ROOT resolution & local imports --------------------
ROOT = Path(os.environ.get("INFOZONE_ROOT", ""))  # injected by launcher
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

# -------------------- Standard libs --------------------
import json
import math
import traceback
import datetime as _dt

# -------------------- Third-party libs --------------------
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

# Matplotlib ≥3.9 shim (PDF builder expects tostring_rgb)
from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
import numpy as _np
_FCA.tostring_rgb = getattr(_FCA, "tostring_rgb", lambda self: _np.asarray(self.buffer_rgba())[..., :3].tobytes())

# -------------------- Helpers from local project --------------------
try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Missing required extractor helper.")
    raise SystemExit(1)

try:
    from zones_process import load_zones, compute_zone_intervals, dwell_in_polygon, make_polygon, sanitize_polygon_points
except Exception:
    print("Error Report:")
    print("Missing required zones_process helper.")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("Missing PDF builder helper.")
    raise SystemExit(1)

try:
    from report_limits import apply_budgets, make_lite
except Exception:
    print("Error Report:")
    print("Missing report_limits helper.")
    raise SystemExit(1)

# -------------------- CLI args --------------------
def parse_args(argv):
    if len(argv) < 3:
        print("Error Report:")
        print("Expected: python script.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
        raise SystemExit(1)
    user_prompt = argv[1]
    csv_paths = [Path(a) for a in argv[2:]]
    return user_prompt, csv_paths

# -------------------- Config loader --------------------
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
        "max_figures": 6,
        "figsize_overlay": (9, 7),
        "figsize_bar": (7, 5),
    }
    try:
        if CONFIG.exists():
            data = json.loads(read_text(CONFIG))
            if isinstance(data, dict):
                cfg.update(data)
    except Exception:
        pass
    # normalize list sizes
    def _tuple(v, default):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (float(v[0]), float(v[1]))
        return default
    cfg["figsize_overlay"] = _tuple(cfg.get("figsize_overlay"), (9, 7))
    cfg["figsize_bar"] = _tuple(cfg.get("figsize_bar"), (7, 5))
    # ints
    for k in ("overlay_subsample", "top_n", "max_figures"):
        try:
            cfg[k] = int(cfg.get(k, cfg[k]))
        except Exception:
            pass
    # floats
    for k in ("floorplan_margin", "overlay_point_size", "overlay_alpha", "zone_face_alpha", "zone_edge_alpha"):
        try:
            cfg[k] = float(cfg.get(k, cfg[k]))
        except Exception:
            pass
    return cfg

# -------------------- Floorplan loader --------------------
def load_floorplan_assets():
    # Prefer explicit floorplan.jpeg under ROOT; fallback to .png/.jpg
    img_path = None
    for name in ["floorplan.jpeg", "floorplan.jpg", "floorplan.png"]:
        p = ROOT / name
        if p.exists():
            img_path = p
            break

    extent = None
    image = None
    if FLOORJSON.exists():
        try:
            data = json.loads(read_text(FLOORJSON))
            # Use first plan
            fp = None
            if isinstance(data, dict):
                fps = data.get("floorplans") or data.get("plans")
                if isinstance(fps, list) and fps:
                    fp = fps[0]
                elif isinstance(data, dict):
                    fp = data
            elif isinstance(data, list) and data:
                fp = data[0]
            if fp:
                width  = float(fp.get("width", 0))
                height = float(fp.get("height", 0))
                x_c    = float(fp.get("image_offset_x", 0))
                y_c    = float(fp.get("image_offset_y", 0))
                image_scale = float(fp.get("image_scale", 0))  # meters/pixel
                scale = image_scale * 100.0  # mm/pixel

                x_min = (x_c - width/2.0)  * scale
                x_max = (x_c + width/2.0)  * scale
                y_min = (y_c - height/2.0) * scale
                y_max = (y_c + height/2.0) * scale
                extent = (x_min, x_max, y_min, y_max)
        except Exception:
            extent = None

    if img_path and img_path.exists():
        try:
            image = plt.imread(str(img_path))
        except Exception:
            image = None

    return extent, image

# -------------------- Color palette --------------------
def colors_for_categories(cats):
    cmap = matplotlib.colormaps.get_cmap("tab10")
    uniq = list(dict.fromkeys([str(c) for c in cats]))
    return {c: cmap(i % 10) for i, c in enumerate(uniq)}

# -------------------- Figures --------------------
def make_floorplan_overlay(points_sample, polygon_pts, extent, image, cfg, title="Floorplan Overlay — Requested Area"):
    if points_sample is None:
        points_sample = []
    fig = plt.figure(figsize=cfg["figsize_overlay"])
    ax = fig.add_subplot(111)

    if extent is not None and image is not None:
        x_min, x_max, y_min, y_max = extent
        ax.imshow(image, extent=[x_min, x_max, y_min, y_max], origin="upper")
        # margin
        mx = float(cfg["floorplan_margin"])
        xr = (x_max - x_min); yr = (y_max - y_min)
        ax.set_xlim(x_min - mx * xr, x_max + mx * xr)
        ax.set_ylim(y_min - mx * yr, y_max + mx * yr)
    else:
        # If no floorplan, auto-scale later from points/polygon
        pass

    # Draw polygon
    try:
        if polygon_pts is not None and len(polygon_pts) >= 3:
            patch = MplPolygon(np.asarray(polygon_pts, dtype=float), closed=True,
                               facecolor=(1, 0, 0, cfg["zone_face_alpha"]),
                               edgecolor=(1, 0, 0, cfg["zone_edge_alpha"]),
                               linewidth=1.2)
            ax.add_patch(patch)
    except Exception:
        pass

    # Plot points by trade color
    trades = [p.get("trade", "") for p in points_sample]
    palette = colors_for_categories(trades)
    # group by trade
    by_trade = {}
    for p in points_sample:
        by_trade.setdefault(p.get("trade", ""), []).append(p)
    for tr, lst in by_trade.items():
        xs = [float(d.get("x", np.nan)) for d in lst]
        ys = [float(d.get("y", np.nan)) for d in lst]
        ax.scatter(xs, ys, s=cfg["overlay_point_size"],
                   alpha=cfg["overlay_alpha"],
                   color=palette.get(tr, (0, 0, 0, 0.6)),
                   label=str(tr) if tr else "unknown")
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 12 and len(labels) > 0:
        ax.legend(loc="upper left", frameon=True, fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def make_trade_zone_bar(trade, zone_to_sec, cfg, top_n=10):
    # Create a bar chart of hours per zone for the given trade
    if not zone_to_sec:
        return None
    items = sorted(zone_to_sec.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None and top_n > 0:
        items = items[:top_n]
    zones = [str(k) for k, _ in items]
    hours = [float(v) / 3600.0 for _, v in items]
    if not zones:
        return None
    fig = plt.figure(figsize=cfg["figsize_bar"])
    ax = fig.add_subplot(111)
    y = np.arange(len(zones))
    ax.barh(y, hours, color="#1f77b4", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(zones)
    ax.invert_yaxis()
    ax.set_xlabel("Hours")
    ax.set_title(f"Dwell by Zone — {trade if trade else 'unknown'}")
    # Add value labels
    for i, v in enumerate(hours):
        ax.text(v + max(hours) * 0.01, i, f"{v:.1f}h", va="center", fontsize=8)
    fig.tight_layout()
    return fig

# -------------------- Reservoir sampler --------------------
class Reservoir:
    def __init__(self, k: int):
        self.k = max(0, int(k))
        self.buf = []
        self.n = 0
        self._rng = np.random.RandomState(42)

    def consider(self, item):
        if self.k <= 0:
            return
        self.n += 1
        if len(self.buf) < self.k:
            self.buf.append(item)
        else:
            j = self._rng.randint(0, self.n)
            if j < self.k:
                self.buf[j] = item

    def items(self):
        return list(self.buf)

# -------------------- Utility --------------------
def get_field(d, names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default

def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

# -------------------- Main --------------------
def main():
    user_prompt, csv_paths = parse_args(sys.argv)

    # Output directory: first CSV's folder
    out_dir = csv_paths[0].parent if csv_paths else Path.cwd().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    # Zones requested by user? Detect via prompt keywords: "zone", "area"
    zones_requested = True  # The user explicitly asked for zones and an area

    # Load zones polygons
    zones_list = load_zones(str(ZONES_JSON), only_active=True)
    have_zones = isinstance(zones_list, list) and len(zones_list) > 0

    # Prepare aggregators
    trade_zone_seconds = {}        # trade -> {zone_name -> sec}
    ad_hoc_trade_seconds = {}      # trade -> sec
    ad_points_sampler = Reservoir(cfg.get("overlay_subsample", 20000))

    # Ad-hoc polygon from user (mm)
    user_poly_points = [(20000,20000),(40000,20000),(40000,40000),(20000,40000)]
    poly_arr = sanitize_polygon_points(user_poly_points)
    if poly_arr.size == 0:
        print("Error Report:")
        print("Invalid ad-hoc polygon points.")
        raise SystemExit(1)
    poly_path = MplPath(poly_arr)

    # For meta
    global_min_ts = None
    global_max_ts = None
    n_files = 0
    first_df_sample = None  # for evidence table

    # Schema checked once after first file
    schema_checked = False

    # JSONL buffer for intervals (if needed)
    jsonl_path = out_dir / "_wk_intervals.jsonl"
    if jsonl_path.exists():
        try:
            jsonl_path.unlink()
        except Exception:
            pass

    # Floorplan assets for overlay
    extent, image = load_floorplan_assets()

    # Process each CSV independently (large-data mode)
    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            # Skip missing files silently, continue others
            continue

        try:
            raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
        except Exception as e:
            print("Error Report:")
            print(f"Extraction failed: {e.__class__.__name__}: {e}")
            raise SystemExit(1)

        audit = raw.get("audit", {}) if isinstance(raw, dict) else {}
        if not audit or not audit.get("mac_map_loaded", False):
            print("Error Report:")
            print("MAC map not loaded; cannot resolve names/trades.")
            raise SystemExit(1)

        rows = raw.get("rows", [])
        df = pd.DataFrame(rows)
        # Duplicate-name guard
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Timestamp canon
        ts_src = df["ts_iso"] if "ts_iso" in df.columns else df.get("ts")
        df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

        # Schema validation after first file
        if not schema_checked:
            cols = set(df.columns.astype(str))
            identity_ok = ("trackable" in cols) or ("trackable_uid" in cols)
            trade_ok = ("trade" in cols)
            xy_ok = ("x" in cols) and ("y" in cols)

            if not (identity_ok and trade_ok and xy_ok):
                print("Error Report:")
                print("Missing required columns for analysis.")
                print(f"Columns detected: {','.join(df.columns.astype(str))}")
                raise SystemExit(1)

            # If zones requested: must have zone_name or be able to compute zones
            if zones_requested:
                if ("zone_name" not in cols) and (not have_zones):
                    print("Error Report:")
                    print("Zones requested but no zones.json polygons are available, and 'zone_name' not present.")
                    print(f"Columns detected: {','.join(df.columns.astype(str))}")
                    raise SystemExit(1)

            schema_checked = True

        # track mapping for trade lookup
        uid_to_trade = {}
        if "trackable_uid" in df.columns and "trade" in df.columns:
            uid_to_trade = dict(zip(df["trackable_uid"].astype(str), df["trade"].astype(str)))

        # time range
        tmin = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").min()
        tmax = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").max()
        if pd.notna(tmin):
            global_min_ts = tmin if (global_min_ts is None or tmin < global_min_ts) else global_min_ts
        if pd.notna(tmax):
            global_max_ts = tmax if (global_max_ts is None or tmax > global_max_ts) else global_max_ts

        if first_df_sample is None:
            first_df_sample = df.copy()

        # Valid subset for spatial/zones operations
        x_num = pd.to_numeric(df.get("x", np.nan), errors="coerce")
        y_num = pd.to_numeric(df.get("y", np.nan), errors="coerce")
        valid = df.loc[x_num.notna() & y_num.notna() & df["ts_utc"].notna(), :].copy()
        if valid.empty:
            n_files += 1
            continue
        valid["x"] = pd.to_numeric(valid["x"], errors="coerce")
        valid["y"] = pd.to_numeric(valid["y"], errors="coerce")

        # Compute official zone intervals per file (without downsampling)
        if zones_requested and have_zones:
            try:
                intervals = compute_zone_intervals(valid, zones_list, id_col="trackable_uid",
                                                   ts_col="ts_utc", x_col="x", y_col="y")
            except Exception:
                intervals = []

            # Stream intervals to JSONL and also aggregate per trade×zone seconds
            if intervals:
                with jsonl_path.open("a", encoding="utf-8", errors="ignore") as f:
                    for it in intervals:
                        zname = str(get_field(it, ["zone_name", "name"], default="")).strip()
                        dur = float(get_field(it, ["duration_sec", "duration", "seconds"], default=0.0) or 0.0)
                        uid = str(get_field(it, ["trackable_uid", "id", "trackableId"], default=""))
                        tr = uid_to_trade.get(uid, "")
                        # update small aggregate
                        if zname:
                            trade_zone_seconds.setdefault(tr, {})
                            trade_zone_seconds[tr][zname] = trade_zone_seconds[tr].get(zname, 0.0) + float(dur)
                        # write jsonl
                        try:
                            f.write(json.dumps({
                                "trackable_uid": uid,
                                "trade": tr,
                                "zone_name": zname,
                                "duration_sec": float(dur),
                                "enter_ts": get_field(it, ["enter_ts", "enter", "start", "start_ts"], default=None),
                                "leave_ts": get_field(it, ["leave_ts", "leave", "end", "end_ts"], default=None),
                            }) + "\n")
                        except Exception:
                            # ignore write errors for single line
                            pass

        # Ad-hoc polygon dwell intervals
        try:
            adhoc = dwell_in_polygon(valid, poly_arr.tolist(), name="Requested Area",
                                     id_col="trackable_uid", ts_col="ts_utc", x_col="x", y_col="y")
            ad_intervals = adhoc.get("intervals", []) if isinstance(adhoc, dict) else []
            for it in ad_intervals:
                dur = float(get_field(it, ["duration_sec", "duration", "seconds"], default=0.0) or 0.0)
                uid = str(get_field(it, ["trackable_uid", "id", "trackableId"], default=""))
                tr = uid_to_trade.get(uid, "")
                ad_hoc_trade_seconds[tr] = ad_hoc_trade_seconds.get(tr, 0.0) + float(dur)
        except Exception:
            # Continue without ad-hoc intervals if computation fails
            pass

        # Collect overlay points inside ad-hoc polygon (reservoir)
        try:
            pts = np.column_stack((pd.to_numeric(valid["x"], errors="coerce").values,
                                   pd.to_numeric(valid["y"], errors="coerce").values))
            inside_mask = poly_path.contains_points(pts)
            inside_df = valid.loc[inside_mask, ["x", "y", "trade"]]
            for _, r in inside_df.iterrows():
                ad_points_sampler.consider({"x": float(r["x"]), "y": float(r["y"]), "trade": str(r.get("trade", ""))})
        except Exception:
            pass

        # Done with this file
        n_files += 1
        del df, valid

        # Close any stray figures created during processing (none yet)
        plt.close('all')

    # ------------- Build figures -------------
    figs = []
    png_paths = []

    # Per-trade bar charts (hours by zone)
    # Only include trades with any zone dwell seconds
    active_trades = [tr for tr, zmap in trade_zone_seconds.items() if sum(zmap.values()) > 0]
    # Sort trades by total hours descending
    active_trades = sorted(active_trades, key=lambda t: sum(trade_zone_seconds.get(t, {}).values()), reverse=True)

    for tr in active_trades:
        fig = make_trade_zone_bar(tr, trade_zone_seconds.get(tr, {}), cfg, top_n=cfg.get("top_n", 10))
        if fig is not None:
            figs.append(fig)

    # Floorplan overlay for ad-hoc points
    points_sample = ad_points_sampler.items()
    if len(points_sample) > 0:
        fig_overlay = make_floorplan_overlay(points_sample, poly_arr.tolist(), extent, image, cfg,
                                             title="Points Inside Requested Area")
        if fig_overlay is not None:
            figs.append(fig_overlay)

    # Respect figure budget before saving PNGs
    # We'll still save only the kept figures after budget application by previewing limits
    max_figs = int(cfg.get("max_figures", 6))
    figs_to_save = figs[:max_figs] if max_figs > 0 else figs

    # Report date tag
    if global_min_ts is not None and global_max_ts is not None:
        # Keep as ISO without double Z in meta; leave timezone-aware
        start_tag = pd.to_datetime(global_min_ts).strftime("%Y%m%d")
        end_tag   = pd.to_datetime(global_max_ts).strftime("%Y%m%d")
        report_date = f"{start_tag}_to_{end_tag}"
        meta_range = f"{pd.to_datetime(global_min_ts).strftime('%Y-%m-%d %H:%MZ')} – {pd.to_datetime(global_max_ts).strftime('%Y-%m-%d %H:%MZ')}"
    else:
        today = _dt.datetime.utcnow().strftime("%Y%m%d")
        report_date = today
        meta_range = "No valid timestamps detected"

    # Save PNGs first (DPI=120, no bbox_inches='tight')
    for i, fig in enumerate(figs_to_save, 1):
        p = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
        try:
            fig.savefig(str(p), dpi=120)
            png_paths.append(p)
        except Exception:
            # Ignore save errors for individual figures
            pass

    # ------------- Build tables -------------
    sections = []

    # Summary bullets (at least 1)
    bullets = []
    bullets.append(f"Processed {n_files} file(s); time range: {meta_range}.")
    if active_trades:
        bullets.append(f"Active trades with zone dwell: {min(len(active_trades), 8)} shown (capped by budgets).")
    if len(points_sample) > 0:
        bullets.append(f"Ad-hoc area points plotted: {len(points_sample)} (reservoir sampled).")
    sections.append({"type": "summary", "title": "Summary", "bullets": bullets})

    # Evidence table (limited rows; from first file)
    if first_df_sample is not None and not first_df_sample.empty:
        try:
            cols = ["trackable","trade","ts_short","x","y","z"]
            cols_present = [c for c in cols if c in first_df_sample.columns]
            # Ensure required are present at least in dict
            df_e = first_df_sample.copy()
            for c in cols:
                if c not in df_e.columns:
                    df_e[c] = ""
            rows = df_e[cols].head(50).fillna("").astype(str).to_dict(orient="records")
            sections.append({"type": "table", "title": "Evidence (first file sample)", "data": rows, "headers": cols, "rows_per_page": 24})
        except Exception:
            pass

    # Ad-hoc area dwell table by trade (hours)
    if ad_hoc_trade_seconds:
        try:
            rows = []
            total_sec = 0.0
            for tr, sec in sorted(ad_hoc_trade_seconds.items(), key=lambda kv: kv[1], reverse=True):
                rows.append({"trade": str(tr if tr else "unknown"), "hours": f"{(float(sec)/3600.0):.2f}"})
                total_sec += float(sec)
            rows.append({"trade": "TOTAL", "hours": f"{(total_sec/3600.0):.2f}"})
            headers = ["trade", "hours"]
            sections.append({"type": "table", "title": "Ad-hoc Area Dwell by Trade (hours)", "data": rows, "headers": headers, "rows_per_page": 24})
        except Exception:
            pass

    # Charts section (only if we have any live figures)
    live_figs = [f for f in figs_to_save if getattr(f, "savefig", None)]
    if len(live_figs) > 0:
        sections.append({"type": "charts", "title": "Figures", "figures": live_figs})

    # ------------- Build report dict -------------
    title = "Walmart Renovation — Zones by Trade (Weekly)"
    meta = f"Prompt: {user_prompt}\nFiles: {n_files}\nRange (UTC): {meta_range}"
    report = {"title": title, "meta": meta, "sections": sections}

    # Apply budgets
    report = apply_budgets(report)

    # Build PDF
    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

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
        # Now safe to close figures
        try:
            plt.close('all')
        except Exception:
            pass

    # ------------- Print links (success only) -------------
    print(f"[Download the PDF]({file_uri(pdf_path)})")
    for i, pth in enumerate(png_paths, 1):
        print(f"[Download Plot {i}]({file_uri(pth)})")

if __name__ == "__main__":
    try:
        # Ensure we read guidelines (mandatory)
        _ = read_text(GUIDELINES)
        main()
    except SystemExit:
        raise
    except Exception as e:
        # Unified error output
        print("Error Report:")
        msg = f"{e.__class__.__name__}: {e}"
        print(msg if len(str(e)) > 0 else "Unexpected error.")
        # If DataFrame 'df' existed and schema failed, the specific handler above already exited.