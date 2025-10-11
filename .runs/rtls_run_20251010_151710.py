#!/usr/bin/env python3
# InfoZoneBuilder – Walmart RTLS Summary Report
import sys, os, json, math, traceback
from pathlib import Path
from datetime import datetime, timezone

# --------------------------- ROOT resolution & local imports ---------------------------
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

# --------------------------- Imports (helpers, libs) ---------------------------
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Error Report:")
    print("Required Python libraries are missing: pandas/matplotlib/numpy.")
    raise SystemExit(1)

try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Local helper 'extractor' not found.")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("Local helper 'pdf_creation_script.safe_build_pdf' not found.")
    raise SystemExit(1)

# Budgets
try:
    from report_limits import apply_budgets, make_lite, DEFAULTS as LIMIT_DEFAULTS
except Exception:
    def apply_budgets(report, caps=None): return report
    def make_lite(report): return report
    LIMIT_DEFAULTS = {"MAX_FIGURES": 6, "MAX_TABLE_ROWS_TOTAL": 180, "MAX_TEXT_LINES_TOTAL": 900, "MAX_PAGES": 12}

# --------------------------- Config loading ---------------------------
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
        "pie_max_trades": 8,
        "pie_max_single_share": 0.90,
        "line_min_points": 2,
        "small_multiples_cols": 2,
        "max_figures": 6,
        "figsize_overlay": (9, 7),
        "figsize_bar": (7, 5),
        "figsize_line": (7, 5),
        "figsize_pie": (5, 5),
        "figsize_box": (7, 5),
    }
    try:
        if CONFIG.exists():
            cfg.update(json.loads(read_text(CONFIG) or "{}"))
    except Exception:
        pass
    # Ensure numeric types
    for k in ("floorplan_margin","overlay_point_size","overlay_alpha","overlay_subsample","trail_seconds","top_n","pie_max_trades","pie_max_single_share","line_min_points","small_multiples_cols","max_figures"):
        try:
            cfg[k] = float(cfg[k]) if k in ("floorplan_margin","overlay_point_size","overlay_alpha","pie_max_single_share") else int(cfg[k])
        except Exception:
            pass
    # Ensure figsize tuples
    for k in ("figsize_overlay","figsize_bar","figsize_line","figsize_pie","figsize_box"):
        v = cfg.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            cfg[k] = (float(v[0]), float(v[1]))
    return cfg

# --------------------------- Floorplan loading & overlay ---------------------------
def _find_floorplan_image():
    # Search only LOCAL paths (ROOT or CWD)
    candidates = [
        ROOT / "floorplan.png",
        ROOT / "floorplan.jpg",
        ROOT / "floorplan.jpeg",
        Path.cwd() / "floorplan.png",
        Path.cwd() / "floorplan.jpg",
        Path.cwd() / "floorplan.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def load_floorplan_assets():
    if not FLOORJSON.exists():
        return None
    try:
        data = json.loads(read_text(FLOORJSON) or "{}")
        plans = data.get("floorplans") or data.get("plans") or []
        if isinstance(plans, list) and plans:
            fp = plans[0]
        elif isinstance(plans, dict):
            fp = plans
        else:
            fp = data if isinstance(data, dict) else None
        if not fp:
            return None
        width  = float(fp.get("width", 0))
        height = float(fp.get("height", 0))
        x_c    = float(fp.get("image_offset_x", 0))
        y_c    = float(fp.get("image_offset_y", 0))
        image_scale = float(fp.get("image_scale", 0))
        scale_mm_per_px = image_scale * 100.0
        x_min = (x_c - width/2.0)  * scale_mm_per_px
        x_max = (x_c + width/2.0)  * scale_mm_per_px
        y_min = (y_c - height/2.0) * scale_mm_per_px
        y_max = (y_c + height/2.0) * scale_mm_per_px
        img_path = _find_floorplan_image()
        if not img_path:
            return None
        img = plt.imread(img_path)
        return {"extent": (x_min, x_max, y_min, y_max), "image": img, "image_path": img_path}
    except Exception:
        return None

def make_floorplan_overlay(overlay_df, fp, cfg, color_by="trade"):
    if overlay_df is None or overlay_df.empty or ("x" not in overlay_df.columns) or ("y" not in overlay_df.columns):
        return None
    # Numeric x/y
    x = pd.to_numeric(overlay_df["x"], errors="coerce")
    y = pd.to_numeric(overlay_df["y"], errors="coerce")
    use = overlay_df.loc[x.notna() & y.notna(), :].copy()
    if use.empty:
        return None

    # Subsample to overlay_subsample
    max_pts = int(cfg.get("overlay_subsample", 20000))
    if len(use) > max_pts:
        idx = np.linspace(0, len(use) - 1, max_pts).astype(int)
        use = use.iloc[idx]

    fig = plt.figure(figsize=cfg.get("figsize_overlay", (9,7)))
    ax = fig.add_subplot(111)
    x_min, x_max, y_min, y_max = fp["extent"]
    ax.imshow(fp["image"], extent=[x_min, x_max, y_min, y_max], origin="upper")

    cat_col = color_by if color_by in use.columns else None
    legend_labels = []
    if cat_col is None:
        ax.scatter(pd.to_numeric(use["x"], errors="coerce"),
                   pd.to_numeric(use["y"], errors="coerce"),
                   s=float(cfg.get("overlay_point_size", 8)),
                   alpha=float(cfg.get("overlay_alpha", 0.85)))
    else:
        # Category coloring with cap on legend labels
        cats = use[cat_col].astype(str).fillna("")
        uniq = list(dict.fromkeys(cats.tolist()))
        palette = plt.cm.get_cmap("tab20")
        for i, cat in enumerate(uniq):
            g = use[cats == cat]
            ax.scatter(pd.to_numeric(g["x"], errors="coerce"),
                       pd.to_numeric(g["y"], errors="coerce"),
                       s=float(cfg.get("overlay_point_size", 8)),
                       alpha=float(cfg.get("overlay_alpha", 0.85)),
                       color=palette(i % 20),
                       label=str(cat) if cat else "Unlabeled")
            legend_labels.append(str(cat) if cat else "Unlabeled")
        if len(legend_labels) <= 12:
            ax.legend(loc="upper left", frameon=True, fontsize=8)

    # Margins & labels
    mx = float(cfg.get("floorplan_margin", 0.10))
    xr = (x_max - x_min); yr = (y_max - y_min)
    ax.set_xlim(x_min - mx * xr, x_max + mx * xr)
    ax.set_ylim(y_min - mx * yr, y_max + mx * yr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Floorplan Overlay")
    fig.tight_layout()
    return fig

# --------------------------- Supporting figures ---------------------------
def make_trade_bar(trade_counts: dict, cfg):
    if not trade_counts:
        return None
    s = pd.Series(trade_counts).sort_values(ascending=False)
    fig = plt.figure(figsize=cfg.get("figsize_bar", (7,5)))
    ax = fig.add_subplot(111)
    top_n = int(cfg.get("top_n", 10))
    s = s.head(top_n)
    s.plot(kind="bar", color="#4C78A8", ax=ax)
    ax.set_ylabel("Samples")
    ax.set_title("Top Trades by Samples")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

def make_hourly_line(hourly_counts: dict, cfg):
    if not hourly_counts or len(hourly_counts) < int(cfg.get("line_min_points", 2)):
        return None
    hours = sorted(hourly_counts.keys())
    vals = [hourly_counts[h] for h in hours]
    idx = pd.to_datetime(pd.Series(hours), utc=True, errors="coerce")
    # For plotting, show as naive UTC to avoid tz issues
    idx_naive = idx.dt.tz_convert('UTC').dt.tz_localize(None)
    fig = plt.figure(figsize=cfg.get("figsize_line", (7,5)))
    ax = fig.add_subplot(111)
    ax.plot(idx_naive, vals, color="#F58518", marker="o", linewidth=1.5)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Samples")
    ax.set_title("Hourly Sample Counts")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

# --------------------------- MAIN ---------------------------
def main():
    try:
        # Read guidelines/context (UTF-8, ignore errors) – compliance
        _ = read_text(GUIDELINES)
        _ = read_text(CONTEXT)

        if len(sys.argv) < 3:
            print("Error Report:")
            print("No CSV inputs provided.")
            raise SystemExit(1)

        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:] if p and Path(p).exists()]
        if not csv_paths:
            print("Error Report:")
            print("No valid CSV paths were provided.")
            raise SystemExit(1)

        out_dir = csv_paths[0].resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_config()
        report_date = datetime.utcnow().strftime("%Y%m%d")

        # Aggregates (memory-safe)
        hourly_counts = {}           # {hour_str: count}
        trade_counts = {}            # {trade: count}
        trackable_uids = set()
        trackable_names = set()
        total_samples = 0
        t_min = None
        t_max = None

        overlay_cap = int(cfg.get("overlay_subsample", 20000))
        overlay_df = None  # bounded reservoir DataFrame with cols ['x','y','trade','trackable']
        evidence_rows = [] # from first file only

        # Zones flag (only if asked)
        zones_requested = any(w in (user_prompt or "").lower() for w in ("zone", "zones", "area", "room"))
        # We will NOT compute zones unless requested (per spec). For this prompt, not requested.

        first_df_cols = None

        for file_idx, csv_path in enumerate(csv_paths, start=1):
            try:
                raw = extract_tracks(str(csv_path))
            except Exception as e:
                print("Error Report:")
                print(f"Failed to extract tracks from CSV: {csv_path.name}")
                raise SystemExit(1)

            df = pd.DataFrame(raw.get("rows", []))
            audit = raw.get("audit", {}) if isinstance(raw, dict) else {}
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
            # Timestamp canon
            src = df["ts_iso"] if "ts_iso" in df.columns else df.get("ts", pd.Series([], dtype=str))
            df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

            # Schema validation after first file
            if file_idx == 1:
                first_df_cols = list(df.columns.astype(str))
                identity_ok = ("trackable" in df.columns) or ("trackable_uid" in df.columns)
                trade_ok = ("trade" in df.columns)
                pos_ok = ("x" in df.columns) and ("y" in df.columns)
                if not (identity_ok and trade_ok and pos_ok):
                    print("Error Report:")
                    print("Missing required columns for analysis.")
                    print("Columns detected: " + ",".join(df.columns.astype(str)))
                    raise SystemExit(1)
                # Evidence rows from first file
                cols = ["trackable","trade","ts_short","x","y","z"]
                for c in cols:
                    if c not in df.columns:
                        df[c] = ""
                try:
                    ev = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
                    evidence_rows = ev
                except Exception:
                    evidence_rows = []

            # Column pruning for downstream
            keep_cols = [c for c in ("trackable","trackable_uid","trade","mac","ts","ts_iso","ts_short","x","y","z","ts_utc") if c in df.columns]
            df = df[keep_cols].copy()

            # Aggregates
            # Valid timestamps
            valid_time = df["ts_utc"].notna()
            if valid_time.any():
                tt = df.loc[valid_time, "ts_utc"]
                tmin_f = tt.min()
                tmax_f = tt.max()
                t_min = tmin_f if t_min is None or (pd.notna(tmin_f) and tmin_f < t_min) else t_min
                t_max = tmax_f if t_max is None or (pd.notna(tmax_f) and tmax_f > t_max) else t_max

                # Hourly counts (UTC)
                hrs = tt.dt.floor("H").astype("datetime64[ns, UTC]")
                vc = hrs.value_counts()
                for h, c in vc.items():
                    k = pd.to_datetime(h).strftime("%Y-%m-%dT%H:00Z")
                    hourly_counts[k] = hourly_counts.get(k, 0) + int(c)

            # Trade counts
            if "trade" in df.columns:
                vc = df["trade"].astype(str).fillna("").value_counts()
                for tr, c in vc.items():
                    trade = tr if tr else "unlabeled"
                    trade_counts[trade] = trade_counts.get(trade, 0) + int(c)

            # Trackable sets
            if "trackable_uid" in df.columns:
                trackable_uids.update([u for u in df["trackable_uid"].astype(str).tolist() if u])
            if "trackable" in df.columns:
                trackable_names.update([n for n in df["trackable"].astype(str).tolist() if n])

            # Overlay reservoir (bounded)
            # take only necessary columns for overlay
            have_cols = [c for c in ("x","y","trade","trackable") if c in df.columns]
            df_xy = df[have_cols].copy()
            # numeric filter quickly
            xs = pd.to_numeric(df_xy["x"], errors="coerce") if "x" in df_xy.columns else pd.Series([], dtype=float)
            ys = pd.to_numeric(df_xy["y"], errors="coerce") if "y" in df_xy.columns else pd.Series([], dtype=float)
            mask = xs.notna() & ys.notna()
            df_xy = df_xy.loc[mask]
            if not df_xy.empty:
                # pre-decimate the batch to at most overlay_cap rows (evenly spaced)
                if len(df_xy) > overlay_cap:
                    idx = np.linspace(0, len(df_xy) - 1, overlay_cap).astype(int)
                    df_xy = df_xy.iloc[idx]
                if overlay_df is None:
                    overlay_df = df_xy.copy()
                    if len(overlay_df) > overlay_cap:
                        idx = np.linspace(0, len(overlay_df) - 1, overlay_cap).astype(int)
                        overlay_df = overlay_df.iloc[idx].reset_index(drop=True)
                else:
                    tmp = pd.concat([overlay_df, df_xy], ignore_index=True)
                    if len(tmp) > overlay_cap:
                        idx = np.linspace(0, len(tmp) - 1, overlay_cap).astype(int)
                        overlay_df = tmp.iloc[idx].reset_index(drop=True)
                    else:
                        overlay_df = tmp

            total_samples += int(len(df))

            # Drop references for memory (per-file processing)
            del df

        # --------------------------- Figures ---------------------------
        figs = []
        png_paths = []
        # Floorplan overlay if available and preferred
        fp = load_floorplan_assets() if bool(cfg.get("prefer_floorplan", True)) else None
        if fp and overlay_df is not None and not overlay_df.empty:
            fig = make_floorplan_overlay(overlay_df, fp, cfg, color_by=str(cfg.get("overlay_color_by", "trade")))
            if fig:
                figs.append(fig)

        # Trade bar
        fig = make_trade_bar(trade_counts, cfg)
        if fig:
            figs.append(fig)

        # Hourly line
        fig = make_hourly_line(hourly_counts, cfg)
        if fig:
            figs.append(fig)

        # Cap number of figures by limits/config
        max_figs = int(cfg.get("max_figures", 6))
        if len(figs) > max_figs:
            figs = figs[:max_figs]

        # Save PNGs (in order), do not close figures
        for i, fig in enumerate(figs, start=1):
            png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(png), dpi=120)
                png_paths.append(png)
            except Exception:
                # If saving fails, skip but keep figure for PDF
                pass

        # --------------------------- Report assembly ---------------------------
        # Meta text time window
        def fmt_ts(dt_obj):
            if pd.isna(dt_obj) or dt_obj is None:
                return ""
            try:
                # Ensure UTC then format without trailing Z
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                return dt_obj.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                return ""

        window = ""
        if t_min is not None and t_max is not None:
            window = f"Window: {fmt_ts(t_min)} → {fmt_ts(t_max)}"
        counts_meta = f"Samples: {total_samples:,} | Trackables: {max(len(trackable_uids), len(trackable_names)):,}"
        meta_text = " | ".join([p for p in (window, counts_meta) if p])

        # Summary bullets
        bullets = []
        if window:
            bullets.append(window)
        bullets.append(counts_meta)
        if trade_counts:
            top = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            share_lines = []
            denom = max(1, sum(trade_counts.values()))
            for k, v in top:
                share = (v / denom) * 100.0
                share_lines.append(f"{k}: {v:,} samples ({share:.1f}%)")
            bullets.append("Top trades: " + "; ".join(share_lines))

        # Evidence table (from first CSV)
        sections = []
        sections.append({"type": "summary", "title": "Quick Summary", "bullets": bullets})
        if evidence_rows:
            cols = ["trackable","trade","ts_short","x","y","z"]
            sections.append({"type":"table","title":"Evidence","data":evidence_rows,"headers":cols,"rows_per_page":24})
        if figs:
            sections.append({"type":"charts","title":"Figures","figures":figs})

        title = "Walmart Renovation RTLS Summary"
        report = {
            "title": title,
            "meta": meta_text,
            "sections": sections
        }

        # Apply budgets
        report = apply_budgets(report, LIMIT_DEFAULTS if isinstance(LIMIT_DEFAULTS, dict) else None)

        # Build PDF
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        try:
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except MemoryError:
            # Minimal-Report Mode
            lite = make_lite(report)
            safe_build_pdf(lite, str(pdf_path), logo_path=str(LOGO))
            png_paths = []  # no PNG links in lite fallback
        except Exception:
            # Try lite fallback
            try:
                lite = make_lite(report)
                safe_build_pdf(lite, str(pdf_path), logo_path=str(LOGO))
                png_paths = []
            except Exception:
                print("Error Report:")
                print("Failed to write PDF report.")
                raise SystemExit(1)

        # --------------------------- Print links (success) ---------------------------
        print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        for i, pth in enumerate(png_paths, 1):
            print(f"[Download Plot {i}](file:///{pth.resolve().as_posix()})")

    except SystemExit:
        raise
    except Exception as e:
        # Generic failure
        msg = str(e).strip() or "Unhandled error."
        print("Error Report:")
        print(msg[:300])

if __name__ == "__main__":
    main()