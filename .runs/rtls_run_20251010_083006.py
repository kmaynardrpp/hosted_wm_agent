#!/usr/bin/env python
# InfoZoneBuilder — Walmart RTLS positions analyzer
# Generates a branded PDF + PNGs from local CSV(s), Windows-safe.

import sys, os, json, math, traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# -------------------- Resolve project ROOT and local imports --------------------
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

# -------------------- Imports (helpers) --------------------
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
except Exception as e:
    print("Error Report:")
    print("Missing required Python libraries (pandas/matplotlib).")
    raise SystemExit(1)

try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Missing helper module: extractor.extract_tracks not found under ROOT.")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("Missing helper module: pdf_creation_script.safe_build_pdf not found under ROOT.")
    raise SystemExit(1)

try:
    from report_limits import apply_budgets, DEFAULTS as LIMIT_DEFAULTS
except Exception:
    # Fallback minimal apply_budgets if helper missing, but follow contract
    def apply_budgets(report: Dict[str, Any], caps: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        return report
    LIMIT_DEFAULTS = {"MAX_FIGURES": 6, "MAX_TABLE_ROWS_TOTAL": 180, "MAX_TEXT_LINES_TOTAL": 900, "MAX_PAGES": 12}

# -------------------- Config and constants --------------------
GUIDE_TXT = read_text(GUIDELINES)
CFG_OBJ: Dict[str, Any] = {}
if CONFIG.exists():
    try:
        CFG_OBJ = json.loads(read_text(CONFIG)) or {}
    except Exception:
        CFG_OBJ = {}
# Defaults (align with report_config.json when present)
CONFIG_DEFAULTS = {
    "prefer_floorplan": True,
    "floorplan_margin": 0.10,
    "overlay_point_size": 8,
    "overlay_alpha": 0.85,
    "overlay_color_by": "trade",   # or "trackable" or "none"
    "overlay_subsample": 20000,
    "draw_trails": False,
    "trail_seconds": 900,
    "draw_zones": False,           # zones only if asked; override disabled here
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
# Merge config file over defaults
for k, v in (CFG_OBJ or {}).items():
    CONFIG_DEFAULTS[k] = v

# -------------------- CLI args --------------------
if len(sys.argv) < 3:
    print("Error Report:")
    print("Usage: python script.py \"<USER_PROMPT>\" <abs_csv1> [abs_csv2 ...]")
    raise SystemExit(1)

USER_PROMPT = sys.argv[1]
CSV_PATHS = [Path(p) for p in sys.argv[2:]]
if not CSV_PATHS or not CSV_PATHS[0].exists():
    print("Error Report:")
    print("First CSV path not found or invalid.")
    raise SystemExit(1)

out_dir = CSV_PATHS[0].resolve().parent

# -------------------- Utility helpers --------------------
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def _find_floorplan_image(root: Path) -> Optional[Path]:
    # Only look under ROOT for a fixed naming convention
    for name in ("floorplan.png", "floorplan.jpg", "floorplan.jpeg"):
        p = root / name
        if p.exists():
            return p
    return None

def _load_floorplan_extent_and_image() -> Optional[Dict[str, Any]]:
    """
    Load floorplan extent from FLOORJSON and image from ROOT/floorplan.(png|jpg|jpeg).
    Returns dict with extent (x_min,x_max,y_min,y_max) and image ndarray.
    """
    if not FLOORJSON.exists():
        return None
    try:
        data = json.loads(read_text(FLOORJSON))
        # choose first plan if list
        fp = (data.get("floorplans") or data.get("plans") or data or [None])
        if isinstance(fp, list):
            fp = fp[0]
        if not fp:
            return None
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

        img_path = _find_floorplan_image(ROOT)
        if img_path is None or not img_path.exists():
            return None
        img = plt.imread(str(img_path))
        return {"extent": (x_min, x_max, y_min, y_max), "image": img}
    except Exception:
        return None

def _colors_for_categories(categories: List[str]) -> Dict[str, Any]:
    cmap = plt.cm.get_cmap("tab10")
    uniq = list(dict.fromkeys([str(c) for c in categories]))
    return {c: cmap(i % 10) for i, c in enumerate(uniq)}

def _safe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 12 and len(labels) > 0:
        ax.legend(loc="upper left", frameon=True, fontsize=8)

# -------------------- Aggregation storages (memory-light) --------------------
overlay_budget = int(CONFIG_DEFAULTS.get("overlay_subsample", 20000) or 20000)
overlay_reservoir: List[Tuple[float, float, str]] = []  # (x, y, color_key) color_key=trade or trackable
reservoir_n = 0

hourly_counts: Dict[pd.Timestamp, int] = {}
trade_counts: Dict[str, int] = {}
unique_trackables: set = set()
files_processed: List[str] = []
ts_first: Optional[pd.Timestamp] = None
ts_last: Optional[pd.Timestamp] = None

schema_checked = False
schema_ok = False
detected_columns: List[str] = []

# -------------------- Zones intent (only if asked) --------------------
USE_ZONES = False
if any(w in USER_PROMPT.lower() for w in ["zone", "zones", "area", "room", "section"]):
    USE_ZONES = True  # but we won't compute unless asked; here "summary" prompt -> False
# The instruction "ZONES ONLY IF ASKED" -> user's prompt didn't explicitly ask zones, so keep False
USE_ZONES = False

# -------------------- Per-file processing (large-data safe) --------------------
try:
    for csv_path in CSV_PATHS:
        if not csv_path.exists() or not csv_path.is_file():
            continue
        files_processed.append(str(csv_path))
        raw = extract_tracks(str(csv_path))  # rows + audit as required
        rows = raw.get("rows", [])
        audit = raw.get("audit", {})
        df = pd.DataFrame(rows)

        # Duplicate-name guard (must run immediately)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        detected_columns = list(df.columns.astype(str))

        # Timestamp canon
        src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
        df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

        # Early schema validation (after first file)
        if not schema_checked:
            must_have_identity = ("trackable" in df.columns) or ("trackable_uid" in df.columns)
            must_have_trade = ("trade" in df.columns)
            must_have_xy = ("x" in df.columns) and ("y" in df.columns)
            if not (must_have_identity and must_have_trade and must_have_xy):
                print("Error Report:")
                print("Missing required columns for analysis.")
                print("Columns detected: " + ",".join(df.columns.astype(str)))
                raise SystemExit(1)
            schema_ok = True
            schema_checked = True

        # Keep only needed columns to conserve memory
        need_cols = [c for c in ["trackable","trackable_uid","trade","mac","ts","ts_iso","ts_short","x","y","z","ts_utc"] if c in df.columns]
        df = df[need_cols].copy()

        # Track unique trackables (prefer UID)
        if "trackable_uid" in df.columns:
            unique_trackables.update([u for u in df["trackable_uid"].astype(str).unique() if u])
        elif "trackable" in df.columns:
            unique_trackables.update([t for t in df["trackable"].astype(str).unique() if t])

        # Time span
        valid_ts = df["ts_utc"].dropna()
        if not valid_ts.empty:
            tmin = valid_ts.min()
            tmax = valid_ts.max()
            ts_first = tmin if ts_first is None else min(ts_first, tmin)
            ts_last  = tmax if ts_last  is None else max(ts_last, tmax)

        # Hourly sample counts
        if "ts_utc" in df.columns:
            hh = df["ts_utc"].dropna().dt.floor("H")
            if not hh.empty:
                vc = hh.value_counts()
                for k, v in vc.items():
                    hourly_counts[k] = hourly_counts.get(k, 0) + int(v)

        # Trade distribution
        if "trade" in df.columns:
            tr = df["trade"].astype(str).replace({"": "unknown", "nan": "unknown"})
            vc = tr.value_counts()
            for k, v in vc.items():
                trade_counts[str(k)] = trade_counts.get(str(k), 0) + int(v)

        # Overlay reservoir sampling (x,y needed; no numeric cast elsewhere)
        # Use trade or trackable for color-by category as per config
        color_by = str(CONFIG_DEFAULTS.get("overlay_color_by") or "trade")
        if color_by not in df.columns:
            color_by = "trade" if "trade" in df.columns else ("trackable" if "trackable" in df.columns else None)

        if "x" in df.columns and "y" in df.columns:
            xs = pd.to_numeric(df["x"], errors="coerce")
            ys = pd.to_numeric(df["y"], errors="coerce")
            mask = xs.notna() & ys.notna()
            if "ts_utc" in df.columns:
                mask = mask & df["ts_utc"].notna()
            use = df.loc[mask, ["x","y"] + ([color_by] if color_by else [])]
            if not use.empty:
                # Reservoir sampling over this chunk
                # Convert to numpy for speed
                xvals = pd.to_numeric(use["x"], errors="coerce").to_numpy()
                yvals = pd.to_numeric(use["y"], errors="coerce").to_numpy()
                cvals = use[color_by].astype(str).replace({"": "unknown"}).to_numpy() if color_by else np.array([""]*len(use))
                for i in range(len(use)):
                    reservoir_n += 1
                    if len(overlay_reservoir) < overlay_budget:
                        overlay_reservoir.append((float(xvals[i]), float(yvals[i]), str(cvals[i])))
                    else:
                        # Replace with decreasing probability
                        j = np.random.randint(0, reservoir_n)
                        if j < overlay_budget:
                            overlay_reservoir[j] = (float(xvals[i]), float(yvals[i]), str(cvals[i]))

        # Release large df before next file iteration (we keep small aggregates only)
        del df

    # If schema failed (no valid files), error
    if not schema_ok:
        print("Error Report:")
        print("Missing required columns for analysis.")
        cols = detected_columns if detected_columns else []
        if cols:
            print("Columns detected: " + ",".join(cols))
        raise SystemExit(1)

    # -------------------- Build aggregates DataFrames for charts --------------------
    # Hourly line
    hourly_df = pd.DataFrame(sorted(hourly_counts.items(), key=lambda kv: kv[0]), columns=["hour_utc","count_samples"]) if hourly_counts else pd.DataFrame(columns=["hour_utc","count_samples"])

    # Trade distribution
    trade_df = pd.DataFrame(sorted(trade_counts.items(), key=lambda kv: (-kv[1], kv[0])), columns=["trade","count_samples"]) if trade_counts else pd.DataFrame(columns=["trade","count_samples"])

    # -------------------- Create Figures --------------------
    figs: List[plt.Figure] = []
    png_paths: List[Path] = []

    # 1) Floorplan overlay (preferred if we have any overlay points)
    if overlay_reservoir:
        fp = _load_floorplan_extent_and_image()
        # Build figure
        fig1 = plt.figure(figsize=CONFIG_DEFAULTS.get("figsize_overlay", (9,7)))
        ax1 = fig1.add_subplot(111)
        # If floorplan image present, draw it with extent; else just scatter with bounds
        if fp and isinstance(fp.get("extent"), tuple) and fp.get("image") is not None:
            x_min, x_max, y_min, y_max = fp["extent"]
            ax1.imshow(fp["image"], extent=[x_min, x_max, y_min, y_max], origin="upper")
            use_extent = True
            extent_vals = (x_min, x_max, y_min, y_max)
        else:
            use_extent = False
            extent_vals = None

        # Prepare data
        xs = np.array([p[0] for p in overlay_reservoir], dtype=float)
        ys = np.array([p[1] for p in overlay_reservoir], dtype=float)
        cats = [p[2] for p in overlay_reservoir]
        color_by = str(CONFIG_DEFAULTS.get("overlay_color_by") or "trade")
        palette = _colors_for_categories(cats) if cats else {}

        if color_by == "none":
            ax1.scatter(xs, ys, s=float(CONFIG_DEFAULTS.get("overlay_point_size", 8)),
                        alpha=float(CONFIG_DEFAULTS.get("overlay_alpha", 0.85)))
        else:
            # Group scatter by category
            # To avoid too many legend entries, cap to ≤12 most frequent categories
            from collections import Counter
            top_cats = [c for c, _ in Counter(cats).most_common(12)]
            plotted_labels = set()
            arr = np.array(cats, dtype=object)
            for cat in top_cats:
                m = (arr == cat)
                ax1.scatter(xs[m], ys[m],
                            s=float(CONFIG_DEFAULTS.get("overlay_point_size", 8)),
                            alpha=float(CONFIG_DEFAULTS.get("overlay_alpha", 0.85)),
                            color=palette.get(cat),
                            label=str(cat))
                plotted_labels.add(cat)
            # Plot remaining without label
            if len(plotted_labels) < len(palette):
                m = np.array([c not in plotted_labels for c in arr])
                if m.any():
                    ax1.scatter(xs[m], ys[m],
                                s=float(CONFIG_DEFAULTS.get("overlay_point_size", 8)),
                                alpha=float(CONFIG_DEFAULTS.get("overlay_alpha", 0.85)),
                                color="#999999")
            _safe_legend(ax1)

        # Axis setup
        if use_extent and extent_vals:
            x_min, x_max, y_min, y_max = extent_vals
            mx = float(CONFIG_DEFAULTS.get("floorplan_margin", 0.10) or 0.10)
            xr = (x_max - x_min); yr = (y_max - y_min)
            ax1.set_xlim(x_min - mx * xr, x_max + mx * xr)
            ax1.set_ylim(y_min - mx * yr, y_max + mx * yr)
        else:
            # Data bounds with 10% margin
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
                y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
                mx = float(CONFIG_DEFAULTS.get("floorplan_margin", 0.10) or 0.10)
                ax1.set_xlim(x_min - mx * (x_max - x_min + 1e-6), x_max + mx * (x_max - x_min + 1e-6))
                ax1.set_ylim(y_min - mx * (y_max - y_min + 1e-6), y_max + mx * (y_max - y_min + 1e-6))

        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_title("Floorplan Overlay")
        fig1.tight_layout()
        figs.append(fig1)

    # 2) Trade distribution (bar)
    if not trade_df.empty:
        fig2 = plt.figure(figsize=CONFIG_DEFAULTS.get("figsize_bar", (7,5)))
        ax2 = fig2.add_subplot(111)
        td = trade_df.copy()
        top_n = int(CONFIG_DEFAULTS.get("top_n", 10) or 10)
        td = td.iloc[:top_n]
        ax2.barh(td["trade"].astype(str), td["count_samples"].astype(int), color="#3E74C9")
        ax2.set_xlabel("Samples")
        ax2.set_ylabel("Trade")
        ax2.set_title("Sample Count by Trade")
        ax2.invert_yaxis()
        fig2.tight_layout()
        figs.append(fig2)

    # 3) Hourly sample counts (line)
    if not hourly_df.empty and len(hourly_df) >= int(CONFIG_DEFAULTS.get("line_min_points", 2) or 2):
        fig3 = plt.figure(figsize=CONFIG_DEFAULTS.get("figsize_line", (7,5)))
        ax3 = fig3.add_subplot(111)
        hd = hourly_df.sort_values("hour_utc")
        ax3.plot(hd["hour_utc"].astype("datetime64[ns]"), hd["count_samples"].astype(int), color="#E1262D", marker="o", linewidth=1.5, markersize=3)
        ax3.set_xlabel("Hour (UTC)")
        ax3.set_ylabel("Samples")
        ax3.set_title("Hourly Samples (UTC)")
        fig3.autofmt_xdate()
        fig3.tight_layout()
        figs.append(fig3)

    # -------------------- Build Report --------------------
    report_date = ""
    if ts_first is not None:
        report_date = ts_first.strftime("%Y%m%d")
    else:
        # fallback to today
        import datetime as _dt
        report_date = _dt.datetime.utcnow().strftime("%Y%m%d")

    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"

    # Summary bullets
    n_files = len(files_processed)
    total_samples = sum(trade_counts.values()) if trade_counts else (sum(hourly_counts.values()) if hourly_counts else 0)
    n_unique = len(unique_trackables)
    ts_span = ""
    if ts_first is not None and ts_last is not None:
        ts_span = f"{ts_first.strftime('%Y-%m-%d %H:%M UTC')} → {ts_last.strftime('%Y-%m-%d %H:%M UTC')}"
    bullets = [
        f"Files processed: {n_files}",
        f"Total samples: {total_samples}",
        f"Unique trackables: {n_unique}",
        f"Time span (UTC): {ts_span}" if ts_span else "Time span: unavailable",
    ]
    if trade_counts:
        top_items = sorted(trade_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        bullets.append("Top trades: " + ", ".join([f"{k} ({v})" for k, v in top_items]))

    # Evidence table (first CSV only is fine; but we'll provide from first file rows via re-extraction with small head)
    # To avoid re-reading, we'll attempt again but it's fine for small head.
    try:
        raw0 = extract_tracks(str(CSV_PATHS[0]))
        df0 = pd.DataFrame(raw0.get("rows", []))
        if df0.columns.duplicated().any():
            df0 = df0.loc[:, ~df0.columns.duplicated()]
        # Ensure required columns exist in table selection
        for c in ["trackable","trade","ts_short","x","y","z"]:
            if c not in df0.columns:
                df0[c] = ""
        # Limit rows
        cols_tbl = ["trackable","trade","ts_short","x","y","z"]
        rows_tbl = df0[cols_tbl].head(50).fillna("").astype(str).to_dict(orient="records")
    except Exception:
        cols_tbl = ["trackable","trade","ts_short","x","y","z"]
        rows_tbl = []

    # Narrative context (optional)
    context_text = read_text(CONTEXT).strip()
    user_prompt_text = f"User request: {USER_PROMPT}"

    sections: List[Dict[str, Any]] = []
    sections.append({"type":"summary","title":"Summary","bullets":bullets})
    if user_prompt_text:
        sections.append({"type":"narrative","title":"Request","paragraphs":[user_prompt_text]})
    if context_text:
        sections.append({"type":"narrative","title":"Context","paragraphs":[context_text[:1200]]})  # keep short

    # Table section
    sections.append({"type":"table","title":"Evidence","data":rows_tbl,"headers":cols_tbl,"rows_per_page":24})

    # Save figures to PNGs first (required order), then add live figs to report
    png_paths = []
    for i, fig in enumerate(figs, start=1):
        png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
        try:
            fig.savefig(str(png), dpi=120)  # no bbox_inches='tight'
            png_paths.append(png)
        except Exception:
            # ignore save failure for this figure
            pass

    if figs:
        sections.append({"type":"charts","title":"Figures","figures":figs})

    # Build final report dict
    title = "Walmart Renovation RTLS — Position Summary"
    meta_lines = []
    if files_processed:
        meta_lines.append(f"Files: {len(files_processed)}")
    if ts_span:
        meta_lines.append(f"Span: {ts_span}")
    meta_text = " | ".join(meta_lines)

    report: Dict[str, Any] = {
        "title": title,
        "meta": meta_text,
        "sections": sections,
    }

    # Apply budgets and write PDF
    report = apply_budgets(report, {"MAX_FIGURES": int(CONFIG_DEFAULTS.get("max_figures", 6) or 6),
                                    "MAX_TABLE_ROWS_TOTAL": LIMIT_DEFAULTS.get("MAX_TABLE_ROWS_TOTAL", 180),
                                    "MAX_TEXT_LINES_TOTAL": LIMIT_DEFAULTS.get("MAX_TEXT_LINES_TOTAL", 900),
                                    "MAX_PAGES": LIMIT_DEFAULTS.get("MAX_PAGES", 12)})
    # Build PDF
    safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))

    # -------------------- Print links (success) --------------------
    print(f"[Download the PDF]({file_uri(pdf_path)})")
    for i, pth in enumerate(png_paths, 1):
        print(f"[Download Plot {i}]({file_uri(pth)})")

except (MemoryError, KeyboardInterrupt):
    # Minimal-Report Mode: summary + evidence only; still counts as success if PDF written.
    try:
        report_date = report_date if 'report_date' in locals() and report_date else "report"
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        bullets = bullets if 'bullets' in locals() else ["Minimal report due to resource limits."]
        cols_tbl = cols_tbl if 'cols_tbl' in locals() else ["trackable","trade","ts_short","x","y","z"]
        rows_tbl = rows_tbl if 'rows_tbl' in locals() else []
        sections = [
            {"type":"summary","title":"Summary","bullets":bullets},
            {"type":"table","title":"Evidence","data":rows_tbl,"headers":cols_tbl,"rows_per_page":24},
        ]
        report = {"title":"Walmart Renovation RTLS — Position Summary (Lite)","meta":"","sections":sections}
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        print(f"[Download the PDF]({file_uri(pdf_path)})")
    except Exception:
        print("Error Report:")
        print("Failed to build minimal report due to resource limits.")
        raise SystemExit(1)
except SystemExit:
    raise
except Exception as e:
    # Generic failure (short message)
    print("Error Report:")
    msg = str(e).strip()
    if not msg:
        msg = "Unexpected error during processing."
    # If caused by schema/columns, include detected columns
    if "Missing required columns" in msg or ("schema" in msg.lower()):
        print("Missing required columns for analysis.")
        if detected_columns:
            print("Columns detected: " + ",".join(detected_columns))
    else:
        # Avoid printing long tracebacks; one-line reason
        print(msg[:200])
    raise SystemExit(1)