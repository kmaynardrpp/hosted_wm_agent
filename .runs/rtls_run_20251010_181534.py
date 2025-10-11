import sys, os
from pathlib import Path

# ------------------------ ROOT resolution and local imports ------------------------
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

# Read guidelines/context silently (Windows-safe UTF-8)
_ = read_text(GUIDELINES)
_ = read_text(CONTEXT)

# ------------------------ Imports (helpers and libs) ------------------------
import json
import math
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for reliable PNG/PDF writes
import matplotlib.pyplot as plt

from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from report_limits import apply_budgets, make_lite

# ------------------------ Utility helpers ------------------------
def load_config() -> dict:
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
            data = json.loads(CONFIG.read_text(encoding="utf-8", errors="ignore"))
            cfg.update({k: v for k, v in data.items() if k in cfg})
    except Exception:
        pass
    return cfg

def parse_floorplan() -> dict | None:
    """
    Load floorplans.json and floorplan image (prefer floorplan.jpeg/png/jpg under ROOT).
    Compute world rectangle in mm.
    Returns dict: {"extent": (x_min, x_max, y_min, y_max), "image": np.ndarray}
    """
    if not FLOORJSON.exists():
        return None
    try:
        data = json.loads(FLOORJSON.read_text(encoding="utf-8", errors="ignore"))
        plans = data.get("floorplans") or data.get("plans") or data
        if isinstance(plans, list):
            fp = plans[0] if plans else None
        else:
            fp = plans
        if not fp:
            return None
        width  = float(fp.get("width", 0))
        height = float(fp.get("height", 0))
        x_c    = float(fp.get("image_offset_x", 0))
        y_c    = float(fp.get("image_offset_y", 0))
        image_scale = float(fp.get("image_scale", 0))  # meters per pixel

        scale_mm_per_px = image_scale * 100.0
        x_min = (x_c - width / 2.0)  * scale_mm_per_px
        x_max = (x_c + width / 2.0)  * scale_mm_per_px
        y_min = (y_c - height / 2.0) * scale_mm_per_px
        y_max = (y_c + height / 2.0) * scale_mm_per_px

        # Find raster under ROOT (prefer JPEG)
        img_path_candidates = [ROOT / "floorplan.jpeg", ROOT / "floorplan.jpg", ROOT / "floorplan.png"]
        img = None
        for p in img_path_candidates:
            if p.exists():
                try:
                    img = plt.imread(str(p))
                    break
                except Exception:
                    continue
        if img is None:
            return None
        return {"extent": (x_min, x_max, y_min, y_max), "image": img}
    except Exception:
        return None

def iso_utc(dt_like: pd.Timestamp | None) -> str:
    if dt_like is None or pd.isna(dt_like):
        return ""
    try:
        # Convert to UTC-naive and add single 'Z' (avoid double 'Z')
        if getattr(dt_like, "tzinfo", None) is not None:
            dt_naive = dt_like.tz_convert("UTC").tz_localize(None)
        else:
            dt_naive = dt_like
        return dt_naive.isoformat(timespec="seconds") + "Z"
    except Exception:
        try:
            return str(dt_like)
        except Exception:
            return ""

def color_map_for_categories(categories: list[str]) -> dict:
    base = plt.cm.get_cmap("tab10")
    uniq = list(dict.fromkeys([str(c) for c in categories]))
    return {c: base(i % 10) for i, c in enumerate(uniq)}

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def main():
    # ------------------------ CLI parsing ------------------------
    if len(sys.argv) < 2:
        # No prompt, no CSVs; produce minimal PDF in ROOT
        user_prompt = "No prompt provided."
        csv_paths: list[Path] = []
        out_dir = ROOT
    else:
        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:] if p and Path(p).exists()]
        out_dir = (csv_paths[0].parent if csv_paths else ROOT)

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    # ------------------------ Aggregation holders ------------------------
    # Large-data mode: per-file processing; keep small aggregates in RAM
    hourly_counts: dict[pd.Timestamp, int] = {}
    trade_counts: dict[str, int] = {}
    total_samples = 0
    uid_set: set[str] = set()
    trackable_set: set[str] = set()
    ts_min: pd.Timestamp | None = None
    ts_max: pd.Timestamp | None = None

    # Overlay reservoir (bounded)
    overlay_points: list[tuple[float, float, str]] = []

    # For evidence table, keep first-file head safely
    first_df_for_evidence: pd.DataFrame | None = None

    # ------------------------ Handle no CSV case early ------------------------
    if not csv_paths:
        # Build a minimal report with a summary only
        report_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        sections = []
        sections.append({
            "type": "summary",
            "title": "Summary",
            "bullets": [
                "No CSV inputs were provided.",
                f"User query: {user_prompt}",
            ],
        })
        report = {
            "title": "Walmart Renovation RTLS — Summary",
            "meta": "Generated with no input data.",
            "sections": sections,
        }
        report = apply_budgets(report, None)
        try:
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except Exception as e:
            print("Error Report:")
            print(f"PDF build failed: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            try:
                report = make_lite(report)
                safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            except Exception as e2:
                print("Error Report:")
                print(f"Lite PDF failed: {e2.__class__.__name__}: {e2}")
                traceback.print_exc()
                raise SystemExit(1)
        print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        return

    # ------------------------ Process CSVs (per-file) ------------------------
    schema_validated = False
    zones_requested = ("zone" in user_prompt.lower())  # zones only if user asked
    for idx, csv_p in enumerate(csv_paths):
        try:
            raw = extract_tracks(str(csv_p), mac_map_path=str(ROOT / "trackable_objects.json"))
        except Exception as e:
            print("Error Report:")
            print(f"Failed to read CSV: {e.__class__.__name__}: {e}")
            raise SystemExit(1)

        audit = raw.get("audit", {}) or {}
        if audit.get("mac_map_loaded") is False:
            print("Error Report:")
            print("MAC mapping could not be loaded (trackable_objects.json).")
            raise SystemExit(1)

        rows = raw.get("rows", []) or []
        df = pd.DataFrame(rows)
        # Duplicate-name guard
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Early continue if empty file
        if df.empty:
            if not schema_validated:
                # Cannot validate schema from empty file; continue scanning other files
                continue

        # Timestamp canon
        src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"] if "ts" in df.columns else None
        if src is None:
            # Create missing ts_utc with NaT to pass schema check? No, schema expects x,y but ts_utc is needed for analytics.
            df["ts_utc"] = pd.NaT
        else:
            df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

        # Schema validation after first non-empty file
        if not schema_validated and not df.empty:
            cols = set(df.columns.astype(str).tolist())
            identity_ok = ("trackable" in cols) or ("trackable_uid" in cols)
            trade_ok = ("trade" in cols)
            xy_ok = ("x" in cols) and ("y" in cols)
            zones_ok = True
            if zones_requested:
                zones_ok = ("zone_name" in cols) or ZONES_JSON.exists()
            if not (identity_ok and trade_ok and xy_ok and zones_ok):
                print("Error Report:")
                print("Missing required columns for analysis.")
                print(f"Columns detected: {','.join(df.columns.astype(str))}")
                raise SystemExit(1)
            schema_validated = True
            if first_df_for_evidence is None:
                first_df_for_evidence = df.copy()

        # Column pruning for performance
        keep_cols = [c for c in ["trackable", "trackable_uid", "trade", "mac", "ts", "ts_iso", "ts_short", "x", "y", "z", "ts_utc", "zone_name"] if c in df.columns]
        df = df.loc[:, keep_cols].copy()

        # Valid rows for analytics (require ts_utc, x, y)
        x_num = safe_numeric(df["x"]) if "x" in df.columns else pd.Series([], dtype=float)
        y_num = safe_numeric(df["y"]) if "y" in df.columns else pd.Series([], dtype=float)
        mask_valid = pd.Series(True, index=df.index)
        if "ts_utc" in df.columns:
            mask_valid &= df["ts_utc"].notna()
        if "x" in df.columns:
            mask_valid &= x_num.notna()
        if "y" in df.columns:
            mask_valid &= y_num.notna()
        valid = df.loc[mask_valid].copy()
        if not valid.empty:
            # Update time range
            vmin = valid["ts_utc"].min()
            vmax = valid["ts_utc"].max()
            ts_min = vmin if ts_min is None else min(ts_min, vmin)  # type: ignore[arg-type]
            ts_max = vmax if ts_max is None else max(ts_max, vmax)  # type: ignore[arg-type]

            # Hourly counts (UTC floor 'h')
            hours = valid["ts_utc"].dt.floor("h")
            vc = hours.value_counts()
            for h, cnt in vc.items():
                hourly_counts[h] = hourly_counts.get(h, 0) + int(cnt)

            # Trade counts
            if "trade" in valid.columns:
                for t, cnt in valid["trade"].fillna("").astype(str).value_counts().items():
                    trade_counts[t] = trade_counts.get(t, 0) + int(cnt)

            # Unique IDs and names (best-effort)
            if "trackable_uid" in valid.columns:
                uid_set.update(set(valid["trackable_uid"].astype(str)))
            if "trackable" in valid.columns:
                trackable_set.update(set(valid["trackable"].astype(str)))

            # Overlay reservoir (bounded)
            color_by = (cfg.get("overlay_color_by") or "trade")
            color_col = color_by if color_by in valid.columns else None
            for _, r in valid.iterrows():
                x_val = pd.to_numeric(r.get("x"), errors="coerce")
                y_val = pd.to_numeric(r.get("y"), errors="coerce")
                if pd.isna(x_val) or pd.isna(y_val):
                    continue
                cat = str(r.get(color_col, "")) if color_col else ""
                overlay_points.append((float(x_val), float(y_val), cat))
                if len(overlay_points) > int(cfg["overlay_subsample"]) * 2:
                    # Decimate in place to keep memory bounded
                    step = 2
                    overlay_points[:] = overlay_points[::step]

            total_samples += int(len(valid))

        # Release memory for this file (figures are created later)
        del df, valid, x_num, y_num

    # If never found a non-empty file to validate schema, create minimal report
    if not schema_validated:
        report_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        sections = []
        sections.append({
            "type": "summary",
            "title": "Summary",
            "bullets": [
                "No valid rows were found in the provided CSV(s).",
                f"CSV files: {len(csv_paths)}",
                f"User query: {user_prompt}"
            ],
        })
        report = {
            "title": "Walmart Renovation RTLS — Summary",
            "meta": "No valid data to analyze.",
            "sections": sections,
        }
        report = apply_budgets(report, None)
        try:
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except Exception as e:
            print("Error Report:")
            print(f"PDF build failed: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            try:
                report = make_lite(report)
                safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            except Exception as e2:
                print("Error Report:")
                print(f"Lite PDF failed: {e2.__class__.__name__}: {e2}")
                traceback.print_exc()
                raise SystemExit(1)
        print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        return

    # ------------------------ Final aggregates ------------------------
    # Hourly DataFrame sorted
    if hourly_counts:
        h_idx = sorted(hourly_counts.keys())
        hourly_df = pd.DataFrame({
            "hour_utc": h_idx,
            "count_samples": [hourly_counts[h] for h in h_idx],
        })
    else:
        hourly_df = pd.DataFrame(columns=["hour_utc", "count_samples"])

    # Trade DataFrame sorted
    if trade_counts:
        items = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)
        trade_df = pd.DataFrame(items, columns=["trade", "count_samples"])
    else:
        trade_df = pd.DataFrame(columns=["trade", "count_samples"])

    # ------------------------ Figures ------------------------
    figs: list[plt.Figure] = []
    png_paths: list[Path] = []
    report_date = (ts_min.tz_convert("UTC").strftime("%Y%m%d") if ts_min is not None else datetime.utcnow().strftime("%Y%m%d"))

    # 1) Floorplan overlay scatter (prefer if floorplan + points available)
    overlay_fig = None
    if overlay_points and cfg.get("prefer_floorplan", True):
        # Subsample to budget
        max_pts = int(cfg.get("overlay_subsample", 20000))
        if len(overlay_points) > max_pts:
            # Evenly spaced selection
            idx = np.linspace(0, len(overlay_points) - 1, max_pts).astype(int)
            overlay_points = [overlay_points[i] for i in idx]

        fp = parse_floorplan()
        if fp is not None:
            try:
                extent = fp["extent"]  # (x_min, x_max, y_min, y_max)
                img = fp["image"]
                overlay_fig = plt.figure(figsize=tuple(cfg.get("figsize_overlay", (9, 7))))
                ax = overlay_fig.add_subplot(111)
                # Draw raster with origin='upper'
                ax.imshow(img, extent=[extent[0], extent[1], extent[2], extent[3]], origin="upper")

                # Colors
                cats = [p[2] for p in overlay_points]
                cmap = color_map_for_categories(cats)
                # Plot points grouped by category for legend control
                from collections import defaultdict
                grp = defaultdict(list)
                for x, y, c in overlay_points:
                    grp[c].append((x, y))
                for c, pts in grp.items():
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    ax.scatter(xs, ys,
                               s=float(cfg.get("overlay_point_size", 8)),
                               alpha=float(cfg.get("overlay_alpha", 0.85)),
                               color=cmap.get(str(c), None),
                               label=str(c) if c else None)
                handles, labels = ax.get_legend_handles_labels()
                if len([l for l in labels if l]) <= 12 and any(labels):
                    ax.legend(loc="upper left", frameon=True, fontsize=8)

                # Axis limits + margin
                x_min, x_max, y_min, y_max = extent
                mx = float(cfg.get("floorplan_margin", 0.10))
                xr = (x_max - x_min); yr = (y_max - y_min)
                ax.set_xlim(x_min - mx * xr, x_max + mx * xr)
                ax.set_ylim(y_min - mx * yr, y_max + mx * yr)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("X (mm)")
                ax.set_ylabel("Y (mm)")
                ax.set_title("Floorplan Overlay")
                overlay_fig.tight_layout()
                figs.append(overlay_fig)
            except Exception:
                if overlay_fig is not None:
                    plt.close(overlay_fig)
                overlay_fig = None

    # 2) Hourly line chart
    line_fig = None
    if not hourly_df.empty and len(hourly_df) >= int(cfg.get("line_min_points", 2)):
        try:
            # Ensure tz-naive for Matplotlib plotting
            hrs = pd.to_datetime(hourly_df["hour_utc"], utc=True, errors="coerce")
            hrs_naive = hrs.dt.tz_convert("UTC").dt.tz_localize(None)
            counts = pd.to_numeric(hourly_df["count_samples"], errors="coerce").fillna(0)

            line_fig = plt.figure(figsize=tuple(cfg.get("figsize_line", (7, 5))))
            ax = line_fig.add_subplot(111)
            ax.plot(hrs_naive, counts, marker="o", linestyle="-", color="#1f77b4")
            ax.set_title("Hourly Sample Counts (UTC)")
            ax.set_xlabel("Hour (UTC)")
            ax.set_ylabel("Samples")
            ax.grid(True, alpha=0.3)
            line_fig.autofmt_xdate()
            line_fig.tight_layout()
            figs.append(line_fig)
        except Exception:
            if line_fig is not None:
                plt.close(line_fig)
            line_fig = None

    # 3) Trade distribution (pie if few, else bar)
    trade_fig = None
    if not trade_df.empty:
        try:
            top_n = int(cfg.get("top_n", 10))
            td = trade_df.sort_values("count_samples", ascending=False).head(top_n)
            n_trades = len(td)
            total = td["count_samples"].sum()
            shares = (td["count_samples"] / max(1, total)).tolist()
            pie_ok = n_trades <= int(cfg.get("pie_max_trades", 8)) and (max(shares) if shares else 0) <= float(cfg.get("pie_max_single_share", 0.90))

            if pie_ok:
                trade_fig = plt.figure(figsize=tuple(cfg.get("figsize_pie", (5, 5))))
                ax = trade_fig.add_subplot(111)
                ax.pie(td["count_samples"], labels=td["trade"], autopct="%1.0f%%", startangle=90, counterclock=False)
                ax.set_title("Trade Share (by samples)")
                trade_fig.tight_layout()
            else:
                trade_fig = plt.figure(figsize=tuple(cfg.get("figsize_bar", (7, 5))))
                ax = trade_fig.add_subplot(111)
                ax.barh(td["trade"], td["count_samples"], color="#2ca02c")
                ax.set_title("Top Trades (by samples)")
                ax.set_xlabel("Samples")
                ax.set_ylabel("Trade")
                ax.invert_yaxis()
                trade_fig.tight_layout()
            figs.append(trade_fig)
        except Exception:
            if trade_fig is not None:
                plt.close(trade_fig)
            trade_fig = None

    # Save PNGs first (MANDATORY) then add to report
    png_paths = []
    for i, fig in enumerate(figs, 1):
        try:
            png_path = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            fig.savefig(str(png_path), dpi=120)  # no bbox_inches='tight'
            png_paths.append(png_path)
        except Exception:
            # If saving fails for a figure, drop it from figs
            png_path = None

    # Filter out any Nones or non-figure objects before passing to PDF
    figs = [f for f in figs if getattr(f, "savefig", None)]

    # ------------------------ Build report ------------------------
    title = "Walmart Renovation RTLS — Summary"
    t_min = iso_utc(ts_min) if ts_min is not None else ""
    t_max = iso_utc(ts_max) if ts_max is not None else ""
    meta_lines = []
    meta_lines.append(f"Files: {len(csv_paths)}")
    meta_lines.append(f"Samples: {total_samples}")
    if t_min and t_max:
        meta_lines.append(f"UTC Range: {t_min} → {t_max}")
    meta = " | ".join(meta_lines)

    bullets = []
    bullets.append(f"Analyzed {len(csv_paths)} file(s); {total_samples} valid position samples.")
    if uid_set:
        bullets.append(f"Unique trackables (UIDs): {len(uid_set)}")
    if trackable_set:
        bullets.append(f"Unique display names: {len(trackable_set)}")
    if not trade_df.empty:
        top_row = trade_df.sort_values("count_samples", ascending=False).iloc[0]
        bullets.append(f"Top trade by samples: {top_row['trade']} ({int(top_row['count_samples'])})")
    if t_min and t_max:
        bullets.append(f"Time window (UTC): {t_min} to {t_max}")
    if not bullets:
        bullets.append("Summary generated (no additional insights available).")

    sections: list[dict] = []
    sections.append({
        "type": "summary",
        "title": "Summary",
        "bullets": bullets,
    })

    # Evidence table from first_df_for_evidence
    try:
        if first_df_for_evidence is not None and not first_df_for_evidence.empty:
            cols = ["trackable","trade","ts_short","x","y","z"]
            present_cols = [c for c in cols if c in first_df_for_evidence.columns]
            if present_cols:
                rows = (first_df_for_evidence[present_cols].head(50).fillna("").astype(str).to_dict(orient="records"))
                sections.append({"type":"table","title":"Evidence","data":rows,"headers":present_cols,"rows_per_page":24})
    except Exception:
        pass

    if figs:
        sections.append({"type":"charts","title":"Figures","figures":figs})

    report = {"title": title, "meta": meta, "sections": sections}
    report = apply_budgets(report, None)

    # ------------------------ Write PDF with fallback ------------------------
    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
    try:
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
    except Exception as e:
        print("Error Report:")
        print(f"PDF build failed: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        try:
            report = make_lite(report)
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except Exception as e2:
            print("Error Report:")
            print(f"Lite PDF failed: {e2.__class__.__name__}: {e2}")
            traceback.print_exc()
            raise SystemExit(1)

    # ------------------------ Success: print links ------------------------
    print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
    for i, p in enumerate(png_paths, 1):
        print(f"[Download Plot {i}](file:///{p.resolve().as_posix()})")

    # ------------------------ Cleanup (close figs AFTER PDF) ------------------------
    try:
        for f in figs:
            plt.close(f)
        plt.close("all")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        # Generic failure (non-schema). Keep output minimal.
        print("Error Report:")
        print(f"Unhandled error: {e.__class__.__name__}: {e}")