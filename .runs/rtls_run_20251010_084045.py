#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, json, math, traceback
from pathlib import Path

# ------------------------ Resolve ROOT and imports ------------------------
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

# ------------------------ Helper imports (validate presence) ------------------------
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Error Report:")
    print("Matplotlib/Pandas/Numpy not available: " + str(e))
    raise SystemExit(1)

try:
    from extractor import extract_tracks
except Exception as e:
    print("Error Report:")
    print("Missing required helper: extractor.extract_tracks")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception as e:
    print("Error Report:")
    print("Missing required helper: pdf_creation_script.safe_build_pdf")
    raise SystemExit(1)

try:
    from report_limits import apply_budgets
except Exception:
    # Fallback: simple passthrough if budgets module missing
    def apply_budgets(report, caps=None):
        return report

# ------------------------ CLI parse ------------------------
def main():
    try:
        # Read and acknowledge guidelines (Windows-safe text)
        _ = read_text(GUIDELINES)

        if len(sys.argv) < 3:
            print("Error Report:")
            print("Usage: python script.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
            return

        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:] if p and Path(p).exists() and Path(p).suffix.lower() == ".csv"]
        if not csv_paths:
            print("Error Report:")
            print("No valid CSV inputs were provided.")
            return

        out_dir = csv_paths[0].resolve().parent

        # Load config if present
        cfg = {}
        try:
            if CONFIG.exists():
                cfg = json.loads(read_text(CONFIG) or "{}")
        except Exception:
            cfg = {}
        # Defaults if missing
        cfg.setdefault("prefer_floorplan", True)
        cfg.setdefault("floorplan_margin", 0.10)
        cfg.setdefault("overlay_point_size", 8)
        cfg.setdefault("overlay_alpha", 0.85)
        cfg.setdefault("overlay_color_by", "trade")
        cfg.setdefault("overlay_subsample", 20000)
        cfg.setdefault("draw_trails", False)
        cfg.setdefault("trail_seconds", 900)
        cfg.setdefault("draw_zones", True)
        cfg.setdefault("zone_face_alpha", 0.20)
        cfg.setdefault("zone_edge_alpha", 0.65)
        cfg.setdefault("top_n", 10)
        cfg.setdefault("pie_max_trades", 8)
        cfg.setdefault("pie_max_single_share", 0.90)
        cfg.setdefault("line_min_points", 2)
        cfg.setdefault("small_multiples_cols", 2)
        cfg.setdefault("max_figures", 6)
        cfg.setdefault("figsize_overlay", [9, 7])
        cfg.setdefault("figsize_bar", [7, 5])
        cfg.setdefault("figsize_line", [7, 5])
        cfg.setdefault("figsize_pie", [5, 5])
        cfg.setdefault("figsize_box", [7, 5])

        prefer_floorplan = bool(cfg.get("prefer_floorplan", True))
        overlay_subsample = int(cfg.get("overlay_subsample", 20000) or 20000)

        # Intent: zones only if asked (user_prompt text check)
        zones_requested = any(w in user_prompt.lower() for w in ["zone", "room", "area"])  # basic intent
        # The policy: If not asked, DO NOT compute zones.
        # We'll not compute zones unless zones_requested is True.

        # Aggregates (small, in-RAM)
        total_samples = 0
        trade_counts = {}
        hour_counts = {}  # key: UTC Timestamp, val: int
        global_min_ts = None
        global_max_ts = None

        # Overlay reservoir (bounded)
        overlay_limit_total = max(1000, overlay_subsample)
        per_file_target = max(1, overlay_limit_total // max(1, len(csv_paths)))
        overlay_data_cols = ["x", "y", "trade"]
        overlay_reservoir = []  # list of tuples (x, y, trade)

        # Evidence buffer (across files up to 50 rows)
        evidence_cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
        evidence_buffer = []

        # For meta
        file_audits = []
        files_processed = 0

        # Process each CSV independently (large-data mode)
        for idx, csv_path in enumerate(csv_paths):
            raw = extract_tracks(str(csv_path))
            rows = raw.get("rows", [])
            audit = raw.get("audit", {})
            file_audits.append(audit)

            df = pd.DataFrame(rows)
            # Duplicate-name guard
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            # Timestamp canon
            src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
            df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

            # Early schema validation after first file
            if idx == 0:
                cols = set(df.columns.astype(str))
                identity_ok = ("trackable" in cols) or ("trackable_uid" in cols)
                trade_ok = ("trade" in cols)
                xy_ok = ("x" in cols) and ("y" in cols)
                # Zones: only if asked
                if not (identity_ok and trade_ok and xy_ok):
                    print("Error Report:")
                    print("Missing required columns for analysis.")
                    print("Columns detected: " + ",".join(df.columns.astype(str)))
                    return
                if zones_requested:
                    # Must have zone_name or be able to compute; we check later if needed
                    pass

            # Minimal column pruning for ops
            keep_cols = [c for c in ["trackable", "trackable_uid", "trade", "ts", "ts_iso", "ts_short", "x", "y", "z", "ts_utc", "zone_name"] if c in df.columns]
            df = df[keep_cols]

            # Evidence (first 50 rows total across files)
            if len(evidence_buffer) < 50:
                have_cols = [c for c in evidence_cols if c in df.columns]
                if have_cols:
                    # fill missing with empty string, to dicts
                    safe = df.copy()
                    for c in evidence_cols:
                        if c not in safe.columns:
                            safe[c] = ""
                    ev_rows = (safe[evidence_cols].head(50 - len(evidence_buffer)).fillna("")
                               .astype(str).to_dict(orient="records"))
                    evidence_buffer.extend(ev_rows)

            # Aggregates
            # Count samples
            total_samples += len(df)

            # Time extents
            ts_valid = df["ts_utc"].dropna()
            if not ts_valid.empty:
                mn = ts_valid.min()
                mx = ts_valid.max()
                global_min_ts = mn if (global_min_ts is None or mn < global_min_ts) else global_min_ts
                global_max_ts = mx if (global_max_ts is None or mx > global_max_ts) else global_max_ts

            # Hourly counts (UTC)
            if not ts_valid.empty:
                hr = ts_valid.dt.floor("H")
                vc = hr.value_counts()
                for k, v in vc.items():
                    hour_counts[k] = hour_counts.get(k, 0) + int(v)

            # Trade counts
            if "trade" in df.columns:
                vc = df["trade"].astype(str).replace({None: "", "None": ""}).fillna("").value_counts()
                for k, v in vc.items():
                    trade_counts[k] = trade_counts.get(k, 0) + int(v)

            # Overlay reservoir for floorplan scatter
            # Cast x,y to numeric for sampling; keep strings elsewhere
            if "x" in df.columns and "y" in df.columns:
                xnum = pd.to_numeric(df["x"], errors="coerce")
                ynum = pd.to_numeric(df["y"], errors="coerce")
                mask = xnum.notna() & ynum.notna()
                if mask.any():
                    sub = df.loc[mask, ["x", "y"] + (["trade"] if "trade" in df.columns else [])].copy()
                    # Ensure 'trade' exists
                    if "trade" not in sub.columns:
                        sub["trade"] = ""
                    n = len(sub)
                    if n > per_file_target:
                        # uniform subsample
                        idxs = np.linspace(0, n - 1, per_file_target).astype(int)
                        sub = sub.iloc[idxs]
                    # Append to reservoir (tuples to be memory-light)
                    overlay_reservoir.extend(list(zip(
                        pd.to_numeric(sub["x"], errors="coerce").astype(float).tolist(),
                        pd.to_numeric(sub["y"], errors="coerce").astype(float).tolist(),
                        sub["trade"].astype(str).tolist()
                    )))
                    # Enforce global cap (simple thinning if needed)
                    if len(overlay_reservoir) > overlay_limit_total:
                        step = max(1, len(overlay_reservoir) // overlay_limit_total)
                        overlay_reservoir = overlay_reservoir[::step]

            files_processed += 1
            # Drop big frame before next
            del df

        # Prepare overlay DataFrame from reservoir
        overlay_df = pd.DataFrame(overlay_reservoir, columns=["x", "y", "trade"]) if overlay_reservoir else pd.DataFrame(columns=["x", "y", "trade"])

        # Build summary bullets
        bullets = []
        bullets.append(f"Files processed: {files_processed}")
        bullets.append(f"Total samples: {total_samples:,}")
        if global_min_ts is not None and global_max_ts is not None:
            # Avoid double 'Z' in meta strings; show ISO but without enforcing 'Z' twice
            bullets.append(f"UTC time span: {str(global_min_ts)} to {str(global_max_ts)}")
        if trade_counts:
            non_empty_trades = [(k or "(unlabeled)") for k in []]  # place-holder to avoid syntax error
        # The above placeholder had a syntax error; rebuild trade bullets cleanly
        if trade_counts:
            tc_sorted = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)
            top_n = min(5, len(tc_sorted))
            top_parts = [f"{(k or '(unlabeled)')}: {v:,}" for k, v in tc_sorted[:top_n]]
            bullets.append("Top trades by sample count: " + "; ".join(top_parts))

        # Narrative meta
        meta_lines = []
        if files_processed:
            meta_lines.append(f"User query: {user_prompt}")
            meta_lines.append(f"CSV inputs: {len(csv_paths)}")
        # Attach any context text
        ctx_text = read_text(CONTEXT).strip()
        if ctx_text:
            meta_lines.append("Context loaded")
        meta_text = " | ".join(meta_lines)

        # --------- Figures generation ---------
        figs = []
        png_paths = []

        def _colors_for_categories(categories):
            base = plt.cm.get_cmap("tab10")
            uniq = list(dict.fromkeys(categories))
            return {c: base(i % 10) for i, c in enumerate(uniq)}

        # Floorplan loader (ROOT / floorplan.(png|jpg|jpeg), extent from FLOORJSON)
        def load_floorplan_assets():
            if not FLOORJSON.exists():
                return None
            try:
                data = json.loads(read_text(FLOORJSON) or "{}")
                fp = data.get("floorplans") or data.get("plans") or data
                if isinstance(fp, list):
                    fp = fp[0] if fp else None
                if not fp:
                    return None
                width  = float(fp.get("width", 0))
                height = float(fp.get("height", 0))
                x_c    = float(fp.get("image_offset_x", 0))
                y_c    = float(fp.get("image_offset_y", 0))
                image_scale = float(fp.get("image_scale", 0))
                if width <= 0 or height <= 0 or image_scale <= 0:
                    return None
                # mm/px
                scale = image_scale * 100.0
                x_min = (x_c - width/2.0)  * scale
                x_max = (x_c + width/2.0)  * scale
                y_min = (y_c - height/2.0) * scale
                y_max = (y_c + height/2.0) * scale
                # try local floorplan image under ROOT
                candidates = [ROOT / "floorplan.png", ROOT / "floorplan.jpg", ROOT / "floorplan.jpeg"]
                img_path = None
                for c in candidates:
                    if c.exists():
                        img_path = c; break
                if img_path is None:
                    return None
                img = plt.imread(str(img_path))
                return {"extent": (x_min, x_max, y_min, y_max), "image": img}
            except Exception:
                return None

        # Chart 1: Floorplan overlay or fallback scatter
        overlay_fig = None
        try:
            if prefer_floorplan and not overlay_df.empty:
                fp = load_floorplan_assets()
                if fp is not None:
                    fig = plt.figure(figsize=tuple(cfg.get("figsize_overlay", [9, 7])))
                    ax = fig.add_subplot(111)
                    x_min, x_max, y_min, y_max = fp["extent"]
                    ax.imshow(fp["image"], extent=[x_min, x_max, y_min, y_max], origin="upper")

                    # Decimate if needed
                    use = overlay_df
                    if len(use) > overlay_subsample:
                        idxs = np.linspace(0, len(use) - 1, overlay_subsample).astype(int)
                        use = use.iloc[idxs]

                    color_by = str(cfg.get("overlay_color_by") or "trade")
                    if color_by not in use.columns:
                        color_by = "none"
                    if color_by == "none":
                        ax.scatter(use["x"], use["y"], s=float(cfg.get("overlay_point_size", 8)),
                                   alpha=float(cfg.get("overlay_alpha", 0.85)))
                    else:
                        palette = _colors_for_categories(use[color_by].astype(str).tolist())
                        for cat, g in use.groupby(color_by):
                            ax.scatter(g["x"], g["y"], s=float(cfg.get("overlay_point_size", 8)),
                                       color=palette.get(cat), alpha=float(cfg.get("overlay_alpha", 0.85)),
                                       label=str(cat))
                        handles, labels = ax.get_legend_handles_labels()
                        if len(labels) <= 12 and len(labels) > 0:
                            ax.legend(loc="upper left", frameon=True, fontsize=8)

                    mx = float(cfg.get("floorplan_margin", 0.10))
                    xr = (x_max - x_min); yr = (y_max - y_min)
                    ax.set_xlim(x_min - mx * xr, x_max + mx * xr)
                    ax.set_ylim(y_min - mx * yr, y_max + mx * yr)
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
                    ax.set_title("Floorplan Overlay")
                    fig.tight_layout()
                    overlay_fig = fig
                else:
                    # Fallback scatter without image
                    if not overlay_df.empty:
                        fig = plt.figure(figsize=tuple(cfg.get("figsize_overlay", [9, 7])))
                        ax = fig.add_subplot(111)
                        use = overlay_df
                        if len(use) > overlay_subsample:
                            idxs = np.linspace(0, len(use) - 1, overlay_subsample).astype(int)
                            use = use.iloc[idxs]
                        color_by = str(cfg.get("overlay_color_by") or "trade")
                        if color_by not in use.columns:
                            color_by = "none"
                        if color_by == "none":
                            ax.scatter(use["x"], use["y"], s=float(cfg.get("overlay_point_size", 8)),
                                       alpha=float(cfg.get("overlay_alpha", 0.85)))
                        else:
                            palette = _colors_for_categories(use[color_by].astype(str).tolist())
                            for cat, g in use.groupby(color_by):
                                ax.scatter(g["x"], g["y"], s=float(cfg.get("overlay_point_size", 8)),
                                           color=palette.get(cat), alpha=float(cfg.get("overlay_alpha", 0.85)),
                                           label=str(cat))
                            handles, labels = ax.get_legend_handles_labels()
                            if len(labels) <= 12 and len(labels) > 0:
                                ax.legend(loc="best", frameon=True, fontsize=8)
                        ax.set_aspect("equal", adjustable="box")
                        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
                        ax.set_title("Position Overlay (no floorplan)")
                        fig.tight_layout()
                        overlay_fig = fig
        except Exception:
            overlay_fig = None

        if overlay_fig is not None:
            figs.append(overlay_fig)

        # Chart 2: Trade distribution (bar)
        try:
            if trade_counts:
                td_sorted = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)
                cats = [k if k else "(unlabeled)" for k, _ in td_sorted[:cfg.get("top_n", 10)]]
                vals = [v for _, v in td_sorted[:cfg.get("top_n", 10)]]
                fig = plt.figure(figsize=tuple(cfg.get("figsize_bar", [7,5])))
                ax = fig.add_subplot(111)
                ax.barh(range(len(cats))[::-1], vals[::-1], color="#4E79A7")
                ax.set_yticks(range(len(cats))[::-1])
                ax.set_yticklabels(cats[::-1])
                ax.set_xlabel("Samples")
                ax.set_title("Trade Distribution (Top)")
                fig.tight_layout()
                figs.append(fig)
        except Exception:
            pass

        # Chart 3: Hourly counts (line) if enough points
        try:
            if len(hour_counts) >= int(cfg.get("line_min_points", 2)):
                idx = sorted(hour_counts.keys())
                vals = [hour_counts[k] for k in idx]
                # Convert tz-aware UTC to naive for plotting (timezone safety)
                times = pd.to_datetime(pd.Series(idx), utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
                fig = plt.figure(figsize=tuple(cfg.get("figsize_line", [7,5])))
                ax = fig.add_subplot(111)
                ax.plot(times, vals, color="#E15759", linewidth=1.8)
                ax.set_xlabel("UTC Hour")
                ax.set_ylabel("Samples")
                ax.set_title("Hourly Sample Counts")
                fig.autofmt_xdate()
                fig.tight_layout()
                figs.append(fig)
        except Exception:
            pass

        # Save PNGs in order (do not close figures yet)
        png_paths = []
        # report_date based on global min timestamp or today
        if global_min_ts is not None:
            report_date = str(pd.to_datetime(global_min_ts).date())
        else:
            report_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        for i, fig in enumerate(figs, 1):
            png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(png), dpi=120)  # no bbox_inches='tight'
                png_paths.append(png)
            except Exception:
                # If save fails, skip that PNG
                continue

        # Build Evidence table (list-of-dicts)
        evidence_rows = evidence_buffer[:50]
        sections = []
        # Summary
        sections.append({"type": "summary", "title": "Summary", "bullets": bullets})
        # Evidence table
        if evidence_rows:
            sections.append({
                "type": "table",
                "title": "Evidence",
                "data": evidence_rows,
                "headers": evidence_cols,
                "rows_per_page": 24
            })
        # Charts section with live figures
        if figs:
            sections.append({"type": "charts", "title": "Figures", "figures": figs})

        title_text = "Walmart Renovation RTLS — Position Summary"
        report = {
            "title": title_text,
            "meta": meta_text,
            "sections": sections
        }

        # Apply budgets
        report = apply_budgets(report, caps=None)

        # Build PDF
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))

        # Print links (success)
        def file_uri(p: Path) -> str:
            return "file:///" + str(p.resolve()).replace("\\", "/")
        print(f"[Download the PDF]({file_uri(pdf_path)})")
        for i, pth in enumerate(png_paths, 1):
            print(f"[Download Plot {i}]({file_uri(pth)})")

        # Do not close figures before PDF is built; safe to close now
        # But no more output allowed, so silently close
        try:
            plt.close('all')
        except Exception:
            pass

    except (MemoryError, KeyboardInterrupt):
        # Minimal-Report Mode
        try:
            # Use minimal bullets
            bullets = ["Minimal report due to resource constraints."]
            pdf_path = Path(sys.argv[2]).resolve().parent / f"info_zone_report_{pd.Timestamp.utcnow().strftime('%Y-%m-%d')}.pdf"
            report = {
                "title": "Walmart Renovation RTLS — Minimal Summary",
                "meta": f"User query: {sys.argv[1] if len(sys.argv)>1 else ''}",
                "sections": [
                    {"type": "summary", "title": "Summary", "bullets": bullets}
                ]
            }
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            def file_uri(p: Path) -> str:
                return "file:///" + str(p.resolve()).replace("\\", "/")
            print(f"[Download the PDF]({file_uri(pdf_path)})")
        except Exception as e2:
            print("Error Report:")
            print("Unable to complete minimal report: " + str(e2))
    except SystemExit:
        raise
    except Exception as e:
        # General failure
        print("Error Report:")
        msg = str(e).strip()
        if not msg:
            msg = "Unexpected error occurred."
        print(msg)

if __name__ == "__main__":
    main()