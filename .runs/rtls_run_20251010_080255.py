#!/usr/bin/env python3
import sys, os
from pathlib import Path

# ------------------- Resolve project root and enable local imports -------------------
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

# ------------------- Imports from helpers -------------------
try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Local helper 'extractor.py' is missing or failed to import.")
    raise SystemExit(1)

# PDF builder and budgets
try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("Local helper 'pdf_creation_script.py' is missing or failed to import.")
    raise SystemExit(1)

try:
    from report_limits import apply_budgets, make_lite
except Exception:
    # Fallback no-op if limits helper missing
    def apply_budgets(report, caps=None): return report
    def make_lite(report): return report

# Optional chart policy
try:
    from chart_policy import choose_charts
    HAVE_CHART_POLICY = True
except Exception:
    HAVE_CHART_POLICY = False

# ------------------- Standard libs -------------------
import json
import math
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------- Utilities -------------------
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def load_config(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
    # defaults mirroring chart_policy DEFAULTS where needed
    return {
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
        "figsize_overlay": [9, 7],
        "figsize_bar": [7, 5],
        "figsize_line": [7, 5],
        "figsize_pie": [5, 5],
        "figsize_box": [7, 5],
    }

def hour_floor(ts: pd.Series) -> pd.Series:
    try:
        return ts.dt.floor("H")
    except Exception:
        return pd.Series([], dtype="datetime64[ns, UTC]")

def ensure_columns(df: pd.DataFrame, cols: list):
    for c in cols:
        if c not in df.columns:
            df[c] = ""

def load_floorplan_extent(img_override: Path | None) -> tuple | None:
    """
    Load floorplan extent and image using floorplans.json and explicit override image.
    Returns (img, (x_min, x_max, y_min, y_max)) or None.
    """
    if not FLOORJSON.exists():
        return None
    try:
        data = json.loads(FLOORJSON.read_text(encoding="utf-8", errors="ignore"))
        fp = (data.get("floorplans") or data.get("plans") or data or [None])
        if isinstance(fp, list):
            fp = fp[0]
        if not fp:
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
        img_path = img_override if img_override and img_override.exists() else None
        if img_path is None:
            return None
        img = plt.imread(str(img_path))
        return img, (x_min, x_max, y_min, y_max)
    except Exception:
        return None

def make_fallback_figures(overlay_df: pd.DataFrame, hourly_df: pd.DataFrame, trade_df: pd.DataFrame, cfg: dict) -> list:
    figs = []

    # Floorplan overlay (fallback)
    try:
        if not overlay_df.empty and {"x","y"}.issubset(set(overlay_df.columns)):
            # Cast numeric and drop na
            x = pd.to_numeric(overlay_df["x"], errors="coerce")
            y = pd.to_numeric(overlay_df["y"], errors="coerce")
            use = overlay_df.loc[x.notna() & y.notna(), :].copy()
            img_tuple = load_floorplan_extent(ROOT / "floorplan.jpeg")
            if img_tuple is not None and not use.empty:
                img, (x_min, x_max, y_min, y_max) = img_tuple
                fig = plt.figure(figsize=tuple(cfg.get("figsize_overlay", (9,7))))
                ax = fig.add_subplot(111)
                ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin="upper")
                # subsample
                max_pts = int(cfg.get("overlay_subsample", 20000))
                if len(use) > max_pts:
                    idx = np.linspace(0, len(use) - 1, max_pts).astype(int)
                    use = use.iloc[idx]
                color_by = cfg.get("overlay_color_by", "trade")
                s = float(cfg.get("overlay_point_size", 8))
                a = float(cfg.get("overlay_alpha", 0.85))
                if color_by in use.columns:
                    cats = use[color_by].astype(str).fillna("")
                    uniq = list(dict.fromkeys(cats.tolist()))[:12]
                    palette = {c: plt.cm.get_cmap("tab10")(i % 10) for i,c in enumerate(uniq)}
                    for cat, g in use.groupby(color_by):
                        ax.scatter(pd.to_numeric(g["x"], errors="coerce"),
                                   pd.to_numeric(g["y"], errors="coerce"),
                                   s=s, alpha=a, color=palette.get(cat, None), label=str(cat))
                    handles, labels = ax.get_legend_handles_labels()
                    if len(labels) <= 12 and len(labels) > 0:
                        ax.legend(loc="upper left", fontsize=8, frameon=True)
                else:
                    ax.scatter(pd.to_numeric(use["x"], errors="coerce"),
                               pd.to_numeric(use["y"], errors="coerce"),
                               s=s, alpha=a)
                mx = float(cfg.get("floorplan_margin", 0.10))
                xr = x_max - x_min; yr = y_max - y_min
                ax.set_xlim(x_min - mx*xr, x_max + mx*xr)
                ax.set_ylim(y_min - mx*yr, y_max + mx*yr)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("X (mm)")
                ax.set_ylabel("Y (mm)")
                ax.set_title("Floorplan Overlay")
                fig.tight_layout()
                figs.append(fig)
    except Exception:
        pass

    # Hourly line
    try:
        if not hourly_df.empty and "count" in hourly_df.columns:
            fig = plt.figure(figsize=tuple(cfg.get("figsize_line", (7,5))))
            ax = fig.add_subplot(111)
            ax.plot(hourly_df["hour"], hourly_df["count"], marker="o", linewidth=1.5)
            ax.set_title("Hourly Sample Counts (UTC)")
            ax.set_xlabel("Hour (UTC)")
            ax.set_ylabel("Samples")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            figs.append(fig)
    except Exception:
        pass

    # Trade bar
    try:
        if not trade_df.empty and {"trade","count"}.issubset(trade_df.columns):
            top_n = int(cfg.get("top_n", 10))
            tdf = trade_df.sort_values("count", ascending=False).head(top_n)
            fig = plt.figure(figsize=tuple(cfg.get("figsize_bar", (7,5))))
            ax = fig.add_subplot(111)
            ax.barh(tdf["trade"].astype(str), tdf["count"].astype(int))
            ax.set_title("Top Trades by Samples")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Trade")
            ax.invert_yaxis()
            fig.tight_layout()
            figs.append(fig)
    except Exception:
        pass

    return figs

def build_figures_with_policy(overlay_df: pd.DataFrame, hourly_df: pd.DataFrame, trade_df: pd.DataFrame, user_query: str, cfg: dict) -> list:
    figs = []
    if HAVE_CHART_POLICY:
        try:
            figs = choose_charts(
                overlay_df,
                hourly_df=hourly_df,
                trade_df=trade_df,
                user_query=user_query or "",
                floorplans_path=str(FLOORJSON),
                floorplan_image_path=str(ROOT / "floorplan.jpeg"),
                zones_path=str(ZONES_JSON),
                config=cfg
            ) or []
        except Exception:
            figs = []
    # Fallback if policy unavailable or produced nothing
    if not figs:
        figs = make_fallback_figures(overlay_df, hourly_df, trade_df, cfg)
    # Cap figures by config
    max_figs = int(cfg.get("max_figures", 6))
    return figs[:max_figs]

def minimal_report_pdf(out_dir: Path, report_date: str, user_prompt: str, first_df: pd.DataFrame, bullets: list) -> Path:
    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
    sections = []
    sections.append({"type":"summary","title":"Summary","bullets":bullets})
    # Evidence table (compact)
    cols = ["trackable","trade","ts_short","x","y","z"]
    ensure_columns(first_df, cols)
    tbl_rows = first_df[cols].head(24).fillna("").astype(str).to_dict(orient="records")
    sections.append({"type":"table","title":"Evidence","data":tbl_rows,"headers":cols,"rows_per_page":24})
    report = {
        "title": "Walmart RTLS Summary",
        "subtitle": "Positions-based analytics",
        "author": "InfoZoneBuilder",
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "query": user_prompt or "",
        "meta": (read_text(GUIDELINES)[:300] + "\n" + read_text(CONTEXT)[:300]).strip(),
        "sections": sections
    }
    report = make_lite(report)
    safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
    return pdf_path

# ------------------- Main -------------------
def main():
    try:
        if len(sys.argv) < 3:
            print("Error Report:")
            print("Usage: python generated.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
            return

        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:] if p and p.strip()]
        if not csv_paths:
            print("Error Report:")
            print("No CSV inputs provided.")
            return
        for p in csv_paths:
            if not p.exists():
                print("Error Report:")
                print(f"CSV not found: {p}")
                return

        out_dir = csv_paths[0].resolve().parent

        cfg = load_config(CONFIG)

        # Aggregates across files (memory-light)
        overlay_limit = int(cfg.get("overlay_subsample", 20000))
        overlay_cols = ["trackable","trackable_uid","trade","ts_iso","ts_short","x","y","z"]
        overlay_buf = pd.DataFrame(columns=overlay_cols)
        hourly_counts = {}   # key: pd.Timestamp (UTC hour), value: int
        trade_counts = {}    # key: trade str, value: int
        uniq_trackable_uids = set()
        uniq_trades = set()
        total_samples = 0
        t_min = None
        t_max = None
        first_df_for_evidence = None
        first_df_columns = None

        # Process each CSV independently
        for idx, csv_path in enumerate(csv_paths):
            try:
                raw = extract_tracks(str(csv_path))
                df = pd.DataFrame(raw.get("rows", []))
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated()]
                # Timestamp canon
                src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
                df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

                # After first file ingestion, validate required columns
                if idx == 0:
                    required_ok = True
                    cols = set(df.columns.astype(str))
                    if not (("trackable" in cols) or ("trackable_uid" in cols)):
                        required_ok = False
                    if "trade" not in cols:
                        required_ok = False
                    if not (("x" in cols) and ("y" in cols)):
                        required_ok = False
                    if not required_ok:
                        print("Error Report:")
                        print("Missing required columns for analysis.")
                        print("Columns detected: " + ",".join(df.columns.astype(str)))
                        raise SystemExit(1)
                    first_df_for_evidence = df.copy()
                    first_df_columns = list(df.columns.astype(str))

                # Stream-safe aggregates
                # Valid rows for time-based metrics
                valid_time_mask = df["ts_utc"].notna()
                # Hourly counts
                if valid_time_mask.any():
                    hrs = hour_floor(df.loc[valid_time_mask, "ts_utc"])
                    vc = hrs.value_counts()
                    for h, c in vc.items():
                        hourly_counts[h] = hourly_counts.get(h, 0) + int(c)

                    # Track min/max ts
                    cur_min = pd.to_datetime(hrs.min(), utc=True)
                    cur_max = pd.to_datetime(hrs.max(), utc=True)
                    if cur_min is not pd.NaT:
                        t_min = cur_min if t_min is None else min(t_min, cur_min)
                    if cur_max is not pd.NaT:
                        t_max = cur_max if t_max is None else max(t_max, cur_max)

                # Trade counts
                if "trade" in df.columns:
                    vc_trade = df["trade"].astype(str).replace("", "unknown").value_counts()
                    for tr, c in vc_trade.items():
                        trade_counts[tr] = trade_counts.get(tr, 0) + int(c)
                    uniq_trades.update(set(df["trade"].astype(str).unique().tolist()))

                # Unique identities
                if "trackable_uid" in df.columns:
                    uniq_trackable_uids.update([u for u in df["trackable_uid"].astype(str).unique().tolist() if u != ""])
                elif "trackable" in df.columns:
                    uniq_trackable_uids.update([u for u in df["trackable"].astype(str).unique().tolist() if u != ""])

                # Overlay reservoir (subsample by downselecting evenly)
                need_cols = [c for c in overlay_cols if c in df.columns]  # avoid duplicate name issues
                part = df.loc[:, need_cols].copy()
                # numeric filter for x,y
                if "x" in part.columns and "y" in part.columns:
                    xnum = pd.to_numeric(part["x"], errors="coerce")
                    ynum = pd.to_numeric(part["y"], errors="coerce")
                    part = part.loc[xnum.notna() & ynum.notna()]
                # Append then downsample if exceeding
                overlay_buf = pd.concat([overlay_buf, part], axis=0, ignore_index=True)
                if len(overlay_buf) > overlay_limit:
                    # Evenly spaced downsample to overlay_limit
                    idx_keep = np.linspace(0, len(overlay_buf) - 1, overlay_limit).astype(int)
                    overlay_buf = overlay_buf.iloc[idx_keep].reset_index(drop=True)

                total_samples += len(df)

                # Free per-file
                del df
                plt.close('all')
            except MemoryError:
                # Minimal-Report Mode
                report_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                if first_df_for_evidence is None:
                    # Create an empty df with required columns for evidence
                    first_df_for_evidence = pd.DataFrame(columns=["trackable","trade","ts_short","x","y","z"])
                bullets = [
                    "Minimal-Report Mode: memory limit reached during processing.",
                    f"Files processed (partial): {idx+1} of {len(csv_paths)}",
                ]
                pdf_path = minimal_report_pdf(out_dir, report_date, user_prompt, first_df_for_evidence, bullets)
                print(f"[Download the PDF]({file_uri(pdf_path)})")
                return
            except KeyboardInterrupt:
                # Minimal-Report Mode on interrupt
                report_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                if first_df_for_evidence is None:
                    first_df_for_evidence = pd.DataFrame(columns=["trackable","trade","ts_short","x","y","z"])
                bullets = [
                    "Minimal-Report Mode: processing interrupted by user.",
                    f"Files processed (partial): {idx+1} of {len(csv_paths)}",
                ]
                pdf_path = minimal_report_pdf(out_dir, report_date, user_prompt, first_df_for_evidence, bullets)
                print(f"[Download the PDF]({file_uri(pdf_path)})")
                return
            except Exception:
                # Continue to next file; if all fail, handle later
                continue

        # Prepare aggregates
        # Overlay df
        overlay_df = overlay_buf.copy()

        # Hourly df
        if hourly_counts:
            h_items = sorted(hourly_counts.items(), key=lambda x: x[0])
            hourly_df = pd.DataFrame({"hour": [k for k,_ in h_items], "count": [v for _,v in h_items]})
        else:
            hourly_df = pd.DataFrame(columns=["hour","count"])

        # Trade df
        if trade_counts:
            tr_items = sorted(trade_counts.items(), key=lambda x: (-x[1], x[0]))
            trade_df = pd.DataFrame({"trade": [k for k,_ in tr_items], "count": [v for _,v in tr_items]})
        else:
            trade_df = pd.DataFrame(columns=["trade","count"])

        # Evidence DataFrame
        if first_df_for_evidence is None:
            # Try re-read first file minimally
            raw = extract_tracks(str(csv_paths[0]))
            tmp_df = pd.DataFrame(raw.get("rows", []))
            if tmp_df.columns.duplicated().any():
                tmp_df = tmp_df.loc[:, ~tmp_df.columns.duplicated()]
            ensure_columns(tmp_df, ["trackable","trade","ts_short","x","y","z"])
            first_df_for_evidence = tmp_df
        else:
            ensure_columns(first_df_for_evidence, ["trackable","trade","ts_short","x","y","z"])

        # Report date label
        if t_min is not None and t_max is not None:
            if t_min.date() == t_max.date():
                report_date = t_min.strftime("%Y%m%d")
            else:
                report_date = t_min.strftime("%Y%m%d") + "_to_" + t_max.strftime("%Y%m%d")
        else:
            report_date = datetime.utcnow().strftime("%Y%m%d")

        # Build figures
        figs = build_figures_with_policy(overlay_df, hourly_df, trade_df, user_prompt, cfg)

        # Save PNGs to out_dir
        png_paths = []
        for i, fig in enumerate(figs, start=1):
            png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(png), dpi=120)
                png_paths.append(png)
            except Exception:
                # Skip this figure if saving fails
                continue

        # Build report sections
        sections = []

        # Summary bullets
        period_str = ""
        if t_min is not None and t_max is not None:
            period_str = f"UTC window: {t_min.strftime('%Y-%m-%d %H:%M')} to {t_max.strftime('%Y-%m-%d %H:%M')}"
        bullets = [
            f"Files analyzed: {len(csv_paths)}",
            f"Total samples: {total_samples:,}",
            f"Unique tags: {len(uniq_trackable_uids)}",
            f"Trades observed: {len([t for t in uniq_trades if t])}",
        ]
        if period_str:
            bullets.insert(1, period_str)
        sections.append({"type":"summary","title":"Summary","bullets":bullets})

        # Evidence table (first 50 rows)
        cols = ["trackable","trade","ts_short","x","y","z"]
        rows = first_df_for_evidence[cols].head(50).fillna("").astype(str).to_dict(orient="records")
        sections.append({"type":"table","title":"Evidence","data":rows,"headers":cols,"rows_per_page":24})

        # Charts section (pass live figures)
        if figs:
            sections.append({"type":"charts","title":"Figures","figures":figs})

        # Narrative/context
        ctx = read_text(CONTEXT)
        if ctx:
            sections.append({"type":"narrative","title":"Context","paragraphs":[ctx[:800]]})

        # Build report dict
        report = {
            "title": "Walmart RTLS Summary",
            "subtitle": "Positions-based analytics",
            "author": "InfoZoneBuilder",
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "query": user_prompt or "",
            "meta": (read_text(GUIDELINES)[:400]).strip(),
            "sections": sections
        }

        # Apply budgets and write PDF
        report = apply_budgets(report)
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"

        try:
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except MemoryError:
            # Minimal mode if PDF creation runs out of memory
            pdf_path = minimal_report_pdf(out_dir, report_date, user_prompt, first_df_for_evidence, bullets)
            png_paths = []
        except Exception:
            # Attempt lite
            try:
                report_lite = make_lite(report)
                safe_build_pdf(report_lite, str(pdf_path), logo_path=str(LOGO))
            except Exception as e2:
                print("Error Report:")
                print("Failed to create the PDF report.")
                return

        # Print links (PDF first, then plots)
        print(f"[Download the PDF]({file_uri(pdf_path)})")
        for i, pth in enumerate(png_paths, 1):
            print(f"[Download Plot {i}]({file_uri(pth)})")

    except SystemExit:
        # already printed error if needed
        return
    except Exception as e:
        # Generic failure
        print("Error Report:")
        msg = str(e).strip() or "Unhandled error during report generation."
        # Keep it 1–2 lines
        msg = msg.splitlines()[0]
        print(msg)
        return

if __name__ == "__main__":
    main()