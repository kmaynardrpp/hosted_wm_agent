import sys, os
from pathlib import Path

# -------------------- ROOT resolution and local imports --------------------
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

# -------------------- Imports requiring ROOT --------------------
import json
import math
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Helpers (local)
try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Missing local helper 'extractor.py'.")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("Missing local helper 'pdf_creation_script.py'.")
    raise SystemExit(1)

# Limits
try:
    from report_limits import DEFAULTS as LIMIT_DEFAULTS, apply_budgets, make_lite
except Exception:
    print("Error Report:")
    print("Missing local helper 'report_limits.py'.")
    raise SystemExit(1)

# Optional chart policy
_have_chart_policy = True
try:
    from chart_policy import choose_charts
except Exception:
    _have_chart_policy = False

# Read and apply guidelines at the very start (as text; behavior enforced below)
_ = read_text(GUIDELINES)

def _zones_requested_from_prompt(prompt: str) -> bool:
    if not prompt:
        return False
    p = prompt.lower()
    # Only consider zones if explicitly asked
    keys = ["zone", "zones", "area", "room", "section", "department"]
    return any(k in p for k in keys)

def _load_config(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
    # Defaults if config missing
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

def _validate_schema_or_error(df: pd.DataFrame) -> None:
    cols = set(df.columns.astype(str))
    ok_identity = ("trackable" in cols) or ("trackable_uid" in cols)
    ok_trade = ("trade" in cols)
    ok_xy = ("x" in cols) and ("y" in cols)
    if not (ok_identity and ok_trade and ok_xy):
        print("Error Report:")
        print("Missing required columns for analysis.")
        print(f"Columns detected: {','.join(df.columns.astype(str))}")
        raise SystemExit(1)

def _iso_noz(dt: pd.Timestamp) -> str:
    # Format UTC time; ensure no double 'Z' in meta strings
    try:
        if dt.tzinfo is not None:
            dt = dt.tz_convert("UTC")
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(dt)

def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def _reservoir_sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    # Evenly-spaced downsample to at most k rows, stable and deterministic
    n = len(df)
    if n <= k:
        return df.copy()
    idx = np.linspace(0, n - 1, k).astype(int)
    return df.iloc[idx].copy()

def main():
    try:
        # -------------------- CLI parse --------------------
        if len(sys.argv) < 3:
            print("Error Report:")
            print("No CSV inputs provided.")
            raise SystemExit(1)
        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:] if p and Path(p).exists()]
        if not csv_paths:
            print("Error Report:")
            print("CSV file(s) not found on disk.")
            raise SystemExit(1)

        out_dir = csv_paths[0].parent.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        config = _load_config(CONFIG)
        budgets = dict(LIMIT_DEFAULTS)

        # Zones requested?
        want_zones = _zones_requested_from_prompt(user_prompt)
        # Enforce "ZONES ONLY IF ASKED"
        config["draw_zones"] = bool(want_zones)

        # -------------------- Aggregators (large-data safe) --------------------
        overlay_cap = int(config.get("overlay_subsample", 20000)) if isinstance(config.get("overlay_subsample", 20000), (int, float)) else 20000
        reservoir = None  # pandas DataFrame with limited rows
        hourly_counts: dict = {}  # {pd.Timestamp(UTC naive or aware): int}
        trade_counts: dict = {}   # {trade: int}
        first_evidence_df = None  # For evidence table
        min_ts = None
        max_ts = None
        uniq_trackable_uids = set()
        uniq_trades = set()
        total_rows = 0

        # meta inputs
        file_list_display = [str(p.name) for p in csv_paths]

        # -------------------- Per-file processing --------------------
        first_file_schema_checked = False
        for csv_path in csv_paths:
            try:
                raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
            except Exception as e:
                print("Error Report:")
                print("Failed to read CSV via extractor.")
                raise SystemExit(1)

            audit = raw.get("audit", {}) if isinstance(raw, dict) else {}
            if not audit or not audit.get("mac_map_loaded", False):
                print("Error Report:")
                print("Required MAC→name map not loaded (trackable_objects.json).")
                raise SystemExit(1)

            rows = raw.get("rows", []) if isinstance(raw, dict) else []
            df = pd.DataFrame(rows)
            # Duplicate-name guard FIRST
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            # Timestamp canon: create ts_utc and use for analytics
            ts_src = df["ts_iso"] if "ts_iso" in df.columns else (df["ts"] if "ts" in df.columns else None)
            if ts_src is None:
                # Create missing ts column to avoid KeyError in pd.to_datetime
                df["ts_utc"] = pd.NaT
            else:
                df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

            # Early schema validation (after first file only)
            if not first_file_schema_checked:
                _validate_schema_or_error(df)
                # If zones were requested, ensure we have zone_name or zones polygons available
                if want_zones:
                    if "zone_name" not in df.columns:
                        # Load zones polygons; if missing/invalid, error
                        if not ZONES_JSON.exists():
                            print("Error Report:")
                            print("Zones requested but zones.json not found and no zone_name column present.")
                            print(f"Columns detected: {','.join(df.columns.astype(str))}")
                            raise SystemExit(1)
                first_file_schema_checked = True

            # Update totals and uniques
            total_rows += len(df)
            if "trackable_uid" in df.columns:
                uniq_trackable_uids.update(set([s for s in df["trackable_uid"].astype(str).tolist() if s]))
            if "trade" in df.columns:
                uniq_trades.update(set([s for s in df["trade"].astype(str).tolist() if s]))

            # Time range
            if "ts_utc" in df.columns:
                valid_ts = df["ts_utc"].dropna()
                if not valid_ts.empty:
                    cur_min = valid_ts.min()
                    cur_max = valid_ts.max()
                    min_ts = cur_min if (min_ts is None or cur_min < min_ts) else min_ts
                    max_ts = cur_max if (max_ts is None or cur_max > max_ts) else max_ts

            # Evidence table: capture first 50 rows across first file
            if first_evidence_df is None:
                first_evidence_df = df.copy()

            # Aggregates: Hourly counts
            if "ts_utc" in df.columns:
                ts_valid = df["ts_utc"].dropna()
                if not ts_valid.empty:
                    # Keep tz-aware; group by floored hours
                    hours = ts_valid.dt.floor("H")
                    g = hours.value_counts().to_dict()
                    # Merge
                    for k, v in g.items():
                        hourly_counts[k] = hourly_counts.get(k, 0) + int(v)

            # Aggregates: trade counts (samples per trade)
            if "trade" in df.columns:
                trades = df["trade"].astype(str).replace("", "unknown")
                g = trades.value_counts().to_dict()
                for k, v in g.items():
                    trade_counts[k] = trade_counts.get(k, 0) + int(v)

            # Overlay reservoir: retain limited sample of rows with valid x,y
            if ("x" in df.columns) and ("y" in df.columns):
                xn = _safe_numeric(df["x"]); yn = _safe_numeric(df["y"])
                use = df.loc[xn.notna() & yn.notna(), ["x", "y", "trade", "trackable", "trackable_uid", "ts_short", "ts_utc"]].copy()
                # Subsample within this file to a manageable size proportional to cap
                if len(use) > 0:
                    # Ensure numeric for plotting
                    use["x"] = _safe_numeric(use["x"])
                    use["y"] = _safe_numeric(use["y"])
                    # Per-file downsample
                    keep_n = min(len(use), max(1, overlay_cap))
                    use_ds = _reservoir_sample(use, keep_n)
                    if reservoir is None:
                        reservoir = use_ds
                    else:
                        # Concatenate and trim to overlay_cap
                        reservoir = pd.concat([reservoir, use_ds], ignore_index=True)
                        if len(reservoir) > overlay_cap:
                            reservoir = _reservoir_sample(reservoir, overlay_cap)

            # Clear per-file df to save memory
            del df
            plt.close('all')
        # End per-file loop

        # Prepare aggregates DataFrames
        # Hourly DataFrame sorted by hour
        if hourly_counts:
            # Sort by datetime key
            hours_sorted = sorted(hourly_counts.items(), key=lambda kv: kv[0])
            hourly_df = pd.DataFrame({
                "hour_utc": [k for k, _ in hours_sorted],
                "count_samples": [v for _, v in hours_sorted],
            })
        else:
            hourly_df = pd.DataFrame(columns=["hour_utc", "count_samples"])

        # Trade summary DataFrame
        if trade_counts:
            trades_sorted = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)
            trade_df = pd.DataFrame({"trade": [k for k, _ in trades_sorted], "count_samples": [v for _, v in trades_sorted]})
        else:
            trade_df = pd.DataFrame(columns=["trade", "count_samples"])

        # Evidence table rows (list of dicts)
        sections = []
        evidence_rows = []
        if first_evidence_df is not None and not first_evidence_df.empty:
            cols = ["trackable","trade","ts_short","x","y","z"]
            for c in cols:
                if c not in first_evidence_df.columns:
                    first_evidence_df[c] = ""
            try:
                evidence_rows = first_evidence_df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
            except Exception:
                # If type conversion fails, fallback to safe conversion
                tmp = first_evidence_df[cols].head(50).copy()
                for c in cols:
                    tmp[c] = tmp[c].astype(str)
                evidence_rows = tmp.to_dict(orient="records")
            sections.append({"type":"table","title":"Evidence","data":evidence_rows,"headers":cols,"rows_per_page":24})

        # Figures: choose via chart_policy if available, else basic fallbacks
        figs = []
        max_figures_allowed = int(budgets.get("MAX_FIGURES", 6))
        try:
            # Build chart-policy figures with floorplan overlay preference; zones only if asked
            if _have_chart_policy:
                # Override config to enforce zones policy and respect limits
                cfg = dict(config)
                cfg["draw_zones"] = bool(want_zones)
                cfg["max_figures"] = max_figures_allowed
                floorplans_path = str(FLOORJSON)
                floorplan_image_path = str(ROOT / "floorplan.jpeg")  # explicit local image path
                zones_path = str(ZONES_JSON)

                plot_df = reservoir if reservoir is not None else pd.DataFrame(columns=["x","y","trade","trackable","trackable_uid","ts_short","ts_utc"])
                figs = choose_charts(
                    plot_df,
                    hourly_df=hourly_df if not hourly_df.empty else None,
                    trade_df=trade_df if not trade_df.empty else None,
                    user_query=str(user_prompt or ""),
                    floorplans_path=floorplans_path,
                    floorplan_image_path=floorplan_image_path,
                    zones_path=zones_path,
                    config=cfg
                ) or []
            # If still no figs, do simple fallbacks
            if not figs:
                # Fallback 1: simple scatter (no floorplan), color by trade if available
                if reservoir is not None and not reservoir.empty:
                    fig1 = plt.figure(figsize=(9, 7))
                    ax1 = fig1.add_subplot(111)
                    r = reservoir.dropna(subset=["x","y"])
                    color_by = "trade" if "trade" in r.columns else None
                    if color_by and r[color_by].nunique() > 0:
                        # cap categories at 12
                        counts = r[color_by].fillna("unknown").value_counts().sort_values(ascending=False)
                        top_cats = counts.index.tolist()[:12]
                        palette = plt.cm.get_cmap("tab10")
                        for i, cat in enumerate(top_cats):
                            g = r[r[color_by] == cat]
                            ax1.scatter(g["x"], g["y"], s=8, alpha=0.6, color=palette(i % 10), label=str(cat))
                        handles, labels = ax1.get_legend_handles_labels()
                        if len(labels) <= 12:
                            ax1.legend(loc="upper right", fontsize=8)
                    else:
                        ax1.scatter(r["x"], r["y"], s=8, alpha=0.6)
                    ax1.set_aspect("equal", adjustable="box")
                    ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)")
                    ax1.set_title("Position Overlay (No Floorplan)")
                    fig1.tight_layout()
                    figs.append(fig1)

                # Fallback 2: bar of top trades
                if not trade_df.empty:
                    top_n = int(config.get("top_n", 10))
                    td = trade_df.sort_values("count_samples", ascending=False).head(top_n)
                    fig2 = plt.figure(figsize=(7, 5))
                    ax2 = fig2.add_subplot(111)
                    ax2.barh(td["trade"].astype(str), td["count_samples"].astype(int), color="#4472C4")
                    ax2.invert_yaxis()
                    ax2.set_xlabel("Samples"); ax2.set_title("Top Trades by Samples")
                    fig2.tight_layout()
                    figs.append(fig2)

                # Fallback 3: hourly line
                if not hourly_df.empty and len(hourly_df) >= 2:
                    fig3 = plt.figure(figsize=(7, 5))
                    ax3 = fig3.add_subplot(111)
                    hr = hourly_df.copy()
                    # Timezone safety: if tz-aware -> convert to UTC then drop tz for plotting
                    try:
                        if isinstance(hr["hour_utc"].iloc[0], pd.Timestamp) and hr["hour_utc"].dt.tz is not None:
                            hr["hour_plot"] = hr["hour_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
                        else:
                            hr["hour_plot"] = hr["hour_utc"]
                    except Exception:
                        hr["hour_plot"] = hr["hour_utc"]
                    ax3.plot(hr["hour_plot"], hr["count_samples"].astype(int), color="#E1262D", linewidth=1.6)
                    ax3.set_xlabel("Hour (UTC)"); ax3.set_ylabel("Samples")
                    ax3.set_title("Hourly Sample Counts")
                    fig3.autofmt_xdate()
                    fig3.tight_layout()
                    figs.append(fig3)
        except (MemoryError, KeyboardInterrupt):
            # Minimal-Report Mode will be handled later (still write a PDF, maybe without figs)
            figs = []
        except Exception:
            # If chart policy fails, fall back to simple plots (handled above)
            if not figs and reservoir is not None and not reservoir.empty:
                try:
                    fig1 = plt.figure(figsize=(9, 7))
                    ax1 = fig1.add_subplot(111)
                    r = reservoir.dropna(subset=["x","y"])
                    ax1.scatter(r["x"], r["y"], s=8, alpha=0.6)
                    ax1.set_aspect("equal", adjustable="box")
                    ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)")
                    ax1.set_title("Position Overlay")
                    fig1.tight_layout()
                    figs.append(fig1)
                except Exception:
                    pass

        # Cap figures to budgets before saving PNG
        if len(figs) > max_figures_allowed:
            figs = figs[:max_figures_allowed]

        # Save PNGs first (DPI=120, no tight bbox)
        png_paths = []
        report_date = ""
        if min_ts is not None and isinstance(min_ts, pd.Timestamp) and not pd.isna(min_ts):
            try:
                if min_ts.tzinfo is not None:
                    min_ts_local = min_ts.tz_convert("UTC")
                else:
                    min_ts_local = pd.Timestamp(min_ts).tz_localize("UTC")
                report_date = min_ts_local.strftime("%Y%m%d")
            except Exception:
                report_date = pd.Timestamp.utcnow().strftime("%Y%m%d")
        else:
            report_date = pd.Timestamp.utcnow().strftime("%Y%m%d")

        for i, fig in enumerate(figs, 1):
            png_path = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(png_path), dpi=120)
                png_paths.append(png_path)
            except Exception:
                # Best-effort; continue
                pass

        # Add charts section with LIVE figures
        if figs:
            sections.append({"type":"charts","title":"Figures","figures":figs})

        # -------------------- Build report dict --------------------
        title = "Walmart RTLS Position Summary"
        meta_parts = []
        meta_parts.append(f"Files: {', '.join(file_list_display)}")
        meta_parts.append(f"Samples: {total_rows:,}")
        meta_parts.append(f"Trackables: {len(uniq_trackable_uids)} | Trades: {len([t for t in uniq_trades if t])}")
        if min_ts is not None and max_ts is not None:
            try:
                min_s = _iso_noz(min_ts)
                max_s = _iso_noz(max_ts)
                meta_parts.append(f"Time range: {min_s} → {max_s}")
            except Exception:
                pass
        meta_text = " | ".join(meta_parts)

        # Narrative context (optional)
        context_txt = read_text(CONTEXT).strip()
        if context_txt:
            sections.append({"type":"narrative","title":"Context","paragraphs":[context_txt]})

        # Summary bullets
        bullets = []
        bullets.append(f"Processed {len(csv_paths)} file(s).")
        bullets.append(f"Collected {total_rows:,} position samples across {len(uniq_trackable_uids)} unique trackables.")
        if min_ts is not None and max_ts is not None:
            try:
                bullets.append(f"Coverage window (UTC): {_iso_noz(min_ts)} to {_iso_noz(max_ts)}.")
            except Exception:
                pass
        if not trade_df.empty:
            top_trade = trade_df.sort_values("count_samples", ascending=False).iloc[0]
            bullets.append(f"Top trade by samples: {str(top_trade['trade'])} ({int(top_trade['count_samples']):,}).")
        sections.insert(0, {"type":"summary","title":"At-a-glance","bullets":bullets})

        report = {
            "title": title,
            "meta": meta_text,
            "sections": sections,
        }

        # Apply budgets BEFORE writing
        report_capped = apply_budgets(report, LIMIT_DEFAULTS)

        # -------------------- Write PDF --------------------
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        wrote_pdf = False
        try:
            safe_build_pdf(report_capped, str(pdf_path), logo_path=str(LOGO))
            wrote_pdf = pdf_path.exists()
        except (MemoryError, KeyboardInterrupt):
            # Minimal-Report Mode
            try:
                report_lite = make_lite(report)
                safe_build_pdf(report_lite, str(pdf_path), logo_path=str(LOGO))
                wrote_pdf = pdf_path.exists()
                # In lite mode, don't promise plots if not created
                png_paths = []
            except Exception:
                wrote_pdf = False
        except Exception:
            # Try lite as fallback
            try:
                report_lite = make_lite(report)
                safe_build_pdf(report_lite, str(pdf_path), logo_path=str(LOGO))
                wrote_pdf = pdf_path.exists()
                png_paths = []
            except Exception:
                wrote_pdf = False

        if not wrote_pdf:
            print("Error Report:")
            print("Failed to write PDF report.")
            raise SystemExit(1)

        # -------------------- Print links (exact format) --------------------
        print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        for i, pth in enumerate(png_paths, 1):
            print(f"[Download Plot {i}](file:///{pth.resolve().as_posix()})")

    except SystemExit:
        raise
    except Exception:
        # Generic failure
        print("Error Report:")
        msg = "Unexpected error during processing."
        try:
            # Keep it to 1–2 lines
            tb = traceback.format_exc().strip().splitlines()[-1]
            if tb:
                msg = tb
        except Exception:
            pass
        print(msg)
        raise SystemExit(1)

if __name__ == "__main__":
    main()