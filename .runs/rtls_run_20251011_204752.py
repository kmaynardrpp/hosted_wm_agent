import sys, os, json, math, traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

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

# Matplotlib ≥3.9 shim (PDF builder expects tostring_rgb)
from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
import numpy as _np
_FCA.tostring_rgb = getattr(_FCA, "tostring_rgb", lambda self: _np.asarray(self.buffer_rgba())[..., :3].tobytes())

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Helpers
from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from report_limits import apply_budgets, make_lite
from chart_policy import choose_charts

def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def main():
    # ------------------------ CLI args ------------------------
    if len(sys.argv) < 3:
        print("Error Report:")
        print("No CSV input provided.")
        return

    user_query = sys.argv[1]
    csv_paths = [Path(p) for p in sys.argv[2:] if p]

    # Output directory: first CSV's directory
    first_csv = Path(csv_paths[0])
    out_dir = (first_csv.parent if first_csv.exists() else Path.cwd()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------ Load config (if any) ------------------------
    cfg: Dict[str, Any] = {}
    try:
        if CONFIG.exists():
            cfg = json.loads(read_text(CONFIG) or "{}")
    except Exception:
        cfg = {}

    # Respect "Zones only if asked"
    zones_requested = (" zone " in f" {user_query.lower()} " or user_query.lower().strip().startswith("zone") or "zones" in user_query.lower())
    cfg_override = dict(cfg or {})
    if not zones_requested:
        cfg_override["draw_zones"] = False

    # ------------------------ Aggregators (large-data friendly) ------------------------
    overlay_limit = _safe_int(cfg_override.get("overlay_subsample", 20000), 20000)
    overlay_rows: List[Dict[str, Any]] = []
    hourly_counts: Dict[pd.Timestamp, int] = {}
    trade_counts: Dict[str, int] = {}
    tag_set: set = set()
    total_samples = 0

    first_df_for_table: Optional[pd.DataFrame] = None
    global_min_ts: Optional[pd.Timestamp] = None
    global_max_ts: Optional[pd.Timestamp] = None

    # ------------------------ Process each CSV ------------------------
    first_file_validated = False
    for csv_path in csv_paths:
        try:
            raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
        except Exception as e:
            # Skip bad file but continue others
            continue

        audit = raw.get("audit", {}) if isinstance(raw, dict) else {}
        if audit is not None and isinstance(audit, dict):
            if audit.get("mac_map_loaded") is False:
                print("Error Report:")
                print("MAC map not loaded; trackable names and trades unavailable.")
                return

        rows = raw.get("rows", []) if isinstance(raw, dict) else []
        df = pd.DataFrame(rows)
        # Duplicate column guard
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # If empty, continue but keep minimal state
        if df.empty:
            if not first_file_validated:
                # We still must validate schema; but with empty, we cannot. Delay until we see a non-empty file.
                pass
            continue

        # Timestamp canon
        src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"] if "ts" in df.columns else None
        if src is None:
            df["ts_utc"] = pd.NaT
        else:
            df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

        # Early schema validation (after first non-empty file)
        if not first_file_validated:
            required_ok = True
            cols = set(df.columns.astype(str))
            if not (("trackable" in cols) or ("trackable_uid" in cols)):
                required_ok = False
            if "trade" not in cols:
                required_ok = False
            if ("x" not in cols) or ("y" not in cols):
                required_ok = False
            if not required_ok:
                print("Error Report:")
                print("Missing required columns for analysis.")
                print(f"Columns detected: {','.join(df.columns.astype(str))}")
                return
            first_file_validated = True
            # Keep first df for evidence table (limit columns)
            first_df_for_table = df.copy()

        # Track total samples
        total_samples += len(df)

        # Unique tags
        if "trackable_uid" in df.columns:
            tag_set.update(set(df["trackable_uid"].astype(str).tolist()))
        elif "trackable" in df.columns:
            tag_set.update(set(df["trackable"].astype(str).tolist()))

        # Min/Max ts
        if "ts_utc" in df.columns:
            ts_valid = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
            if ts_valid.notna().any():
                mn = ts_valid.min()
                mx = ts_valid.max()
                if isinstance(mn, pd.Timestamp):
                    global_min_ts = mn if global_min_ts is None else min(global_min_ts, mn)
                if isinstance(mx, pd.Timestamp):
                    global_max_ts = mx if global_max_ts is None else max(global_max_ts, mx)

        # Hourly counts
        if "ts_utc" in df.columns:
            ts_series = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
            if ts_series.notna().any():
                hours = ts_series.dt.floor("h")
                counts = hours.value_counts()
                for h, c in counts.items():
                    # Ensure tz-aware UTC timestamps as keys
                    if not isinstance(h, pd.Timestamp):
                        continue
                    hourly_counts[h] = hourly_counts.get(h, 0) + int(c)

        # Trade counts
        if "trade" in df.columns:
            tc = df["trade"].astype(str).replace({None: "", "None": ""}).fillna("")
            vc = tc.value_counts()
            for k, v in vc.items():
                trade_counts[str(k)] = trade_counts.get(str(k), 0) + int(v)

        # Overlay reservoir (x,y valid)
        use_cols = ["x", "y", "trade", "trackable", "trackable_uid", "ts_short", "ts_iso", "ts_utc", "z"]
        for col in use_cols:
            if col not in df.columns:
                df[col] = "" if col not in ("x", "y", "z", "ts_utc") else (np.nan if col in ("x", "y", "z") else pd.NaT)

        x = pd.to_numeric(df["x"], errors="coerce")
        y = pd.to_numeric(df["y"], errors="coerce")
        mask = x.notna() & y.notna()
        sub = df.loc[mask, use_cols].copy()
        if not sub.empty:
            # decide how many to take to respect overlay_limit
            can_take = max(0, overlay_limit - len(overlay_rows))
            if can_take > 0:
                if len(sub) > can_take:
                    idx = np.linspace(0, len(sub) - 1, can_take).astype(int)
                    sub = sub.iloc[idx]
                # Append to overlay_rows
                for _, r in sub.iterrows():
                    overlay_rows.append({
                        "x": pd.to_numeric(r.get("x"), errors="coerce"),
                        "y": pd.to_numeric(r.get("y"), errors="coerce"),
                        "z": pd.to_numeric(r.get("z"), errors="coerce"),
                        "trade": str(r.get("trade", "")),
                        "trackable": str(r.get("trackable", "")),
                        "trackable_uid": str(r.get("trackable_uid", "")),
                        "ts_short": str(r.get("ts_short", "")),
                        "ts_iso": str(r.get("ts_iso", "")),
                        "ts_utc": pd.to_datetime(r.get("ts_utc"), utc=True, errors="coerce")
                    })

        # Release large df before next loop
        del df
        plt.close('all')

    # If no valid files or schema not validated
    if not first_file_validated:
        # Build minimal report with no data
        report_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        title = "Walmart Renovation RTLS Summary"
        meta = "No valid data found."
        sections: List[Dict[str, Any]] = []
        sections.append({"type": "summary", "title": "Summary", "bullets": ["No valid CSV data was found or inputs were empty."]})
        report = {"title": title, "meta": meta, "sections": sections}
        report = apply_budgets(report)

        try:
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except Exception as e:
            print("Error Report:")
            print(f"{e.__class__.__name__}: {e}")
            traceback.print_exc(limit=2)
            try:
                report_lite = make_lite(report)
                safe_build_pdf(report_lite, str(pdf_path), logo_path=str(LOGO))
            except Exception as e2:
                print("Error Report:")
                print(f"{e2.__class__.__name__}: {e2}")
                traceback.print_exc(limit=2)
                raise SystemExit(1)

        print(f"[Download the PDF]({file_uri(pdf_path)})")
        return

    # ------------------------ Finalization aggregates ------------------------
    # Overlay DataFrame
    overlay_df = pd.DataFrame(overlay_rows)
    if not overlay_df.empty:
        # Ensure ts_utc dtype proper
        if "ts_utc" in overlay_df.columns:
            overlay_df["ts_utc"] = pd.to_datetime(overlay_df["ts_utc"], utc=True, errors="coerce")

    # Hourly DataFrame
    if hourly_counts:
        hours_sorted = sorted(hourly_counts.keys())
        hourly_df = pd.DataFrame({
            "hour_utc": hours_sorted,
            "count_samples": [hourly_counts[h] for h in hours_sorted]
        })
    else:
        hourly_df = pd.DataFrame(columns=["hour_utc", "count_samples"])

    # Trade DataFrame
    if trade_counts:
        trades_sorted = sorted(trade_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        trade_df = pd.DataFrame({"trade": [k for k, _ in trades_sorted], "count": [v for _, v in trades_sorted]})
    else:
        trade_df = pd.DataFrame(columns=["trade", "count"])

    # Evidence table rows
    evidence_rows: List[Dict[str, Any]] = []
    try:
        df0 = first_df_for_table if first_df_for_table is not None else overlay_df
        if df0 is None or df0.empty:
            evidence_rows = []
        else:
            cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
            for col in cols:
                if col not in df0.columns:
                    df0[col] = ""
            evidence_rows = df0[cols].head(50).fillna("").astype(str).to_dict(orient="records")
    except Exception:
        evidence_rows = []

    # ------------------------ Figures via chart_policy ------------------------
    # Provide explicit local paths for floorplan and zones
    floorplans_path = str(FLOORJSON)
    floorplan_image_path = str(ROOT / "floorplan.jpeg")
    zones_path = str(ZONES_JSON)

    try:
        figs_all = choose_charts(
            overlay_df if isinstance(overlay_df, pd.DataFrame) else pd.DataFrame(),
            hourly_df=hourly_df,
            trade_df=trade_df,
            user_query=user_query,
            floorplans_path=floorplans_path,
            floorplan_image_path=floorplan_image_path,
            zones_path=zones_path,
            config=cfg_override
        )
    except Exception:
        # Fallback: simple bar chart for trades if chart_policy fails
        figs_all = []
        if not trade_df.empty:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
            ax.bar(trade_df["trade"].astype(str), trade_df["count"].astype(int))
            ax.set_ylabel("Samples")
            ax.set_title("Trade Distribution")
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            fig.tight_layout()
            figs_all.append(fig)

    # Filter to live matplotlib figures
    figs: List[plt.Figure] = [f for f in figs_all if getattr(f, "savefig", None)]

    # ------------------------ Save PNGs then build PDF ------------------------
    # Report date token
    if global_min_ts is not None:
        report_date = pd.to_datetime(global_min_ts, utc=True).strftime("%Y%m%d")
    else:
        report_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    png_paths: List[Path] = []
    for i, fig in enumerate(figs, start=1):
        p = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
        try:
            fig.savefig(str(p), dpi=120)
            png_paths.append(p)
        except Exception:
            # continue without this figure path
            continue

    # ------------------------ Build report dict ------------------------
    # Title and meta
    title = "Walmart Renovation RTLS Summary"
    range_text = ""
    if global_min_ts is not None and global_max_ts is not None:
        mn_str = pd.to_datetime(global_min_ts, utc=True).strftime("%Y-%m-%d %H:%M:%S Z")
        mx_str = pd.to_datetime(global_max_ts, utc=True).strftime("%Y-%m-%d %H:%M:%S Z")
        range_text = f"{mn_str} to {mx_str}"
    meta_parts = []
    if range_text:
        meta_parts.append(f"UTC Window: {range_text}")
    meta_parts.append(f"Total samples: {total_samples}")
    meta_parts.append(f"Unique tags: {len(tag_set)}")
    if not trade_df.empty:
        meta_parts.append(f"Trades observed: {len(trade_df)}")
    meta = " | ".join(meta_parts)

    # Summary bullets
    bullets = []
    bullets.append(f"Processed {len(csv_paths)} file(s) with {total_samples:,} position samples.")
    if range_text:
        bullets.append(f"UTC window: {range_text}.")
    bullets.append(f"Unique tags observed: {len(tag_set)}.")
    if not trade_df.empty:
        top3 = trade_df.head(3)
        parts = [f"{row['trade']} ({int(row['count'])})" for _, row in top3.iterrows()]
        bullets.append(f"Top trades by samples: {', '.join(parts)}.")

    sections: List[Dict[str, Any]] = []
    sections.append({"type": "summary", "title": "Summary", "bullets": bullets})

    if evidence_rows:
        sections.append({
            "type": "table",
            "title": "Evidence",
            "data": evidence_rows,
            "headers": ["trackable", "trade", "ts_short", "x", "y", "z"],
            "rows_per_page": 24
        })

    if figs:
        sections.append({"type": "charts", "title": "Figures", "figures": figs})

    report: Dict[str, Any] = {"title": title, "meta": meta, "sections": sections}
    report = apply_budgets(report)

    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"

    # Build PDF with fallback
    try:
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
    except Exception as e:
        print("Error Report:")
        print(f"{e.__class__.__name__}: {e}")
        traceback.print_exc(limit=2)
        try:
            report_lite = make_lite(report)
            safe_build_pdf(report_lite, str(pdf_path), logo_path=str(LOGO))
        except Exception as e2:
            print("Error Report:")
            print(f"{e2.__class__.__name__}: {e2}")
            traceback.print_exc(limit=2)
            raise SystemExit(1)

    # After successful PDF build, print links
    print(f"[Download the PDF]({file_uri(pdf_path)})")
    for i, pth in enumerate(png_paths, 1):
        print(f"[Download Plot {i}]({file_uri(pth)})")

    # Safe to close figures now
    plt.close('all')

if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        # Minimal-report mode on memory pressure
        try:
            # Build minimal PDF
            user_query = sys.argv[1] if len(sys.argv) > 1 else ""
            csv0 = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path.cwd()
            out_dir = (csv0.parent if csv0.exists() else Path.cwd()).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            report_date = datetime.now(timezone.utc).strftime("%Y%m%d")
            pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
            sections = [{"type": "summary", "title": "Summary", "bullets": ["Minimal report mode: memory constraints encountered during processing."]}]
            report = {"title": "Walmart Renovation RTLS Summary", "meta": "Minimal mode", "sections": sections}
            report = apply_budgets(report)
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            print(f"[Download the PDF]({file_uri(pdf_path)})")
        except Exception as e:
            print("Error Report:")
            print(f"{e.__class__.__name__}: {e}")
            traceback.print_exc(limit=2)
            raise SystemExit(1)
    except KeyboardInterrupt:
        print("Error Report:")
        print("Interrupted by user.")