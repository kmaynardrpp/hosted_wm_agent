import sys, os, json, traceback
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------- ROOT resolution and local imports ----------------------
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

# ---------------------- Helper imports ----------------------
try:
    from extractor import extract_tracks
except Exception as e:
    print("Error Report:")
    print(f"Missing extractor: {e.__class__.__name__}: {e}")
    raise SystemExit(1)

try:
    from chart_policy import choose_charts
except Exception as e:
    print("Error Report:")
    print(f"Missing chart_policy: {e.__class__.__name__}: {e}")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception as e:
    print("Error Report:")
    print(f"Missing pdf builder: {e.__class__.__name__}: {e}")
    raise SystemExit(1)

try:
    from report_limits import apply_budgets, make_lite, DEFAULTS as RL_DEFAULTS
except Exception as e:
    print("Error Report:")
    print(f"Missing report_limits: {e.__class__.__name__}: {e}")
    raise SystemExit(1)

# ---------------------- Utility ----------------------
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def _fmt_utc(ts: pd.Timestamp) -> str:
    try:
        if ts is None or pd.isna(ts):
            return ""
        if not getattr(ts, "tzinfo", None):
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""

def _linear_subsample_indices(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, k).astype(int)

# ---------------------- Main ----------------------
def main():
    try:
        # Read guidelines/context (do not print)
        _ = read_text(GUIDELINES)
        _ = read_text(CONTEXT)

        # Parse args
        if len(sys.argv) < 2:
            print("Error Report:")
            print("No user prompt provided.")
            raise SystemExit(1)
        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:] if p]

        # Determine out_dir
        out_dir = None
        if csv_paths:
            out_dir = csv_paths[0].resolve().parent
        else:
            # Fallback to ROOT if no CSVs provided
            out_dir = ROOT
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load config (optional)
        cfg = {}
        try:
            if CONFIG.exists():
                cfg = json.loads(CONFIG.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            cfg = {}
        overlay_limit = int(cfg.get("overlay_subsample", 20000)) if isinstance(cfg.get("overlay_subsample"), (int, float)) else 20000
        max_figures_budget = int(RL_DEFAULTS.get("MAX_FIGURES", 6))

        # Aggregates
        total_samples = 0
        unique_tracks = set()
        unique_trades = set()
        earliest_ts = None
        latest_ts = None
        hourly_counts: Dict[pd.Timestamp, int] = {}
        trade_counts: Dict[str, int] = {}
        evidence_rows: List[Dict[str, Any]] = []
        overlay_rows: List[Dict[str, Any]] = []

        first_nonempty_df_columns: List[str] = []
        schema_checked = False
        files_processed = 0
        files_missing: List[str] = []

        # Process each CSV one at a time (large-data mode)
        for csv_path in csv_paths:
            if not csv_path.exists():
                files_missing.append(str(csv_path))
                continue

            # Extract with MAC map path explicitly
            raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
            audit = raw.get("audit", {}) if isinstance(raw, dict) else {}
            if not audit or not audit.get("mac_map_loaded", False):
                print("Error Report:")
                print("MAC map was not loaded; cannot resolve trackable identities.")
                raise SystemExit(1)

            df = pd.DataFrame(raw.get("rows", []))
            # Duplicate-name guard
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
            # Timestamp canon
            src = df["ts_iso"] if "ts_iso" in df.columns else (df["ts"] if "ts" in df.columns else pd.Series([], dtype=str))
            df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

            # Skip if empty
            if df.empty:
                continue

            # Schema validation after first non-empty file
            if not schema_checked:
                cols = set(df.columns.astype(str))
                has_identity = ("trackable" in cols) or ("trackable_uid" in cols)
                has_trade = ("trade" in cols)
                has_xy = ("x" in cols) and ("y" in cols)
                if not (has_identity and has_trade and has_xy):
                    print("Error Report:")
                    print("Missing required columns for analysis.")
                    print(f"Columns detected: {','.join(df.columns.astype(str))}")
                    raise SystemExit(1)
                schema_checked = True
                first_nonempty_df_columns = list(df.columns.astype(str))

            files_processed += 1

            # Update aggregates
            total_samples += len(df)
            if "trackable_uid" in df.columns:
                unique_tracks.update([t for t in df["trackable_uid"].astype(str).tolist() if t])
            elif "trackable" in df.columns:
                unique_tracks.update([t for t in df["trackable"].astype(str).tolist() if t])

            if "trade" in df.columns:
                trades = df["trade"].astype(str).fillna("")
                for t, cnt in trades.value_counts().items():
                    if t:
                        trade_counts[t] = trade_counts.get(t, 0) + int(cnt)
                        unique_trades.add(t)

            # Time range
            ts_valid = df["ts_utc"].dropna()
            if not ts_valid.empty:
                mn = ts_valid.min()
                mx = ts_valid.max()
                earliest_ts = mn if earliest_ts is None else min(earliest_ts, mn)
                latest_ts = mx if latest_ts is None else max(latest_ts, mx)

                # Hourly counts
                hours = ts_valid.dt.floor("h")
                for h, cnt in hours.value_counts().items():
                    # ensure timezone-aware UTC
                    if not getattr(h, "tzinfo", None):
                        h = h.tz_localize("UTC")
                    else:
                        h = h.tz_convert("UTC")
                    hourly_counts[h] = hourly_counts.get(h, 0) + int(cnt)

            # Evidence table rows (bounded)
            try:
                cols_ev = ["trackable", "trade", "ts_short", "x", "y", "z"]
                missing_cols = [c for c in cols_ev if c not in df.columns]
                if missing_cols:
                    # reshape to at least columns present
                    avail = [c for c in cols_ev if c in df.columns]
                    if avail:
                        take = df[avail].head(max(0, 50 - len(evidence_rows))).fillna("").astype(str).to_dict(orient="records")
                        # pad dict keys to expected headers for table consistency
                        for r in take:
                            for c in cols_ev:
                                if c not in r:
                                    r[c] = ""
                        evidence_rows.extend(take)
                else:
                    if len(evidence_rows) < 50:
                        rows = df[cols_ev].head(50 - len(evidence_rows)).fillna("").astype(str).to_dict(orient="records")
                        evidence_rows.extend(rows)
            except Exception:
                # Skip evidence enrichment errors
                pass

            # Overlay reservoir (bounded by overlay_limit)
            try:
                if overlay_limit > 0 and ("x" in df.columns and "y" in df.columns):
                    x_num = pd.to_numeric(df["x"], errors="coerce")
                    y_num = pd.to_numeric(df["y"], errors="coerce")
                    mask = x_num.notna() & y_num.notna()
                    use = df.loc[mask, ["x", "y", "trade", "trackable", "trackable_uid", "ts_utc"]].copy()
                    if not use.empty:
                        remaining = max(0, overlay_limit - len(overlay_rows))
                        if remaining > 0:
                            idx = _linear_subsample_indices(len(use), remaining)
                            sub = use.iloc[idx]
                            overlay_rows.extend(sub.to_dict(orient="records"))
            except Exception:
                pass

            # Clear per-file large DataFrame before next file
            del df
            plt.close('all')

        # Build aggregates DataFrames
        overlay_df = pd.DataFrame(overlay_rows) if overlay_rows else pd.DataFrame(columns=["x", "y", "trade", "trackable", "trackable_uid", "ts_utc"])
        if "ts_utc" in overlay_df.columns:
            overlay_df["ts_utc"] = pd.to_datetime(overlay_df["ts_utc"], utc=True, errors="coerce")

        if hourly_counts:
            hours_sorted = sorted(hourly_counts.items(), key=lambda kv: kv[0])
            hourly_df = pd.DataFrame({"hour_utc": [h for h, _ in hours_sorted], "count": [c for _, c in hours_sorted]})
        else:
            hourly_df = None

        if trade_counts:
            trade_df = pd.DataFrame({"trade": list(trade_counts.keys()), "count": list(trade_counts.values())})
        else:
            trade_df = None

        # Determine report date string
        if earliest_ts and latest_ts:
            d0 = pd.Timestamp(earliest_ts).tz_convert("UTC").date()
            d1 = pd.Timestamp(latest_ts).tz_convert("UTC").date()
            report_date = f"{d0.isoformat()}" if d0 == d1 else f"{d0.isoformat()}_to_{d1.isoformat()}"
        else:
            report_date = "undated"

        # Figures: use chart_policy with explicit local paths
        figs: List[Any] = []
        try:
            figs = choose_charts(
                overlay_df,
                hourly_df=hourly_df,
                trade_df=trade_df,
                user_query=str(user_prompt or ""),
                floorplans_path=str(FLOORJSON),
                floorplan_image_path=str(ROOT / "floorplan.jpeg"),
                zones_path=str(ZONES_JSON),
                config=cfg if isinstance(cfg, dict) else None
            ) or []
        except Exception:
            figs = []
        # Filter live figures and cap to budget
        live_figs = [f for f in figs if getattr(f, "savefig", None)]
        live_figs = live_figs[:max_figures_budget]

        # Save PNGs before PDF
        png_paths: List[Path] = []
        for i, fig in enumerate(live_figs, 1):
            p = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(p), dpi=120)
                png_paths.append(p)
            except Exception:
                # If save fails, drop this fig from list
                continue

        # Build report
        title = "Walmart Renovation RTLS Summary"
        time_bullet = "No valid timestamps found."
        if earliest_ts and latest_ts:
            time_bullet = f"Time window (UTC): { _fmt_utc(pd.Timestamp(earliest_ts)) } to { _fmt_utc(pd.Timestamp(latest_ts)) }"

        bullets = [
            time_bullet,
            f"Samples: {total_samples:,}",
            f"Unique trackables: {len(unique_tracks)}",
            f"Trades observed: {len(unique_trades)}",
            f"Files processed: {files_processed}" + (f" (missing: {len(files_missing)})" if files_missing else "")
        ]
        sections: List[Dict[str, Any]] = []
        sections.append({"type": "summary", "title": "Summary", "bullets": bullets})

        # Evidence table
        try:
            cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
            if evidence_rows:
                # Ensure all rows have these keys
                norm_rows = []
                for r in evidence_rows[:50]:
                    rr = {k: str(r.get(k, "")) for k in cols}
                    norm_rows.append(rr)
                sections.append({"type": "table", "title": "Evidence", "data": norm_rows, "headers": cols, "rows_per_page": 24})
        except Exception:
            pass

        # Charts section only if there are valid figures
        if live_figs:
            sections.append({"type": "charts", "title": "Figures", "figures": live_figs})

        # Meta string
        meta_parts = []
        if earliest_ts and latest_ts:
            meta_parts.append(f"UTC range: { _fmt_utc(pd.Timestamp(earliest_ts)) } → { _fmt_utc(pd.Timestamp(latest_ts)) }")
        meta_parts.append(f"Samples={total_samples:,} • Trackables={len(unique_tracks)} • Trades={len(unique_trades)}")
        if files_processed or files_missing:
            meta_parts.append(f"Files: ok={files_processed}, missing={len(files_missing)}")
        meta = " | ".join(meta_parts)

        report = {
            "title": title,
            "meta": meta,
            "sections": sections
        }

        # Apply budgets
        report = apply_budgets(report)

        # Ensure out dir
        out_dir.mkdir(parents=True, exist_ok=True)
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
        finally:
            # Close figures AFTER PDF build
            try:
                for f in live_figs:
                    try:
                        plt.close(f)
                    except Exception:
                        pass
            except Exception:
                pass

        # Success links (PDF first, then plots if any)
        print(f"[Download the PDF]({file_uri(pdf_path)})")
        for i, pth in enumerate(png_paths, 1):
            print(f"[Download Plot {i}]({file_uri(pth)})")

    except SystemExit:
        raise
    except Exception as e:
        print("Error Report:")
        msg = f"{e.__class__.__name__}: {e}"
        print(msg)
        # short traceback
        traceback.print_exc(limit=2)
        raise SystemExit(1)

if __name__ == "__main__":
    main()