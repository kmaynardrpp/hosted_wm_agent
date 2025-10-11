#!/usr/bin/env python3
# InfoZoneBuilder — Walmart RTLS positions analyzer
import sys, os, json, math, traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- ROOT resolution & local imports -----------------------
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

# ------------------------------ Helper imports --------------------------------
try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Missing local helper 'extractor.py' under ROOT.")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("Missing local helper 'pdf_creation_script.py' or 'safe_build_pdf'.")
    raise SystemExit(1)

# chart policy (optional; we'll handle fallback figs ourselves if needed)
try:
    from chart_policy import choose_charts
    HAVE_CHART_POLICY = True
except Exception:
    HAVE_CHART_POLICY = False

# budgets
try:
    from report_limits import apply_budgets, make_lite
except Exception:
    def apply_budgets(report: Dict[str, Any], caps: Optional[Dict[str, int]]=None) -> Dict[str, Any]:
        return report
    def make_lite(report: Dict[str, Any]) -> Dict[str, Any]:
        return report

# --------------------------------- Utilities ----------------------------------
def _parse_args():
    if len(sys.argv) < 2:
        print("Error Report:")
        print("No arguments provided. Expect: python script.py \"<USER_PROMPT>\" <CSV1> [CSV2 ...]")
        raise SystemExit(1)
    user_query = sys.argv[1]
    csv_paths = [Path(p) for p in sys.argv[2:]]
    if not csv_paths:
        print("Error Report:")
        print("No CSV inputs provided.")
        raise SystemExit(1)
    for p in csv_paths:
        if not p.exists() or not p.is_file():
            print("Error Report:")
            print(f"CSV not found: {p}")
            raise SystemExit(1)
    return user_query, csv_paths

def _load_config() -> Dict[str, Any]:
    try:
        cfg = json.loads(read_text(CONFIG)) if CONFIG.exists() else {}
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception:
        return {}

def _ts_range_str(t0: Optional[pd.Timestamp], t1: Optional[pd.Timestamp]) -> str:
    if t0 is None or pd.isna(t0) or t1 is None or pd.isna(t1):
        return "N/A"
    try:
        t0u = pd.to_datetime(t0, utc=True)
        t1u = pd.to_datetime(t1, utc=True)
        return f"{t0u.isoformat()} to {t1u.isoformat()}"
    except Exception:
        return "N/A"

def _report_date_label(t0: Optional[pd.Timestamp], t1: Optional[pd.Timestamp]) -> str:
    try:
        if t0 is None or pd.isna(t0) or t1 is None or pd.isna(t1):
            return pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        t0u = pd.to_datetime(t0, utc=True)
        t1u = pd.to_datetime(t1, utc=True)
        d0 = t0u.date().isoformat()
        d1 = t1u.date().isoformat()
        return d0 if d0 == d1 else f"{d0}_to_{d1}"
    except Exception:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d")

def _make_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_utc" not in df.columns:
        return pd.DataFrame(columns=["hour_utc", "count"])
    s = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return pd.DataFrame(columns=["hour_utc", "count"])
    hours = s.dt.floor("H")
    out = hours.value_counts().sort_index()
    return pd.DataFrame({"hour_utc": out.index, "count": out.values})

def _make_trade_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "trade" not in df.columns:
        return pd.DataFrame(columns=["trade", "count"])
    g = df["trade"].fillna("").astype(str).value_counts()
    return pd.DataFrame({"trade": g.index, "count": g.values})

def _make_line_hourly_figure(hourly_df: pd.DataFrame) -> Optional[plt.Figure]:
    try:
        if hourly_df is None or hourly_df.empty:
            return None
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        x = pd.to_datetime(hourly_df["hour_utc"], utc=True, errors="coerce")
        y = pd.to_numeric(hourly_df["count"], errors="coerce").fillna(0).astype(int)
        if x.notna().sum() < 2:
            plt.close(fig)
            return None
        ax.plot(x.dt.tz_convert('UTC').dt.tz_localize(None), y, color="#1f77b4", linewidth=1.8)
        ax.set_title("Hourly Sample Counts (UTC)")
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Samples")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig
    except Exception:
        return None

def _make_trade_bar_figure(trade_df: pd.DataFrame, top_n: int = 10) -> Optional[plt.Figure]:
    try:
        if trade_df is None or trade_df.empty:
            return None
        td = trade_df.copy()
        td["count"] = pd.to_numeric(td["count"], errors="coerce").fillna(0).astype(int)
        td = td.sort_values("count", ascending=False).head(top_n)
        if td.empty:
            return None
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        ax.barh(td["trade"].astype(str), td["count"].astype(int), color="#4e79a7")
        ax.invert_yaxis()
        ax.set_title("Top Trades by Sample Count")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Trade")
        fig.tight_layout()
        return fig
    except Exception:
        return None

def _evidence_rows(df: pd.DataFrame) -> (List[str], List[Dict[str, str]]):
    want_cols = ["trackable","trade","ts_short","x","y","z"]
    cols = [c for c in want_cols if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if c in ("trackable","trade","ts_iso","x","y","z","ts","ts_short")][:6]
    rows: List[Dict[str, str]] = []
    try:
        rows = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
    except Exception:
        # very defensive fallback
        try:
            rows = df.head(50).fillna("").astype(str).to_dict(orient="records")
            cols = list(rows[0].keys()) if rows else cols
        except Exception:
            rows = []
    return cols, rows

def _schema_validate(df: pd.DataFrame) -> None:
    cols = set(df.columns.astype(str))
    identity_ok = ("trackable" in cols) or ("trackable_uid" in cols)
    trade_ok = ("trade" in cols)
    xy_ok = ("x" in cols) and ("y" in cols)
    if not (identity_ok and trade_ok and xy_ok):
        print("Error Report:")
        print("Missing required columns for analysis.")
        print("Columns detected: " + ",".join(df.columns.astype(str)))
        raise SystemExit(1)

def _reservoir_append(res: List[Dict[str, Any]], rows: List[Dict[str, Any]], cap: int, rng_seed: Optional[int]=None) -> None:
    # Simple reservoir sampling: keep at most 'cap' items
    import random
    rnd = random.Random(rng_seed)
    for r in rows:
        if len(res) < cap:
            res.append(r)
        else:
            j = rnd.randint(0, max(0, len(res)))
            if j < cap:
                res[j] = r

def main():
    # Ensure guidelines read (encoding-safe)
    _ = read_text(GUIDELINES)

    user_query, csv_paths = _parse_args()
    out_dir = csv_paths[0].resolve().parent

    config = _load_config()
    # Zones only if asked: force disable if not asked
    user_wants_zones = ("zone" in user_query.lower()) or ("zones" in user_query.lower()) or ("area" in user_query.lower())
    if not user_wants_zones:
        config = dict(config or {})
        config["draw_zones"] = False

    # floorplan & refs
    floorplans_path = str(FLOORJSON)
    floorplan_image_path = str(ROOT / "floorplan.jpeg")  # provided locally
    zones_path = str(ZONES_JSON)

    # Aggregates across files (memory-safe)
    total_samples = 0
    t0: Optional[pd.Timestamp] = None
    t1: Optional[pd.Timestamp] = None
    per_trade_counts: Dict[str, int] = {}
    hourly_counts: Dict[pd.Timestamp, int] = {}
    overlay_reservoir: List[Dict[str, Any]] = []
    overlay_cap = int(config.get("overlay_subsample", 20000))
    evidence_cols: List[str] = []
    evidence_rows: List[Dict[str, str]] = []

    # Minimal audit collection
    audits: List[str] = []

    first_file_validated = False

    try:
        for idx, csv_path in enumerate(csv_paths, 1):
            # Extract with local MAC map
            raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
            audit = raw.get("audit", {}) or {}
            if not audit.get("mac_map_loaded", False):
                print("Error Report:")
                print("MAC map could not be loaded (trackable_objects.json).")
                raise SystemExit(1)
            audits.append(f"{csv_path.name}: rows={audit.get('rows', 'n/a')} mac_map_loaded={audit.get('mac_map_loaded')}")

            df = pd.DataFrame(raw.get("rows", []))
            # Duplicate-name guard
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            # Timestamp canon (single source of truth)
            src = df["ts_iso"] if "ts_iso" in df.columns else (df["ts"] if "ts" in df.columns else pd.Series([], dtype=str))
            df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

            # Early schema validation (after first file)
            if not first_file_validated:
                _schema_validate(df)
                first_file_validated = True

            # Update time range
            if "ts_utc" in df.columns and not df["ts_utc"].empty:
                v = df["ts_utc"].dropna()
                if not v.empty:
                    vmin = v.min()
                    vmax = v.max()
                    t0 = vmin if (t0 is None or vmin < t0) else t0
                    t1 = vmax if (t1 is None or vmax > t1) else t1

            # Update totals
            total_samples += len(df)

            # Per-trade counts
            if "trade" in df.columns:
                vc = df["trade"].fillna("").astype(str).value_counts()
                for k, c in vc.items():
                    per_trade_counts[k] = per_trade_counts.get(k, 0) + int(c)

            # Hourly counts
            if "ts_utc" in df.columns:
                ts = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dropna()
                hrs = ts.dt.floor("H")
                vc = hrs.value_counts()
                for h, c in vc.items():
                    # h is tz-aware; keep as UTC
                    hour = pd.to_datetime(h, utc=True)
                    hourly_counts[hour] = hourly_counts.get(hour, 0) + int(c)

            # Overlay reservoir (x,y plus color by trade)
            keep_cols = [c for c in ["x","y","z","trackable","trade","trackable_uid","ts_utc"] if c in df.columns]
            if "x" in df.columns and "y" in df.columns:
                # Only rows with valid numeric x,y
                xn = pd.to_numeric(df["x"], errors="coerce")
                yn = pd.to_numeric(df["y"], errors="coerce")
                mask = xn.notna() & yn.notna()
                df_use = df.loc[mask, keep_cols].copy()
                # Convert to plain dicts (strings okay for colors/labels)
                rows = df_use.to_dict(orient="records")
                _reservoir_append(overlay_reservoir, rows, cap=overlay_cap, rng_seed=idx)

            # Evidence table from first ingested file
            if idx == 1:
                evidence_cols, evidence_rows = _evidence_rows(df)

            # drop large frames for memory
            del df
            plt.close('all')

        # Final aggregates to DataFrames
        # Overlay DF (small reservoir)
        overlay_df = pd.DataFrame(overlay_reservoir) if overlay_reservoir else pd.DataFrame(columns=["x","y"])
        # Hourly DF
        if hourly_counts:
            hours_sorted = sorted(hourly_counts.items(), key=lambda kv: kv[0])
            hourly_df = pd.DataFrame({"hour_utc": [h for h, _ in hours_sorted],
                                      "count": [c for _, c in hours_sorted]})
        else:
            hourly_df = pd.DataFrame(columns=["hour_utc","count"])
        # Trade DF
        if per_trade_counts:
            trades_sorted = sorted(per_trade_counts.items(), key=lambda kv: kv[1], reverse=True)
            trade_df = pd.DataFrame({"trade": [k for k, _ in trades_sorted],
                                     "count": [v for _, v in trades_sorted]})
        else:
            trade_df = pd.DataFrame(columns=["trade","count"])

        # Figures
        figs: List[plt.Figure] = []
        png_paths: List[Path] = []

        # Try chart_policy overlay (if available)
        if HAVE_CHART_POLICY:
            try:
                figs_policy = choose_charts(
                    overlay_df,
                    hourly_df=hourly_df if not hourly_df.empty else None,
                    trade_df=trade_df if not trade_df.empty else None,
                    user_query=user_query or "",
                    floorplans_path=floorplans_path,
                    floorplan_image_path=floorplan_image_path,
                    zones_path=zones_path,
                    config=config
                )
                # Expect a list of Figures; be defensive
                if isinstance(figs_policy, list):
                    # Filter only matplotlib Figures
                    figs_policy = [f for f in figs_policy if hasattr(f, "savefig")]
                    figs.extend(figs_policy)
            except Exception:
                # Fallback to our own figures below
                pass

        # If no figures yet, create at least hourly line and/or trade bar
        if not figs:
            f1 = _make_line_hourly_figure(hourly_df)
            if f1 is not None:
                figs.append(f1)
            f2 = _make_trade_bar_figure(trade_df)
            if f2 is not None:
                figs.append(f2)

        # Build report
        report_date = _report_date_label(t0, t1)
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"

        # Save PNGs first (mandatory order)
        for i, fig in enumerate(figs, 1):
            png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(png), dpi=120)
                png_paths.append(png)
            except Exception:
                # continue without saving this fig
                continue

        # Sections
        sections: List[Dict[str, Any]] = []

        # Summary bullets
        bullets = []
        bullets.append(f"CSV files analyzed: {len(csv_paths)}")
        bullets.append(f"Total samples: {total_samples}")
        bullets.append(f"Time range (UTC): { _ts_range_str(t0, t1) }")
        if not trade_df.empty:
            top_trade = trade_df.sort_values("count", ascending=False).iloc[0]
            bullets.append(f"Top trade by samples: {top_trade['trade']} ({int(top_trade['count'])})")
        else:
            bullets.append("Trade distribution unavailable.")

        sections.append({"type": "summary", "title": "Summary", "bullets": bullets})

        # Evidence table
        if evidence_rows:
            sections.append({
                "type": "table",
                "title": "Evidence (first 50 rows)",
                "data": evidence_rows,
                "headers": evidence_cols,
                "rows_per_page": 24
            })

        # Charts section (pass live figs)
        if figs:
            sections.append({"type": "charts", "title": "Figures", "figures": figs})

        # Narrative / appendix (optional light)
        appendix_text = "\n".join(audits[:8])
        if appendix_text:
            sections.append({"type":"appendix","title":"Audit (truncated)","text":appendix_text})

        # Compose report dict
        meta_lines = [
            f"Generated from local Walmart RTLS positions",
            f"Date label: {report_date}",
            f"Root: {str(ROOT)}",
        ]
        report: Dict[str, Any] = {
            "title": "Walmart Renovation RTLS Summary",
            "meta": " | ".join(meta_lines),
            "sections": sections
        }

        # Apply budgets
        report = apply_budgets(report)

        # Build PDF (write after PNGs; keep figures alive)
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))

        # Success links
        print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        for i, pth in enumerate(png_paths, 1):
            print(f"[Download Plot {i}](file:///{pth.resolve().as_posix()})")

    except (MemoryError, KeyboardInterrupt):
        # Minimal-Report Mode: summary + evidence + audit (no PNGs)
        try:
            report_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
            pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
            bullets = [
                f"CSV files analyzed: {len(csv_paths)}",
                f"Total samples (approx): {total_samples}",
                f"Time range (UTC): { _ts_range_str(t0, t1) }",
                "Minimal mode due to resource limits."
            ]
            sections = [{"type":"summary","title":"Summary","bullets":bullets}]
            if evidence_rows:
                sections.append({
                    "type":"table","title":"Evidence (first 50 rows)",
                    "data": evidence_rows,"headers": evidence_cols,"rows_per_page":24
                })
            if audits:
                sections.append({"type":"appendix","title":"Audit (truncated)","text":"\n".join(audits[:8])})
            report = {"title":"Walmart Renovation RTLS Summary (Lite)",
                      "meta":"Minimal write path","sections":sections}
            report = make_lite(report)
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        except Exception:
            print("Error Report:")
            print("Failed to write minimal PDF after resource limit.")
            raise SystemExit(1)
    except SystemExit:
        # already printed a specific error; just propagate
        raise
    except Exception as e:
        # Generic failure
        try:
            err = str(e).strip() or e.__class__.__name__
        except Exception:
            err = "Unknown error"
        print("Error Report:")
        print(err[:300])

if __name__ == "__main__":
    main()