#!/usr/bin/env python3
# InfoZoneBuilder — Walmart RTLS Analyzer
import sys, os
from pathlib import Path

# ----------------------------- ROOT & LOCAL IMPORTS -----------------------------
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
try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
    import numpy as _np
    _FCA.tostring_rgb = getattr(_FCA, "tostring_rgb", lambda self: _np.asarray(self.buffer_rgba())[..., :3].tobytes())
except Exception:
    pass

# ----------------------------- STANDARD IMPORTS --------------------------------
import json
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
    print("Missing local extractor helper.")
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
    # Fallbacks if apply_budgets isn't available
    def apply_budgets(report, caps=None): return report
    def make_lite(report): return report

# Chart policy (prefer floorplan overlay when available)
try:
    from chart_policy import choose_charts
    _HAS_CHART_POLICY = True
except Exception:
    _HAS_CHART_POLICY = False

# ----------------------------- UTILS -------------------------------------------
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def parse_args(argv):
    if len(argv) < 3:
        print("Error Report:")
        print("Usage: python script.py \"<USER_PROMPT>\" <csv1> [csv2 ...]")
        raise SystemExit(1)
    user_prompt = argv[1]
    csv_paths = [Path(a) for a in argv[2:]]
    # Validate existence
    for p in csv_paths:
        if not p.exists():
            print("Error Report:")
            print(f"CSV not found: {p}")
            raise SystemExit(1)
    return user_prompt, csv_paths

def _iso_z(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    try:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return str(ts)

def _safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def _ensure_ts_short(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_short" not in df.columns:
        try:
            tsu = pd.to_datetime(df.get("ts_utc"), utc=True, errors="coerce")
            # For display only; do not mutate tz-awareness elsewhere
            df = df.copy()
            df["ts_short"] = tsu.dt.tz_convert("UTC").dt.tz_localize(None).dt.strftime("%m-%d\n%H:%M")
        except Exception:
            df["ts_short"] = ""
    return df

# ----------------------------- MAIN --------------------------------------------
def main():
    # Read guidelines/context (Windows-safe; ignore errors)
    _ = read_text(GUIDELINES)
    _ = read_text(CONTEXT)

    user_prompt, csv_paths = parse_args(sys.argv)
    out_dir = csv_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config (optional)
    cfg = {}
    try:
        if CONFIG.exists():
            cfg = json.loads(read_text(CONFIG)) or {}
    except Exception:
        cfg = {}
    # Zones only if asked
    want_zones = any(k in user_prompt.lower() for k in ["zone", "zones", "area", "room", "aisle"]) and False  # default OFF unless explicitly asked
    # Force config for this run
    cfg = dict(cfg or {})
    cfg["draw_zones"] = bool(want_zones)  # do not draw zones unless asked

    overlay_cap = int(cfg.get("overlay_subsample", 20000))
    max_figures = int(cfg.get("max_figures", 6))

    # Aggregates (bounded)
    overlay_have = 0
    overlay_samples: list[pd.DataFrame] = []
    hour_counts: dict[pd.Timestamp, int] = {}
    trade_counts: dict[str, int] = {}
    unique_ids: set[str] = set()
    total_samples = 0
    global_min_ts = None
    global_max_ts = None
    evidence_rows: list[dict] = []

    first_df_for_schema: pd.DataFrame | None = None
    first_audit: dict | None = None

    # Per-file processing (large-data mode)
    for idx, csv_path in enumerate(csv_paths, start=1):
        try:
            raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
        except Exception as e:
            print("Error Report:")
            print(f"Extractor failed: {e}")
            raise SystemExit(1)

        audit = raw.get("audit", {}) or {}
        if idx == 1:
            first_audit = audit

        # Enforce MAC→trackable map loaded
        if not audit.get("mac_map_loaded", False):
            print("Error Report:")
            print("MAC map not loaded from trackable_objects.json.")
            raise SystemExit(1)

        rows = raw.get("rows", []) or []
        df = pd.DataFrame(rows)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Timestamp canon (single source of truth)
        src = df["ts_iso"] if "ts_iso" in df.columns else df.get("ts")
        df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

        # Early schema validation after first file
        if idx == 1:
            first_df_for_schema = df
            cols = df.columns.astype(str)
            has_identity = ("trackable" in cols) or ("trackable_uid" in cols)
            has_trade = ("trade" in cols)
            has_xy = ("x" in cols) and ("y" in cols)
            if not (has_identity and has_trade and has_xy):
                print("Error Report:")
                print("Missing required columns for analysis.")
                print(f"Columns detected: {','.join(df.columns.astype(str))}")
                raise SystemExit(1)

        # Aggregates
        total_samples += len(df)

        # Min/max time window (ts_utc)
        tsu = df["ts_utc"]
        try:
            mn = tsu.min() if len(df) else pd.NaT
            mx = tsu.max() if len(df) else pd.NaT
        except Exception:
            mn, mx = pd.NaT, pd.NaT
        if pd.notna(mn):
            global_min_ts = mn if (global_min_ts is None or mn < global_min_ts) else global_min_ts
        if pd.notna(mx):
            global_max_ts = mx if (global_max_ts is None or mx > global_max_ts) else global_max_ts

        # Overlay reservoir (bounded)
        # Keep rows with valid x,y (numeric) and any ts_utc (not strictly required for scatter)
        if "x" in df.columns and "y" in df.columns:
            xnum = _safe_to_numeric(df["x"])
            ynum = _safe_to_numeric(df["y"])
            valid_mask = xnum.notna() & ynum.notna()
            use = df.loc[valid_mask, ["x", "y", "trade", "trackable", "trackable_uid", "ts_utc"]].copy() if valid_mask.any() else pd.DataFrame(columns=["x","y","trade","trackable","trackable_uid","ts_utc"])
            if len(use) > 0 and overlay_have < overlay_cap:
                remain = overlay_cap - overlay_have
                take_n = min(remain, len(use))
                if take_n > 0:
                    if take_n < len(use):
                        idxs = np.linspace(0, len(use) - 1, take_n).astype(int)
                        samp = use.iloc[idxs]
                    else:
                        samp = use
                    overlay_samples.append(samp)
                    overlay_have += len(samp)

        # Hourly counts
        try:
            if df["ts_utc"].notna().any():
                h = df["ts_utc"].dt.floor("h")
                vc = h.value_counts()
                for k, v in vc.items():
                    hour_counts[k] = hour_counts.get(k, 0) + int(v)
        except Exception:
            pass

        # Trade counts
        try:
            if "trade" in df.columns:
                vc = df["trade"].astype(str).fillna("").value_counts()
                for k, v in vc.items():
                    trade_counts[str(k)] = trade_counts.get(str(k), 0) + int(v)
        except Exception:
            pass

        # Unique IDs
        id_col = "trackable_uid" if "trackable_uid" in df.columns else ("trackable" if "trackable" in df.columns else None)
        if id_col:
            try:
                vals = set(str(x) for x in df[id_col].dropna().unique().tolist())
                unique_ids |= vals
            except Exception:
                pass

        # Evidence rows (up to 50 across files)
        if len(evidence_rows) < 50:
            try:
                df_ev = _ensure_ts_short(df.copy())
                cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
                for c in cols:
                    if c not in df_ev.columns:
                        df_ev[c] = ""
                rows = df_ev[cols].head(50 - len(evidence_rows)).fillna("").astype(str).to_dict(orient="records")
                evidence_rows.extend(rows)
            except Exception:
                pass

        # Release per-file DF (keep only aggregates)
        del df

    # Final aggregates to DataFrames
    if overlay_samples:
        overlay_df = pd.concat(overlay_samples, ignore_index=True)
    else:
        overlay_df = pd.DataFrame(columns=["x","y","trade","trackable","trackable_uid","ts_utc"])

    if hour_counts:
        hours_sorted = sorted(hour_counts.items(), key=lambda kv: kv[0])
        hourly_df = pd.DataFrame({"hour_utc": [k for k, _ in hours_sorted], "count_samples": [v for _, v in hours_sorted]})
    else:
        hourly_df = pd.DataFrame(columns=["hour_utc", "count_samples"])

    if trade_counts:
        trades_sorted = sorted(trade_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        trade_df = pd.DataFrame({"trade": [k for k, _ in trades_sorted], "count_samples": [v for _, v in trades_sorted]})
    else:
        trade_df = pd.DataFrame(columns=["trade", "count_samples"])

    # ----------------------------- FIGURES -------------------------------------
    figs: list = []
    png_paths: list[Path] = []
    report_date = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    if _HAS_CHART_POLICY and not overlay_df.empty:
        try:
            figs = choose_charts(
                overlay_df,
                hourly_df=hourly_df if not hourly_df.empty else None,
                trade_df=trade_df if not trade_df.empty else None,
                user_query=user_prompt,
                floorplans_path=str(FLOORJSON),
                floorplan_image_path=str(ROOT / "floorplan.jpeg"),
                zones_path=str(ZONES_JSON),
                config=cfg
            ) or []
        except Exception:
            figs = []
    # Filter only live figures and cap by config
    figs = [f for f in figs if getattr(f, "savefig", None)]
    if len(figs) > max_figures:
        figs = figs[:max_figures]

    # Save PNGs FIRST (mandatory order)
    for i, fig in enumerate(figs, start=1):
        try:
            p = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            fig.savefig(str(p), dpi=120)
            png_paths.append(p)
        except Exception:
            # If saving a figure fails, skip it for PDF, but keep others
            continue

    # ----------------------------- REPORT BUILD --------------------------------
    # Meta and sections
    start_iso = _iso_z(global_min_ts) if global_min_ts is not None else ""
    end_iso   = _iso_z(global_max_ts) if global_max_ts is not None else ""
    file_list = "; ".join([p.name for p in csv_paths])
    bullets = []
    bullets.append(f"Analyzed {len(csv_paths)} file(s) with {total_samples} sample(s).")
    if start_iso or end_iso:
        bullets.append(f"UTC window: {start_iso} to {end_iso}")
    if unique_ids:
        bullets.append(f"Unique tags: {len(unique_ids)}")
    if not trade_df.empty:
        top_n = min(5, len(trade_df))
        tops = ", ".join([f"{trade_df.iloc[i]['trade']} ({int(trade_df.iloc[i]['count_samples'])})" for i in range(top_n)])
        bullets.append(f"Top trades by samples: {tops}")

    # Ensure at least one bullet
    if not bullets:
        bullets = ["No valid data rows were found to analyze."]

    sections: list[dict] = []
    sections.append({"type": "summary", "title": "Summary", "bullets": bullets})

    # Evidence table (list-of-dicts)
    if evidence_rows:
        cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
        sections.append({
            "type": "table",
            "title": "Evidence",
            "data": evidence_rows[:50],
            "headers": cols,
            "rows_per_page": 24
        })

    # Charts section only if non-empty
    if figs:
        sections.append({"type": "charts", "title": "Figures", "figures": figs})

    report_title = "Walmart RTLS Position Summary"
    meta_str = f"Files: {len(csv_paths)} | Samples: {total_samples} | Tags: {len(unique_ids)} | Window (UTC): {start_iso} → {end_iso}\n{file_list}"
    report = {
        "title": report_title,
        "meta": meta_str,
        "sections": sections
    }

    # Apply budgets (safe cap)
    report = apply_budgets(report)

    # Build PDF (with fallback)
    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
    try:
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
    except Exception as e:
        print("Error Report:")
        print(f"{e.__class__.__name__}: {e}")
        traceback.print_exc(limit=2)
        try:
            report2 = make_lite(report)
            safe_build_pdf(report2, str(pdf_path), logo_path=str(LOGO))
        except Exception as e2:
            print("Error Report:")
            print(f"{e2.__class__.__name__}: {e2}")
            traceback.print_exc(limit=2)
            raise SystemExit(1)
    finally:
        # Now it is safe to close figures
        try:
            for f in figs:
                try:
                    plt.close(f)
                except Exception:
                    pass
            plt.close("all")
        except Exception:
            pass

    # ----------------------------- LINKS (SUCCESS) ------------------------------
    print(f"[Download the PDF]({file_uri(pdf_path)})")
    for i, pth in enumerate(png_paths, 1):
        print(f"[Download Plot {i}]({file_uri(pth)})")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as ex:
        # Generic failure path
        print("Error Report:")
        print(f"{ex.__class__.__name__}: {ex}")
        raise SystemExit(1)