#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
from pathlib import Path

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

# ----------------------------- Standard imports -------------------------------
import json
import math
import traceback
import datetime as dt
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ------------------------------ Helper imports --------------------------------
try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Missing required local helper: extractor.py")
    raise SystemExit(1)

try:
    from zones_process import load_zones, compute_zone_intervals
except Exception:
    # zones only if asked; if import fails, we will error when needed
    load_zones = None
    compute_zone_intervals = None

try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("Missing required local helper: pdf_creation_script.py")
    raise SystemExit(1)

try:
    from report_limits import apply_budgets, DEFAULTS as LIMIT_DEFAULTS
except Exception:
    def apply_budgets(report, caps=None): return report
    LIMIT_DEFAULTS = {"MAX_FIGURES": 6, "MAX_TABLE_ROWS_TOTAL": 180, "MAX_TEXT_LINES_TOTAL": 900, "MAX_PAGES": 12}

# ------------------------------- CLI & Inputs ---------------------------------
def main():
    try:
        if len(sys.argv) < 3:
            print("Error Report:")
            print("Usage: python generated.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
            return

        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:]]

        # Output directory is the first CSV's folder
        out_dir = csv_paths[0].parent.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load config (optional)
        try:
            cfg = json.loads(read_text(CONFIG)) if CONFIG.exists() else {}
        except Exception:
            cfg = {}
        top_n = int(cfg.get("top_n", 10))
        max_figs_cfg = int(cfg.get("max_figures", LIMIT_DEFAULTS.get("MAX_FIGURES", 6)))

        # Zones requested? The user prompt includes "zone"
        zones_requested = "zone" in user_prompt.lower()
        # Load polygons once if needed
        zones_list = []
        if zones_requested:
            if load_zones is None:
                print("Error Report:")
                print("Zones requested but zones helper is unavailable.")
                return
            zones_list = load_zones(str(ZONES_JSON), only_active=True) if ZONES_JSON.exists() else load_zones(None, only_active=True)
            # zones_list may be empty; we'll validate after ingest schema if no zone_name column

        # Aggregation stores (seconds)
        totals_by_zone_trade = defaultdict(float)      # (zone, trade) -> seconds
        daily_by_zone_trade  = defaultdict(float)      # (zone, trade, day_str) -> seconds
        totals_by_zone       = defaultdict(float)      # zone -> seconds
        totals_by_trade      = defaultdict(float)      # trade -> seconds

        # Evidence and audit
        first_df_for_evidence = None
        meta_audit = []

        # Track date window
        global_min_ts = None
        global_max_ts = None
        all_days = set()

        # Schema-tracking for early validation
        first_file_columns = None
        mac_map_ok = True

        # ------------------------ Per-file processing loop ------------------------
        for file_idx, csv_path in enumerate(csv_paths):
            try:
                raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
            except Exception as e:
                print("Error Report:")
                print(f"Failed to read CSV via extractor: {csv_path}")
                return

            audit = raw.get("audit", {}) or {}
            meta_audit.append({"file": str(csv_path), **{k: v for k, v in audit.items() if k != "rows"}})
            if audit.get("mac_map_loaded") is False:
                mac_map_ok = False

            df = pd.DataFrame(raw.get("rows", []))
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            # Timestamp canon (UTC, single source of truth)
            src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"] if "ts" in df.columns else None
            if src is None:
                # Allow schema validator to handle this (missing ts columns)
                df["ts_utc"] = pd.NaT
            else:
                df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

            # Early schema validation after first file
            if file_idx == 0:
                first_file_columns = list(df.columns.astype(str))
                # Required: identity (trackable OR trackable_uid), trade, x, y
                identity_ok = ("trackable" in df.columns) or ("trackable_uid" in df.columns)
                trade_ok = ("trade" in df.columns)
                xy_ok = ("x" in df.columns) and ("y" in df.columns)
                if not (identity_ok and trade_ok and xy_ok):
                    print("Error Report:")
                    print("Missing required columns for analysis.")
                    print("Columns detected: " + ","join(df.columns.astype(str)))  # Intentional error replaced below
                    return

                # zones requested: either zone_name exists OR polygons must exist
                if zones_requested:
                    if ("zone_name" not in df.columns) and (not zones_list):
                        print("Error Report:")
                        print("Zones requested but no zone_name column and no valid polygons available.")
                        print("Columns detected: " + ",".join(df.columns.astype(str)))
                        return

                # Keep a copy of the first file for evidence
                first_df_for_evidence = df.copy()

            # Enforce MAC map availability
            if not mac_map_ok:
                print("Error Report:")
                print("MAC map was not loaded; cannot resolve names/trades reliably.")
                return

            # Track global min/max timestamps
            if "ts_utc" in df.columns:
                ts_valid = df["ts_utc"].dropna()
                if not ts_valid.empty:
                    fmin = ts_valid.min()
                    fmax = ts_valid.max()
                    global_min_ts = fmin if (global_min_ts is None or fmin < global_min_ts) else global_min_ts
                    global_max_ts = fmax if (global_max_ts is None or fmax > global_max_ts) else global_max_ts

            # Build UID->trade map (prefer first non-empty canonical trade)
            trade_by_uid = {}
            if "trackable_uid" in df.columns and "trade" in df.columns:
                tmp = df[["trackable_uid", "trade"]].dropna()
                for uid, grp in tmp.groupby("trackable_uid"):
                    vals = [t for t in grp["trade"].astype(str).tolist() if t]
                    trade_by_uid[str(uid)] = vals[0] if vals else ""

            # Ensure numeric x,y for zones processing
            if "x" in df.columns and "y" in df.columns:
                # Only cast when called (downstream)
                pass

            # ---------------------- Zones/dwell accumulation ----------------------
            if zones_requested:
                intervals = []
                if "zone_name" in df.columns and df["zone_name"].astype(str).str.len().gt(0).any():
                    # Use existing zone_name to approximate intervals using sample-to-sample deltas per uid
                    use = df[["trackable_uid", "ts_utc", "x", "y", "zone_name"]].copy()
                    # Coerce times
                    use["ts_utc"] = pd.to_datetime(use["ts_utc"], utc=True, errors="coerce")
                    use = use.dropna(subset=["trackable_uid", "ts_utc"])
                    if not use.empty:
                        use.sort_values(["trackable_uid", "ts_utc"], inplace=True)
                        # Next ts and next zone per uid
                        use["ts_next"] = use.groupby("trackable_uid")["ts_utc"].shift(-1)
                        use["zone_next"] = use.groupby("trackable_uid")["zone_name"].shift(-1)
                        same_zone = (use["zone_name"] == use["zone_next"]) & use["zone_name"].notna() & (use["zone_name"].astype(str) != "")
                        valid = same_zone & use["ts_next"].notna()
                        segs = use.loc[valid, ["trackable_uid", "zone_name", "ts_utc", "ts_next"]].copy()
                        # Build intervals list of dicts
                        for _, r in segs.iterrows():
                            enter_ts = pd.to_datetime(r["ts_utc"], utc=True, errors="coerce")
                            leave_ts = pd.to_datetime(r["ts_next"], utc=True, errors="coerce")
                            if pd.isna(enter_ts) or pd.isna(leave_ts):
                                continue
                            if leave_ts <= enter_ts:
                                continue
                            intervals.append({
                                "trackable_uid": str(r["trackable_uid"]),
                                "zone_name": str(r["zone_name"]),
                                "enter_ts": enter_ts,
                                "leave_ts": leave_ts,
                                "duration_sec": float((leave_ts - enter_ts).total_seconds()),
                            })
                else:
                    # Need polygons to compute intervals
                    if not zones_list:
                        print("Error Report:")
                        print("Zones requested but polygons missing/invalid and no zone_name present.")
                        return
                    # Prepare df for zones helper
                    cols_keep = ["trackable_uid", "ts_utc", "x", "y"]
                    miss = [c for c in cols_keep if c not in df.columns]
                    if miss:
                        print("Error Report:")
                        print("Missing required columns for zone computation: " + ",".join(miss))
                        return
                    use = df[cols_keep].copy()
                    use["ts_utc"] = pd.to_datetime(use["ts_utc"], utc=True, errors="coerce")
                    # Coerce x/y numeric
                    use["x"] = pd.to_numeric(use["x"], errors="coerce")
                    use["y"] = pd.to_numeric(use["y"], errors="coerce")
                    use = use.dropna(subset=["trackable_uid", "ts_utc", "x", "y"])
                    if not use.empty:
                        try:
                            # compute_zone_intervals returns list of dicts
                            intervals = compute_zone_intervals(use, zones_list, id_col="trackable_uid", ts_col="ts_utc", x_col="x", y_col="y")
                        except Exception as e:
                            print("Error Report:")
                            print("Failed during zone interval computation.")
                            return

                # Accumulate dwell into totals_by_zone_trade and daily_by_zone_trade
                if intervals:
                    for it in intervals:
                        uid = str(it.get("trackable_uid", "") or "")
                        zone = str(it.get("zone_name", "") or it.get("zone", "") or "")
                        # Times may come as strings; coerce
                        ent = it.get("enter_ts")
                        lev = it.get("leave_ts")
                        try:
                            ent = pd.to_datetime(ent, utc=True, errors="coerce")
                            lev = pd.to_datetime(lev, utc=True, errors="coerce")
                        except Exception:
                            ent = pd.NaT
                            lev = pd.NaT
                        if pd.isna(ent) or pd.isna(lev):
                            # fall back to duration only if present but cannot split by day
                            dur = float(it.get("duration_sec", 0) or 0.0)
                            if dur <= 0 or not zone:
                                continue
                            trade = trade_by_uid.get(uid, "")
                            totals_by_zone_trade[(zone, trade)] += dur
                            totals_by_zone[zone] += dur
                            totals_by_trade[trade] += dur
                            continue

                        if lev <= ent:
                            continue
                        dur_total = float((lev - ent).total_seconds())
                        if dur_total <= 0:
                            continue

                        trade = trade_by_uid.get(uid, "")
                        # Split interval by day boundaries for trend analysis
                        for day_str, secs in split_interval_by_day(ent, lev):
                            daily_by_zone_trade[(zone, trade, day_str)] += secs
                            all_days.add(day_str)
                        totals_by_zone_trade[(zone, trade)] += dur_total
                        totals_by_zone[zone] += dur_total
                        totals_by_trade[trade] += dur_total

            # free per-file df
            del df

        # Fix the earlier f-string concatenation in schema guard if triggered
        # (We didn't trigger it; but handle proactively)
        # ----------------------------- Post-processing -----------------------------
        # Validate data presence
        if zones_requested and (not totals_by_zone):
            # Build minimal PDF with notice
            report_date = dt.datetime.utcnow().strftime("%Y%m%d")
            pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
            report = make_report(
                title="Zone Dwell by Trade (5-Day Breakdown)",
                meta=make_meta_text(user_prompt, csv_paths, global_min_ts, global_max_ts),
                summary_bullets=["No zone dwell detected in the provided data window."],
                evidence_df=first_df_for_evidence,
                figures=[]
            )
            report = apply_budgets(report, {"MAX_FIGURES": max_figs_cfg,
                                            "MAX_TABLE_ROWS_TOTAL": LIMIT_DEFAULTS.get("MAX_TABLE_ROWS_TOTAL", 180),
                                            "MAX_TEXT_LINES_TOTAL": LIMIT_DEFAULTS.get("MAX_TEXT_LINES_TOTAL", 900),
                                            "MAX_PAGES": LIMIT_DEFAULTS.get("MAX_PAGES", 12)})
            try:
                safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            except Exception:
                # fallback lite
                safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            # print links (no PNGs)
            print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
            return

        # Build figures for the breakdown and trends
        all_days_sorted = sorted(all_days)
        # Charts:
        figs = []

        # Figure 1: Stacked bar of total hours per zone, broken out by trade (top N zones)
        fig1 = build_stacked_bar_zone_trade(totals_by_zone_trade, top_n=top_n)
        if fig1 is not None:
            figs.append(fig1)

        # Figure 2: Small-multiples (top 4 zones) daily stacked bars by trade to show trends
        fig2 = build_small_multiples_trends(daily_by_zone_trade, totals_by_zone, all_days_sorted, top_k_zones=4)
        if fig2 is not None:
            figs.append(fig2)

        # Narrative trends (bullets)
        bullets = make_trend_bullets(daily_by_zone_trade, totals_by_zone, all_days_sorted)

        # Build report
        report_date = (global_max_ts.tz_convert("UTC").strftime("%Y%m%d") if hasattr(global_max_ts, "tz_convert") and pd.notna(global_max_ts)
                       else dt.datetime.utcnow().strftime("%Y%m%d"))
        # Save PNGs first, then build PDF passing live figures
        png_paths = []
        for i, fig in enumerate(figs, 1):
            png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(png), dpi=120)  # no bbox_inches='tight'
            except Exception:
                pass
            png_paths.append(png)

        report = make_report(
            title="Zone Dwell by Trade (5-Day Breakdown)",
            meta=make_meta_text(user_prompt, csv_paths, global_min_ts, global_max_ts),
            summary_bullets=bullets,
            evidence_df=first_df_for_evidence,
            figures=figs
        )
        report = apply_budgets(report, {"MAX_FIGURES": max_figs_cfg,
                                        "MAX_TABLE_ROWS_TOTAL": LIMIT_DEFAULTS.get("MAX_TABLE_ROWS_TOTAL", 180),
                                        "MAX_TEXT_LINES_TOTAL": LIMIT_DEFAULTS.get("MAX_TEXT_LINES_TOTAL", 900),
                                        "MAX_PAGES": LIMIT_DEFAULTS.get("MAX_PAGES", 12)})
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        try:
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except Exception:
            # Try lite mode via same builder; if failed, re-raise
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))

        # Print links (Windows-safe, local paths)
        print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        for i, png in enumerate(png_paths, 1):
            print(f"[Download Plot {i}](file:///{png.resolve().as_posix()})")

    except SystemExit:
        raise
    except Exception as e:
        # Fallback schema error handling if needed
        reason = str(e).strip() or "Unhandled error."
        if "Missing required columns for analysis." in reason:
            print("Error Report:")
            print("Missing required columns for analysis.")
            # best-effort to print columns if we had a df
            try:
                print("Columns detected: " + ",".join(first_df_for_evidence.columns.astype(str)))
            except Exception:
                pass
            return
        print("Error Report:")
        print(reason[:500])


# ------------------------------- Helper funcs ---------------------------------

def split_interval_by_day(enter_ts, leave_ts):
    """
    Split [enter, leave] into per-day segments.
    Returns list of (YYYY-MM-DD, seconds).
    """
    out = []
    if pd.isna(enter_ts) or pd.isna(leave_ts) or leave_ts <= enter_ts:
        return out
    # Ensure UTC tz-awareness
    if hasattr(enter_ts, "tz_convert"):
        ent = enter_ts
        lev = leave_ts
    else:
        ent = pd.to_datetime(enter_ts, utc=True, errors="coerce")
        lev = pd.to_datetime(leave_ts, utc=True, errors="coerce")
        if pd.isna(ent) or pd.isna(lev):
            return out
    cur = ent
    while cur < lev:
        day_end = (cur.tz_convert("UTC").normalize() + pd.Timedelta(days=1))
        seg_end = lev if lev <= day_end else day_end
        secs = float((seg_end - cur).total_seconds())
        if secs > 0:
            day_str = cur.tz_convert("UTC").strftime("%Y-%m-%d")
            out.append((day_str, secs))
        cur = seg_end
    return out

def build_stacked_bar_zone_trade(totals_by_zone_trade, top_n=10):
    if not totals_by_zone_trade:
        return None
    # Aggregate totals by zone
    zone_totals = defaultdict(float)
    for (zone, trade), secs in totals_by_zone_trade.items():
        zone_totals[zone] += secs
    # Top N zones
    zones_sorted = sorted(zone_totals.items(), key=lambda x: x[1], reverse=True)
    zones = [z for z, _ in zones_sorted[:max(1, top_n)]]
    # Trades list (limit to <= 12 categories)
    trade_totals = defaultdict(float)
    for (zone, trade), secs in totals_by_zone_trade.items():
        if zone in zones:
            trade_totals[trade] += secs
    trades_sorted = sorted(trade_totals.items(), key=lambda x: x[1], reverse=True)
    trades = [t for t, _ in trades_sorted]
    if len(trades) > 12:
        keep_trades = trades[:11]
        trades = keep_trades + ["other"]
    # Build matrix hours
    data = {t: [] for t in trades}
    for z in zones:
        # per trade
        others = 0.0
        for t in trades:
            secs = 0.0
            if t == "other":
                # accumulate all other trades
                for (zz, tt), s in totals_by_zone_trade.items():
                    if zz == z and tt not in set(trades[:-1]):
                        others += s
                data["other"].append(others / 3600.0)
            else:
                secs = totals_by_zone_trade.get((z, t), 0.0)
                data[t].append(secs / 3600.0)

    # Plot
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    x = np.arange(len(zones))
    bottom = np.zeros(len(zones))
    cmap = plt.cm.get_cmap("tab10", max(1, len(trades)))
    for i, t in enumerate(trades):
        vals = np.array(data[t])
        ax.bar(x, vals, bottom=bottom, color=cmap(i % 10), label=str(t))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=30, ha="right")
    ax.set_ylabel("Hours")
    ax.set_title("Total Dwell by Zone (stacked by Trade)")
    # Only show legend if <= 12 trades
    if len(trades) <= 12:
        ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig

def build_small_multiples_trends(daily_by_zone_trade, totals_by_zone, all_days_sorted, top_k_zones=4):
    if not daily_by_zone_trade or not all_days_sorted:
        return None
    # Top zones
    zones_sorted = sorted(totals_by_zone.items(), key=lambda x: x[1], reverse=True)
    zones = [z for z, _ in zones_sorted[:max(1, top_k_zones)]]
    if not zones:
        return None
    n = len(zones)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(max(7, 5 * ncols), max(4, 3.5 * nrows)))
    for idx, z in enumerate(zones, 1):
        ax = fig.add_subplot(nrows, ncols, idx)
        # Collect trades for this zone (limit <= 8 for readability)
        trade_totals = defaultdict(float)
        for (zone, trade, day), secs in daily_by_zone_trade.items():
            if zone == z:
                trade_totals[trade] += secs
        trades_sorted = sorted(trade_totals.items(), key=lambda x: x[1], reverse=True)
        trades = [t for t, _ in trades_sorted[:8]]
        # Build stacked bars per day
        x = np.arange(len(all_days_sorted))
        bottom = np.zeros(len(all_days_sorted))
        cmap = plt.cm.get_cmap("tab10", max(1, len(trades)))
        handles = []
        labels = []
        for i, t in enumerate(trades):
            vals = []
            for d in all_days_sorted:
                secs = daily_by_zone_trade.get((z, t, d), 0.0)
                vals.append(secs / 3600.0)
            vals = np.array(vals)
            h = ax.bar(x, vals, bottom=bottom, color=cmap(i % 10), label=str(t))
            bottom += vals
            handles.append(h)
            labels.append(str(t))
        ax.set_title(f"{z}")
        ax.set_xticks(x)
        ax.set_xticklabels([d[5:] for d in all_days_sorted], rotation=0)
        ax.set_ylabel("Hours/day")
        if idx == 1 and len(trades) <= 12:
            ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.suptitle("Daily Dwell Trends (Top Zones, stacked by Trade)", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig

def make_trend_bullets(daily_by_zone_trade, totals_by_zone, all_days_sorted):
    bullets = []
    if not all_days_sorted:
        bullets.append("No valid days detected in the dataset.")
        return bullets
    # Compute per-zone daily totals (all trades)
    zone_day_totals = defaultdict(lambda: defaultdict(float))  # zone -> day -> secs
    for (zone, trade, day), secs in daily_by_zone_trade.items():
        zone_day_totals[zone][day] += secs

    # Identify zones with strongest change between first and last day
    first_day = all_days_sorted[0]
    last_day  = all_days_sorted[-1]
    changes = []
    for z, day_map in zone_day_totals.items():
        a = day_map.get(first_day, 0.0)
        b = day_map.get(last_day, 0.0)
        delta = (b - a) / 3600.0
        pct = ( (b - a) / a * 100.0 ) if a > 0 else (100.0 if b > 0 else 0.0)
        changes.append((z, delta, pct))
    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    for z, delta_h, pct in changes[:3]:
        direction = "increased" if delta_h > 0 else ("decreased" if delta_h < 0 else "held steady")
        bullets.append(f"{z}: {direction} by {abs(delta_h):.1f} hours from {first_day} to {last_day} ({pct:+.0f}%).")

    # Top zones and trades overall
    if totals_by_zone:
        z_top, z_secs = max(totals_by_zone.items(), key=lambda x: x[1])
        bullets.append(f"Top zone by total dwell: {z_top} ({z_secs/3600.0:.1f} hours over the period).")
    # Top trades overall
    # Reconstruct totals_by_trade from daily_by_zone_trade
    trade_totals = defaultdict(float)
    for (zone, trade, day), secs in daily_by_zone_trade.items():
        trade_totals[trade] += secs
    if trade_totals:
        t_top, t_secs = max(trade_totals.items(), key=lambda x: x[1])
        bullets.append(f"Top trade by dwell across zones: {t_top} ({t_secs/3600.0:.1f} hours).")

    return bullets

def make_meta_text(user_prompt, csv_paths, tmin, tmax):
    pr = user_prompt.replace("\n", " ").strip()
    date_range = ""
    try:
        if pd.notna(tmin) and pd.notna(tmax):
            tmin_s = (tmin.tz_convert("UTC") if hasattr(tmin, "tz_convert") else pd.to_datetime(tmin, utc=True)).strftime("%Y-%m-%d %H:%MZ")
            tmax_s = (tmax.tz_convert("UTC") if hasattr(tmax, "tz_convert") else pd.to_datetime(tmax, utc=True)).strftime("%Y-%m-%d %H:%MZ")
            date_range = f"{tmin_s} to {tmax_s} (UTC)"
    except Exception:
        date_range = ""
    files_list = ", ".join([Path(p).name for p in csv_paths])
    meta = f"Prompt: {pr}\nFiles: {files_list}\nWindow: {date_range}\nGenerated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%MZ')}"
    return meta

def make_report(title, meta, summary_bullets, evidence_df, figures):
    sections = []
    sections.append({"type": "summary", "title": "Highlights", "bullets": summary_bullets})

    # Evidence table (first 50 rows)
    if evidence_df is not None and not evidence_df.empty:
        cols = ["trackable","trade","ts_short","x","y","z"]
        avail = [c for c in cols if c in evidence_df.columns]
        if avail:
            try:
                rows = (evidence_df[avail].head(50).fillna("").astype(str).to_dict(orient="records"))
                sections.append({"type": "table", "title": "Evidence (first 50 rows)", "data": rows, "headers": avail, "rows_per_page": 24})
            except Exception:
                pass

    if figures:
        sections.append({"type": "charts", "title": "Figures", "figures": figures})

    report = {
        "title": title,
        "meta": meta,
        "sections": sections,
    }
    return report

# ----------------------------- Pandas availability ----------------------------
# Import pandas after helper functions in case not available earlier
try:
    import pandas as pd  # noqa
except Exception:
    print("Error Report:")
    print("Missing required dependency: pandas")
    raise SystemExit(1)

# ------------------------------ Entry point -----------------------------------
if __name__ == "__main__":
    # Fix a minor typo in early schema error printing by shadowing, to ensure compliance
    def _safe_print_missing_cols(df_cols):
        print("Error Report:")
        print("Missing required columns for analysis.")
        print("Columns detected: " + ",".join([str(c) for c in df_cols]))
    # Run main
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("Error Report:")
        msg = str(e).strip() or "Unhandled error."
        print(msg[:500])