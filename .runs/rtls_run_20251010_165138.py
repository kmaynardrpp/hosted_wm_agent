#!/usr/bin/env python3
# InfoZoneBuilder — Zone dwell by trade over 5 days, charts->PNGs->PDF (local-only)

import sys, os, json, math, traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
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

# ------------------------------ helper imports --------------------------------
try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Missing local helper: extractor.py not found or failed to import.")
    raise SystemExit(1)

try:
    from zones_process import load_zones, compute_zone_intervals
except Exception:
    print("Error Report:")
    print("Missing local helper: zones_process.py not found or failed to import.")
    raise SystemExit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("Missing local helper: pdf_creation_script.py not found or failed to import.")
    raise SystemExit(1)

try:
    from report_limits import apply_budgets, make_lite
except Exception:
    print("Error Report:")
    print("Missing local helper: report_limits.py not found or failed to import.")
    raise SystemExit(1)

# ------------------------------ utility funcs ---------------------------------
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def safe_to_num(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_ts_utc(obj) -> pd.Timestamp:
    return pd.to_datetime(obj, utc=True, errors="coerce")

def most_frequent_nonempty(series: pd.Series) -> str:
    s = series.astype(str).replace("", np.nan).dropna()
    if s.empty:
        return ""
    return s.mode().iloc[0] if not s.mode().empty else s.iloc[-1]

def compute_dwell_from_zone_name(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Approximate dwell durations using existing 'zone_name' labels.
    For each trackable_uid, sum time differences between consecutive samples where the zone_name stays the same.
    Returns a list of intervals-like dicts with: trackable_uid, trackable, zone_name, enter_ts, leave_ts, duration_sec
    """
    out: List[Dict[str, Any]] = []
    if df.empty or "zone_name" not in df.columns:
        return out
    work = df.loc[:, ["trackable_uid", "trackable", "zone_name", "ts_utc"]].copy()
    work = work.dropna(subset=["trackable_uid", "zone_name", "ts_utc"])
    if work.empty:
        return out
    work = work.sort_values(["trackable_uid", "ts_utc"])
    # Compute transitions
    for uid, g in work.groupby("trackable_uid", sort=False):
        g = g.copy()
        g["next_ts"] = g["ts_utc"].shift(-1)
        g["next_zone"] = g["zone_name"].shift(-1)
        g = g[g["next_ts"].notna()]
        if g.empty:
            continue
        # Keep only spans where zone remains the same across consecutive samples
        same = g["zone_name"] == g["next_zone"]
        gg = g[same]
        if gg.empty:
            continue
        dur = (gg["next_ts"] - gg["ts_utc"]).dt.total_seconds()
        # discard negative or absurdly long intervals? Keep as-is per spec (no downsampling)
        ok = dur > 0
        gg = gg[ok]
        if gg.empty:
            continue
        for _, r in gg.iterrows():
            out.append({
                "trackable_uid": uid,
                "trackable": r.get("trackable", ""),
                "zone_name": r.get("zone_name", ""),
                "enter_ts": r.get("ts_utc"),
                "leave_ts": r.get("next_ts"),
                "duration_sec": float(r.get("next_ts").to_datetime64() - r.get("ts_utc").to_datetime64()) / 1e9 if isinstance(r.get("next_ts"), pd.Timestamp) else float(dur),
            })
    return out

def add_or_increment(d: Dict[Tuple[str, str], float], key: Tuple[str, str], val: float):
    d[key] = d.get(key, 0.0) + float(val if pd.notna(val) else 0.0)

def add_or_increment_1d(d: Dict[str, float], key: str, val: float):
    d[key] = d.get(key, 0.0) + float(val if pd.notna(val) else 0.0)

def pick_top_categories(df: pd.DataFrame, cat_col: str, val_col: str, top_n: int) -> List[str]:
    if df.empty or cat_col not in df.columns or val_col not in df.columns:
        return []
    agg = df.groupby(cat_col, as_index=False)[val_col].sum().sort_values(val_col, ascending=False)
    return agg[cat_col].head(max(1, int(top_n))).astype(str).tolist()

# ----------------------------------- main -------------------------------------
def main():
    try:
        # CLI parse
        if len(sys.argv) < 3:
            print("Error Report:")
            print("Expected: python script.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
            raise SystemExit(1)
        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:] if p and p.strip()]
        if not csv_paths:
            print("Error Report:")
            print("No CSV files provided.")
            raise SystemExit(1)
        out_dir = csv_paths[0].resolve().parent

        # Load config if present
        cfg = {}
        try:
            if CONFIG.exists():
                cfg = json.loads(CONFIG.read_text(encoding="utf-8", errors="ignore") or "{}")
        except Exception:
            cfg = {}
        top_n_trades = int(cfg.get("top_n", 10))
        max_figures = int(cfg.get("max_figures", 6))

        # Zones requested by the user? Yes (intent: "zone" in prompt)
        zones_intent = True

        # Load zones polygons (local-only) ahead of schema validation if needed
        zones_list = []
        if zones_intent:
            try:
                zones_list = load_zones(str(ZONES_JSON), only_active=True)
            except Exception:
                zones_list = []

        # Aggregators
        dwell_by_zone_trade_sec: Dict[Tuple[str, str], float] = {}
        per_day_trade_sec: Dict[Tuple[str, str], float] = {}      # (day_str, trade) -> seconds
        per_zone_total_sec: Dict[str, float] = {}                 # zone_name -> seconds
        trade_overall_sec: Dict[str, float] = {}
        ts_min: Optional[pd.Timestamp] = None
        ts_max: Optional[pd.Timestamp] = None

        # Evidence rows reservoir (from first non-empty file)
        first_df_for_evidence: Optional[pd.DataFrame] = None

        # Process files one-by-one (large-data mode)
        mac_map_checked = False
        have_zone_name_any = False

        for csv_path in csv_paths:
            if not csv_path.exists():
                continue
            # Ingest via extractor with local mac_map path
            raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
            audit = raw.get("audit", {}) if isinstance(raw, dict) else {}
            if not mac_map_checked:
                mac_map_checked = True
                if not audit or not bool(audit.get("mac_map_loaded", False)):
                    print("Error Report:")
                    print("MAC→name map could not be loaded; trackable names/trades unavailable.")
                    raise SystemExit(1)

            df = pd.DataFrame(raw.get("rows", []))
            # Duplicate-name guard
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            # Timestamp canon
            src = df["ts_iso"] if "ts_iso" in df.columns else (df["ts"] if "ts" in df.columns else None)
            if src is None:
                df["ts_utc"] = pd.NaT
            else:
                df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

            # Early schema validation after first file read
            if first_df_for_evidence is None:
                cols = df.columns.astype(str).tolist()
                identity_ok = ("trackable" in df.columns) or ("trackable_uid" in df.columns)
                trade_ok = "trade" in df.columns
                pos_ok = ("x" in df.columns) and ("y" in df.columns)
                if not (identity_ok and trade_ok and pos_ok):
                    print("Error Report:")
                    print("Missing required columns for analysis.")
                    print(f"Columns detected: {','.join(df.columns.astype(str))}")
                    raise SystemExit(1)
                # Zones requirement
                if zones_intent:
                    have_zone_name_any = "zone_name" in df.columns
                    if not have_zone_name_any and not zones_list:
                        print("Error Report:")
                        print("Zones requested but no polygons available and 'zone_name' column missing.")
                        print(f"Columns detected: {','.join(df.columns.astype(str))}")
                        raise SystemExit(1)

            # Track evidence from first non-empty df
            if first_df_for_evidence is None and not df.empty:
                first_df_for_evidence = df.copy()

            # Update global ts_min/max
            if "ts_utc" in df.columns and not df["ts_utc"].isna().all():
                fmin = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").min()
                fmax = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").max()
                if pd.notna(fmin):
                    ts_min = fmin if ts_min is None or fmin < ts_min else ts_min
                if pd.notna(fmax):
                    ts_max = fmax if ts_max is None or fmax > ts_max else ts_max

            # Build per-file trade_by_uid map (most frequent non-empty)
            trade_by_uid: Dict[str, str] = {}
            if "trackable_uid" in df.columns and "trade" in df.columns:
                tmp = df.loc[:, ["trackable_uid", "trade"]].copy()
                tmp = tmp[tmp["trackable_uid"].astype(str).str.len() > 0]
                if not tmp.empty:
                    for uid, g in tmp.groupby("trackable_uid", sort=False):
                        trade_by_uid[uid] = most_frequent_nonempty(g["trade"])

            # Zones logic
            intervals: List[Dict[str, Any]] = []
            used_zone_name_path = False
            if "zone_name" in df.columns:
                # Use existing zone_name to approximate dwell (no recomputation)
                used_zone_name_path = True
                # Ensure ts_utc is valid
                df = df.dropna(subset=["ts_utc"])
                if not df.empty:
                    # sort and compute dwell via consecutive timestamps per uid
                    intervals = compute_dwell_from_zone_name(df)
            else:
                # Compute zones via polygons (no downsampling)
                # Prepare numeric x,y and valid ts_utc
                cols_keep = ["trackable_uid", "trackable", "trade", "ts_utc", "x", "y"]
                use = df.loc[:, [c for c in cols_keep if c in df.columns]].copy()
                if not use.empty:
                    use["x"] = pd.to_numeric(use["x"], errors="coerce")
                    use["y"] = pd.to_numeric(use["y"], errors="coerce")
                    use = use.dropna(subset=["trackable_uid", "ts_utc", "x", "y"])
                if not use.empty and zones_list:
                    try:
                        intervals = compute_zone_intervals(use, zones_list, id_col="trackable_uid", ts_col="ts_utc", x_col="x", y_col="y")
                    except Exception:
                        intervals = []

            # Aggregate durations across intervals
            for it in intervals:
                uid = str(it.get("trackable_uid", "") or "")
                zone_name = it.get("zone_name") or it.get("zone") or it.get("name") or ""
                zone_name = str(zone_name or "").strip()
                dur = it.get("duration_sec")
                # Parse duration
                try:
                    dsec = float(dur)
                except Exception:
                    dsec = 0.0
                if not zone_name or dsec <= 0:
                    continue
                # Trade from uid map (fallback to interval trackable/trade if present)
                trade = trade_by_uid.get(uid, "")
                if not trade:
                    trade = str(it.get("trade", "") or "")
                trade = str(trade or "").strip() or "unknown"

                # Aggregate totals by (zone, trade)
                add_or_increment(dwell_by_zone_trade_sec, (zone_name, trade), dsec)
                add_or_increment_1d(per_zone_total_sec, zone_name, dsec)
                add_or_increment_1d(trade_overall_sec, trade, dsec)

                # Per-day trend (by trade across all zones) — use enter_ts day if present else leave_ts
                ts_enter = it.get("enter_ts")
                ts_leave = it.get("leave_ts")
                ts_for_day = parse_ts_utc(ts_enter) if ts_enter is not None else parse_ts_utc(ts_leave)
                if pd.isna(ts_for_day):
                    continue
                day_str = ts_for_day.tz_convert("UTC").strftime("%Y-%m-%d")
                add_or_increment(per_day_trade_sec, (day_str, trade), dsec)

            # Clean up per-file
            del df
            plt.close("all")

        # If after processing nothing was aggregated, build a minimal report
        if not dwell_by_zone_trade_sec:
            # Still build an empty but valid PDF
            report_date = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
            pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
            title = "Zone Dwell by Trade (No Data)"
            meta = f"Query: {user_prompt}"
            report = {
                "title": title,
                "meta": meta,
                "sections": [
                    {"type":"summary","title":"Summary","bullets":[
                        "No zone dwell intervals could be derived from the provided inputs."
                    ]},
                ],
            }
            try:
                report = apply_budgets(report)
                safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
                print(f"[Download the PDF]({file_uri(pdf_path)})")
            except Exception:
                print("Error Report:")
                print("Failed to write PDF for empty dataset.")
            raise SystemExit(0)

        # ---------------------- Build aggregated DataFrames ----------------------
        rows = []
        for (zone, trade), sec in dwell_by_zone_trade_sec.items():
            rows.append({"zone_name": zone, "trade": trade, "hours": float(sec)/3600.0})
        df_zone_trade = pd.DataFrame(rows)
        if df_zone_trade.empty:
            df_zone_trade = pd.DataFrame(columns=["zone_name","trade","hours"])

        # Determine top trades overall to display (collapse the rest as 'other')
        trade_totals = pd.DataFrame([
            {"trade": t, "hours": float(v)/3600.0} for t, v in trade_overall_sec.items()
        ])
        top_trades = pick_top_categories(trade_totals, "trade", "hours", min(top_n_trades, 8))
        if not top_trades:
            top_trades = trade_totals.sort_values("hours", ascending=False)["trade"].head(5).astype(str).tolist()

        # Collapse other trades
        df_plot = df_zone_trade.copy()
        df_plot["trade_plot"] = np.where(df_plot["trade"].isin(top_trades), df_plot["trade"], "other")
        df_plot = (df_plot.groupby(["zone_name","trade_plot"], as_index=False)["hours"].sum())

        # Per-day trend by trade (top trades only)
        row_trend = []
        for (day_str, trade), sec in per_day_trade_sec.items():
            if trade in top_trades:
                row_trend.append({"day": day_str, "trade": trade, "hours": float(sec)/3600.0})
        df_trend = pd.DataFrame(row_trend)
        if not df_trend.empty:
            df_trend = df_trend.sort_values("day")

        # Top zones listing
        zone_tot_rows = [{"zone_name": z, "hours": float(s)/3600.0} for z, s in per_zone_total_sec.items()]
        df_zone_tot = pd.DataFrame(zone_tot_rows).sort_values("hours", ascending=False) if zone_tot_rows else pd.DataFrame(columns=["zone_name","hours"])

        # ------------------------------ Build figures ----------------------------
        figs: List[plt.Figure] = []
        png_paths: List[Path] = []

        # Figure 1: Stacked bar — hours per zone, split by trade (top trades + other)
        if not df_plot.empty:
            # Limit zones to top N by total hours for readability (e.g., top 12)
            max_zones = 12
            top_z = df_zone_tot["zone_name"].head(max_zones).astype(str).tolist() if not df_zone_tot.empty else df_plot["zone_name"].unique().tolist()
            use = df_plot[df_plot["zone_name"].isin(top_z)].copy()
            piv = use.pivot_table(index="zone_name", columns="trade_plot", values="hours", aggfunc="sum", fill_value=0.0)
            piv = piv.loc[top_z] if len(top_z) and set(top_z).issubset(piv.index) else piv
            fig1 = plt.figure(figsize=(10, 6))
            ax1 = fig1.add_subplot(111)
            bottom = np.zeros(len(piv), dtype=float)
            labels = list(piv.columns.astype(str))
            colors = plt.cm.get_cmap("tab10", max(3, len(labels)))
            for i, col in enumerate(labels):
                vals = piv[col].values
                ax1.bar(np.arange(len(piv)), vals, bottom=bottom, color=colors(i), label=str(col))
                bottom += vals
            ax1.set_xticks(np.arange(len(piv)))
            ax1.set_xticklabels([str(z) for z in piv.index], rotation=35, ha="right")
            ax1.set_ylabel("Hours")
            ax1.set_title("Zone dwell (hours) by trade — top zones")
            handles, llabels = ax1.get_legend_handles_labels()
            if len(llabels) <= 12 and len(llabels) > 0:
                ax1.legend(loc="upper right", frameon=True, fontsize=8)
            fig1.tight_layout()
            figs.append(fig1)

        # Figure 2: Line chart — per-day hours by trade (all zones), top trades
        if not df_trend.empty and df_trend["day"].nunique() >= 2:
            # Pivot day x trade
            days_sorted = sorted(df_trend["day"].unique())
            piv2 = df_trend.pivot_table(index="day", columns="trade", values="hours", aggfunc="sum", fill_value=0.0)
            piv2 = piv2.loc[days_sorted]
            fig2 = plt.figure(figsize=(9, 5))
            ax2 = fig2.add_subplot(111)
            colors2 = plt.cm.get_cmap("tab10", max(3, len(piv2.columns)))
            for i, col in enumerate(piv2.columns):
                ax2.plot(piv2.index, piv2[col].values, marker="o", color=colors2(i), label=str(col))
            ax2.set_xlabel("Day (UTC)")
            ax2.set_ylabel("Hours")
            ax2.set_title("Daily trend — hours by trade (all zones)")
            ax2.grid(True, alpha=0.3)
            handles, llabels = ax2.get_legend_handles_labels()
            if len(llabels) <= 12 and len(llabels) > 0:
                ax2.legend(loc="upper left", frameon=True, fontsize=8)
            fig2.tight_layout()
            figs.append(fig2)

        # ---------------------------- Narrative & tables --------------------------
        # Summary bullets (trends)
        bullets: List[str] = []
        # Time window
        if ts_min is not None and ts_max is not None and pd.notna(ts_min) and pd.notna(ts_max):
            bullets.append(f"Window (UTC): {ts_min.strftime('%Y-%m-%d %H:%M')} → {ts_max.strftime('%Y-%m-%d %H:%M')}")
        # Top zones
        if not df_zone_tot.empty:
            topz = df_zone_tot.head(3)
            ztxt = "; ".join([f"{r['zone_name']} ({r['hours']:.1f} h)" for _, r in topz.iterrows()])
            bullets.append(f"Top zones by dwell: {ztxt}")
        # Top trades
        if not trade_totals.empty:
            topt = trade_totals.sort_values("hours", ascending=False).head(3)
            ttxt = "; ".join([f"{r['trade']} ({r['hours']:.1f} h)" for _, r in topt.iterrows()])
            bullets.append(f"Top trades by dwell: {ttxt}")
        # Trend highlight (increase/decrease from first to last day)
        if not df_trend.empty and df_trend["day"].nunique() >= 2:
            days_sorted = sorted(df_trend["day"].unique())
            trend_msg = []
            for tr in df_trend["trade"].unique():
                s = df_trend[df_trend["trade"] == tr].set_index("day").reindex(days_sorted)["hours"].fillna(0.0)
                if len(s) >= 2:
                    delta = s.iloc[-1] - s.iloc[0]
                    if abs(delta) >= 0.5:  # at least 0.5 hours change to mention
                        trend_msg.append(f"{tr}: {'↑' if delta>0 else '↓'} {abs(delta):.1f} h")
            if trend_msg:
                bullets.append("Trend (first→last day): " + "; ".join(trend_msg))

        # Evidence table from first df
        evidence_section: Optional[Dict[str, Any]] = None
        if first_df_for_evidence is not None and not first_df_for_evidence.empty:
            df_evi = first_df_for_evidence.copy()
            for col in ["x", "y", "z"]:
                if col in df_evi.columns:
                    # keep as-is (string); evidence table uses raw text
                    pass
            cols = ["trackable","trade","ts_short","x","y","z"]
            existing = [c for c in cols if c in df_evi.columns]
            # ensure the columns exist; fill missing
            for c in cols:
                if c not in df_evi.columns:
                    df_evi[c] = ""
            rows = df_evi[cols].head(50).fillna("").astype(str).to_dict(orient="records")
            evidence_section = {"type":"table","title":"Evidence (first file sample)","data":rows,"headers":cols,"rows_per_page":24}

        # ------------------------------ Save PNGs --------------------------------
        report_date = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
        for i, fig in enumerate(figs, 1):
            png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(png), dpi=120)
                png_paths.append(png)
            except Exception:
                # continue without PNG save
                pass

        # ------------------------------ Build PDF --------------------------------
        title = "Walmart Renovation — Zone Dwell by Trade (5-day)"
        meta_lines = [
            f"Query: {user_prompt}",
            f"Files: {len(csv_paths)}",
            f"Zones source: {'zone_name column' if have_zone_name_any else 'polygons from zones.json'}",
        ]
        if ts_min is not None and ts_max is not None and pd.notna(ts_min) and pd.notna(ts_max):
            meta_lines.append(f"UTC Range: {ts_min.strftime('%Y-%m-%d %H:%M')} to {ts_max.strftime('%Y-%m-%d %H:%M')}")
        meta = " | ".join(meta_lines)

        sections: List[Dict[str, Any]] = []
        if bullets:
            sections.append({"type":"summary","title":"Highlights","bullets":bullets})
        if figs:
            sections.append({"type":"charts","title":"Figures","figures":figs})
        if evidence_section:
            sections.append(evidence_section)

        report = {"title": title, "meta": meta, "sections": sections}

        # Apply budgets
        try:
            report = apply_budgets(report)
        except Exception:
            pass

        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"

        # Write PDF
        try:
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except (MemoryError, KeyboardInterrupt):
            # Minimal-Report Mode fallback
            lite = {
                "title": title,
                "meta": meta,
                "sections": [{"type":"summary","title":"Summary","bullets":bullets[:4] if bullets else []}],
            }
            try:
                lite = make_lite(lite)
                safe_build_pdf(lite, str(pdf_path), logo_path=str(LOGO))
            except Exception:
                print("Error Report:")
                print("PDF generation failed in fallback mode.")
                raise SystemExit(1)
        except Exception:
            # Try lite fallback
            lite = {
                "title": title,
                "meta": meta,
                "sections": [{"type":"summary","title":"Summary","bullets":bullets[:4] if bullets else []}],
            }
            try:
                lite = make_lite(lite)
                safe_build_pdf(lite, str(pdf_path), logo_path=str(LOGO))
            except Exception:
                print("Error Report:")
                print("PDF generation failed.")
                raise SystemExit(1)

        # ------------------------------ Print links ------------------------------
        print(f"[Download the PDF]({file_uri(pdf_path)})")
        for i, pth in enumerate(png_paths, 1):
            print(f"[Download Plot {i}]({file_uri(pth)})")

    except SystemExit:
        raise
    except Exception as e:
        # On failure: print only Error Report with 1–2 line reason
        msg = str(e).strip()
        if not msg:
            msg = "Unexpected error."
        print("Error Report:")
        # If schema-related, we don't have df here; handled earlier. Just print reason.
        print(msg)

if __name__ == "__main__":
    main()