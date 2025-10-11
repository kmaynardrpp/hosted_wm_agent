#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
from pathlib import Path

# project root injected by launcher; fallback to script’s folder or parent
ROOT = Path(os.environ.get("INFOZONE_ROOT", ""))
if not ROOT or not (ROOT / "guidelines.txt").exists():
    script_dir = Path(__file__).resolve().parent
    ROOT = script_dir if (script_dir / "guidelines.txt").exists() else script_dir.parent

# make "import extractor", etc. work when running from .runs/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# helper paths
GUIDELINES = ROOT / "guidelines.txt"
CONTEXT    = ROOT / "context.txt"
FLOORJSON  = ROOT / "floorplans.json"
LOGO       = ROOT / "redpoint_logo.png"
CONFIG     = ROOT / "report_config.json"
LIMITS_PY  = ROOT / "report_limits.py"
ZONES_JSON = ROOT / "zones.json"

# robust text read: Windows-safe UTF-8
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

import json
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from report_limits import apply_budgets, DEFAULTS as RL_DEFAULTS

# --------------------------- Utilities ---------------------------

def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def find_floorplan_image(root: Path) -> Path | None:
    # Only look for allowed names in local root
    for name in ("floorplan.png", "floorplan.jpg", "floorplan.jpeg"):
        cand = root / name
        if cand.exists():
            return cand
    return None

def load_config() -> dict:
    try:
        txt = read_text(CONFIG)
        return json.loads(txt) if txt else {}
    except Exception:
        return {}

def load_floorplan_extent() -> dict | None:
    if not FLOORJSON.exists():
        return None
    try:
        data = json.loads(read_text(FLOORJSON) or "{}")
        fp = (data.get("floorplans") or data.get("plans") or data or None)
        if isinstance(fp, list):
            fp = fp[0] if fp else None
        if not fp:
            return None
        width  = float(fp.get("width", 0) or 0)
        height = float(fp.get("height", 0) or 0)
        x_c    = float(fp.get("image_offset_x", 0) or 0)
        y_c    = float(fp.get("image_offset_y", 0) or 0)
        image_scale = float(fp.get("image_scale", 0) or 0)  # meters per pixel
        scale = image_scale * 100.0  # mm/px
        x_min = (x_c - width/2.0)  * scale
        x_max = (x_c + width/2.0)  * scale
        y_min = (y_c - height/2.0) * scale
        y_max = (y_c + height/2.0) * scale
        return {"extent": (x_min, x_max, y_min, y_max)}
    except Exception:
        return None

def legend_if_any(ax):
    handles, labels = ax.get_legend_handles_labels()
    if labels and len(labels) <= 12:
        ax.legend(loc="upper left", frameon=True, fontsize=8)

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def fmt_utc(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    try:
        # Ensure UTC and format to avoid double-Z
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    except Exception:
        pass
    return ts.strftime("%Y-%m-%d %H:%M UTC")

def build_summary_bullets(meta: dict) -> list[str]:
    bullets = []
    bullets.append(f"Total samples: {meta.get('total_samples', 0):,}")
    bullets.append(f"Distinct trackables: {meta.get('distinct_trackables', 0):,}")
    if meta.get("time_min") or meta.get("time_max"):
        bullets.append(f"Time span (UTC): {fmt_utc(meta.get('time_min'))} → {fmt_utc(meta.get('time_max'))}")
    if meta.get("top_trades"):
        tops = meta["top_trades"]
        top_str = ", ".join(f"{k} ({v:,})" for k, v in tops[:5])
        if top_str:
            bullets.append(f"Top trades by samples: {top_str}")
    return bullets

def cap_rows_for_table(rows: list[dict], max_rows: int) -> list[dict]:
    return rows[:max_rows] if rows else []

# --------------------------- Figures ---------------------------

def make_floorplan_overlay_figure(overlay_df: pd.DataFrame,
                                  extent: tuple[float, float, float, float],
                                  floor_img_path: Path | None,
                                  cfg: dict) -> plt.Figure | None:
    try:
        if overlay_df.empty or "x" not in overlay_df.columns or "y" not in overlay_df.columns:
            return None
        # numeric
        x = pd.to_numeric(overlay_df["x"], errors="coerce")
        y = pd.to_numeric(overlay_df["y"], errors="coerce")
        use = overlay_df.loc[x.notna() & y.notna(), :].copy()
        if use.empty:
            return None

        # color by trade (default)
        color_by = str(cfg.get("overlay_color_by", "trade") or "trade")
        if color_by not in use.columns:
            color_by = "none"

        fig = plt.figure(figsize=tuple(cfg.get("figsize_overlay", (9,7))))
        ax = fig.add_subplot(111)

        # draw floorplan raster if provided
        if floor_img_path and floor_img_path.exists():
            try:
                img = plt.imread(str(floor_img_path))
                x_min, x_max, y_min, y_max = extent
                ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin="upper")
            except Exception:
                pass

        if color_by == "none":
            ax.scatter(pd.to_numeric(use["x"], errors="coerce"),
                       pd.to_numeric(use["y"], errors="coerce"),
                       s=float(cfg.get("overlay_point_size", 8)),
                       alpha=float(cfg.get("overlay_alpha", 0.85)))
        else:
            cats = use[color_by].astype(str).fillna("")
            uniq = list(dict.fromkeys(cats))
            cmap = plt.cm.get_cmap("tab10")
            palette = {c: cmap(i % 10) for i, c in enumerate(uniq)}
            for cat, g in use.groupby(color_by):
                ax.scatter(pd.to_numeric(g["x"], errors="coerce"),
                           pd.to_numeric(g["y"], errors="coerce"),
                           s=float(cfg.get("overlay_point_size", 8)),
                           alpha=float(cfg.get("overlay_alpha", 0.85)),
                           color=palette.get(cat), label=str(cat))
            legend_if_any(ax)

        x_min, x_max, y_min, y_max = extent
        mx = float(cfg.get("floorplan_margin", 0.10))
        xr = (x_max - x_min); yr = (y_max - y_min)
        ax.set_xlim(x_min - mx*xr, x_max + mx*xr)
        ax.set_ylim(y_min - mx*yr, y_max + mx*yr)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Floorplan Overlay (positions)")
        fig.tight_layout()
        return fig
    except Exception:
        return None

def make_hourly_line_figure(hour_counts: dict, cfg: dict) -> plt.Figure | None:
    if not hour_counts:
        return None
    try:
        items = sorted(hour_counts.items(), key=lambda kv: kv[0])
        xs = [pd.to_datetime(k).to_pydatetime() for k,_ in items]
        ys = [safe_int(v) for _,v in items]
        fig = plt.figure(figsize=tuple(cfg.get("figsize_line", (7,5))))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label="Samples")
        legend_if_any(ax)
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Samples")
        ax.set_title("Hourly Samples (UTC)")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig
    except Exception:
        return None

def make_trade_bar_figure(trade_counts: dict, cfg: dict) -> plt.Figure | None:
    if not trade_counts:
        return None
    try:
        items = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)
        cats = [k if k else "(unknown)" for k,_ in items]
        vals = [int(v) for _,v in items]
        fig = plt.figure(figsize=tuple(cfg.get("figsize_bar", (7,5))))
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(cats))
        ax.barh(y_pos, vals, color="#4C78A8")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cats, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Samples")
        ax.set_title("Trade Distribution (by samples)")
        fig.tight_layout()
        return fig
    except Exception:
        return None

# --------------------------- Main ---------------------------

def main():
    # Read guidance and context (UTF-8) for meta
    _guidelines_text = read_text(GUIDELINES)
    _context_text = read_text(CONTEXT)

    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python generated.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
        user_prompt = sys.argv[1]
        csv_paths = [Path(p) for p in sys.argv[2:] if p and Path(p).exists()]
        if not csv_paths:
            raise FileNotFoundError("No readable CSV inputs were found.")

        out_dir = csv_paths[0].resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_config()
        overlay_subsample = int(cfg.get("overlay_subsample", 20000))
        max_figs_allowed = int(cfg.get("max_figures", RL_DEFAULTS.get("MAX_FIGURES", 6)))

        # Per-file cap (large-data mode): avoid holding too many points in RAM
        per_file_cap = max(500, overlay_subsample // max(1, len(csv_paths)))

        # Aggregates
        total_samples = 0
        time_min_utc: pd.Timestamp | None = None
        time_max_utc: pd.Timestamp | None = None
        distinct_trackables: set[str] = set()
        trade_counts: dict[str, int] = {}
        hourly_counts: dict[str, int] = {}
        evidence_rows: list[dict] = []

        # Overlay samples reservoir (bounded)
        overlay_samples: list[dict] = []

        # Keep a small audit record
        audits: list[dict] = []

        # Process each file (streaming style; no giant concatenations)
        for csv_path in csv_paths:
            try:
                raw = extract_tracks(str(csv_path))
                rows = raw.get("rows", [])
                audit = raw.get("audit", {})
                audits.append({"file": str(csv_path), **({k: audit.get(k) for k in audit} if isinstance(audit, dict) else {})})

                # Build a DataFrame for this file only
                df = pd.DataFrame(rows)
                # Duplicate-name guard
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated()]

                if df.empty:
                    del df
                    continue

                # Timestamp canon
                ts_src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
                df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

                # Evidence rows (first 50 across all files)
                if len(evidence_rows) < 50:
                    cols = ["trackable","trade","ts_short","x","y","z"]
                    use_cols = [c for c in cols if c in df.columns]
                    if use_cols:
                        chunk = (df[use_cols].head(50 - len(evidence_rows))
                                    .fillna("")
                                    .astype(str)
                                    .to_dict(orient="records"))
                        evidence_rows.extend(chunk)

                # Aggregates (only required columns)
                total_samples += len(df)

                # Time min/max
                tmin = df["ts_utc"].min(skipna=True)
                tmax = df["ts_utc"].max(skipna=True)
                if pd.notna(tmin):
                    time_min_utc = tmin if (time_min_utc is None or tmin < time_min_utc) else time_min_utc
                if pd.notna(tmax):
                    time_max_utc = tmax if (time_max_utc is None or tmax > time_max_utc) else time_max_utc

                # Distinct trackables
                if "trackable_uid" in df.columns:
                    distinct_trackables.update(df["trackable_uid"].dropna().astype(str).unique().tolist())
                elif "trackable" in df.columns:
                    distinct_trackables.update(df["trackable"].dropna().astype(str).unique().tolist())

                # Trade counts (by samples)
                if "trade" in df.columns:
                    vc = df["trade"].astype(str).fillna("").value_counts()
                    for k, v in vc.items():
                        trade_counts[k] = trade_counts.get(k, 0) + int(v)

                # Hourly counts using ts_utc
                if "ts_utc" in df.columns:
                    hours = df["ts_utc"].dropna().dt.floor("H").astype("datetime64[ns, UTC]")
                    vc_h = hours.value_counts()
                    for k, v in vc_h.items():
                        key = pd.Timestamp(k).isoformat()
                        hourly_counts[key] = hourly_counts.get(key, 0) + int(v)

                # Overlay sampling: numeric x/y only (bounded per file)
                need_cols = [c for c in ("x","y","trade") if c in df.columns]
                if all(c in df.columns for c in ("x","y")):
                    # filter valid numeric x/y without mutating df types
                    xv = pd.to_numeric(df["x"], errors="coerce")
                    yv = pd.to_numeric(df["y"], errors="coerce")
                    valid_idx = (xv.notna() & yv.notna()).to_numpy()
                    if valid_idx.any():
                        use_df = df.loc[valid_idx, need_cols].copy()
                        n = len(use_df)
                        if n > per_file_cap:
                            # Even subsample across the file
                            idx = np.linspace(0, n-1, per_file_cap).astype(int)
                            use_df = use_df.iloc[idx]
                        overlay_samples.extend(use_df.to_dict(orient="records"))

                # Clean per-file memory
                del df
                plt.close('all')
            except MemoryError:
                raise
            except KeyboardInterrupt:
                raise
            except Exception:
                # Continue to next file; keep partial aggregates
                continue

        # Finalization pass: assemble report pieces
        report_date = (time_max_utc or pd.Timestamp.utcnow()).strftime("%Y-%m-%d")
        title = "Walmart Renovation RTLS — Data Summary"
        meta_text = f"User query: {user_prompt}\nContext: {(_context_text[:180] + '...') if _context_text else ''}"

        # Build overlay DataFrame (bounded to overlay_subsample)
        overlay_df = pd.DataFrame(overlay_samples)
        if not overlay_df.empty and len(overlay_df) > overlay_subsample:
            idx = np.linspace(0, len(overlay_df)-1, overlay_subsample).astype(int)
            overlay_df = overlay_df.iloc[idx].reset_index(drop=True)

        # Top trades list
        top_trades_sorted = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)

        # Figures
        figs: list[plt.Figure] = []

        # Try floorplan overlay if assets exist
        fp_info = load_floorplan_extent()
        fp_img_path = find_floorplan_image(ROOT)
        if fp_info and fp_img_path and not overlay_df.empty:
            fig_overlay = make_floorplan_overlay_figure(overlay_df, tuple(fp_info["extent"]), fp_img_path, cfg)
            if fig_overlay is not None:
                figs.append(fig_overlay)

        # Hourly line
        fig_hour = make_hourly_line_figure(hourly_counts, cfg)
        if fig_hour is not None:
            figs.append(fig_hour)

        # Trade bar
        fig_trade = make_trade_bar_figure(trade_counts, cfg)
        if fig_trade is not None:
            figs.append(fig_trade)

        # Enforce figure budget before writing PDF
        figs = figs[:max_figs_allowed]

        # Save PNGs first (mandatory order)
        png_paths: list[Path] = []
        for i, f in enumerate(figs, start=1):
            png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                f.savefig(str(png), dpi=120)  # no bbox_inches='tight'
                png_paths.append(png)
            except Exception:
                # Ignore PNG write errors; continue
                pass

        # Sections assembly
        sections: list[dict] = []

        # Summary bullets
        meta = {
            "total_samples": total_samples,
            "distinct_trackables": len(distinct_trackables),
            "time_min": time_min_utc,
            "time_max": time_max_utc,
            "top_trades": top_trades_sorted,
        }
        sections.append({"type": "summary", "title": "Key takeaways", "bullets": build_summary_bullets(meta)})

        # Evidence table (list-of-dicts)
        cols = ["trackable","trade","ts_short","x","y","z"]
        # Ensure all keys present in each row to satisfy table renderers
        rows = []
        for r in evidence_rows[:50]:
            row = {c: str(r.get(c, "")) for c in cols}
            rows.append(row)
        sections.append({"type": "table", "title": "Evidence", "data": rows, "headers": cols, "rows_per_page": 24})

        # Audit appendix (compact)
        if audits:
            try:
                audits_text = "\n".join(json.dumps(a, ensure_ascii=False) for a in audits[:6])
            except Exception:
                audits_text = "\n".join(str(a) for a in audits[:6])
            sections.append({"type": "appendix", "title": "Audit (first few entries)", "text": audits_text})

        # Charts section (pass live figures)
        if figs:
            sections.append({"type": "charts", "title": "Figures", "figures": figs})

        # Report object
        report = {
            "title": title,
            "date": report_date,
            "meta": meta_text,
            "sections": sections,
        }

        # Apply budgets to cap figures/tables/text
        report_capped = apply_budgets(report)

        # Build the PDF
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        safe_build_pdf(report_capped, str(pdf_path), logo_path=str(LOGO))

        # Success links
        print(f"[Download the PDF]({file_uri(pdf_path)})")
        for i, png in enumerate(png_paths, 1):
            print(f"[Download Plot {i}]({file_uri(png)})")

        # Close figures after write
        plt.close('all')

    except (MemoryError, KeyboardInterrupt) as e:
        # Minimal-Report Mode
        try:
            report_date = datetime.utcnow().strftime("%Y-%m-%d")
            title = "Walmart Renovation RTLS — Data Summary (Lite)"
            bullets = [
                "Minimal-Report Mode due to system constraints.",
                f"Processed inputs: {', '.join(str(p) for p in sys.argv[2:])}",
            ]
            sections = [{"type":"summary","title":"Key takeaways","bullets":bullets}]
            # Attempt a tiny evidence section if possible
            sections.append({"type":"table","title":"Evidence","data":[],"headers":["trackable","trade","ts_short","x","y","z"],"rows_per_page":24})
            report = {"title":title,"date":report_date,"meta":"", "sections":sections}
            pdf_path = Path(sys.argv[2]).resolve().parent / f"info_zone_report_{report_date}.pdf"
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            print(f"[Download the PDF]({file_uri(pdf_path)})")
        except Exception as e2:
            print("Error Report:")
            print("Insufficient resources to build even a minimal report.")
    except Exception as e:
        # Failure: print only an Error Report with a brief reason (1–2 lines)
        reason = str(e).strip() or e.__class__.__name__
        print("Error Report:")
        print(reason)

if __name__ == "__main__":
    main()