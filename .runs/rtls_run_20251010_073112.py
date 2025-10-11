#!/usr/bin/env python
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

# --------------------------- Imports (local helpers) ---------------------------
import json
import math
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from report_limits import apply_budgets, DEFAULTS as LIMIT_DEFAULTS

# --------------------------- Utility helpers ---------------------------
def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def parse_args(argv: List[str]) -> Tuple[str, List[Path]]:
    if len(argv) < 3:
        raise ValueError("Usage: python generated.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
    user_prompt = argv[1]
    csv_paths = [Path(a) for a in argv[2:]]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
    return user_prompt, csv_paths

def load_config() -> Dict[str, object]:
    try:
        if CONFIG.exists():
            text = read_text(CONFIG)
            if text.strip():
                cfg = json.loads(text)
                return cfg if isinstance(cfg, dict) else {}
    except Exception:
        pass
    # default minimal config
    return {
        "prefer_floorplan": True,
        "floorplan_margin": 0.10,
        "overlay_point_size": 8,
        "overlay_alpha": 0.85,
        "overlay_color_by": "trade",
        "overlay_subsample": 20000,
        "draw_trails": False,
        "trail_seconds": 900,
        "draw_zones": False,
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

def find_floorplan_image(root: Path) -> Optional[Path]:
    for name in ("floorplan.png", "floorplan.jpg", "floorplan.jpeg"):
        p = root / name
        if p.exists():
            return p
    return None

def load_floorplan_extent() -> Optional[Tuple[float, float, float, float]]:
    # Fallback to first plan entry
    try:
        if not FLOORJSON.exists():
            return None
        data = json.loads(read_text(FLOORJSON) or "{}")
        fp = data.get("floorplans") or data.get("plans") or []
        if isinstance(fp, list) and fp:
            obj = fp[0]
        elif isinstance(fp, dict):
            obj = fp
        else:
            return None
        width  = float(obj.get("width", 0))
        height = float(obj.get("height", 0))
        x_c    = float(obj.get("image_offset_x", 0))
        y_c    = float(obj.get("image_offset_y", 0))
        image_scale = float(obj.get("image_scale", 0))  # meters per pixel
        if width <= 0 or height <= 0 or image_scale <= 0:
            return None
        s = image_scale * 100.0  # mm/px
        x_min = (x_c - width/2.0)  * s
        x_max = (x_c + width/2.0)  * s
        y_min = (y_c - height/2.0) * s
        y_max = (y_c + height/2.0) * s
        return (x_min, x_max, y_min, y_max)
    except Exception:
        return None

def safe_to_datetime_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")

def build_evidence_table(df: pd.DataFrame) -> Dict[str, object]:
    cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    rows = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
    return {"type":"table","title":"Evidence","data":rows,"headers":cols,"rows_per_page":24}

def has_labeled_artists(ax: plt.Axes) -> bool:
    handles, labels = ax.get_legend_handles_labels()
    return len([l for l in labels if l]) > 0

# --------------------------- Chart builders ---------------------------
def make_floorplan_overlay(overlay_df: pd.DataFrame,
                           extent: Optional[Tuple[float,float,float,float]],
                           image_path: Optional[Path],
                           cfg: Dict[str, object]) -> Optional[plt.Figure]:
    if overlay_df is None or overlay_df.empty:
        return None
    if "x" not in overlay_df.columns or "y" not in overlay_df.columns:
        return None

    x = pd.to_numeric(overlay_df["x"], errors="coerce")
    y = pd.to_numeric(overlay_df["y"], errors="coerce")
    use = overlay_df.loc[x.notna() & y.notna(), :].copy()
    if use.empty:
        return None

    max_pts = int(cfg.get("overlay_subsample", 20000))
    if len(use) > max_pts:
        idx = np.linspace(0, len(use) - 1, max_pts).astype(int)
        use = use.iloc[idx]

    figsize = tuple(cfg.get("figsize_overlay", (9,7)))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if extent and image_path and image_path.exists():
        try:
            img = plt.imread(str(image_path))
            x_min, x_max, y_min, y_max = extent
            ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin="upper")
            xr = (x_max - x_min); yr = (y_max - y_min)
            margin = float(cfg.get("floorplan_margin", 0.10))
            ax.set_xlim(x_min - margin*xr, x_max + margin*xr)
            ax.set_ylim(y_min - margin*yr, y_max + margin*yr)
        except Exception:
            # Plot without raster if loading failed
            pass

    color_by = str(cfg.get("overlay_color_by", "trade") or "none")
    if color_by not in use.columns:
        color_by = "none"

    if color_by == "none":
        ax.scatter(pd.to_numeric(use["x"], errors="coerce"),
                   pd.to_numeric(use["y"], errors="coerce"),
                   s=float(cfg.get("overlay_point_size", 8)),
                   alpha=float(cfg.get("overlay_alpha", 0.85)))
    else:
        cats = use[color_by].astype(str).fillna("")
        uniq = list(dict.fromkeys(cats.tolist()))
        cmap = plt.cm.get_cmap("tab10")
        lbl_count = 0
        for i, cat in enumerate(uniq):
            g = use.loc[cats == cat]
            ax.scatter(pd.to_numeric(g["x"], errors="coerce"),
                       pd.to_numeric(g["y"], errors="coerce"),
                       s=float(cfg.get("overlay_point_size", 8)),
                       color=cmap(i % 10),
                       alpha=float(cfg.get("overlay_alpha", 0.85)),
                       label=str(cat))
            lbl_count += 1
        if lbl_count <= 12 and has_labeled_artists(ax):
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Floorplan Overlay")
    fig.tight_layout()
    return fig

def make_top_trades_bar(trade_counts: Dict[str, int], cfg: Dict[str, object]) -> Optional[plt.Figure]:
    if not trade_counts:
        return None
    items = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_n = int(cfg.get("top_n", 10))
    items = items[:top_n]
    labels = [k if k else "(unknown)" for k, _ in items]
    values = [v for _, v in items]
    if not values or sum(values) == 0:
        return None
    figsize = tuple(cfg.get("figsize_bar", (7,5)))
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    ax.barh(y, values, color="#2E86AB", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sample Count")
    ax.set_title("Top Trades by Samples")
    fig.tight_layout()
    return fig

# --------------------------- Main analysis ---------------------------
def main():
    user_prompt, csv_paths = parse_args(sys.argv)
    out_dir = csv_paths[0].parent

    # Load context/guidelines for meta
    guidelines_txt = read_text(GUIDELINES)
    context_txt = read_text(CONTEXT)
    cfg = load_config()

    # Aggregates (in-RAM small dicts)
    total_rows = 0
    total_valid_xy = 0
    unique_trackables = set()
    unique_trades = set()
    trade_counts: Dict[str, int] = {}
    hour_counts: Dict[pd.Timestamp, int] = {}
    ts_min_global: Optional[pd.Timestamp] = None
    ts_max_global: Optional[pd.Timestamp] = None

    # Overlay sample buffer
    overlay_target = int(cfg.get("overlay_subsample", 20000))
    overlay_remaining = overlay_target
    overlay_buf: List[pd.DataFrame] = []

    # Evidence from first file
    evidence_rows_df: Optional[pd.DataFrame] = None

    # Per-file streaming
    try:
        for csv_path in csv_paths:
            raw = extract_tracks(str(csv_path))
            df = pd.DataFrame(raw.get("rows", []))
            audit = raw.get("audit", {})  # may use later

            # Duplicate-name guard
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            # Timestamp canon
            src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
            df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

            # Math-only cast for x,y
            df["x"] = pd.to_numeric(df.get("x"), errors="coerce")
            df["y"] = pd.to_numeric(df.get("y"), errors="coerce")

            # Update aggregates
            n_rows = len(df)
            total_rows += n_rows

            # Valid xy rows with valid ts_utc
            valid_mask = df["ts_utc"].notna() & df["x"].notna() & df["y"].notna()
            df_valid = df.loc[valid_mask, :]

            total_valid_xy += int(valid_mask.sum())

            if "trackable_uid" in df.columns:
                unique_trackables.update(df["trackable_uid"].dropna().astype(str).unique().tolist())
            elif "trackable" in df.columns:
                unique_trackables.update(df["trackable"].dropna().astype(str).unique().tolist())

            if "trade" in df.columns:
                trades = df["trade"].dropna().astype(str)
                unique_trades.update(trades.unique().tolist())
                vc = trades.value_counts()
                for k, v in vc.items():
                    trade_counts[k] = trade_counts.get(k, 0) + int(v)

            # Time range
            if "ts_utc" in df.columns:
                tmin = df["ts_utc"].min()
                tmax = df["ts_utc"].max()
                if pd.notna(tmin):
                    ts_min_global = tmin if ts_min_global is None or tmin < ts_min_global else ts_min_global
                if pd.notna(tmax):
                    ts_max_global = tmax if ts_max_global is None or tmax > ts_max_global else ts_max_global

                # Hourly counts (lightweight)
                hours = df["ts_utc"].dt.floor("H")
                vc_h = hours.value_counts()
                for h, c in vc_h.items():
                    if pd.isna(h):
                        continue
                    hour_counts[h] = hour_counts.get(h, 0) + int(c)

            # Overlay sampling: take up to overlay_remaining evenly spaced points from df_valid
            if overlay_remaining > 0 and not df_valid.empty:
                take = min(overlay_remaining, len(df_valid))
                if take > 0:
                    if take == len(df_valid):
                        sample = df_valid.copy()
                    else:
                        idx = np.linspace(0, len(df_valid) - 1, take).astype(int)
                        sample = df_valid.iloc[idx].copy()
                    # Keep only needed columns to reduce memory
                    keep_cols = [c for c in ["x","y","trade","trackable","trackable_uid"] if c in sample.columns]
                    sample = sample[keep_cols]
                    overlay_buf.append(sample)
                    overlay_remaining -= len(sample)

            # Evidence table source: first file's df
            if evidence_rows_df is None:
                # Ensure required columns exist
                for c in ["trackable","trade","ts_short","x","y","z"]:
                    if c not in df.columns:
                        df[c] = ""
                evidence_rows_df = df[["trackable","trade","ts_short","x","y","z"]].copy()

            # Clear per-file heavy objects
            del df, df_valid
            plt.close('all')

        # Finalization
        overlay_df = pd.concat(overlay_buf, ignore_index=True) if overlay_buf else pd.DataFrame(columns=["x","y"])
        report_date = (ts_max_global or pd.Timestamp.utcnow()).date().isoformat()

        # Build figures list
        figs: List[plt.Figure] = []
        png_paths: List[Path] = []

        # Attempt floorplan overlay
        extent = load_floorplan_extent()
        fp_img = find_floorplan_image(ROOT)
        overlay_fig = make_floorplan_overlay(overlay_df, extent, fp_img, cfg)
        if overlay_fig is not None:
            figs.append(overlay_fig)

        # Top trades bar
        bar_fig = make_top_trades_bar(trade_counts, cfg)
        if bar_fig is not None:
            figs.append(bar_fig)

        # Save PNGs first (mandatory order)
        for i, fig in enumerate(figs, start=1):
            png = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            fig.savefig(str(png), dpi=120)  # no bbox_inches='tight'
            png_paths.append(png)

        # Compose report
        title = "InfoZone — Walmart RTLS Summary"
        meta_lines = []
        if ts_min_global is not None and ts_max_global is not None:
            meta_lines.append(f"Coverage (UTC): {ts_min_global.isoformat()} → {ts_max_global.isoformat()}")
        meta_lines.append(f"Files: {len(csv_paths)}")
        meta_lines.append(f"Rows: {total_rows:,}")
        meta_lines.append(f"Valid XY: {total_valid_xy:,}")
        meta_lines.append(f"Trackables: {len(unique_trackables)}")
        meta_lines.append(f"Trades: {len(unique_trades)}")
        meta_text = " | ".join(meta_lines)

        # Summary bullets
        bullets = []
        bullets.append(f"Total samples: {total_rows:,} ({total_valid_xy:,} with valid X,Y).")
        if ts_min_global is not None and ts_max_global is not None:
            bullets.append(f"Time span (UTC): {ts_min_global.strftime('%Y-%m-%d %H:%M')} to {ts_max_global.strftime('%Y-%m-%d %H:%M')}.")
        bullets.append(f"Unique trackables: {len(unique_trackables)}; unique trades: {len(unique_trades)}.")
        if trade_counts:
            top_items = sorted(trade_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            bullets.append("Top trades by sample count: " + ", ".join([f"{k or '(unknown)'} ({v:,})" for k, v in top_items]) + ".")

        sections: List[Dict[str, object]] = []
        sections.append({"type":"summary","title":"Summary","bullets":bullets})

        # Evidence table (compact)
        if evidence_rows_df is not None:
            sections.append(build_evidence_table(evidence_rows_df))

        # Charts section
        if figs:
            sections.append({"type":"charts","title":"Figures","figures":figs})

        # Narrative/context (light)
        if context_txt.strip():
            sections.append({"type":"narrative","title":"Context","paragraphs":[context_txt.strip()[:1500]]})

        report = {
            "title": title,
            "date": report_date,
            "meta": meta_text,
            "sections": sections,
        }

        # Apply budgets
        # Use max figures from limits/config if present
        budgets = dict(LIMIT_DEFAULTS)
        try:
            if "max_figures" in cfg and isinstance(cfg["max_figures"], int):
                budgets["MAX_FIGURES"] = int(cfg["max_figures"])
        except Exception:
            pass
        report = apply_budgets(report, budgets)

        # Build PDF
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))

        # Success links
        print(f"[Download the PDF]({file_uri(pdf_path)})")
        for i, png in enumerate(png_paths, 1):
            print(f"[Download Plot {i}]({file_uri(png)})")

    except (MemoryError, KeyboardInterrupt):
        # Minimal-Report Mode: summary + compact evidence; no PNGs
        try:
            report_date = datetime.now(timezone.utc).date().isoformat()
            bullets = [
                f"Total samples: {total_rows:,} ({total_valid_xy:,} with valid X,Y).",
                f"Files processed: {len(csv_paths)}.",
                f"Unique trackables: {len(unique_trackables)}; unique trades: {len(unique_trades)}.",
            ]
            if 'ts_min_global' in locals() and ts_min_global is not None and ts_max_global is not None:
                bullets.append(f"Time span (UTC): {ts_min_global.strftime('%Y-%m-%d %H:%M')} to {ts_max_global.strftime('%Y-%m-%d %H:%M')}.")

            sections = [{"type":"summary","title":"Summary","bullets":bullets}]
            if evidence_rows_df is not None:
                sections.append(build_evidence_table(evidence_rows_df))

            report = {
                "title": "InfoZone — Walmart RTLS Summary (Lite)",
                "date": report_date,
                "meta": "Minimal-Report Mode",
                "sections": sections,
            }
            pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            print(f"[Download the PDF]({file_uri(pdf_path)})")
        except Exception:
            err = "Error Report:\nMinimal-Report Mode failed to write the PDF."
            print(err)
    except Exception as e:
        # Hard failure — print only a short Error Report
        reason = str(e).strip() or e.__class__.__name__
        msg = "Error Report:\n" + reason[:300]
        print(msg)

if __name__ == "__main__":
    main()