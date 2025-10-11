#!/usr/bin/env python
# InfoZoneBuilder – Walmart RTLS analyzer
import sys, os, json, math, traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------- ROOT resolution and local imports (MANDATORY) ----------
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

# ---------- External local helpers ----------
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Error Report:")
    print("Missing required Python packages for plotting or pandas.")
    sys.exit(1)

try:
    from extractor import extract_tracks
except Exception:
    print("Error Report:")
    print("Local extractor helper not found or failed to import.")
    sys.exit(1)

try:
    from pdf_creation_script import safe_build_pdf
except Exception:
    print("Error Report:")
    print("PDF builder helper not found or failed to import.")
    sys.exit(1)

try:
    from report_limits import apply_budgets, make_lite
except Exception:
    # Minimal fallbacks if report_limits is missing
    def apply_budgets(report: Dict[str, Any], caps: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        return report
    def make_lite(report: Dict[str, Any]) -> Dict[str, Any]:
        return report

# ---------- Config loader ----------
def load_config() -> Dict[str, Any]:
    try:
        if CONFIG.exists():
            return json.loads(read_text(CONFIG)) or {}
    except Exception:
        pass
    # Defaults mirroring helper defaults
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

# ---------- Floorplan utilities (LOCAL paths only) ----------
def load_floorplan_extent_and_image(fp_json: Path, image_candidates: List[Path]) -> Optional[Dict[str, Any]]:
    if not fp_json.exists():
        return None
    try:
        data = json.loads(read_text(fp_json))
        fp = (data.get("floorplans") or data.get("plans") or data or [None])
        if isinstance(fp, list):
            fp = fp[0]
        if not fp:
            return None
        width  = float(fp.get("width", 0))
        height = float(fp.get("height", 0))
        x_c    = float(fp.get("image_offset_x", 0))
        y_c    = float(fp.get("image_offset_y", 0))
        image_scale = float(fp.get("image_scale", 0))  # meters per pixel
        scale = image_scale * 100.0  # mm per pixel

        x_min = (x_c - width/2.0)  * scale
        x_max = (x_c + width/2.0)  * scale
        y_min = (y_c - height/2.0) * scale
        y_max = (y_c + height/2.0) * scale

        img_path: Optional[Path] = None
        for cand in image_candidates:
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            return None
        img = plt.imread(str(img_path))
        return {"extent": (x_min, x_max, y_min, y_max), "image": img}
    except Exception:
        return None

# ---------- Plot helpers ----------
def safe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0 and len(labels) <= 12:
        ax.legend(loc="upper left", frameon=True, fontsize=8)

def make_floorplan_overlay(figsize, overlay_points: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[plt.Figure]:
    """
    overlay_points must contain columns: x, y, cat (category string), and optionally raw_x/raw_y for fallback.
    """
    if overlay_points.empty:
        return None

    # Attempt floorplan drawing using local floorplans.json + floorplan image
    candidates = [
        ROOT / "floorplan.png",
        ROOT / "floorplan.jpg",
        ROOT / "floorplan.jpeg",
    ]
    fp = load_floorplan_extent_and_image(FLOORJSON, candidates)

    fig = plt.figure(figsize=tuple(cfg.get("figsize_overlay", (9, 7))))
    ax = fig.add_subplot(111)

    plotted = False
    if fp:
        x_min, x_max, y_min, y_max = fp["extent"]
        xr = (x_max - x_min)
        yr = (y_max - y_min)
        dx0 = -x_min
        dy0 = -y_min

        # Background image: draw with origin='upper' and extent in display coords [0, xr] x [0, yr]
        ax.imshow(fp["image"], extent=[0, xr, 0, yr], origin="upper")

        # Transform points into display coords (x' = x + dx0; y' = y + dy0)
        xs = pd.to_numeric(overlay_points["x"], errors="coerce")
        ys = pd.to_numeric(overlay_points["y"], errors="coerce")
        mask = xs.notna() & ys.notna()
        use = overlay_points.loc[mask].copy()
        if not use.empty:
            use["_xp"] = xs[mask] + dx0
            use["_yp"] = ys[mask] + dy0

            # Color mapping per category
            cats = use["cat"].astype(str).tolist()
            uniq = list(dict.fromkeys(cats))
            cmap = plt.cm.get_cmap("tab10")
            color_map = {c: cmap(i % 10) for i, c in enumerate(uniq)}

            for cat, g in use.groupby("cat"):
                ax.scatter(g["_xp"], g["_yp"],
                           s=float(cfg.get("overlay_point_size", 8)),
                           alpha=float(cfg.get("overlay_alpha", 0.85)),
                           color=color_map.get(cat, (0.1, 0.1, 0.1, 0.85)),
                           label=str(cat))
            safe_legend(ax)

            # Limits with margin
            mx = float(cfg.get("floorplan_margin", 0.10))
            ax.set_xlim(-mx * xr, xr * (1 + mx))
            ax.set_ylim(-mx * yr, yr * (1 + mx))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
            ax.set_title("Floorplan Overlay")
            fig.tight_layout()
            plotted = True

    # Fallback: scatter in world coords (no image)
    if not plotted:
        xs = pd.to_numeric(overlay_points["x"], errors="coerce")
        ys = pd.to_numeric(overlay_points["y"], errors="coerce")
        mask = xs.notna() & ys.notna()
        use = overlay_points.loc[mask].copy()
        if use.empty:
            plt.close(fig)
            return None
        cats = use["cat"].astype(str).tolist()
        uniq = list(dict.fromkeys(cats))
        cmap = plt.cm.get_cmap("tab10")
        color_map = {c: cmap(i % 10) for i, c in enumerate(uniq)}
        for cat, g in use.groupby("cat"):
            ax.scatter(pd.to_numeric(g["x"], errors="coerce"),
                       pd.to_numeric(g["y"], errors="coerce"),
                       s=float(cfg.get("overlay_point_size", 8)),
                       alpha=float(cfg.get("overlay_alpha", 0.85)),
                       color=color_map.get(cat, (0.1, 0.1, 0.1, 0.85)),
                       label=str(cat))
        safe_legend(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.set_title("Position Overlay (No Floorplan)")
        fig.tight_layout()
    return fig

def make_hourly_line(hourly_counts: pd.Series, figsize) -> Optional[plt.Figure]:
    if hourly_counts is None or len(hourly_counts) < 2:
        return None
    try:
        fig = plt.figure(figsize=tuple(figsize))
        ax = fig.add_subplot(111)
        idx = pd.to_datetime(hourly_counts.index)
        # Ensure timezone-naive for plotting
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        ax.plot(idx, hourly_counts.values, marker="o", linewidth=1.5, color="#1f77b4")
        ax.set_title("Hourly Sample Counts (UTC)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Samples")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig
    except Exception:
        return None

def make_trade_dist(trade_counts: pd.Series, cfg) -> Optional[plt.Figure]:
    if trade_counts is None or trade_counts.empty:
        return None
    try:
        pie_max_trades = int(cfg.get("pie_max_trades", 8))
        pie_max_single_share = float(cfg.get("pie_max_single_share", 0.90))
        figsize_pie = tuple(cfg.get("figsize_pie", (5, 5)))
        figsize_bar = tuple(cfg.get("figsize_bar", (7, 5)))

        share = (trade_counts / trade_counts.sum()).fillna(0.0)
        if len(trade_counts) <= pie_max_trades and (share.max() <= pie_max_single_share):
            fig = plt.figure(figsize=figsize_pie)
            ax = fig.add_subplot(111)
            ax.pie(trade_counts.values, labels=trade_counts.index.tolist(), autopct="%1.0f%%", startangle=90)
            ax.set_title("Trade Share")
            fig.tight_layout()
            return fig
        else:
            fig = plt.figure(figsize=figsize_bar)
            ax = fig.add_subplot(111)
            trade_counts.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#2ca02c")
            ax.set_title("Samples by Trade")
            ax.set_xlabel("Trade")
            ax.set_ylabel("Samples")
            fig.tight_layout()
            return fig
    except Exception:
        return None

# ---------- Main processing ----------
def main():
    # CLI: python generated.py "<USER_PROMPT>" /abs/csv1 [/abs/csv2 ...]
    if len(sys.argv) < 3:
        print("Error Report:")
        print("Usage: python generated.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]")
        return

    user_prompt = sys.argv[1]
    csv_paths = [Path(p) for p in sys.argv[2:] if p and str(p).strip()]

    # Output directory from first CSV
    out_dir = csv_paths[0].resolve().parent if csv_paths else Path.cwd().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    cfg = load_config()

    # Aggregates (memory-light)
    total_samples = 0
    uniq_ids = set()
    trade_counter: Dict[str, int] = {}
    hourly_counts_all: Optional[pd.Series] = None
    ts_min: Optional[pd.Timestamp] = None
    ts_max: Optional[pd.Timestamp] = None

    # Overlay reservoir
    overlay_limit = int(cfg.get("overlay_subsample", 20000))
    overlay_records: List[Dict[str, Any]] = []

    # Evidence rows (from first file)
    evidence_df: Optional[pd.DataFrame] = None

    # Schema flag
    schema_validated = False

    # Mac map audit
    mac_map_ok = True

    try:
        for file_idx, csv_path in enumerate(csv_paths):
            # Ingest via helper (MAC → trackable → trade already handled by extractor)
            try:
                raw = extract_tracks(str(csv_path), mac_map_path=str(ROOT / "trackable_objects.json"))
            except Exception as e:
                print("Error Report:")
                print("Failed to read or parse the CSV via extractor.")
                return

            audit = raw.get("audit", {}) or {}
            if audit.get("mac_map_loaded") is False:
                print("Error Report:")
                print("MAC map not loaded; trackable name/UID inference is required.")
                return

            rows = raw.get("rows", []) or []
            df = pd.DataFrame(rows)

            # Duplicate-name guard
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            # Timestamp canon (single source of truth)
            try:
                src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
            except Exception:
                src = pd.Series([], dtype=str)
            df["ts_utc"] = pd.to_datetime(src, utc=True, errors="coerce")

            # Early schema validation after first file
            if not schema_validated:
                cols = set(df.columns.astype(str).tolist())
                have_identity = ("trackable" in cols) or ("trackable_uid" in cols)
                have_trade = ("trade" in cols)
                have_pos = ("x" in cols) and ("y" in cols)
                if not (have_identity and have_trade and have_pos):
                    print("Error Report:")
                    print("Missing required columns for analysis.")
                    print("Columns detected: " + ",".join(df.columns.astype(str)))
                    return
                schema_validated = True

            # Update time range
            valid_ts = df["ts_utc"].dropna()
            if not valid_ts.empty:
                _min = valid_ts.min()
                _max = valid_ts.max()
                ts_min = _min if (ts_min is None or _min < ts_min) else ts_min
                ts_max = _max if (ts_max is None or _max > ts_max) else ts_max

            # Evidence dataframe (keep small)
            if evidence_df is None:
                # Ensure columns for evidence exist
                for col in ["trackable", "trade", "ts_short", "x", "y", "z"]:
                    if col not in df.columns:
                        if col == "ts_short":
                            # create from ts_utc
                            if "ts_utc" in df.columns:
                                try:
                                    ts_short = df["ts_utc"].dt.tz_convert("UTC").dt.tz_localize(None).dt.strftime("%m-%d\n%H:%M")
                                except Exception:
                                    ts_short = pd.Series([""] * len(df), dtype=str)
                                df["ts_short"] = ts_short
                            else:
                                df["ts_short"] = ""
                        else:
                            df[col] = ""
                evidence_df = df[["trackable","trade","ts_short","x","y","z"]].copy()

            # Aggregates
            total_samples += len(df)

            if "trackable_uid" in df.columns:
                uniq_ids.update([v for v in df["trackable_uid"].astype(str).tolist() if v])
            elif "trackable" in df.columns:
                uniq_ids.update([v for v in df["trackable"].astype(str).tolist() if v])

            # Hourly counts
            if "ts_utc" in df.columns:
                ts_clean = df["ts_utc"].dropna()
                if not ts_clean.empty:
                    hours = ts_clean.dt.floor("1H")
                    per = hours.value_counts().sort_index()
                    hourly_counts_all = per if hourly_counts_all is None else hourly_counts_all.add(per, fill_value=0)

            # Trade counts
            if "trade" in df.columns:
                vc = df["trade"].fillna("").replace("", "unknown").astype(str).value_counts()
                for k, v in vc.items():
                    trade_counter[k] = trade_counter.get(k, 0) + int(v)

            # Overlay reservoir: limit points, color by cfg["overlay_color_by"]
            color_by = str(cfg.get("overlay_color_by", "trade") or "trade")
            if color_by not in df.columns:
                color_by = "trade" if "trade" in df.columns else None

            if "x" in df.columns and "y" in df.columns:
                xs = pd.to_numeric(df["x"], errors="coerce")
                ys = pd.to_numeric(df["y"], errors="coerce")
                mask = xs.notna() & ys.notna()
                overlay_df = df.loc[mask, ["x","y"]].copy()
                if color_by:
                    overlay_df["cat"] = df.loc[mask, color_by].fillna("").replace("", "unknown").astype(str)
                else:
                    overlay_df["cat"] = "points"
                # Decimate if too many rows to fit in remaining budget
                remaining = max(0, overlay_limit - len(overlay_records))
                if remaining > 0 and len(overlay_df) > 0:
                    take = min(remaining, len(overlay_df))
                    if len(overlay_df) > take:
                        idx = np.linspace(0, len(overlay_df) - 1, take).astype(int)
                        overlay_df = overlay_df.iloc[idx].copy()
                    overlay_records.extend(overlay_df.to_dict(orient="records"))

            # Cleanup large per-file objects
            del df
            plt.close('all')

        # Build small summaries
        trade_series = pd.Series(trade_counter).sort_values(ascending=False) if trade_counter else pd.Series(dtype=float)
        if hourly_counts_all is not None:
            hourly_counts_all = hourly_counts_all.sort_index()
        # Prepare overlay DataFrame
        overlay_df_small = pd.DataFrame(overlay_records) if overlay_records else pd.DataFrame(columns=["x","y","cat"])

        # Figures construction
        figs: List[plt.Figure] = []
        png_paths: List[Path] = []

        # Floorplan/position overlay
        try:
            if not overlay_df_small.empty:
                figs.append(make_floorplan_overlay(tuple(cfg.get("figsize_overlay", (9,7))), overlay_df_small, cfg))
        except Exception:
            figs.append(None)

        # Hourly line
        try:
            if hourly_counts_all is not None and len(hourly_counts_all) >= int(cfg.get("line_min_points", 2)):
                figs.append(make_hourly_line(hourly_counts_all, tuple(cfg.get("figsize_line", (7,5)))))
        except Exception:
            figs.append(None)

        # Trade distribution
        try:
            if trade_series is not None and not trade_series.empty:
                figs.append(make_trade_dist(trade_series, cfg))
        except Exception:
            figs.append(None)

        # Filter out Nones
        figs = [f for f in figs if f is not None]

        # Evidence table
        sections: List[Dict[str, Any]] = []
        if evidence_df is None:
            evidence_df = pd.DataFrame(columns=["trackable","trade","ts_short","x","y","z"])
        cols = ["trackable","trade","ts_short","x","y","z"]
        for c in cols:
            if c not in evidence_df.columns:
                evidence_df[c] = ""
        table_rows = evidence_df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
        sections.append({"type": "table", "title": "Evidence", "data": table_rows, "headers": cols, "rows_per_page": 24})

        # Summary bullets
        time_range_str = ""
        if ts_min is not None and ts_max is not None:
            try:
                t0 = ts_min.tz_convert("UTC").tz_localize(None) if getattr(ts_min, "tz", None) is not None else ts_min
                t1 = ts_max.tz_convert("UTC").tz_localize(None) if getattr(ts_max, "tz", None) is not None else ts_max
            except Exception:
                t0, t1 = ts_min, ts_max
            time_range_str = f"Time range (UTC): {t0} to {t1}"
        bullets = [
            f"Files analyzed: {len(csv_paths)}",
            f"Total samples: {int(total_samples)}",
            f"Unique trackables: {len(uniq_ids)}",
        ]
        if time_range_str:
            bullets.append(time_range_str)
        if trade_series is not None and not trade_series.empty:
            top_trades = ", ".join([f"{k} ({v})" for k, v in trade_series.head(5).items()])
            bullets.append(f"Top trades by samples: {top_trades}")
        sections.insert(0, {"type": "summary", "title": "Summary", "bullets": bullets})

        # Charts → PNGs
        report_date = ""
        if ts_min is not None:
            try:
                report_date = ts_min.tz_convert("UTC").tz_localize(None).strftime("%Y%m%d")
            except Exception:
                report_date = pd.Timestamp.utcnow().strftime("%Y%m%d")
        else:
            report_date = pd.Timestamp.utcnow().strftime("%Y%m%d")

        for i, fig in enumerate(figs, 1):
            png_path = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
            try:
                fig.savefig(str(png_path), dpi=120)
            except Exception:
                pass
            png_paths.append(png_path)

        if figs:
            sections.append({"type": "charts", "title": "Figures", "figures": figs})

        # Report metadata
        meta_lines = []
        if user_prompt:
            meta_lines.append(f"Query: {user_prompt}")
        meta_lines.append("CSV files:")
        for p in csv_paths[:6]:
            try:
                meta_lines.append(f"- {str(p)}")
            except Exception:
                continue
        meta_text = "\n".join(meta_lines)

        title = "Walmart RTLS Position Report"
        report: Dict[str, Any] = {"title": title, "meta": meta_text, "sections": sections}

        # Apply budgets
        report = apply_budgets(report, None)

        # Build PDF
        pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
        try:
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
        except MemoryError:
            # Minimal-Report Mode
            report_lite = make_lite({"title": title, "meta": meta_text, "sections": sections[:2]})
            safe_build_pdf(report_lite, str(pdf_path), logo_path=str(LOGO))
            png_paths = []  # no PNGs in lite mode
        except Exception:
            # Fallback to lite
            report_lite = make_lite(report)
            safe_build_pdf(report_lite, str(pdf_path), logo_path=str(LOGO))

        # Print links (success)
        print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        for i, pth in enumerate(png_paths, 1):
            print(f"[Download Plot {i}](file:///{pth.resolve().as_posix()})")

    except (MemoryError, KeyboardInterrupt):
        # Minimal-Report Mode (graceful)
        try:
            sections = []
            bullets = [
                f"Files analyzed: {len(csv_paths)}",
                f"Total samples (approx): {int(total_samples)}",
                f"Unique trackables (approx): {len(uniq_ids)}",
            ]
            sections.append({"type":"summary","title":"Summary","bullets":bullets})
            # Evidence minimal
            if evidence_df is None:
                evidence_df = pd.DataFrame(columns=["trackable","trade","ts_short","x","y","z"])
            cols = ["trackable","trade","ts_short","x","y","z"]
            for c in cols:
                if c not in evidence_df.columns:
                    evidence_df[c] = ""
            rows = evidence_df[cols].head(30).fillna("").astype(str).to_dict(orient="records")
            sections.append({"type":"table","title":"Evidence","data":rows,"headers":cols,"rows_per_page":24})
            meta_text = f"Query: {user_prompt}"
            report_date = pd.Timestamp.utcnow().strftime("%Y%m%d")
            pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
            report = {"title":"Walmart RTLS Position Report (Lite)","meta":meta_text,"sections":sections}
            report = make_lite(report)
            safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))
            print(f"[Download the PDF](file:///{pdf_path.resolve().as_posix()})")
        except Exception:
            print("Error Report:")
            print("Processing interrupted; failed to write minimal report.")
    except Exception as e:
        print("Error Report:")
        msg = str(e).strip() or "Unexpected error."
        print(msg)

if __name__ == "__main__":
    main()