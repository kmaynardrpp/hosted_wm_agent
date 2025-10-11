# chart_policy.py
"""
Intent-driven chart selection for the new project with **floorplan overlays** as first-class visuals.

UPDATE:
- Floorplan raster is now ALWAYS named one of:
    /mnt/data/floorplan.png
    /mnt/data/floorplan.jpg
    /mnt/data/floorplan.jpeg
  We no longer scan arbitrary images (prevents accidental use of redpoint_logo.png).

Key behavior
- Prefer a floorplan scatter overlay (world mm) when x/y and floorplan are available.
- Optionally overlay translucent zone polygons from zones.json.
- Then (if useful) add supporting charts: bar/pie/line/box; **no quota**—return 1..N figures.
- All charts are matplotlib only; one chart per figure (small-multiples allowed as a single figure).

API
    from chart_policy import choose_charts
    figs = choose_charts(
        df,                      # pandas DataFrame (trackable, trackable_uid, mac, ts, x, y, z, trade…)
        hourly_df=None,          # hourly aggregation (optional)
        trade_df=None,           # per-trade summary (optional)
        user_query="",           # the plain user question for intent
        floorplans_path="/mnt/data/floorplans.json",
        floorplan_image_path=None,  # optional override; else /mnt/data/floorplan.(png|jpg|jpeg)
        zones_path="/mnt/data/zones.json",
        config=None              # dict to override DEFAULTS
    )
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

try:
    # optional zones helper (if present)
    from zones_process import load_zones
except Exception:  # pragma: no cover
    def load_zones(*args, **kwargs):  # fallback no-op
        return []

DEFAULTS: Dict[str, object] = {
    # floorplan & overlay
    "prefer_floorplan": True,
    "floorplan_margin": 0.10,         # 10% margin
    "overlay_point_size": 8,
    "overlay_alpha": 0.85,            # scatter alpha
    "overlay_color_by": "trade",      # "trade" | "trackable" | "none"
    "overlay_subsample": 20000,       # max points on overlay (decimate if more)
    "draw_trails": False,             # not heavy by default
    "trail_seconds": 900,             # last N seconds to plot as trail if draw_trails=True
    # zones
    "draw_zones": True,
    "zone_face_alpha": 0.20,
    "zone_edge_alpha": 0.65,
    # supporting figs
    "top_n": 10,
    "pie_max_trades": 8,
    "pie_max_single_share": 0.90,
    "line_min_points": 2,
    # grids/small-multiples
    "small_multiples_cols": 2,
    # figure limits
    "max_figures": 6,
    # figsize presets
    "figsize_overlay": (9, 7),
    "figsize_bar": (7, 5),
    "figsize_line": (7, 5),
    "figsize_pie": (5, 5),
    "figsize_box": (7, 5),
}

# --------------------------- Floorplan helpers ---------------------------

FLOORPLAN_CANDIDATES = [
    "/mnt/data/floorplan.png",
    "/mnt/data/floorplan.jpg",
    "/mnt/data/floorplan.jpeg",
]

def _find_floorplan_image(image_path: Optional[str]) -> Optional[str]:
    """
    Returns the explicit floorplan path:
      /mnt/data/floorplan.png|jpg|jpeg
    If an explicit override path is provided and exists, use it.
    We DO NOT scan arbitrary images to avoid picking redpoint_logo.png.
    """
    if image_path and os.path.exists(image_path):
        return image_path
    for p in FLOORPLAN_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def _load_floorplan(floorplans_path: str, override_image_path: Optional[str]=None) -> Optional[Dict[str, object]]:
    """
    Load extent from floorplans.json and raster from fixed floorplan filename(s).
    Extent calculation per provided spec (world mm rectangle).
    """
    if not os.path.exists(floorplans_path):
        return None
    try:
        with open(floorplans_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fp = (data.get("floorplans") or data.get("plans") or data or [None])
        if isinstance(fp, list):
            fp = fp[0]
        if not fp:
            return None

        width  = float(fp.get("width", 0))
        height = float(fp.get("height", 0))
        x_c    = float(fp.get("image_offset_x", 0))
        y_c    = float(fp.get("image_offset_y", 0))
        image_scale = float(fp.get("image_scale", 0))  # typically meters/pixel

        scale = image_scale * 100.0  # mm/pixel
        x_min = (x_c - width/2.0)  * scale
        x_max = (x_c + width/2.0)  * scale
        y_min = (y_c - height/2.0) * scale
        y_max = (y_c + height/2.0) * scale

        img_path = _find_floorplan_image(override_image_path)
        if not img_path:
            return None
        img = plt.imread(img_path)
        return {"extent": (x_min, x_max, y_min, y_max), "image": img}
    except Exception:
        return None

def _colors_for_categories(categories: List[str]) -> Dict[str, str]:
    base = plt.cm.get_cmap("tab10")
    uniq = list(dict.fromkeys(categories))
    return {c: base(i % 10) for i, c in enumerate(uniq)}

def _make_floorplan_overlay(df: pd.DataFrame,
                            fp: Dict[str, object],
                            cfg: Dict[str, object],
                            zones_list: Optional[List[Dict[str, object]]] = None) -> Optional[plt.Figure]:
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        return None
    x = pd.to_numeric(df["x"], errors="coerce"); y = pd.to_numeric(df["y"], errors="coerce")
    use = df.loc[x.notna() & y.notna(), :].copy()
    if use.empty:
        return None

    # subsample if necessary
    max_pts = int(cfg["overlay_subsample"])
    if len(use) > max_pts:
        idx = np.linspace(0, len(use) - 1, max_pts).astype(int)
        use = use.iloc[idx]

    # color mapping
    color_by = str(cfg["overlay_color_by"] or "trade")
    if color_by not in use.columns:
        color_by = "none"
    if color_by != "none":
        palette = _colors_for_categories(use[color_by].astype(str).tolist())

    # figure
    fig = plt.figure(figsize=cfg["figsize_overlay"])  # type: ignore
    ax = fig.add_subplot(111)
    x_min, x_max, y_min, y_max = fp["extent"]  # type: ignore

    # draw image (origin='upper' matches the flipped Y rule)
    ax.imshow(fp["image"], extent=[x_min, x_max, y_min, y_max], origin="upper")

    # zones overlay
    if zones_list and bool(cfg["draw_zones"]):
        for z in zones_list:
            poly: np.ndarray = z["polygon"]  # mm
            patch = Polygon(poly, closed=True,
                            facecolor=(0,0,0,cfg["zone_face_alpha"]),
                            edgecolor=(0,0,0,cfg["zone_edge_alpha"]), linewidth=1.0)
            ax.add_patch(patch)

    # points overlay
    if color_by == "none":
        ax.scatter(use["x"], use["y"], s=float(cfg["overlay_point_size"]),
                   alpha=float(cfg["overlay_alpha"]))
    else:
        for cat, g in use.groupby(color_by):
            ax.scatter(g["x"], g["y"], s=float(cfg["overlay_point_size"]),
                       color=palette.get(cat), alpha=float(cfg["overlay_alpha"]),
                       label=str(cat))
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= 12:
            ax.legend(loc="upper left", frameon=True, fontsize=8)

    # axis limits + margin
    mx = float(cfg["floorplan_margin"])
    xr = (x_max - x_min); yr = (y_max - y_min)
    ax.set_xlim(x_min - mx * xr, x_max + mx * xr)
    ax.set_ylim(y_min - mx * yr, y_max + mx * yr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title("Floorplan Overlay")
    fig.tight_layout()
    return fig

# --------------------------- Supporting charts ---------------------------

def _bar(df: pd.DataFrame, cat: str, value: str, top_n: int, title: str, cfg: Dict[str, object]) -> Optional[plt.Figure]:
    try:
        g = (df.groupby(cat)[value]
               .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
               .sort_values(ascending=False)
               .head(top_n))
        if g.empty:
            return None
        fig = plt.figure(figsize=cfg["figsize_bar"])  # type: ignore
        ax = fig.add_subplot(111)
        ax.bar(g.index.astype(str), g.values)
        ax.set_title(title); ax.set_xlabel(cat.title()); ax.set_ylabel(value.replace("_"," ").title())
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None

def _pie_or_bar_trade(trade_df: pd.DataFrame, tcol: str, cfg: Dict[str, object]) -> Optional[plt.Figure]:
    vals = pd.to_numeric(trade_df.get("total_dwell_min"), errors="coerce").fillna(0)
    if vals.sum() <= 0:
        return None
    n = len(trade_df)
    share_ok = (float(vals.max())/float(vals.sum())) <= float(cfg["pie_max_single_share"])
    if n <= int(cfg["pie_max_trades"]) and share_ok:
        fig = plt.figure(figsize=cfg["figsize_pie"])  # type: ignore
        ax = fig.add_subplot(111)
        ax.pie(vals.values, labels=trade_df[tcol].astype(str).values, autopct="%1.1f%%", startangle=90)
        ax.axis("equal"); ax.set_title("Trade Share (Dwell)"); fig.tight_layout(); return fig
    return _bar(trade_df, tcol, "total_dwell_min", int(cfg["top_n"]), "Total Dwell by Trade (Top)", cfg)

def _line_hourly(hourly_df: pd.DataFrame, cfg: Dict[str, object]) -> Optional[plt.Figure]:
    if hourly_df is None or hourly_df.empty:
        return None
    if "hour" not in hourly_df.columns or "person_minutes" not in hourly_df.columns:
        return None
    if hourly_df["hour"].nunique() < int(cfg["line_min_points"]):
        return None
    if np.nanstd(pd.to_numeric(hourly_df["person_minutes"], errors="coerce").fillna(0).values) == 0:
        return None
    fig = plt.figure(figsize=cfg["figsize_line"])  # type: ignore
    ax = fig.add_subplot(111)
    try:
        ax.plot(hourly_df["hour"].dt.tz_convert("UTC"), pd.to_numeric(hourly_df["person_minutes"], errors="coerce").fillna(0))
    except Exception:
        ax.plot(pd.to_datetime(hourly_df["hour"], errors="coerce"), pd.to_numeric(hourly_df["person_minutes"], errors="coerce").fillna(0))
    ax.set_title("Hourly Total Person-Minutes (UTC)")
    ax.set_xlabel("Hour"); ax.set_ylabel("Person-Minutes")
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig

# --------------------------- Intent parsing ---------------------------

def _intent(user_query: str) -> Dict[str, bool]:
    q = (user_query or "").lower()
    return {
        "map":  any(k in q for k in ["map","floor","floorplan","position","where","layout","coordinate","x y"]),
        "peaks": any(k in q for k in ["peak","busiest","when","schedule","hour","rush"]),
        "mix":   any(k in q for k in ["mix","share","split","composition","who","staff","trade"]),
        "trend": any(k in q for k in ["trend","change","compare","delta","vs","before","after"]),
        "dur":   any(k in q for k in ["duration","dwell","quality","long","short","spread","distribution"]),
    }

# --------------------------- Main selection ---------------------------

def choose_charts(
    df: pd.DataFrame,
    hourly_df: Optional[pd.DataFrame] = None,
    trade_df: Optional[pd.DataFrame] = None,
    user_query: str = "",
    floorplans_path: str = "/mnt/data/floorplans.json",
    floorplan_image_path: Optional[str] = None,
    zones_path: str = "/mnt/data/zones.json",
    config: Optional[Dict[str, object]] = None
) -> List[plt.Figure]:

    cfg = dict(DEFAULTS)
    if config:
        cfg.update({k: v for k, v in config.items() if k in cfg})
    figs: List[plt.Figure] = []
    intent = _intent(user_query)

    # 1) Floorplan overlay (preferred)
    overlay_possible = "x" in df.columns and "y" in df.columns and not df.empty
    if cfg["prefer_floorplan"] and overlay_possible:
        fp = _load_floorplan(floorplans_path, override_image_path=floorplan_image_path)
        if fp is not None:
            zones_list = load_zones(zones_path, only_active=True) if bool(cfg["draw_zones"]) else []
            ov = _make_floorplan_overlay(df, fp, cfg, zones_list=zones_list)
            if ov is not None:
                figs.append(ov)
                if len(figs) >= int(cfg["max_figures"]):
                    return figs

    # 2) Supporting charts based on intent (no quotas)
    tcol = "trade" if "trade" in df.columns else None
    if intent["mix"] and trade_df is not None and len(figs) < int(cfg["max_figures"]):
        pb = _pie_or_bar_trade(trade_df, tcol or "trade", cfg)
        if pb is not None: figs.append(pb)

    if intent["peaks"] and hourly_df is not None and len(figs) < int(cfg["max_figures"]):
        l = _line_hourly(hourly_df, cfg)
        if l is not None: figs.append(l)

    # If user asked "where" but overlay unavailable, fallback to zone/trade bars
    if intent["map"] and (not overlay_possible or len(figs) == 0) and len(figs) < int(cfg["max_figures"]):
        zcol = "zone_name" if "zone_name" in df.columns else None
        if zcol:
            bz = _bar(df, zcol, "x", int(cfg["top_n"]), "Zone Activity (count proxy)", cfg)
            if bz is not None: figs.append(bz)

    # Duration quality
    if intent["dur"] and len(figs) < int(cfg["max_figures"]):
        col = "duration_min" if "duration_min" in df.columns else None
        if col:
            v = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(v) >= 20 and np.nanstd(v) > 0:
                fig = plt.figure(figsize=cfg["figsize_box"])  # type: ignore
                ax = fig.add_subplot(111)
                ax.hist(v, bins=20)
                ax.set_title("Dwell Distribution (minutes)")
                ax.set_xlabel("Minutes"); ax.set_ylabel("Count")
                ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                fig.tight_layout()
                figs.append(fig)

    # If nothing selected, add one high-signal bar by zone or trade
    if not figs:
        zcol = "zone_name" if "zone_name" in df.columns else None
        if zcol:
            b = _bar(df, zcol, "x", int(cfg["top_n"]), "Top Zones (count proxy)", cfg)
            if b is not None: figs.append(b)
        elif tcol and trade_df is not None:
            b = _pie_or_bar_trade(trade_df, tcol, cfg)
            if b is not None: figs.append(b)

    return figs[: int(cfg["max_figures"])]
