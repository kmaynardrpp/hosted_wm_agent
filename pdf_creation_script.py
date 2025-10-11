# pdf_creation_script.py
"""
PDF builder (Letter-like) with Redpoint header using matplotlib PdfPages.

Now with:
- Exact page size 8.1 × 11.0 in
- Tiny header meta (top-right, ~5× smaller) with safe wrapping
- Table anti-overflow (column width calc + cell wrapping)
- Minutes formatter (max 6 chars)
- Atomic write helper + fallback ('lite' mode)
- Local filesystem only (NO /mnt/data, NO sandbox): logo auto-resolves from local ROOT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------------------------------------------------------------------
# ROOT resolution (LOCAL ONLY — never use /mnt/data)
# ---------------------------------------------------------------------
def _resolve_root() -> Path:
    root = Path(os.environ.get("INFOZONE_ROOT", "")).resolve()
    if not root or not root.exists():
        root = Path(__file__).resolve().parent
    if not root or not root.exists():
        root = Path.cwd().resolve()
    return root

ROOT = _resolve_root()

# ---------------------------------------------------------------------
# HARD PAGE SIZE & GLOBAL STYLE
# ---------------------------------------------------------------------
PAGE_W_IN, PAGE_H_IN = 8.1, 11.0               # inches (exact)
PAGE_SIZE = (PAGE_W_IN, PAGE_H_IN)

# Physical regions (inches)
MARGIN_L_IN = 0.50
MARGIN_R_IN = 0.50
MARGIN_TOP_IN = 0.35
MARGIN_BOT_IN = 0.50

HEADER_H_IN = 1.00
FOOTER_H_IN = 0.40

DEFAULT_DPI = 120  # crisp but reasonable speed

plt.rcParams.update({
    "figure.dpi": DEFAULT_DPI,
    "savefig.dpi": DEFAULT_DPI,
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    # Embed TrueType → crisp selectable text
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Compress PDF streams to shrink output & speed writes
    "pdf.compression": 9,
})

# Brand colors
REDPOINT_RED = "#E1262D"
INK = "#222222"
MUTED = "#666A73"

# ---------------------------------------------------------------------
# LAYOUT HELPERS
# ---------------------------------------------------------------------
def _inset_rect_in_fig(x_in: float, y_in: float, w_in: float, h_in: float) -> List[float]:
    """Convert an inch-rect into a Matplotlib 'add_axes' rect [x0,y0,w,h] in figure-fractions."""
    return [
        x_in / PAGE_W_IN,
        y_in / PAGE_H_IN,
        w_in / PAGE_W_IN,
        h_in / PAGE_H_IN,
    ]


def _regions() -> Tuple[List[float], List[float], List[float]]:
    """Return (header_rect, content_rect, footer_rect) in figure fractions."""
    header_rect = _inset_rect_in_fig(
        MARGIN_L_IN,
        PAGE_H_IN - (MARGIN_TOP_IN + HEADER_H_IN),
        PAGE_W_IN - MARGIN_L_IN - MARGIN_R_IN,
        HEADER_H_IN,
    )
    footer_rect = _inset_rect_in_fig(
        MARGIN_L_IN,
        MARGIN_BOT_IN,
        PAGE_W_IN - MARGIN_L_IN - MARGIN_R_IN,
        FOOTER_H_IN,
    )
    content_rect = _inset_rect_in_fig(
        MARGIN_L_IN,
        MARGIN_BOT_IN + FOOTER_H_IN,
        PAGE_W_IN - MARGIN_L_IN - MARGIN_R_IN,
        PAGE_H_IN - (MARGIN_TOP_IN + HEADER_H_IN + MARGIN_BOT_IN + FOOTER_H_IN),
    )
    return header_rect, content_rect, footer_rect

# ---------------------------------------------------------------------
# HEADER / FOOTER
# ---------------------------------------------------------------------
def _load_logo(logo_path: Optional[str]) -> Any:
    """
    Load logo image. If logo_path is None or missing, auto-resolve from ROOT/redpoint_logo.png.
    Returns ndarray or None.
    """
    # Preferred explicit path
    if logo_path:
        p = Path(logo_path)
        if p.exists():
            try:
                return plt.imread(str(p))
            except Exception:
                return None
    # Local fallback
    p = ROOT / "redpoint_logo.png"
    if p.exists():
        try:
            return plt.imread(str(p))
        except Exception:
            return None
    return None

def _wrap_meta_text(text: str, max_chars: int = 55) -> str:
    import textwrap
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False, replace_whitespace=False))

def _draw_header(fig: plt.Figure, header_ax, title_text: str, meta_text: Optional[str], logo_img) -> None:
    header_ax.axis("off")

    # Logo (≈1.4" width, preserve AR)
    if logo_img is not None:
        logo_w_in = 1.4
        ratio = logo_img.shape[0] / max(1, logo_img.shape[1])
        logo_h_in = logo_w_in * ratio
        parent = header_ax.get_position()
        hw, hh = parent.width, parent.height
        ax_logo = fig.add_axes([
            parent.x0 + 0.00 * hw,
            parent.y0 + (1 - min(0.9, (logo_h_in / HEADER_H_IN)) - 0.05) * hh,
            (logo_w_in / (PAGE_W_IN - MARGIN_L_IN - MARGIN_R_IN)) * hw,
            min(0.9, (logo_h_in / HEADER_H_IN)) * hh
        ])
        ax_logo.imshow(logo_img); ax_logo.set_aspect("equal"); ax_logo.axis("off")

    # Title & tiny meta
    header_ax.text(0.22, 0.70, title_text or "", fontsize=18, fontweight="bold",
                   color=INK, va="center", ha="left", transform=header_ax.transAxes)

    if meta_text:
        tiny = _wrap_meta_text(meta_text, max_chars=55)
        header_ax.text(0.995, 0.78, tiny, fontsize=5, color=MUTED, va="top", ha="right",
                       linespacing=0.85, transform=header_ax.transAxes)

    # Red underline
    header_ax.plot([0.00, 1.00], [0.10, 0.10], color=REDPOINT_RED, linewidth=4, clip_on=False)

def _draw_footer(footer_ax, page_num: int, total_pages: int) -> None:
    footer_ax.axis("off")
    footer_ax.text(0.5, 0.5, f"Page {page_num}/{total_pages}",
                   fontsize=9, color=MUTED, va="center", ha="center", transform=footer_ax.transAxes)

# ---------------------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------------------
def _wrap_text(text: str, width: int = 92) -> List[str]:
    import textwrap
    return textwrap.wrap(text or "", width=width, replace_whitespace=False, drop_whitespace=False)

def add_paragraph(fig: plt.Figure, content_ax, text: str, width: int = 92, fontsize: int = 10, top_y: float = 0.96) -> float:
    y = top_y
    for line in _wrap_text(text, width=width):
        content_ax.text(0.02, y, line, fontsize=fontsize, va="top", color=INK, transform=content_ax.transAxes)
        y -= 0.035
    return y

def add_bullets(fig: plt.Figure, content_ax, items: Sequence[str], width: int = 92, fontsize: int = 10, top_y: float = 0.96) -> float:
    y = top_y
    for it in (items or []):
        lines = _wrap_text(it, width=width)
        if not lines:
            continue
        content_ax.text(0.02, y, f"• {lines[0]}", fontsize=fontsize, va="top", color=INK, transform=content_ax.transAxes)
        for ln in lines[1:]:
            y -= 0.035
            content_ax.text(0.04, y, ln, fontsize=fontsize, va="top", color=INK, transform=content_ax.transAxes)
        y -= 0.050
    return y

# ---------------------------------------------------------------------
# TABLE HELPERS (wrapping + minutes formatter)
# ---------------------------------------------------------------------
def _as_table_data(
    data: Union[pd.DataFrame, List[Dict[str, Any]], List[List[Any]]],
    headers: Optional[List[str]] = None
) -> Tuple[List[str], List[List[str]]]:
    if isinstance(data, pd.DataFrame):
        cols = list(data.columns if headers is None else headers)
        rows = data[cols].astype(str).values.tolist()
        return cols, rows
    if isinstance(data, list) and data and isinstance(data[0], dict):
        cols = list(data[0].keys()) if headers is None else headers
        rows = [[str(r.get(c, "")) for c in cols] for r in data]
        return cols, rows
    rows = [[str(x) for x in r] for r in (data or [])]
    if headers is None and rows:
        headers = [f"col_{i}" for i in range(len(rows[0]))]
    return headers or [], rows

# Minutes formatter: ensure ≤ 6 chars (e.g., '123.4', '999999', '12.34')
def _fmt_minutes_str(val: Any) -> str:
    try:
        f = float(str(val))
        if f < 0:
            f = 0.0
        s = f"{f:.2f}"
        if len(s) > 6:
            s = f"{f:.0f}"
        if len(s) > 6:
            # final hard cap; avoid scientific notation
            s = s[:6]
            if s.endswith("."):
                s = s[:-1]
        return s
    except Exception:
        s = str(val)
        return s[:6] if len(s) > 6 else s

# Apply minutes formatter to any minutes-like column
def _format_minutes_columns(cols: List[str], rows: List[List[str]]) -> List[List[str]]:
    minutes_idx = [i for i,c in enumerate(cols) if "duration_min" in c.lower() or c.lower().endswith("minutes")]
    if not minutes_idx:
        return rows
    out = []
    for r in rows:
        r2 = list(r)
        for i in minutes_idx:
            if i < len(r2):
                r2[i] = _fmt_minutes_str(r2[i])
        out.append(r2)
    return out

# ---- Table cell wrapping to prevent overflow ----
CHAR_PER_IN = 10.0        # char-per-inch estimate at ~8–10pt
MAX_CELL_LINES = 3
COL_GUTTER_FRAC = 0.98

def _wrap_table_cells(cols: List[str], rows: List[List[str]], content_w_in: float) -> Tuple[List[float], List[List[str]]]:
    import textwrap
    n = max(1, len(cols))
    total_w_in = content_w_in * COL_GUTTER_FRAC
    col_w_in = [total_w_in / n] * n
    col_widths_frac = [w / content_w_in for w in col_w_in]

    wrapped = []
    for r in rows:
        new_r = []
        for i, cell in enumerate(r[:n] + ([""] * max(0, n - len(r)))):
            cell = "" if cell is None else str(cell)
            max_chars = max(6, int(col_w_in[i] * CHAR_PER_IN))
            wrapped_text = textwrap.fill(cell, width=max_chars, break_long_words=False, replace_whitespace=False)
            lines = wrapped_text.splitlines()
            if len(lines) > MAX_CELL_LINES:
                lines = lines[:MAX_CELL_LINES]
                if not lines[-1].endswith("…"):
                    if len(lines[-1]) >= 1:
                        lines[-1] = (lines[-1][:-1] + "…") if len(lines[-1]) > 1 else "…"
            new_r.append("\n".join(lines))
        wrapped.append(new_r)
    return col_widths_frac, wrapped

def add_table_pages(
    pdf: PdfPages,
    title: str,
    data: Union[pd.DataFrame, List[Dict[str, Any]], List[List[Any]]],
    headers: Optional[List[str]] = None,
    logo_img=None,
    meta_text: Optional[str] = None,
    rows_per_page: int = 24,
    start_page_number: int = 1,
    total_pages_hint: Optional[int] = None,
    dpi: Optional[int] = None
) -> int:
    cols, rows = _as_table_data(data, headers=headers)
    # Format minutes before wrapping
    rows = _format_minutes_columns(cols, rows)

    pages = max(1, (len(rows) + rows_per_page - 1) // rows_per_page)
    content_w_in = PAGE_W_IN - MARGIN_L_IN - MARGIN_R_IN
    col_widths_frac, wrapped_rows = _wrap_table_cells(cols, rows, content_w_in)

    for p in range(pages):
        fig = plt.figure(figsize=PAGE_SIZE, dpi=(dpi or DEFAULT_DPI))
        header_rect, content_rect, footer_rect = _regions()
        header_ax = fig.add_axes(header_rect)
        content_ax = fig.add_axes(content_rect)
        footer_ax = fig.add_axes(footer_rect)

        _draw_header(fig, header_ax, title, meta_text, logo_img)
        _draw_footer(footer_ax, start_page_number + p, total_pages_hint or (start_page_number + pages - 1))

        content_ax.axis("off")
        chunk = wrapped_rows[p * rows_per_page:(p + 1) * rows_per_page]
        tbl = content_ax.table(
            cellText=chunk,
            colLabels=cols,
            colWidths=col_widths_frac,
            loc="upper left",
            cellLoc="left"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)      # ~10% smaller
        tbl.scale(1.00, 1.10)    # slightly tighter rows

        pdf.savefig(fig)  # exact 8.1 × 11
        plt.close(fig)

    return pages

# ---------------------------------------------------------------------
# CHART COMPOSITION
# ---------------------------------------------------------------------
def _draw_chart_page(
    pdf: PdfPages,
    title: str,
    fig_src: plt.Figure,
    logo_img,
    meta_text: Optional[str],
    page_number: int,
    total_pages: int,
    dpi: Optional[int]
) -> None:
    fig_src.set_dpi(dpi or DEFAULT_DPI)
    canvas = FigureCanvas(fig_src); canvas.draw()
    w, h = fig_src.canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))

    fig = plt.figure(figsize=PAGE_SIZE, dpi=(dpi or DEFAULT_DPI))
    header_rect, content_rect, footer_rect = _regions()
    header_ax = fig.add_axes(header_rect)
    content_ax = fig.add_axes(content_rect)
    footer_ax = fig.add_axes(footer_rect)

    _draw_header(fig, header_ax, title, meta_text, logo_img)
    _draw_footer(footer_ax, page_number, total_pages)

    content_ax.imshow(buf); content_ax.set_aspect("auto"); content_ax.axis("off")
    pdf.savefig(fig); plt.close(fig); plt.close(fig_src)

# ---------------------------------------------------------------------
# CORE BUILDER
# ---------------------------------------------------------------------
def build_pdf(
    report: Dict[str, Any],
    output_path: str,
    logo_path: Optional[str] = None,
    dpi: Optional[int] = DEFAULT_DPI
) -> str:
    """
    Build the full PDF. Paths must be local (no sandbox). If logo_path is None,
    we attempt ROOT/redpoint_logo.png automatically.
    """
    out_dir = Path(output_path).resolve().parent
    os.makedirs(str(out_dir), exist_ok=True)
    logo_img = _load_logo(logo_path)

    # Estimate pages
    total_estimate = 0
    for sec in report.get("sections", []):
        t = sec.get("type")
        if t in ("summary", "narrative", "recommendations", "appendix"):
            total_estimate += 1
        elif t == "table":
            data = sec.get("data", [])
            cols, rows = _as_table_data(data, headers=sec.get("headers"))
            rpp = int(sec.get("rows_per_page") or 24)
            total_estimate += max(1, (len(rows) + rpp - 1) // rpp)
        elif t == "chart":
            total_estimate += 1
        elif t == "charts":
            figs = sec.get("figures") or []
            total_estimate += max(1, len(figs))
        else:
            total_estimate += 1

    page_no = 1
    title = report.get("title") or "Report"
    meta_text = report.get("meta") or ""

    with PdfPages(output_path) as pdf:
        for sec in report.get("sections", []):
            stype = sec.get("type")
            stitle = sec.get("title") or title

            if stype == "summary":
                fig = plt.figure(figsize=PAGE_SIZE, dpi=(dpi or DEFAULT_DPI))
                header_rect, content_rect, footer_rect = _regions()
                header_ax = fig.add_axes(header_rect)
                content_ax = fig.add_axes(content_rect)
                footer_ax = fig.add_axes(footer_rect)
                _draw_header(fig, header_ax, stitle, meta_text, logo_img)
                _draw_footer(footer_ax, page_no, total_estimate)
                content_ax.axis("off")
                add_bullets(fig, content_ax, sec.get("bullets") or [], width=92, fontsize=10, top_y=0.96)
                pdf.savefig(fig); plt.close(fig); page_no += 1

            elif stype == "narrative":
                fig = plt.figure(figsize=PAGE_SIZE, dpi=(dpi or DEFAULT_DPI))
                header_rect, content_rect, footer_rect = _regions()
                header_ax = fig.add_axes(header_rect)
                content_ax = fig.add_axes(content_rect)
                footer_ax = fig.add_axes(footer_rect)
                _draw_header(fig, header_ax, stitle, meta_text, logo_img)
                _draw_footer(footer_ax, page_no, total_estimate)
                content_ax.axis("off")
                y = 0.96
                for para in (sec.get("paragraphs") or []):
                    y = add_paragraph(fig, content_ax, para, width=92, fontsize=10, top_y=y) - 0.02
                pdf.savefig(fig); plt.close(fig); page_no += 1

            elif stype == "recommendations":
                fig = plt.figure(figsize=PAGE_SIZE, dpi=(dpi or DEFAULT_DPI))
                header_rect, content_rect, footer_rect = _regions()
                header_ax = fig.add_axes(header_rect)
                content_ax = fig.add_axes(content_rect)
                footer_ax = fig.add_axes(footer_rect)
                _draw_header(fig, header_ax, stitle, meta_text, logo_img)
                _draw_footer(footer_ax, page_no, total_estimate)
                content_ax.axis("off")
                add_bullets(fig, content_ax, sec.get("bullets") or [], width=92, fontsize=10, top_y=0.96)
                pdf.savefig(fig); plt.close(fig); page_no += 1

            elif stype == "appendix":
                fig = plt.figure(figsize=PAGE_SIZE, dpi=(dpi or DEFAULT_DPI))
                header_rect, content_rect, footer_rect = _regions()
                header_ax = fig.add_axes(header_rect)
                content_ax = fig.add_axes(content_rect)
                footer_ax = fig.add_axes(footer_rect)
                _draw_header(fig, header_ax, stitle, meta_text, logo_img)
                _draw_footer(footer_ax, page_no, total_estimate)
                content_ax.axis("off")
                text = sec.get("text") or ""
                mono = bool(sec.get("mono"))
                if mono:
                    content_ax.text(0.02, 0.96, text, family="monospace", fontsize=9, va="top", color=INK, transform=content_ax.transAxes)
                else:
                    _ = add_paragraph(fig, content_ax, text, width=92, fontsize=10, top_y=0.96)
                pdf.savefig(fig); plt.close(fig); page_no += 1

            elif stype == "table":
                pages = add_table_pages(
                    pdf, stitle, sec.get("data", []),
                    headers=sec.get("headers"), logo_img=logo_img, meta_text=meta_text,
                    rows_per_page=int(sec.get("rows_per_page") or 24),
                    start_page_number=page_no, total_pages_hint=total_estimate, dpi=dpi
                )
                page_no += pages

            elif stype == "chart":
                fig_src = sec.get("figure")
                if fig_src is None:
                    fig_src = plt.figure(); ax = fig_src.add_subplot(111)
                    ax.axis("off"); ax.text(0.5,0.5,"No chart provided", ha="center", va="center")
                _draw_chart_page(pdf, stitle, fig_src, logo_img, meta_text, page_no, total_estimate, dpi=dpi)
                page_no += 1

            elif stype == "charts":
                figs = sec.get("figures") or []
                if not figs:
                    fig_src = plt.figure(); ax = fig_src.add_subplot(111)
                    ax.axis("off"); ax.text(0.5,0.5,"No charts provided", ha="center", va="center")
                    _draw_chart_page(pdf, stitle, fig_src, logo_img, meta_text, page_no, total_estimate, dpi=dpi)
                    page_no += 1
                else:
                    for fig_src in figs:
                        _draw_chart_page(pdf, stitle, fig_src, logo_img, meta_text, page_no, total_estimate, dpi=dpi)
                        page_no += 1

            else:
                fig = plt.figure(figsize=PAGE_SIZE, dpi=(dpi or DEFAULT_DPI))
                header_rect, content_rect, footer_rect = _regions()
                header_ax = fig.add_axes(header_rect)
                content_ax = fig.add_axes(content_rect)
                footer_ax = fig.add_axes(footer_rect)
                _draw_header(fig, header_ax, stitle, meta_text, logo_img)
                _draw_footer(footer_ax, page_no, total_estimate)
                content_ax.axis("off")
                _ = add_paragraph(fig, content_ax, sec.get("text") or "", width=92, fontsize=10, top_y=0.96)
                pdf.savefig(fig); plt.close(fig); page_no += 1

    return output_path

# ---------------------------------------------------------------------
# ATOMIC WRITE WRAPPER + FALLBACK
# ---------------------------------------------------------------------
def safe_build_pdf(report: Dict[str, Any],
                   output_path: str,
                   logo_path: Optional[str] = None,
                   dpi: Optional[int] = DEFAULT_DPI,
                   budgets: Optional[Dict[str, Any]] = None) -> str:
    """
    Writes to temp first; enforces budgets; retries once in 'lite' mode on failure.
    All paths are local; report_limits is imported locally (no sandbox path).
    """
    from report_limits import apply_budgets, make_lite

    tmp = output_path + ".tmp"  # keep as string concat; output_path is str

    # Enforce budgets up front
    capped = apply_budgets(report, budgets)

    # First attempt
    try:
        build_pdf(capped, output_path=tmp, logo_path=logo_path, dpi=dpi)
        if not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
            raise RuntimeError("Temp PDF is missing or empty after write.")
        os.replace(tmp, output_path)
        return output_path
    except Exception:
        # Fallback: lite mode
        try:
            lite = make_lite(capped)
            low_dpi = min(96, int(dpi or DEFAULT_DPI))
            build_pdf(lite, output_path=tmp, logo_path=logo_path, dpi=low_dpi)
            if not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
                raise RuntimeError("Temp PDF is missing or empty after lite write.")
            os.replace(tmp, output_path)
            return output_path
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
