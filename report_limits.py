# report_limits.py
"""
Report-size budgets and light shaping before writing the PDF.

DEFAULTS limit total figures, table rows, and text lines so the writer remains
fast and robust. Use `apply_budgets()` before calling pdf_creation_script.safe_build_pdf().

API
    from report_limits import DEFAULTS, apply_budgets, make_lite

Budgets (tune in report_config.json and load them into the main, or pass as `budgets=`):
- MAX_PAGES: hard cap on perceived pages (approximate by section counts)
- MAX_TABLE_ROWS_TOTAL: total rows across all table sections
- MAX_TEXT_LINES_TOTAL: approximate line budget across summary/narrative/appendix
- MAX_FIGURES: cap the number of figures across all "charts" sections
"""

from __future__ import annotations
from typing import Dict, Any, List

DEFAULTS: Dict[str, int] = {
    "MAX_PAGES": 16,
    "MAX_TABLE_ROWS_TOTAL": 180,
    "MAX_TEXT_LINES_TOTAL": 900,
    "MAX_FIGURES": 15,  # allow richer overlays + supporting charts
}

def _count_text_lines(sections: List[Dict[str, Any]]) -> int:
    lines = 0
    for s in sections:
        t = s.get("type")
        if t == "summary":
            lines += sum(len(str(b).splitlines()) for b in (s.get("bullets") or []))
        elif t == "narrative":
            lines += sum(len(str(p).splitlines()) for p in (s.get("paragraphs") or []))
        elif t == "appendix":
            lines += len(str(s.get("text") or "").splitlines())
    return lines

def _cap_list(xs: list, n: int) -> list:
    return xs[: max(0, n)]

def apply_budgets(report: Dict[str, Any], caps: Dict[str, int] | None = None) -> Dict[str, Any]:
    caps = dict(DEFAULTS if caps is None else caps)
    out = dict(report)
    sections = [dict(s) for s in report.get("sections", [])]

    # Cap figures (across all charts sections)
    figs_total = 0
    for s in sections:
        if s.get("type") == "charts":
            figs = s.get("figures") or []
            keep = max(0, caps["MAX_FIGURES"] - figs_total)
            s["figures"] = _cap_list(figs, keep)
            figs_total += len(s["figures"])

    # Cap total table rows
    rows_left = caps["MAX_TABLE_ROWS_TOTAL"]
    for s in sections:
        if s.get("type") == "table":
            data = s.get("data") or []
            s["data"] = data[: rows_left]
            rows_left = max(0, rows_left - len(s["data"]))

    # Cap text lines (summary + narrative + appendix)
    def trim_text():
        nonlocal sections
        while _count_text_lines(sections) > caps["MAX_TEXT_LINES_TOTAL"]:
            # drop last narrative paragraph, then last summary bullet, then shorten appendix
            for t in ("narrative", "summary", "appendix"):
                for i in range(len(sections) - 1, -1, -1):
                    sec = sections[i]
                    if sec.get("type") != t:
                        continue
                    if t == "narrative":
                        paras = sec.get("paragraphs") or []
                        if paras:
                            sec["paragraphs"] = paras[:-1]; return
                    if t == "summary":
                        bulls = sec.get("bullets") or []
                        if bulls:
                            sec["bullets"] = bulls[:-1]; return
                    if t == "appendix":
                        text = (sec.get("text") or "").splitlines()
                        if text:
                            sec["text"] = "\n".join(text[:-max(1, len(text)//10)]); return
            break
    trim_text()

    out["sections"] = sections
    return out

def make_lite(report: Dict[str, Any]) -> Dict[str, Any]:
    """Aggressive shrink for fallback write: fewer figures + fewer rows + shorter text."""
    caps = {"MAX_FIGURES": 2, "MAX_TABLE_ROWS_TOTAL": 60, "MAX_TEXT_LINES_TOTAL": 300, "MAX_PAGES": 8}
    return apply_budgets(report, caps)
