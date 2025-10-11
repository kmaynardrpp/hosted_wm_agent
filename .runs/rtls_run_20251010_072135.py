import sys, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from report_limits import apply_budgets
from chart_policy import choose_charts
import json

# project root injected by launcher; fallback to script’s folder or parent
ROOT = Path(os.environ.get("INFOZONE_ROOT", ""))
if not ROOT or not (ROOT / "guidelines.txt").exists():
    script_dir = Path(__file__).resolve().parent
    ROOT = script_dir if (script_dir / "guidelines.txt").exists() else script_dir.parent

# make "import extractor", etc. work when running from .runs/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Read guidelines
GUIDELINES = ROOT / "guidelines.txt"

# robust text read: Windows-safe UTF-8
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

# CLI args
USER_PROMPT = sys.argv[1]
csv_paths = sys.argv[2:]

# Validate input
if not csv_paths:
    print("Error Report: No CSV paths provided")
    sys.exit(1)

# Prepare output directory
out_dir = Path(csv_paths[0]).parent

# Section to collect report data
sections = []

for csv_path in csv_paths:
    try:
        raw = extract_tracks(csv_path)
        df = pd.DataFrame(raw.get("rows", []))
        audit = raw.get("audit", {})
        
        # Duplicate-name guard
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Timestamp canon
        ts_src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
        df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

        # Only keeping needed numeric columns for operations
        df["x"] = pd.to_numeric(df.get("x"), errors="coerce")
        df["y"] = pd.to_numeric(df.get("y"), errors="coerce")
        
        # Filter out invalid rows
        df.dropna(subset=["ts_utc", "x", "y"], inplace=True)

        # Generate evidence table
        cols = ["trackable", "trade", "ts_iso", "x", "y", "z"]
        rows = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
        sections.append({"type": "table", "title": "Evidence", "data": rows, "headers": cols, "rows_per_page": 24})

        # Generate charts
        figs = choose_charts(df, floorplans_path=str(ROOT / "floorplans.json"), floorplan_image_path=str(ROOT / "floorplan.jpeg"), user_query=USER_PROMPT)
        
        # Save figures as PNGs
        png_paths = []
        for i, fig in enumerate(figs):
            png_path = out_dir / f"info_zone_report_{csv_path.stem}_plot{i:02d}.png"
            fig.savefig(png_path, dpi=120)
            png_paths.append(png_path)
        
        # Add chart section
        sections.append({"type": "charts", "title": "Figures", "figures": figs})

    except Exception as e:
        print(f"Error Report: {str(e)}")
        sys.exit(1)

# Final report for PDF
report = {"title": "Walmart Renovation RTLS Report", "sections": sections}
report = apply_budgets(report)

# PDF output path
pdf_path = out_dir / f"info_zone_report_{Path(csv_paths[0]).stem}.pdf"
safe_build_pdf(report, str(pdf_path), logo_path=str(ROOT / "redpoint_logo.png"))

# Output success messages
print(f"[Download the PDF](file:///{str(pdf_path).replace('\\', '/')})")
for i, png in enumerate(png_paths, 1):
    print(f"[Download Plot {i}](file:///{str(png).replace('\\', '/')})")