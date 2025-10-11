import sys, os
from pathlib import Path

# --- locate project root ---
ROOT = Path(os.environ.get('INFOZONE_ROOT', ''))
if not ROOT or not (ROOT / 'guidelines.txt').exists():
    script_dir = Path(__file__).resolve().parent
    ROOT = script_dir if (script_dir / 'guidelines.txt').exists() else script_dir.parent

# --- make local imports work (extractor, pdf_creation_script, etc.) ---
sys.path.insert(0, str(ROOT))

# helper file paths (never use /mnt/data)
GUIDELINES = ROOT / 'guidelines.txt'
CONTEXT    = ROOT / 'context.txt'
FLOORJSON  = ROOT / 'floorplans.json'
LOGO       = ROOT / 'redpoint_logo.png'
CONFIG     = ROOT / 'report_config.json'
LIMITS_PY  = ROOT / 'report_limits.py'
ZONES_JSON = ROOT / 'zones.json'

from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
import pandas as pd
import json
import matplotlib.pyplot as plt
from chart_policy import choose_charts
from report_limits import apply_budgets

def read_text(p): 
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Walmart renovation RTLS position data.')
    parser.add_argument('user_prompt', type=str, help='User prompt for analysis')
    parser.add_argument('csv_paths', nargs='+', help='Paths to CSV files')
    args = parser.parse_args()

    csv_path = args.csv_paths[0]
    out_dir = Path(csv_path).parent
    report_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Load data
    raw = extract_tracks(csv_path)
    df = pd.DataFrame(raw.get("rows", []))
    audit = raw.get("audit", {})

    # Duplicate-name guard
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Create ts_utc
    ts_src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
    df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

    # Filter for valid data
    df = df.dropna(subset=["ts_utc", "x", "y"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    # Prepare evidence table
    cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
    rows = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
    sections = [{"type": "table", "title": "Evidence", "data": rows, "headers": cols, "rows_per_page": 24}]

    # Load floorplan
    with open(FLOORJSON, 'r', encoding='utf-8') as f:
        floorplan_data = json.load(f)
    
    # Choose charts
    figs = choose_charts(df, floorplans_path=FLOORJSON, floorplan_image_path=str(ROOT / "floorplan.jpeg"), user_query=args.user_prompt)

    # Save figures
    png_paths = []
    for i, fig in enumerate(figs):
        png_path = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
        fig.savefig(png_path, dpi=120)
        png_paths.append(png_path)

    # Build PDF report
    report = {"sections": sections}
    report = apply_budgets(report)
    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
    safe_build_pdf(report, pdf_path, logo_path=str(LOGO))

    # Print links
    print(f"[Download the PDF](file://{pdf_path})")
    for png in png_paths:
        print(f"[Download Plot {png_paths.index(png) + 1}](file://{png})")

if __name__ == "__main__":
    main()