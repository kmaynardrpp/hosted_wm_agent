import sys, os
from pathlib import Path

# --- locate project root ---
ROOT = Path(os.environ.get('INFOZONE_ROOT', ''))
if not ROOT or not (ROOT / 'guidelines.txt').exists():
    # script is saved under .runs/, helpers are one level up
    script_dir = Path(__file__).resolve().parent
    ROOT = script_dir if (script_dir / 'guidelines.txt').exists() else script_dir.parent

# --- make local imports work (extractor, pdf_creation_script, etc.) ---
sys.path.insert(0, str(ROOT))

# helper file paths (never use /mnt/data)
GUIDELINES = ROOT / 'guidelines.txt'
CONTEXT    = ROOT / 'context.txt'
FLOORJSON  = ROOT / 'floorplans.json'
LOGO       = ROOT / 'redpoint_logo.png'

import pandas as pd
from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from report_limits import apply_budgets
from chart_policy import choose_charts
import json
import matplotlib.pyplot as plt

def main(user_prompt, *csv_paths):
    # Read and apply guidelines
    with open(GUIDELINES, 'r') as f:
        guidelines = f.read()
    
    # Initialize report sections
    sections = []
    total_figures = 0
    report_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    for csv_path in csv_paths:
        # Extract tracks from CSV
        df, audit = extract_tracks(csv_path)
        
        # Duplicate-name guard
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Timestamp canon
        ts_src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
        df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")
        
        # Filter out invalid rows
        df = df.dropna(subset=["ts_utc", "x", "y"])
        
        # Load floorplan
        with open(FLOORJSON, 'r') as f:
            floorplans = json.load(f)
        
        # Choose charts based on user prompt
        figs = choose_charts(df, floorplans_path=FLOORJSON, floorplan_image_path=ROOT / 'floorplan.jpeg', user_query=user_prompt)
        
        # Append figures to sections
        sections.append({"type": "charts", "figures": figs})
        total_figures += len(figs)
        
        # Prepare evidence table
        cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
        rows = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
        sections.append({"type": "table", "title": "Evidence", "data": rows, "headers": cols, "rows_per_page": 24})
    
    # Apply budgets
    report = {"sections": sections}
    report = apply_budgets(report)
    
    # Save PDF
    pdf_path = os.path.join(os.path.dirname(csv_paths[0]), f"info_zone_report_{report_date}.pdf")
    safe_build_pdf(report, pdf_path, logo_path=str(LOGO))
    
    # Print links
    print(f"[Download the PDF](file://{pdf_path})")
    for i in range(total_figures):
        print(f"[Download Plot {i+1}](file://{os.path.dirname(csv_paths[0])}/info_zone_report_{report_date}_plot{i:02d}.png)")

if __name__ == "__main__":
    main(*sys.argv[1:])