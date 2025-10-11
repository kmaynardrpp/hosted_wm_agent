import sys, os
from pathlib import Path
# add project root (parent of .runs) to import path so `import extractor` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from report_limits import apply_budgets
from chart_policy import choose_charts
import json
import matplotlib.pyplot as plt

def main(user_prompt, *csv_paths):
    # Load guidelines
    with open("/mnt/data/guidelines.txt", "r") as f:
        guidelines = f.read()

    # Initialize report sections
    sections = []

    # Process each CSV file
    for csv_path in csv_paths:
        df, encoding, pre, engine = extract_tracks(csv_path)
        
        # Duplicate-name guard
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Timestamp canon
        ts_src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
        df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

        # Filter out invalid rows
        df = df.dropna(subset=["ts_utc", "x", "y"])

        # Load floorplan
        with open("/mnt/data/floorplans.json", "r") as f:
            floorplans = json.load(f)
        floorplan_path = "/mnt/data/floorplan.jpeg"  # Assuming JPEG is the format used
        floorplan_info = next((fp for fp in floorplans["floorplans"] if fp["filename"] == "Updated Floorplan 09032025.jpeg"), None)

        # Choose charts based on user prompt
        figs = choose_charts(df, floorplan_image_path=floorplan_path, user_query=user_prompt)

        # Save figures as PNGs
        png_paths = []
        for i, fig in enumerate(figs):
            png_path = f"{os.path.dirname(csv_path)}/info_zone_report_plot{i:02d}.png"
            fig.savefig(png_path, dpi=120)
            png_paths.append(png_path)

        # Build report
        report = {
            "sections": sections,
            "title": "Walmart Renovation RTLS Analysis",
            "bullets": ["Summary of position data", "Visualizations included"],
        }
        report = apply_budgets(report)
        pdf_path = f"{os.path.dirname(csv_path)}/info_zone_report.pdf"
        safe_build_pdf(report, pdf_path, logo_path="/mnt/data/redpoint_logo.png")

        # Print links
        print(f"[Download the PDF](file://{pdf_path})")
        for png in png_paths:
            print(f"[Download Plot {len(png_paths)}](file://{png})")

if __name__ == "__main__":
    main(*sys.argv[1:])