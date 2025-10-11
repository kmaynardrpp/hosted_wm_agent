import sys, os
from pathlib import Path
# add project root (parent of .runs) to import path so `import extractor` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from chart_policy import choose_charts
from report_limits import apply_budgets
import json

def main():
    if len(sys.argv) < 2:
        print("Error: No user prompt provided.")
        return

    user_prompt = sys.argv[1]
    csv_paths = sys.argv[2:]

    # Read guidelines
    with open("/mnt/data/guidelines.txt", "r") as f:
        guidelines = f.read()

    # Process each CSV file
    report_sections = []
    for csv_path in csv_paths:
        df, encoding, pre, engine = extract_tracks(csv_path)

        # Apply duplicate-name guard
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Create ts_utc
        ts_src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
        df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

        # Filter out invalid rows
        df = df.dropna(subset=["ts_utc", "x", "y"])

        # Load floorplan and zones
        with open("/mnt/data/floorplans.json", "r") as f:
            floorplans = json.load(f)
        floorplan_image_path = "/mnt/data/floorplan.jpeg"

        # Choose charts based on user prompt
        charts = choose_charts(df, floorplan_image_path=floorplan_image_path, user_query=user_prompt)

        # Prepare report sections
        if charts:
            report_sections.append({"type": "charts", "figures": charts})

        # Convert DataFrame to list-of-dicts for tables
        cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
        rows = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
        report_sections.append({"type": "table", "title": "Evidence", "data": rows, "headers": cols, "rows_per_page": 24})

    # Build the report
    report = {"sections": report_sections}
    report = apply_budgets(report)
    
    # Save PDF
    pdf_path = os.path.join(os.path.dirname(csv_paths[0]), "report.pdf")
    safe_build_pdf(report, pdf_path, logo_path="/mnt/data/redpoint_logo.png")

    # Print links
    print(f"[Download the PDF](file://{pdf_path})")
    for i, chart in enumerate(report_sections):
        if chart["type"] == "charts":
            for j, fig in enumerate(chart["figures"]):
                png_path = os.path.join(os.path.dirname(csv_paths[0]), f"info_zone_report_plot{j:02d}.png")
                plt.savefig(png_path, dpi=120)
                print(f"[Download Plot {i * len(chart['figures']) + j + 1}](file://{png_path})")

if __name__ == "__main__":
    main()