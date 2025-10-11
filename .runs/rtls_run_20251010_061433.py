import sys
import os
import pandas as pd
from extractor import extract_tracks
from pdf_creation_script import safe_build_pdf
from chart_policy import choose_charts
from report_limits import apply_budgets
import json

def main():
    if len(sys.argv) < 2:
        print("Error: No CSV file provided.")
        return

    user_prompt = sys.argv[1]
    csv_paths = sys.argv[2:]

    # Load guidelines
    with open("/mnt/data/guidelines.txt", "r") as f:
        guidelines = f.read()

    # Process each CSV file
    all_figures = []
    report_sections = []
    for csv_path in csv_paths:
        # Extract tracks
        df = extract_tracks(csv_path)
        
        # Apply guidelines
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        ts_src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
        df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

        # Choose charts based on user prompt
        figures = choose_charts(df, floorplans_path="/mnt/data/floorplans.json", user_query=user_prompt)
        all_figures.extend(figures)

        # Prepare report sections
        cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
        rows = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
        report_sections.append({"type": "table", "title": "Evidence", "data": rows, "headers": cols, "rows_per_page": 24})

    # Create PDF report
    report = {
        "sections": report_sections,
        "title": "Walmart Renovation RTLS Report",
        "bullets": ["Summary of position data", "Figures included as per request"]
    }
    
    # Apply budgets
    report = apply_budgets(report)
    
    # Save figures and build PDF
    pdf_path = os.path.join(os.path.dirname(csv_paths[0]), "walmart_report.pdf")
    safe_build_pdf(report, pdf_path, logo_path="/mnt/data/redpoint_logo.png")

    # Print output links
    print(f"[Download the PDF](file://{pdf_path})")
    for i, fig in enumerate(all_figures):
        png_path = os.path.join(os.path.dirname(csv_paths[0]), f"info_zone_report_plot{i:02d}.png")
        fig.savefig(png_path, dpi=120)
        print(f"[Download Plot {i + 1}](file://{png_path})")

if __name__ == "__main__":
    main()