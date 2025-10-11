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

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def file_uri(p: Path) -> str:
    return "file:///" + str(p.resolve()).replace("\\", "/")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("csv_paths", nargs='+')
    args = parser.parse_args()

    out_dir = Path(args.csv_paths[0]).parent
    sections = []
    png_paths = []
    report_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    for csv_path in args.csv_paths:
        try:
            raw = extract_tracks(csv_path)
            df = pd.DataFrame(raw.get("rows", []))
            audit = raw.get("audit", {})

            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            ts_src = df["ts_iso"] if "ts_iso" in df.columns else df["ts"]
            df["ts_utc"] = pd.to_datetime(ts_src, utc=True, errors="coerce")

            df["x"] = pd.to_numeric(df.get("x"), errors="coerce")
            df["y"] = pd.to_numeric(df.get("y"), errors="coerce")

            cols = ["trackable", "trade", "ts_short", "x", "y", "z"]
            rows = df[cols].head(50).fillna("").astype(str).to_dict(orient="records")
            sections.append({"type": "table", "title": "Evidence", "data": rows, "headers": cols, "rows_per_page": 24})

            # Load floorplan
            with open(FLOORJSON, "r", encoding="utf-8") as f:
                floorplan_data = json.load(f)
            floorplan_image_path = ROOT / floorplan_data['floorplans'][0]['filename']

            # Create charts
            figs = choose_charts(df, floorplans_path=FLOORJSON, floorplan_image_path=str(floorplan_image_path), user_query=args.prompt)
            for i, fig in enumerate(figs):
                png_path = out_dir / f"info_zone_report_{report_date}_plot{i:02d}.png"
                fig.savefig(png_path, dpi=120)
                png_paths.append(png_path)

            sections.append({"type": "charts", "title": "Figures", "figures": figs})

        except Exception as e:
            print(f"Error Report:\n{str(e)}")
            return

    report = {"sections": sections}
    report = apply_budgets(report)

    pdf_path = out_dir / f"info_zone_report_{report_date}.pdf"
    safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))

    print(f"[Download the PDF]({file_uri(pdf_path)})")
    for i, png in enumerate(png_paths, 1):
        print(f"[Download Plot {i}]({file_uri(png)})")

if __name__ == "__main__":
    main()