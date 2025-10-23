This repository hosts a llm powered analysis assistant for Walmart RTLS position data.
You provide a prompt (e.g., “show electrician hours per zone last week”) and one or more CSVs; the system generates a PDF plus PNGs with an overlay on the store floor plan and any requested charts/tables.

It supports multiple sites (e.g., GC and GV). The code path is the same; site-specific assets & rules live in JSON/PNG and prompt files called out below.

Repository map (what each file does)

Tip: Items marked 🔧 are site-specific and should be customized per store/site.

Core runtime

server.py
FastAPI app that connects the web client to the backend. Handles:

receiving a prompt + files (uploads)

arranging work directories (e.g., uploads/ and .runs/)

invoking main.py and returning logs/artifacts (PDF/PNGs)

main.py
The orchestrator. It:

builds the prompt (system + user)

calls the OpenAI Responses API

validates/repairs the returned generated analysis script

executes it, captures output, and prints file links (file:///…)

supports DB auto-select (choose CSVs from db/ based on dates in the prompt)

extractor.py
Fast CSV → normalized rows:

MAC/UID → trackable mapping (via 🔧trackable_objects.json)

trade inference from trackable names

timestamp canon (ts_utc, ts_iso, ts_short)

zone name normalization (includes zone number when present, e.g., 2.1SalesFloor; removes Trailer)

polygon fallback classification when only (x,y) are present (via 🔧zones.json)

GV only: single-point ignore (drops x==5818 & y==2877)
GC uses a global crop (x≥12 000 & y≥15 000) — that rule lives in GC’s prompt/guidelines.

chart_policy.py
Helpers for figure selection and drawing (overlay, hourly line, bar/pie, etc.). Reads floor plan metadata and draws overlays in world mm. It will:

select the floor plan entry with "selected": 1 that matches the raster filename

favor zones-bbox extent so the full plan is visible (axes in mm)

render points directly in world mm (no extra scaling)

pdf_creation_script.py
Composes the report:

Summary, optional Charts (live matplotlib.Figures), optional Table

outputs a single PDF

also saves each figure as a PNG (DPI 120) before building the PDF

zones_process.py
Loads 🔧zones.json and computes dwell intervals with
compute_zone_intervals(df, zones, id_col, ts_col='ts_utc', x_col='x', y_col='y').

report_limits.py
Budgets (max figures, rows) and the lite fallback shaping.

report_config.json
Optional chart layout config (figure sizes, overlay settings, etc.).

requirements.txt
Python dependencies (FastAPI/uvicorn, pandas/numpy/matplotlib, etc.).

Prompts & guidance (where per-site behavior is defined)

system_prompt.txt (🔧 often differs by site)
Rules for the generator (e.g., GC crop vs GV one-point ignore; floor-plan policy; overlay ↔ table parity).

guidelines.txt
Hard contracts enforced by the generated script (paths, plotting rules, PDFs/PNGs order, error reporting).

context.txt
Optional background text included in prompts (non-code narrative).

Site-specific assets (replace per site)

floorplans.json 🔧
Metadata for one or more floor plans (width/height, image_offset_x/y, image_scale).
Important:

The entry corresponding to the raster you actually use must have "selected": 1.

Its display_name/filename basename must match the raster PNG/JPG you ship.

zones.json 🔧
Polygons with names/uids (set active as needed). Used for zone dwell, naming, and the zones-bbox (full plan extent).

trackable_objects.json 🔧
Device MACs and/or UIDs mapped to friendly names (and IDs). Used by the extractor for identity/trade mapping.

floorplan.png (or .jpg/.jpeg) 🔧
The raster drawing used in overlays. Must match the selected: 1 plan entry in floorplans.json.

redpoint_logo.png
Logo for the report header (optional; PDF will be built without if missing).

Project data / outputs

db/ (optional)
A local database of CSVs named positions_YYYY-MM-DD.csv (or tolerant postions_…).
When a user asks “between 09-14 and 10-02”, the generator can auto-select files from db/.

uploads/
The web server stores uploaded files here (volume-mounted in Docker).

.runs/
Where main.py writes/executes generated scripts and drops final artifacts (PDF/PNGs).

Web client / reverse proxy (optional)

web/infozone-web/ (or web/infozone-web-gv/)
Vite/React SPA for a simple “ChatGPT-like” page to submit prompts & files.

Caddyfile, docker-compose.yml, Dockerfile
Turnkey hosting (HTTPS with Caddy + the API container).
The compose file binds uploads/ and .runs/ as volumes so results persist.

Files to customize per site

floorplan.png / .jpg / .jpeg 🔧
The drawing to show under the overlay.

floorplans.json 🔧

Ensure the correct plan entry has "selected": 1.

Its display_name/filename should basename-match floorplan.png.

image_scale units: default mm/px. If a units field exists:

mm_per_px → as-is

cm_per_px → ×10

m_per_px → ×1000

zones.json 🔧

Real store polygons & names; used for zone dwell and to compute the world extent for the raster (so axes are tens of thousands of mm, not hundreds).

trackable_objects.json 🔧

Per-site device inventory: MACs/UIDs and names (used by extractor.py).

system_prompt.txt (recommend per site) 🔧

GV: one-point ignore (drop exactly x==5818 & y==2877) and full-plan overlay (zones-bbox extent).

GC: emergency floor crop (keep x≥12000 & y≥15000), also full-plan overlay.

How the overlay gets the right scale

The overlay draws the raster in world millimeters (not pixels).

It prefers the zones-bbox from zones.json (union of polygons) with a small margin — this shows the entire plan at store scale (e.g., width ≈ 80 000 mm).

If zones are missing, it will try metadata from floorplans.json. If the implied world width/height are tiny (< 10 000 mm), it prints an Error Report rather than render a misleading postage stamp.

Running locally (CLI)

Requires Python 3.11–3.12.

Create venv & install

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt


Set environment (minimum)

# macOS/Linux
export OPENAI_API_KEY="sk-..."
# optional:
export OPENAI_MODEL="gpt-5"
export OPENAI_REASONING_EFFORT="medium"
export INFOZONE_ROOT="$(pwd)"             # if running outside repo root
# use this to write PDFs/PNGs elsewhere:
# export INFOZONE_OUT_DIR="/absolute/path/out"


(Windows PowerShell)

$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_MODEL="gpt-5"
$env:OPENAI_REASONING_EFFORT="medium"
# optional:
$env:INFOZONE_ROOT="$PWD"
# $env:INFOZONE_OUT_DIR="C:\path\to\out"


Run from CLI

# Example: ask for carpenters on two files
python main.py "Show me carpenters positions on 09/25–09/26, one color per tag." \
  /abs/path/positions_2025-09-25.csv /abs/path/positions_2025-09-26.csv


Or let the tool auto-select from db/:

python main.py "Show electricians positions from 09/23 to 09/26."


You’ll get console links like:

[Download the PDF](file:///.../info_zone_report_2025-09-23_to_2025-09-26.pdf)
[Download Plot 1](file:///.../info_zone_report_2025-09-23_to_2025-09-26_plot01.png)

Running locally (web API + optional SPA)

API server

uvicorn server:app --reload --port 8000


Web client (if present)

cd web/infozone-web
npm install
npm run dev   # http://localhost:5173


Configure the client to call your API host (set VITE_API_BASE in the web app if needed).

Docker (API only, or API + Caddy proxy)
Build the API image
docker build -t infozone-api:latest .

Minimal run
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e OPENAI_MODEL=gpt-5 \
  -v "$(pwd)/uploads:/app/uploads" \
  -v "$(pwd)/.runs:/app/.runs" \
  -v "$(pwd):/app" \
  infozone-api:latest uvicorn server:app --host 0.0.0.0 --port 8000

docker-compose (API + Caddy with HTTPS)

Make sure uploads/ and .runs/ exist on the host (so results persist).

Set your domain (DuckDNS or real) in Caddyfile and DNS record.

docker compose up -d --build
docker compose logs -f api
docker compose logs -f caddy

Environment variables (summary)
Variable	Required	Purpose	Typical
OPENAI_API_KEY	✅	OpenAI API key for the Responses API	sk-…
OPENAI_MODEL		Default model selection	gpt-5
OPENAI_REASONING_EFFORT		Reasoning budget	low | medium | high
RTLS_CODE_TIMEOUT_SEC		Max seconds for generated code run	1800
INFOZONE_ROOT		Force project root if running elsewhere	repo absolute path
INFOZONE_OUT_DIR		Where to write PDF/PNGs (overrides default)	absolute path
IZ_MAX_OUTPUT_TOKENS		Cap model output size	24000
IZ_CONTEXT_CAP, IZ_HELPER_CAP		Size caps for context/helper excerpts	30000

Site-specific assets (floorplan*.png, floorplans.json, zones.json, trackable_objects.json) must exist under ROOT. In floorplans.json the correct plan must have "selected": 1 and its display_name/filename must match the raster file you ship.

Quick checklist (per site)

🔧 floorplan.png present & matches selected:1 entry

🔧 floorplans.json has correct offsets/scale (units in mm/px or tagged)

🔧 zones.json polygons loaded; names match your conventions

🔧 trackable_objects.json up-to-date with MACs/UIDs → names

system_prompt.txt is the correct variant:

GV → single-point ignore only (x==5818 & y==2877)

GC → global floor crop (x≥12000 & y≥15000)

OPENAI_API_KEY set, uploads/ & .runs/ exist (writable)