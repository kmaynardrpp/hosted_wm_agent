# InfoZone RTLS Assistant

A local analysis assistant for Walmart RTLS position data.  
Give it a natural-language prompt and one or more CSVs; it generates a PDF report plus PNG charts/overlays on a store floor plan.

This guide is site-agnostic (works for **GC** and **GV**). Site differences live in a few JSON/PNG assets and (optionally) the system prompt.

---

## Contents

- [Repository Layout](#repository-layout)
- [Core Runtime](#core-runtime)
- [Prompts & Guidance](#prompts--guidance)
- [Site-Specific Assets (replace/customize per site)](#site-specific-assets-replacecustomize-per-site)
- [Data & Outputs](#data--outputs)
- [Web Client / Reverse Proxy (optional)](#web-client--reverse-proxy-optional)
- [How the Overlay Uses the Right Scale](#how-the-overlay-uses-the-right-scale)
- [Run Locally (CLI)](#run-locally-cli)
- [Run Locally (Web API + Optional SPA)](#run-locally-web-api--optional-spa)
- [Docker](#docker)
  - [Build the API Image](#build-the-api-image)
  - [Run the API Container](#run-the-api-container)
  - [docker-compose (API + Caddy HTTPS)](#docker-compose-api--caddy-https)
- [Environment Variables](#environment-variables)
- [Per-Site Checklist](#per-site-checklist)

---

## Repository Layout

```
.
├─ server.py                 # FastAPI app (HTTP endpoints)
├─ main.py                   # Orchestrator (prompting, code-gen, execution)
├─ extractor.py              # CSV -> normalized rows
├─ chart_policy.py           # Figure selection/drawing helpers
├─ pdf_creation_script.py    # PDF composer (Summary/Charts/Tables)
├─ zones_process.py          # Zone intervals & dwell computations
├─ report_limits.py          # Budgets and "lite" fallback shaping
├─ report_config.json        # Optional chart/layout tuning
├─ requirements.txt          # Python dependencies
│
├─ floorplan.png|jpg|jpeg    # Site raster for overlay
├─ floorplans.json           # Floor-plan metadata & scale
├─ zones.json                # Store polygons, zone uids/names
├─ trackable_objects.json    # MAC/UID mapping & trade inference
├─ redpoint_logo.png         # Optional PDF header logo
│
├─ db/                       # Optional CSV library (positions_YYYY-MM-DD.csv)
├─ uploads/                  # Web uploads (volume-mounted in Docker)
└─ .runs/                    # Generated scripts & final artifacts (PDF/PNGs)
```

If present:

```
web/infozone-web/            # Vite/React SPA client (per-site folder allowed)
Dockerfile
docker-compose.yml
Caddyfile
```

---

## Core Runtime

### `server.py`
FastAPI app that exposes HTTP endpoints for the web client. It:
- Receives prompts & uploads
- Creates per-job directories under `uploads/` and `.runs/`
- Invokes `main.py`, streams logs, and returns artifacts (PDF & PNGs)

### `main.py`
The orchestrator. It:
- Builds the full prompt (system + user + helper excerpts)
- Calls the OpenAI Responses API
- Validates/repairs the generated analysis script (compile + runtime repair loops)
- Executes that script and prints `file:///…` links to the artifacts
- Supports DB auto-select (picks CSVs from `db/` based on dates in the prompt)

### `extractor.py`
CSV → normalized rows:
- MAC/UID → trackable mapping via `trackable_objects.json`
- Trade inference from trackable labels
- Timestamp canon (`ts_utc`, `ts_iso`, `ts_short`)
- Zone name normalization; preserves zone numbers (e.g., `2.1SalesFloor`) and removes `Trailer`
- Polygon fallback when only `(x, y)` exist (`zones.json`)
- **GV:** drops one bad point only (`x == 5818` & `y == 2877`)  
  *(GC’s emergency floor crop is defined in the GC prompt, not here.)*

### `chart_policy.py`
Figure selection/drawing helpers (overlay, hourly, bar/pie). Reads floor-plan metadata and draws overlays in **world millimeters**.

### `pdf_creation_script.py`
Report composer (Summary / Charts / Tables) → **PDF**. Saves each figure to PNG (DPI 120) before building the PDF.

### `zones_process.py`
Loads store zones and computes dwell/intervals:

```python
compute_zone_intervals(df, zones, id_col, ts_col="ts_utc", x_col="x", y_col="y")
```

### `report_limits.py`
Budgets (max figures, rows) and “lite” fallback shaping.

### `report_config.json`
Optional chart/layout tuning (figure sizes, overlay options, etc.).

### `requirements.txt`
Python dependencies.

---

## Prompts & Guidance

- `system_prompt.txt` (site-tunable)  
  Strong rules for the generator (e.g., GC crop vs GV single-point ignore; full-plan overlay; overlay↔table zone parity; DB auto-select behavior).

- `guidelines.txt`  
  Non-negotiable execution contracts (path rules, plotting order, error reporting, etc.).

- `context.txt`  
  Optional background text included in prompts.

---

## Site-Specific Assets (replace/customize per site)

These are the only files you should swap per site.

- **`floorplan.png` / `.jpg` / `.jpeg`**  
  Raster used under the overlay.

- **`floorplans.json`**  
  Floor-plan metadata: `width`/`height`, `image_offset_x`/`y`, `image_scale` (units).  
  **Important:** The plan that corresponds to your raster must have `"selected": 1` and its `display_name`/filename **basename** must match the raster filename.

- **`zones.json`**  
  Store polygons (names/uids; set `active` as needed). Also used to compute the **full-plan world extent** so axes are tens of thousands of mm (not hundreds).

- **`trackable_objects.json`**  
  Device MACs/UIDs → friendly names/IDs for mapping and trade inference.

- **`redpoint_logo.png`**  
  Optional logo for the PDF header.

---

## Data & Outputs

- **`db/` (optional)**  
  CSV library named `positions_YYYY-MM-DD.csv` (tolerant of slight variations like `positions_…`).  
  When the prompt mentions dates/ranges, the tool can auto-select matching files from here.

- **`uploads/`**  
  Web server deposits uploaded CSVs here (volume-mounted in Docker).

- **`.runs/`**  
  Generated scripts and final artifacts (PDF & PNGs) are written here.

---

## Web Client / Reverse Proxy (optional)

- **`web/infozone-web/` (or per-site folder)**  
  Vite/React SPA for a simple chat-style UI.

- **`Dockerfile`, `docker-compose.yml`, `Caddyfile`**  
  Container build & optional HTTPS proxy (Caddy). Compose mounts `uploads/` & `.runs/` to persist results.

---

## How the Overlay Uses the Right Scale

- Overlays render the raster in **world millimeters**, not pixels.
- Preferred extent is **zones-bbox** (union of `zones.json` polygons) with a small margin — this shows the entire plan at store scale (e.g., width ≈ **80 000 mm**).
- If zones are missing, metadata from `floorplans.json` is used.  
  If metadata would yield a tiny extent (**< 10 000 mm**), the generator prints an **Error Report** rather than draw a misleading postage stamp.
- Points are plotted directly in mm (no extra display-space shifts).

---

## Run Locally (CLI)

Requires **Python 3.11–3.12**.

**1) Create and activate a venv**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

**2) Install deps**
```bash
pip install -r requirements.txt
```

**3) Set env (see full list below)**
```bash
export OPENAI_API_KEY="sk-..."           # PowerShell: $env:OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5"              # optional
export OPENAI_REASONING_EFFORT="medium"  # optional
# optional:
# export INFOZONE_OUT_DIR="/abs/output/dir"
# export INFOZONE_ROOT="$(pwd)"
```

**4) Run with explicit CSVs**
```bash
python main.py "Show me electricians by zone from 09/23–09/26." \
  /abs/path/positions_2025-09-23.csv /abs/path/positions_2025-09-24.csv
```

**…or let it pick from `db/`**
```bash
python main.py "Show electricians by zone from 09/23 to 09/26."
```

The script prints `file:///…` links to the PDF and PNGs on success.

---

## Run Locally (Web API + Optional SPA)

**API**
```bash
uvicorn server:app --reload --port 8000
```

**Web client (if present)**
```bash
cd web/infozone-web
npm install
npm run dev   # http://localhost:5173
```

Configure the client to call your API via `VITE_API_BASE` if not on the same host/port.

---

## Docker

### Build the API Image
```bash
docker build -t infozone-api:latest .
```

### Run the API Container
```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e OPENAI_MODEL=gpt-5 \
  -v "$(pwd)/uploads:/app/uploads" \
  -v "$(pwd)/.runs:/app/.runs" \
  -v "$(pwd):/app" \
  infozone-api:latest \
  uvicorn server:app --host 0.0.0.0 --port 8000
```

### docker-compose (API + Caddy HTTPS)

Ensure `uploads/` and `.runs/` exist on the host for persistence.  
Set your domain (DuckDNS or other) in **Caddyfile** and DNS.

```bash
docker compose up -d --build
docker compose logs -f api
docker compose logs -f caddy
```

---

## Environment Variables

| Name                      | Required | Description                               | Example                 |
|---------------------------|:--------:|-------------------------------------------|-------------------------|
| `OPENAI_API_KEY`          |    ✅     | OpenAI API key (Responses API)            | `sk-...`                |
| `OPENAI_MODEL`            |    ❌     | Default model                              | `gpt-5`                 |
| `OPENAI_REASONING_EFFORT` |    ❌     | Reasoning budget                           | `low` \| `medium` \| `high` |
| `RTLS_CODE_TIMEOUT_SEC`   |    ❌     | Max seconds for generated script run       | `1800`                  |
| `INFOZONE_ROOT`           |    ❌     | Forces project root if running elsewhere   | `/app`                  |
| `INFOZONE_OUT_DIR`        |    ❌     | Output dir for PDF/PNGs                    | `/app/.runs/job_*`      |
| `IZ_MAX_OUTPUT_TOKENS`    |    ❌     | Model output cap                           | `24000`                 |
| `IZ_CONTEXT_CAP`          |    ❌     | Max chars for `context.txt` excerpt        | `30000`                 |
| `IZ_HELPER_CAP`           |    ❌     | Max chars per helper excerpt               | `30000`                 |

---

## Per-Site Checklist

- `floorplan.png` (or `.jpg`/`.jpeg`) exists under the repo root.

- **`floorplans.json`**
  - The correct plan has `"selected": 1`.
  - `display_name`/filename **basename** matches the raster filename.
  - `image_scale` uses **mm/px** (or specify units: `cm_per_px` → ×10, `m_per_px` → ×1000).

- **`zones.json`** contains the real store polygons (set `active` appropriately).

- **`trackable_objects.json`** contains the store’s MAC/UID inventory.

- **`system_prompt.txt`** is set for the site:
  - **GV:** one-point ignore only; full-plan (zones-bbox) overlay.
  - **GC:** floor crop `x ≥ 12000` & `y ≥ 15000`; full-plan overlay.

If these are correct, the overlay axes should be **store-scale** (e.g., width ≈ **80 000 mm**), tables and overlays will label zones consistently, and the report will save to `.runs/` (and print `file:///…` links).
