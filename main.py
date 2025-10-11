#!/usr/bin/env python3
# main.py
r"""
Responses-API generator/runner with debug, validation, and auto-retries:
  1) Builds a strong system message from system_prompt.txt and a grounded user message
     (guidelines + helper excerpts + local-asset list).
  2) Calls the Responses API to get ONE Python script (single code block).
  3) Validates the script. If it’s “half-baked” (too short, missing imports, bad paths),
     auto-reprompts with a stricter repair prompt; then a minimal emergency prompt.
  4) Falls back to alternate models if your OPENAI_MODEL is unavailable.
  5) Saves script to .runs/ and executes locally with project root on PYTHONPATH.

Usage:
  python main.py "Prompt message" C:\\path\\to\\positions_YYYY-MM-DD.csv [more.csv ...]

Env:
  OPENAI_API_KEY        : required
  OPENAI_MODEL          : preferred model (e.g., "gpt-5" if your key has it)
  RTLS_CODE_TIMEOUT_SEC : child-script exec timeout (default 1800s)
  OPENAI_REASONING_EFFORT : "low" | "medium" | "high" (default "high")
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

# ---- OpenAI SDK ----
try:
    from openai import OpenAI
except Exception:
    print("ERROR: OpenAI SDK import failed. Install: pip install --upgrade openai", file=sys.stderr)
    raise

# ---------- Config ----------
ENV_MODEL = os.environ.get("OPENAI_MODEL", "").strip()
FALLBACK_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]
TIMEOUT_SEC = int(os.environ.get("RTLS_CODE_TIMEOUT_SEC", "1800"))
RUNS_DIR = ".runs"

# ---------- Reasoning config ----------
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "medium").lower()

def model_supports_reasoning(model: str) -> bool:
    """
    Heuristic: only attach the 'reasoning' param for models that support it.
    Covers GPT-5 and OpenAI 'o' reasoning families; safe no-op for others.
    """
    m = (model or "").lower()
    return any(tag in m for tag in ("gpt-5", "o4", "o3", "reasoning", "thinking"))

# ---------- Helpers ----------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path, max_chars: int | None = None) -> str:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return txt if max_chars is None else txt[:max_chars]
    except Exception:
        return ""

def first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def extract_code_block(text: str) -> str:
    """Extract the first ```python ...``` block; else first ```...```; else full text."""
    if not text:
        return ""
    fence_py = "```python"
    fence = "```"
    if fence_py in text:
        i = text.find(fence_py) + len(fence_py)
        j = text.find("```", i)
        if j != -1:
            return text[i:j].strip()
    if fence in text:
        i = text.find(fence) + len(fence)
        j = text.find("```", i)
        if j != -1:
            return text[i:j].strip()
    return text.strip()

def code_is_skeletal(code: str) -> Tuple[bool, List[str]]:
    """
    Heuristic validator:
    - length threshold
    - must import extractor & pdf_creation_script
    - NO '/mnt/data' or 'sandbox:'
    - must add ROOT/PYTHONPATH bootstrap (Path(__file__).resolve() or INFOZONE_ROOT)
    """
    issues: List[str] = []
    if len(code) < 1500:  # ~50 lines minimum; tune as needed
        issues.append(f"too short ({len(code)} chars)")
    low = code.lower()
    if "from extractor import extract_tracks" not in code:
        issues.append("missing 'from extractor import extract_tracks'")
    if "from pdf_creation_script import safe_build_pdf" not in code:
        issues.append("missing 'from pdf_creation_script import safe_build_pdf'")
    if "/mnt/data" in low or "sandbox:" in low:
        issues.append("contains forbidden path tokens (/mnt/data or sandbox:)")
    if "infozone_root" not in low and "path(__file__)" not in low:
        issues.append("missing project-root resolution / sys.path bootstrap")
    return (len(issues) > 0), issues

# ---------- Prompt builders ----------
def build_system_message(project_dir: Path) -> str:
    sys_prompt = read_text(project_dir / "system_prompt.txt").strip()
    if not sys_prompt:
        sys_prompt = "You are a code generator that returns one Python script as a single code block."
    sys_prompt += (
        "\n\nMANDATE: You must return ONE Python script in a single code block and nothing else. "
        "The script must obey guidelines.txt and the Output contract exactly."
    )
    return sys_prompt

def build_user_message(user_prompt: str, csv_paths: List[str], project_dir: Path, trimmed: bool=False) -> str:
    # Full guidelines + excerpted helpers; list binaries present locally.
    guidelines = read_text(project_dir / "guidelines.txt", max_chars=None)
    context = read_text(project_dir / "context.txt", max_chars=(4000 if trimmed else 8000))

    helper_files = [
        ("chart_policy.py",        4000 if trimmed else 8000),
        ("extractor.py",           4000 if trimmed else 8000),
        ("pdf_creation_script.py", 4000 if trimmed else 8000),
        ("zones_process.py",       4000 if trimmed else 8000),
        ("report_limits.py",       2000 if trimmed else 4000),
        ("report_config.json",     2000 if trimmed else 4000),
        ("floorplans.json",        2000 if trimmed else 4000),
        ("zones.json",             2000 if trimmed else 4000),
    ]
    helper_snips = []
    for fname, cap in helper_files:
        snip = read_text(project_dir / fname, max_chars=cap)
        if snip:
            helper_snips += [f"\n>>> {fname}\n", snip]

    floorplan = first_existing([
        project_dir / "floorplan.jpeg",
        project_dir / "floorplan.jpg",
        project_dir / "floorplan.png"
    ])
    logo = project_dir / "redpoint_logo.png"
    assets = [
        ("example.csv", (project_dir / "example.csv").exists()),
        ((floorplan.name if floorplan else "floorplan.(jpeg|jpg|png)"), bool(floorplan)),
        ("redpoint_logo.png", logo.exists()),
    ]

    csv_lines = "\n".join(f" - {p}" for p in csv_paths)
    assets_lines = "\n".join(f" - {n}: {'present' if ok else 'missing'}" for n, ok in assets)

    parts: List[str] = []
    parts += [
        "USER PROMPT",
        "-----------",
        user_prompt,
        "",
        "CSV INPUTS (absolute paths)",
        "---------------------------",
        csv_lines,
        "",
        "LOCAL ASSETS (binary/CSV — DO NOT UPLOAD; read from disk at runtime)",
        "---------------------------------------------------------------------",
        assets_lines,
        "",
        "MANDATORY RULES (guidelines.txt — full text)",
        "--------------------------------------------",
        guidelines if not trimmed else guidelines[:4000],
        "",
        "BACKGROUND CONTEXT (excerpt)",
        "----------------------------",
        context,
        "",
        "HELPER EXCERPTS (READ-ONLY; use these APIs — do NOT re-implement)",
        "-----------------------------------------------------------------",
    ]
    parts += helper_snips
 # --- replace ONLY the "parts += [...]" block at the end of build_user_message(...) with this ---

# DELIVERABLE + hard rules for codegen (as ONE big instruction block)
    INSTR = (
    "(ONE user-visible execution in Analysis ONLY — zero tolerance)\n"
    "\n"
    "You are **InfoZoneBuilder**. Generate ONE self-contained **Python script** that analyzes Walmart "
    "renovation **RTLS position** data and writes one branded PDF plus PNGs for any charts. Return **ONE** code "
    "block and nothing else. Use our **local helper modules** exactly as specified and follow every rule below. "
    "The script must be robust on Windows, handle multi-day CSVs, and **never** assume sandbox paths.\n"
    "\n"
    "CRITICAL CONTRACTS (READ FIRST)\n"
    "- ONE code block only (no commentary).\n"
    "- On success the script must print *only* these two kinds of lines in order (use exactly this formatting when you print):\n"
    "    # PDF line\n"
    "    print(f\"[Download the PDF](file:///{pdf_path.resolve().as_posix()})\")\n"
    "    # Plot lines (1..N)\n"
    "    print(f\"[Download Plot {i}](file:///{png_path.resolve().as_posix()})\")\n"
    "  If there are no figures, print only the PDF line.\n"
    "- On failure the script must print:\n"
    "    print(\"Error Report:\")\n"
    "    print(\"<1–2 line reason>\")\n"
    "  If caused by missing/invalid columns, also print exactly one extra line:\n"
    "    print(\"Columns detected: \" + \",\".join(df.columns.astype(str)))\n"
    "- Do not print anything else.\n"
    "\n"
    "LOCAL PATHS ONLY (NEVER “/mnt/data” or “sandbox:”)\n"
    "- At the very top of the script resolve the **project root** and enable local imports:\n"
    "    import sys, os\n"
    "    from pathlib import Path\n"
    "\n"
    "    ROOT = Path(os.environ.get(\"INFOZONE_ROOT\", \"\"))  # injected by launcher\n"
    "    if not ROOT or not (ROOT / \"guidelines.txt\").exists():\n"
    "        script_dir = Path(__file__).resolve().parent\n"
    "        ROOT = script_dir if (script_dir / \"guidelines.txt\").exists() else script_dir.parent\n"
    "\n"
    "    if str(ROOT) not in sys.path:\n"
    "        sys.path.insert(0, str(ROOT))\n"
    "\n"
    "    GUIDELINES = ROOT / \"guidelines.txt\"\n"
    "    CONTEXT    = ROOT / \"context.txt\"\n"
    "    FLOORJSON  = ROOT / \"floorplans.json\"\n"
    "    LOGO       = ROOT / \"redpoint_logo.png\"\n"
    "    CONFIG     = ROOT / \"report_config.json\"\n"
    "    LIMITS_PY  = ROOT / \"report_limits.py\"\n"
    "    ZONES_JSON = ROOT / \"zones.json\"\n"
    "\n"
    "    def read_text(p: Path) -> str:\n"
    "        return p.read_text(encoding=\"utf-8\", errors=\"ignore\") if p.exists() else \"\"\n"
    "- All text reads must use encoding=\"utf-8\", errors=\"ignore\".\n"
    "\n"
    "LAUNCH & ARGUMENTS\n"
    "- The CLI will call your script like:\n"
    "    python generated.py \"<USER_PROMPT>\" /abs/csv1 [/abs/csv2 ...]\n"
    "- Treat the **first CSV’s directory** as the **output directory** (out_dir).\n"
    "\n"
    "INGEST (ONLY PATH, REQUIRED COLUMNS CHECK)\n"
    "- Use the helper for each CSV:\n"
    "    from extractor import extract_tracks\n"
    "    raw = extract_tracks(csv_path)\n"
    "    import pandas as pd\n"
    "    df  = pd.DataFrame(raw.get(\"rows\", []))\n"
    "    audit = raw.get(\"audit\", {})\n"
    "- Duplicate-name guard:\n"
    "    if df.columns.duplicated().any():\n"
    "        df = df.loc[:, ~df.columns.duplicated()]\n"
    "- Timestamp canon:\n"
    "    src = df[\"ts_iso\"] if \"ts_iso\" in df.columns else df[\"ts\"]\n"
    "    df[\"ts_utc\"] = pd.to_datetime(src, utc=True, errors=\"coerce\")\n"
    "- Required fields after the first file’s ingestion (schema validator). The script must ensure:\n"
    "  • At least one of \"trackable\" OR \"trackable_uid\" exists (identity)\n"
    "  • \"trade\" column exists (string; can be empty)\n"
    "  • \"x\" AND \"y\" exist (positions)\n"
    "  • If zones are requested: either \"zone_name\" exists OR the script computes zones via polygons\n"
    "  If any requirement fails, print:\n"
    "    print(\"Error Report:\")\n"
    "    print(\"Missing required columns for analysis.\")\n"
    "    print(\"Columns detected: \" + \",\".join(df.columns.astype(str)))\n"
    "    raise SystemExit(1)\n"
    "\n"
    "EVIDENCE TABLES (JSON, NOT DataFrame)\n"
    "- Build tables as list-of-dicts:\n"
    "    cols = [\"trackable\",\"trade\",\"ts_short\",\"x\",\"y\",\"z\"]\n"
    "    rows = df[cols].head(50).fillna(\"\").astype(str).to_dict(orient=\"records\")\n"
    "    sections.append({\"type\":\"table\",\"title\":\"Evidence\",\"data\":rows,\"headers\":cols,\"rows_per_page\":24})\n"
    "\n"
    "CHARTS → PNGs → PDF (MANDATORY ORDER)\n"
    "- Create Matplotlib figures → save PNGs to out_dir (no bbox_inches='tight'):\n"
    "    png = out_dir / f\"info_zone_report_{report_date}_plot{i:02d}.png\"\n"
    "    fig.savefig(str(png), dpi=120)\n"
    "- Then pass the live Figure objects into a \"charts\" section:\n"
    "    sections.append({\"type\":\"charts\",\"title\":\"Figures\",\"figures\":figs})\n"
    "- Build the PDF with string paths (Windows-safe):\n"
    "    pdf_path = out_dir / f\"info_zone_report_{report_date}.pdf\"\n"
    "    from pdf_creation_script import safe_build_pdf\n"
    "    safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))\n"
    "\n"
    "PRINT LINKS (SUCCESS, WINDOWS-SAFE)\n"
    "- Use absolute file:/// URIs:\n"
    "    def file_uri(p): return \"file:///\" + str(p.resolve()).replace(\"\\\\\", \"/\")\n"
    "    print(f\"[Download the PDF]({file_uri(pdf_path)})\")\n"
    "    for i, pth in enumerate(png_paths, 1):\n"
    "        print(f\"[Download Plot {i}]({file_uri(pth)})\")\n"
    "\n"
    "LARGE-DATA MODE (5 days / ~200MB)\n"
    "- Per-file streaming: process each CSV independently; keep only small aggregates or a bounded overlay reservoir. Do not concatenate all rows in RAM.\n"
    "- Cast numeric only when needed (e.g., for x/y filtering); keep strings otherwise.\n"
    "\n"
    "ZONES (ONLY IF ASKED)\n"
    "- If the user asked about zones:\n"
    "  • If df has \"zone_name\", use it.\n"
    "  • Else compute membership using zones_process and add/attach a zone_name per interval (or at least use it for charts/tables), with:\n"
    "      id_col=\"trackable_uid\", ts_col=\"ts_utc\", x_col=\"x\", y_col=\"y\", no downsampling.\n"
    "  • If polygons are missing/invalid and \"zone_name\" is absent, print Error Report and the detected columns.\n"
    "\n"
    "LEGEND & QUALITY\n"
    "- Only call legend() if there are labeled artists; cap categories at ≤ 12.\n"
    "\n"
      "MAC → TRACKABLE → TRADE (STRICT):\n"
    "  - MAC normalization and lookup in trackable_objects.json to fill final `trackable` and `trackable_uid` (the extractor does this)\n"
    "  - Infer canonical `trade` from the **final trackable** via regex (the extractor does this)\n"
    "  - Use extractor outputs: trackable, trackable_uid, trade, mac, ts, ts_iso, ts_short, x, y, z  "
    "(see extractor helper)\n"
    "ROBUSTNESS\n"
    "- Validate helpers under ROOT; LOGO is optional.\n"
    "- Never rename ts_utc to \"ts\"; never create duplicate column names.\n"
    "- Never reference \"/mnt/data\" or print \"sandbox:\" links.\n"
    "\n"
    "ONE BLOCK RULE\n"
    "- Your reply MUST be one Python code block (no commentary outside).\n"
    )

    # Replace previous deliverable block with this single append:
    parts += [
    "",
    "INSTRUCTIONS (MANDATORY — FOLLOW EXACTLY)",
    "-----------------------------------------",
    INSTR,
    ]

    # keep final join unchanged
    return "\n".join(parts)


def build_minimal_user_message(user_prompt: str, csv_paths: List[str]) -> str:
    """Emergency, minimal re-prompt to force code block."""
    csv_lines = "\n".join(f" - {p}" for p in csv_paths)
    return f"""
Return ONE Python script in a single code block and nothing else.

Requirements:
- CLI: python generated.py "<USER_PROMPT>" /abs/csv1 [/abs/csv2 ...]
- Import local helpers (extractor, pdf_creation_script, zones_process, chart_policy, report_limits, report_config).
- Read CSVs from argv; save PDF/PNGs next to the **first CSV**.
- Print file:/// links exactly as:
    [Download the PDF](file:///ABS/PATH/TO/PDF)
    [Download Plot 1](file:///ABS/PATH/TO/PNG1)
- Implement per-file streaming for large data.
- Use only local paths; never /mnt/data.

CSV INPUTS:
{csv_lines}

Return ONLY the code block (start with ```python).
""".strip()

# ---------- Responses API ----------
def responses_create_text(client: OpenAI, model: str, system_msg: str, user_msg: str) -> str:
    print(f"[{now_ts()}] [DEBUG] Calling Responses.create with model={model}")
    print(f"[{now_ts()}] [DEBUG] System chars: {len(system_msg)} | User chars: {len(user_msg)}")

    # Build kwargs so we only include params supported by the target model.
    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_output_tokens": 12000,  # allow large scripts
    }
    if model_supports_reasoning(model):
        kwargs["reasoning"] = {"effort": REASONING_EFFORT}
        print(f"[{now_ts()}] [DEBUG] reasoning.effort={REASONING_EFFORT}")

    resp = client.responses.create(**kwargs)

    raw = getattr(resp, "output_text", "") or ""
    if not raw:
        # fallback parse for other SDK shapes
        parts = []
        for out in getattr(resp, "output", []) or []:
            for c in getattr(out, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    parts.append(getattr(c, "text", "") or "")
        raw = "\n\n".join([p for p in parts if p])
    print(f"[{now_ts()}] [DEBUG] Raw response length: {len(raw)}")
    return raw

def try_models_with_retries(
    client: OpenAI,
    models: List[str],
    system_msg: str,
    user_full: str,
    user_trimmed: str,
    user_min: str
) -> Tuple[str, str]:
    errors: List[str] = []
    for m in models:
        for variant, msg in [("full", user_full), ("trimmed", user_trimmed)]:
            try:
                raw = responses_create_text(client, m, system_msg, msg)
                code = extract_code_block(raw)
                skeletal, issues = code_is_skeletal(code)
                if code and not skeletal:
                    print(f"[{now_ts()}] [INFO] Code block OK with model={m} variant={variant}")
                    return m, code
                # else repair attempt
                print(f"[{now_ts()}] [WARN] Code failed validation ({issues}). Retrying with REPAIR prompt.")
                repair_prompt = msg + "\n\nREPAIR:\n- Expand to a full, production-quality script.\n- Fix: " + "; ".join(issues) + "\n- Return ONE code block only."
                raw2 = responses_create_text(client, m, system_msg, repair_prompt)
                code2 = extract_code_block(raw2)
                skeletal2, issues2 = code_is_skeletal(code2)
                if code2 and not skeletal2:
                    print(f"[{now_ts()}] [INFO] Code block OK after repair with model={m} variant={variant}")
                    return m, code2
                print(f"[{now_ts()}] [WARN] Still skeletal after repair ({issues2}).")
            except Exception as e:
                print(f"[{now_ts()}] [ERROR] Responses call failed (model={m}, variant={variant}): {e}", file=sys.stderr)
                errors.append(f"{m}/{variant}: {e}")

        # minimal emergency
        try:
            raw3 = responses_create_text(client, m, system_msg, user_min)
            code3 = extract_code_block(raw3)
            skeletal3, issues3 = code_is_skeletal(code3)
            if code3 and not skeletal3:
                print(f"[{now_ts()}] [INFO] Code block OK with MINIMAL prompt (model={m})")
                return m, code3
            print(f"[{now_ts()}] [WARN] Minimal prompt also skeletal ({issues3}).")
        except Exception as e:
            print(f"[{now_ts()}] [ERROR] Minimal prompt failed (model={m}): {e}", file=sys.stderr)
            errors.append(f"{m}/minimal: {e}")

    raise RuntimeError("All model attempts failed. Last errors:\n" + "\n".join(errors))

# ---------- generate + run ----------
def generate_and_run(user_prompt: str, csv_paths: List[str], project_dir: Path) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key)

    # Build prompts
    system_msg       = build_system_message(project_dir)
    user_msg_full    = build_user_message(user_prompt, csv_paths, project_dir, trimmed=False)
    user_msg_trimmed = build_user_message(user_prompt, csv_paths, project_dir, trimmed=True)
    user_msg_minimal = build_minimal_user_message(user_prompt, csv_paths)

    # Model ladder
    model_list: List[str] = []
    if ENV_MODEL:
        model_list.append(ENV_MODEL)
    model_list += [m for m in FALLBACK_MODELS if m not in model_list]
    print(f"[{now_ts()}] [INFO] Model preference order: {model_list}")

    # Request code with validation + repair + fallbacks
    model_used, code_text = try_models_with_retries(
        client, model_list, system_msg, user_msg_full, user_msg_trimmed, user_msg_minimal
    )
    print(f"[{now_ts()}] [INFO] Using model: {model_used}")
    print(f"[{now_ts()}] [INFO] Final code length: {len(code_text)} chars")

    # Save code
    runs_dir = project_dir / RUNS_DIR
    ensure_dir(runs_dir)
    script_path = runs_dir / f"rtls_run_{now_stamp()}.py"
    script_path.write_text(code_text, encoding="utf-8")
    print(f"[{now_ts()}] [INFO] Wrote generated script to {script_path}")

    # Execute with project root on PYTHONPATH and INFOZONE_ROOT
    cmd = [sys.executable, str(script_path), user_prompt] + csv_paths
    print(f"[{now_ts()}] [INFO] Executing generated code:\n$ {' '.join(cmd)}\n(CWD) {project_dir}\n", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"]     = str(project_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["INFOZONE_ROOT"]  = str(project_dir)

    try:
        proc = subprocess.run(
            cmd, cwd=str(project_dir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=TIMEOUT_SEC
        )
    except subprocess.TimeoutExpired:
        print(f"[{now_ts()}] [ERROR] Generated script timed out after {TIMEOUT_SEC}s", file=sys.stderr)
        return 3

    # Echo output
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return proc.returncode

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate analysis code with the Responses API (validation + auto-retry) and execute locally.")
    ap.add_argument("prompt", help="User prompt for the analysis (quoted)")
    ap.add_argument("csv", nargs="+", help="CSV path(s)")
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parent
    csv_paths   = [str(Path(p).resolve()) for p in args.csv]

    rc = generate_and_run(args.prompt, csv_paths, project_dir)
    if rc != 0:
        print(f"\nERROR: Generated script exited with code {rc}.", file=sys.stderr)
    sys.exit(rc)

if __name__ == "__main__":
    main()
