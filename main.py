#!/usr/bin/env python3
# main.py — Responses-API generator/runner with validation + container-safe paths
from __future__ import annotations

import argparse
import os
import re
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
FALLBACK_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
TIMEOUT_SEC = int(os.environ.get("RTLS_CODE_TIMEOUT_SEC", "1800"))
RUNS_DIR = ".runs"
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "medium").lower()

def model_supports_reasoning(model: str) -> bool:
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

def strip_fences(code: str) -> str:
    return re.sub(r'^\s*```(?:python)?\s*|\s*```\s*$', '', code, flags=re.IGNORECASE | re.DOTALL).strip()

def code_is_skeletal(code: str) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if len(code) < 1500:
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
        "The script must obey guidelines.txt and the Output contract exactly. "
        "OUTPUT FORMAT (MANDATORY): Emit ONLY raw Python source — no prose, no Markdown fences."
    )
    return sys_prompt

def build_user_message(user_prompt: str, csv_paths: List[str], project_dir: Path, trimmed: bool=False) -> str:
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

    # --- Deliverables & hard rules (container-safe, with OUT_DIR support) ---
    INSTR = (
    "INSTRUCTIONS (MANDATORY — FOLLOW EXACTLY)\n"
    "-----------------------------------------\n"
    "You are **InfoZoneBuilder**. Generate ONE self-contained Python script that analyzes RTLS position data and writes\n"
    "one branded PDF plus PNGs for any charts. Return ONE code block and nothing else. Use local helper modules.\n"
    "\n"
    "PRINT LINKS ON SUCCESS ONLY (exact format):\n"
    "  print(f\"[Download the PDF](file:///{pdf_path.resolve().as_posix()})\")\n"
    "  print(f\"[Download Plot {i}](file:///{png_path.resolve().as_posix()})\")\n"
    "On failure, print only:\n"
    "  print(\"Error Report:\"); print(\"<short reason>\")\n"
    "  If schema-related, also: print(\"Columns detected: \" + \",\".join(df.columns.astype(str)))\n"
    "\n"
    "PATHS (container-safe):\n"
    "  # Resolve project root and enable local imports\n"
    "  import sys, os; from pathlib import Path\n"
    "  ROOT = Path(os.environ.get(\"INFOZONE_ROOT\", \"\")).resolve() if os.environ.get(\"INFOZONE_ROOT\") else Path(__file__).resolve().parent\n"
    "  if not (ROOT / \"guidelines.txt\").exists(): ROOT = ROOT.parent\n"
    "  if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\n"
    "  # OUT_DIR priority: INFOZONE_OUT_DIR (if set) else directory of first CSV\n"
    "  OUT_ENV = os.environ.get(\"INFOZONE_OUT_DIR\", \"\").strip()\n"
    "  out_dir = Path(OUT_ENV).resolve() if OUT_ENV else Path(csv_paths[0]).resolve().parent\n"
    "  out_dir.mkdir(parents=True, exist_ok=True)\n"
    "  # Asset paths\n"
    "  LOGO = ROOT / \"redpoint_logo.png\"\n"
    "\n"
    "MATPLOTLIB ≥3.9 SHIM (before any PDF rendering):\n"
    "  import matplotlib; matplotlib.use(\"Agg\")\n"
    "  from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA; import numpy as _np\n"
    "  _FCA.tostring_rgb = getattr(_FCA, \"tostring_rgb\", lambda self: _np.asarray(self.buffer_rgba())[..., :3].tobytes())\n"
    "  cmap_get = getattr(getattr(matplotlib, \"colormaps\", matplotlib), \"get_cmap\", None)\n"
    "\n"
    "INGEST (per-file, memory-safe):\n"
    "  from extractor import extract_tracks\n"
    "  raw = extract_tracks(csv_path)\n"
    "  import pandas as pd\n"
    "  df = pd.DataFrame(raw.get(\"rows\", []))\n"
    "  if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]\n"
    "  src = df[\"ts_iso\"] if \"ts_iso\" in df.columns else (df[\"ts\"] if \"ts\" in df.columns else pd.Series([], dtype=str))\n"
    "  df[\"ts_utc\"] = pd.to_datetime(src, utc=True, errors=\"coerce\")\n"
    "  # Required columns after first file: (trackable OR trackable_uid), trade, x, y\n"
    "  cols = set(df.columns.astype(str));\n"
    "  if not ((\"trackable\" in cols) or (\"trackable_uid\" in cols)) or (\"trade\" not in cols) or (\"x\" not in cols) or (\"y\" not in cols):\n"
    "      print(\"Error Report:\"); print(\"Missing required columns for analysis.\")\n"
    "      print(\"Columns detected: \" + \",\".join(df.columns.astype(str))); raise SystemExit(1)\n"
    "\n"
    "CHARTS → PNGs → PDF (order):\n"
    "  png = out_dir / f\"info_zone_report_{report_date}_plot{i:02d}.png\"\n"
    "  fig.savefig(str(png), dpi=120)  # do NOT close yet\n"
    "  sections.append({\"type\":\"charts\",\"title\":\"Figures\",\"figures\":figs})  # pass live Figures\n"
    "  pdf_path = out_dir / f\"info_zone_report_{report_date}.pdf\"\n"
    "  from pdf_creation_script import safe_build_pdf\n"
    "  try:\n"
    "      safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))\n"
    "  except Exception as e:\n"
    "      import traceback; print(\"Error Report:\"); print(f\"PDF build failed: {e.__class__.__name__}: {e}\"); traceback.print_exc(limit=2)\n"
    "      from report_limits import make_lite\n"
    "      try:\n"
    "          report = make_lite(report); safe_build_pdf(report, str(pdf_path), logo_path=str(LOGO))\n"
    "      except Exception as e2:\n"
    "          print(\"Error Report:\"); print(f\"Lite PDF failed: {e2.__class__.__name__}: {e2}\"); traceback.print_exc(limit=2); raise SystemExit(1)\n"
    "\n"
    "TIME:\n"
    "  Use dt.floor(\"h\") (lowercase h) — never \"H\".\n"
    "\n"
    "LEGEND/QUALITY:\n"
    "  Only call legend() when there are labeled artists; cap categories at ≤ 12.\n"
    "\n"
    "ONE BLOCK RULE:\n"
    "  Emit ONLY raw Python (no Markdown fences / prose). Start with imports.\n"
    )

    parts += ["", INSTR]
    return "\n".join(parts)

def build_minimal_user_message(user_prompt: str, csv_paths: List[str]) -> str:
    csv_lines = "\n".join(f" - {p}" for p in csv_paths)
    return f"""
Return ONE Python script in a single code block and nothing else.

Requirements:
- CLI: python generated.py "<USER_PROMPT>" /abs/csv1 [/abs/csv2 ...]
- Resolve ROOT from INFOZONE_ROOT or __file__; OUT_DIR = INFOZONE_OUT_DIR or first CSV dir.
- Import local helpers; save PDF/PNGs to OUT_DIR; print file:/// links exactly.
- Per-file processing (memory-safe); no /mnt/data; use dt.floor("h").

CSV INPUTS:
{csv_lines}
""".strip()

# ---------- Responses API ----------
def responses_create_text(client: OpenAI, model: str, system_msg: str, user_msg: str) -> str:
    print(f"[{now_ts()}] [DEBUG] Calling Responses.create with model={model}")
    print(f"[{now_ts()}] [DEBUG] System chars: {len(system_msg)} | User chars: {len(user_msg)}")
    kwargs = {
        "model": model,
        "input": [{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        "max_output_tokens": 12000,
    }
    if model_supports_reasoning(model):
        kwargs["reasoning"] = {"effort": REASONING_EFFORT}
        print(f"[{now_ts()}] [DEBUG] reasoning.effort={REASONING_EFFORT}")
    resp = client.responses.create(**kwargs)

    raw = getattr(resp, "output_text", "") or ""
    if not raw:
        parts = []
        for out in getattr(resp, "output", []) or []:
            for c in getattr(out, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    parts.append(getattr(c, "text", "") or "")
        raw = "\n\n".join([p for p in parts if p])
    print(f"[{now_ts()}] [DEBUG] Raw response length: {len(raw)}")
    return raw

def try_models_with_retries(client: OpenAI, models: List[str],
                            system_msg: str, user_full: str, user_trimmed: str, user_min: str) -> Tuple[str, str]:
    errors: List[str] = []
    for m in models:
        for variant, msg in [("full", user_full), ("trimmed", user_trimmed)]:
            try:
                raw = responses_create_text(client, m, system_msg, msg)
                code = strip_fences(extract_code_block(raw))
                skeletal, issues = code_is_skeletal(code)
                if code and not skeletal:
                    print(f"[{now_ts()}] [INFO] Code block OK with model={m} variant={variant}")
                    return m, code
                # repair attempt
                print(f"[{now_ts()}] [WARN] Code failed validation ({issues}). Retrying with REPAIR prompt.")
                repair_prompt = msg + "\n\nREPAIR:\n- Expand to full, production-quality script.\n- Fix: " + "; ".join(issues) + "\n- Return ONE code block only (no fences)."
                raw2 = responses_create_text(client, m, system_msg, repair_prompt)
                code2 = strip_fences(extract_code_block(raw2))
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
            code3 = strip_fences(extract_code_block(raw3))
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

    system_msg       = build_system_message(project_dir)
    user_msg_full    = build_user_message(user_prompt, csv_paths, project_dir, trimmed=False)
    user_msg_trimmed = build_user_message(user_prompt, csv_paths, project_dir, trimmed=True)
    user_msg_minimal = build_minimal_user_message(user_prompt, csv_paths)

    model_list: List[str] = []
    if ENV_MODEL:
        model_list.append(ENV_MODEL)
    model_list += [m for m in FALLBACK_MODELS if m not in model_list]
    print(f"[{now_ts()}] [INFO] Model preference order: {model_list}")

    model_used, code_text = try_models_with_retries(client, model_list, system_msg, user_msg_full, user_msg_trimmed, user_msg_minimal)
    print(f"[{now_ts()}] [INFO] Using model: {model_used}")
    print(f"[{now_ts()}] [INFO] Final code length: {len(code_text)} chars")

    # Compile preflight (stop fence bugs & obvious syntax)
    try:
        compile(code_text, "<generated>", "exec")
    except SyntaxError as e:
        print(f"[{now_ts()}] [ERROR] Compile failed: {e.msg} at line {e.lineno}: {e.text}", file=sys.stderr)
        return 4

    runs_dir = project_dir / RUNS_DIR
    ensure_dir(runs_dir)
    script_path = runs_dir / f"rtls_run_{now_stamp()}.py"
    script_path.write_text(code_text, encoding="utf-8")
    print(f"[{now_ts()}] [INFO] Wrote generated script to {script_path}")

    # Execute with project root on PYTHONPATH and INFOZONE_ROOT; pass through INFOZONE_OUT_DIR if set
    cmd = [sys.executable, str(script_path), user_prompt] + csv_paths
    print(f"[{now_ts()}] [INFO] Executing generated code:\n$ {' '.join(cmd)}\n(CWD) {project_dir}\n", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"]    = str(project_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["INFOZONE_ROOT"] = str(project_dir)

    proc = subprocess.run(
        cmd, cwd=str(project_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=TIMEOUT_SEC
    )

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")

    return proc.returncode

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate analysis code and execute locally (container-safe paths).")
    ap.add_argument("prompt", help="User prompt for the analysis (quoted)")
    ap.add_argument("csv", nargs="+", help="CSV path(s)")
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parent
    # Always pass absolute, strict CSV paths
    csv_paths   = [str(Path(p).resolve()) for p in args.csv]

    rc = generate_and_run(args.prompt, csv_paths, project_dir)
    if rc != 0:
        print(f"\nERROR: Generated script exited with code {rc}.", file=sys.stderr)
    sys.exit(rc)

if __name__ == "__main__":
    main()
