#!/usr/bin/env python3
# main.py
"""
CLI launcher that uses the **OpenAI Assistants API** to ask an Assistant
(to which we attach your system prompt + local helper files) to generate
ONE self-contained Python script. We then execute that returned script
**locally** against your CSVs and save the PDF + PNGs into the directory
of the first CSV.

USAGE
------
python main.py "Prompt message here" /abs/path/to/positions_YYYY-MM-DD.csv [/abs/path/to/another.csv ...]

WHAT THIS DOES
--------------
1) Creates (or reuses) an Assistant with your system prompt.
2) Uploads your **local helper files** as Assistant knowledge, so the model
   can read the rules and shape the generated code correctly:
     - guidelines.txt
     - context.txt
     - chart_policy.py
     - extractor.py
     - floorplans.json
     - pdf_creation_script.py
     - report_config.json
     - report_limits.py
     - zones_process.py
     - zones.json
     - floorplan.(jpeg|jpg|png)  (whichever exists)
     - example.csv               (for schema orientation only)
3) Sends a user message containing your prompt and the **absolute paths** to your CSVs.
   (We do NOT upload data files; paths are provided so the generated code reads them **locally**.)
4) Retrieves the Assistant’s reply, extracts the single Python code block, writes it to
   `.runs/rtls_run_<timestamp>.py`, and executes:
     python rtls_run_<ts>.py "<PROMPT>" <CSV1> <CSV2> ...
5) Prints the model’s own output, which must end with the file:// links for the PDF/PNGs.

REQUIREMENTS
------------
- Python 3.9+
- OpenAI Python SDK (>=1.0):  pip install openai
- Environment variable:  OPENAI_API_KEY=<your key>

NOTES
-----
- The Assistant **must not** rely on code_interpreter; we want code **text** only.
- The generated code must import and use the local helpers (do not re-implement).
- The generated code must obey **guidelines.txt** and your **system prompt**.
- The generated code must save the PDF and PNGs into the **directory of the first CSV**.
- This launcher is structured so a future web UI can call the `generate_and_run()` function directly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
import textwrap
from typing import List, Optional

# OpenAI SDK (>=1.0)
try:
    from openai import OpenAI
except Exception:
    print("ERROR: OpenAI SDK not found. Install with: pip install openai", file=sys.stderr)
    raise

# ------------------------- Config -------------------------
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # use a modern reasoning-capable model if available
TIMEOUT_SEC = int(os.environ.get("RTLS_CODE_TIMEOUT_SEC", "1800"))  # 30 minutes for large weekly runs
RUNS_DIR = ".runs"
ASSISTANT_NAME = os.environ.get("RTLS_ASSISTANT_NAME", "InfoZoneBuilder CodeGen")

# --------------------- Helpers ----------------------------
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def extract_code_block(text: str) -> str:
    """
    Extract the first ```python ... ``` block, or first generic ``` ... ``` block.
    If none found, return the whole text (best effort).
    """
    fence_py = "```python"
    fence_any = "```"
    if fence_py in text:
        start = text.find(fence_py) + len(fence_py)
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    if fence_any in text:
        start = text.find(fence_any) + len(fence_any)
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return text.strip()

# --------------------- OpenAI: Assistants ------------------
def upload_file_for_assistants(client: OpenAI, path: Path) -> Optional[str]:
    try:
        with path.open("rb") as f:
            fi = client.files.create(purpose="assistants", file=f)
        return fi.id
    except Exception as e:
        # Non-fatal; we just skip missing assets like optional floorplan type
        print(f"[warn] Could not upload {path.name}: {e}", file=sys.stderr)
        return None

def build_assistant(client: OpenAI, system_prompt: str, file_ids: List[str]) -> str:
    """
    Create an Assistant each run for simplicity (keeps instructions & files in sync).
    You can persist assistant.id if you prefer reusing it.
    """
    asst = client.beta.assistants.create(
        name=ASSISTANT_NAME,
        model=DEFAULT_MODEL,
        instructions=system_prompt,
        tools=[],                 # no code_interpreter; we want code text only
        file_ids=file_ids or []   # attach helper files for grounding
    )
    return asst.id

def run_assistant_code_request(
    client: OpenAI,
    assistant_id: str,
    user_prompt: str,
    csv_paths: List[str],
    guidelines_txt: str,
    context_txt: str,
    floorplans_excerpt: str
) -> str:
    """
    Send a message that instructs the Assistant to emit ONE Python file that:
      - imports & uses local helpers
      - respects guidelines.txt
      - reads CSVs from argv
      - saves outputs next to first CSV
      - prints file:// links in the required order
    """
    # Build the request message
    csv_lines = "\n".join(f" - {p}" for p in csv_paths)
    user_message = f"""
USER PROMPT
-----------
{user_prompt}

CSV INPUTS (absolute paths)
---------------------------
{csv_lines}

MANDATORY RULES (verbatim from guidelines.txt)
----------------------------------------------
{guidelines_txt}

BACKGROUND CONTEXT (excerpt)
----------------------------
{context_txt[:2000]}

FLOORPLAN JSON (excerpt)
------------------------
{floorplans_excerpt[:2000]}

DELIVERABLE
-----------
Return **ONE** Python script in a single code block. It must:
- Accept CLI: python generated.py "<USER_PROMPT>" /abs/csv1 [/abs/csv2 ...]
- Import and use ONLY the local helpers co-located with main.py:
    extractor.py, pdf_creation_script.py, zones_process.py,
    chart_policy.py, report_limits.py, report_config.json,
    floorplans.json, floorplan.(jpeg|jpg|png), redpoint_logo.png,
    context.txt, guidelines.txt
- Read CSVs **locally** from argv; do **not** upload data.
- Save the primary PDF and any PNGs in the **directory of the first CSV**.
- On success, print exactly:
    [Download the PDF](file:///ABS/PATH/TO/PDF)
    [Download Plot 1](file:///ABS/PATH/TO/PNG1)
    [Download Plot 2](file:///ABS/PATH/TO/PNG2)
    ...
  If no figures: only the PDF line. On failure: a short Error Report.
- Respect **large-data mode** (process per-file, stream intervals/aggregates, do not hold everything in RAM).
- Follow **guidelines.txt** precisely (duplicate-name guard; ts_utc; polygon list[tuple(float,float)] only; tables list-of-dicts; figures→PNG→PDF; link order).
No commentary—**return one code block only**.
""".strip()

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread.id, role="user", content=user_message)
    run = client.beta.threads.runs.create(thread.id, assistant_id=assistant_id)

    # Poll until done
    while True:
        run = client.beta.threads.runs.retrieve(thread.id, run.id)
        if run.status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(1.2)

    if run.status != "completed":
        raise RuntimeError(f"Assistant run did not complete: {run.status}")

    msgs = client.beta.threads.messages.list(thread.id)
    # The latest assistant message typically contains the code
    code_text = ""
    for m in msgs.data:
        if m.role == "assistant":
            # Concatenate all text parts from the message
            parts = []
            for c in m.content:
                if getattr(c, "type", None) == "text":
                    parts.append(c.text.value)
            if parts:
                code_text = "\n\n".join(parts)
                break

    if not code_text.strip():
        raise RuntimeError("Assistant returned empty content (no code).")
    return extract_code_block(code_text)

# --------------------- Generation+Execution ----------------------
def generate_and_run(user_prompt: str, csv_paths: List[str], project_dir: Path, model: str) -> int:
    """End-to-end: upload helpers, create assistant, request code, save & execute locally."""
    # 1) Prepare OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key)

    # 2) Resolve files to upload
    helpers = [
        "guidelines.txt",
        "context.txt",
        "chart_policy.py",
        "extractor.py",
        "floorplans.json",
        "pdf_creation_script.py",
        "report_config.json",
        "report_limits.py",
        "zones_process.py",
        "zones.json",
        "example.csv",
    ]
    # Include a floorplan raster if present (jpeg/jpg/png)
    floorplan_candidates = [project_dir / "floorplan.jpeg", project_dir / "floorplan.jpg", project_dir / "floorplan.png"]
    floorplan_file = first_existing(floorplan_candidates)
    if floorplan_file:
        helpers.append(floorplan_file.name)
    logo = project_dir / "redpoint_logo.png"
    if logo.exists():
        helpers.append(logo.name)

    file_ids = []
    for fname in helpers:
        p = project_dir / fname
        if not p.exists():
            print(f"[warn] helper missing: {fname}", file=sys.stderr)
            continue
        fid = upload_file_for_assistants(client, p)
        if fid:
            file_ids.append(fid)

    # 3) Build Assistant with your **system prompt**
    system_prompt = read_text(project_dir / "system_prompt.txt")
    if not system_prompt.strip():
        # fallback to a minimal injected prompt if file missing
        system_prompt = "You are a code generator that returns one Python script as a single code block."

    assistant_id = build_assistant(client, system_prompt, file_ids)

    # 4) Build user message + include excerpts to ground generation
    guidelines_txt = read_text(project_dir / "guidelines.txt")
    context_txt = read_text(project_dir / "context.txt")
    floorplans_excerpt = read_text(project_dir / "floorplans.json")[:4000]

    # 5) Request code from Assistant
    code_text = run_assistant_code_request(
        client=client,
        assistant_id=assistant_id,
        user_prompt=user_prompt,
        csv_paths=csv_paths,
        guidelines_txt=guidelines_txt,
        context_txt=context_txt,
        floorplans_excerpt=floorplans_excerpt,
    )

    # 6) Save generated code to .runs and execute locally
    runs_dir = project_dir / ".runs"
    ensure_dir(runs_dir)
    script_path = runs_dir / f"rtls_run_{now_stamp()}.py"
    script_path.write_text(code_text, encoding="utf-8")

    cmd = [sys.executable, str(script_path), user_prompt] + csv_paths
    print(f"\nExecuting generated code:\n$ {' '.join(cmd)}\n(CWD) {project_dir}\n", flush=True)
    proc = subprocess.run(cmd, cwd=str(project_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=TIMEOUT_SEC)

    # Echo Assistant script output
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return proc.returncode

# --------------------------- CLI ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate analysis code with the Assistants API and execute it locally.")
    ap.add_argument("prompt", help="User prompt for the analysis (quoted)")
    ap.add_argument("csv", nargs="+", help="CSV path(s)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Assistant model (default: {DEFAULT_MODEL})")
    ap.add_argument("--dry", action="store_true", help="Create Assistant & files but do not execute generated code")
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parent
    # Ensure a copy of your system prompt is stored next to main.py
    sys_prompt_file = project_dir / "system_prompt.txt"
    if not sys_prompt_file.exists():
        print("[warn] system_prompt.txt not found. Using a minimal placeholder.", file=sys.stderr)

    # Resolve absolute CSV paths (we do not upload them)
    csv_paths = [str(Path(p).resolve()) for p in args.csv]

    # Generate & run
    rc = generate_and_run(args.prompt, csv_paths, project_dir, args.model)
    if rc != 0:
        print(f"\nERROR: Generated script exited with code {rc}.", file=sys.stderr)
    sys.exit(rc)

if __name__ == "__main__":
    main()
