#!/usr/bin/env python3
# main.py — InfoZone generator/runner with compile+runtime repair (GC)
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

# ---------- OpenAI SDK ----------
try:
    from openai import OpenAI
except Exception:
    print("ERROR: OpenAI SDK import failed. Install: pip install --upgrade openai", file=sys.stderr)
    raise

# ---------- Tunables / Defaults ----------
FALLBACK_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
ENV_MODEL = os.environ.get("OPENAI_MODEL", "").strip()

# Reasoning effort & output tokens (YOUR NEW LIMITS KEPT)
REASONING_EFFORT = os.environ.get(
    "IZ_REASONING_EFFORT",
    os.environ.get("OPENAI_REASONING_EFFORT", "medium")
).lower()
MAX_OUTPUT_TOKENS = int(os.environ.get(
    "IZ_MAX_OUTPUT_TOKENS",
    os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "24000")  # default 24k
))

# Size caps (YOUR NEW LIMITS KEPT)
GUIDELINES_CAP = int(os.environ.get("IZ_GUIDELINES_CAP", "0") or "0")  # 0 = no cap
CONTEXT_CAP    = int(os.environ.get("IZ_CONTEXT_CAP", "30000"))
HELPER_CAP     = int(os.environ.get("IZ_HELPER_CAP", "30000"))

# Repair attempts
COMPILE_RETRIES = int(os.environ.get("IZ_COMPILE_RETRIES", "3"))
RUNTIME_RETRIES = int(os.environ.get("IZ_REPAIR_RETRIES", "3"))

# Caps for repair payloads
REPAIR_CODE_CAP = int(os.environ.get("IZ_REPAIR_CODE_CAP", "40000"))
REPAIR_LOG_CAP  = int(os.environ.get("IZ_REPAIR_LOG_CAP", "20000"))

TIMEOUT_SEC = int(os.environ.get("RTLS_CODE_TIMEOUT_SEC", "1800"))
RUNS_DIR = ".runs"

# ---------- Small helpers ----------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path, max_chars: int | None = None) -> str:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if max_chars and max_chars > 0 and len(txt) > max_chars:
        return txt[:max_chars]
    return txt

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

def _clip(s: str, cap: int) -> str:
    if cap and len(s) > cap:
        return s[:cap] + f"\n\n…[truncated to {cap} chars]…"
    return s

def code_is_skeletal(code: str) -> Tuple[bool, List[str]]:
    """Lightweight validation to nudge repairs before execution."""
    issues: List[str] = []
    if len(code) < 1500:
        issues.append(f"too short ({len(code)} chars)")
    low = code.lower()
    if "from extractor import extract_tracks" not in low:
        issues.append("missing extractor import")
    if "from pdf_creation_script import safe_build_pdf" not in low:
        issues.append("missing pdf builder import")
    if "/mnt/data" in low or "sandbox:" in low:
        issues.append("forbidden path token (/mnt_data or sandbox:)")
    if "infozone_out_dir" not in low and "out_env" not in low:
        issues.append("missing OUT_DIR (INFOZONE_OUT_DIR) logic")
    return (len(issues) > 0), issues

def model_supports_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return any(tag in m for tag in ("gpt-5", "o4", "o3", "reasoning", "thinking"))

def _cap_size(trimmed: bool, full_default: int, trim_default: int) -> int:
    return trim_default if trimmed else full_default

# ---------- Prompt builders ----------
def build_system_message(project_dir: Path) -> str:
    sys_prompt = read_text(
        project_dir / "system_prompt.txt",
        max_chars=(GUIDELINES_CAP if GUIDELINES_CAP > 0 else None)
    ).strip()
    if not sys_prompt:
        sys_prompt = "You are a code generator that returns one Python script as a single code block."
    sys_prompt += (
        "\n\nOUTPUT FORMAT (MANDATORY): Emit ONLY raw Python source — no prose, no Markdown fences."
        " Begin directly with imports; if you would include fences, OMIT them."
    )
    return sys_prompt

def build_user_message(user_prompt: str, csv_paths: List[str], project_dir: Path, trimmed: bool=False) -> str:
    context_cap = _cap_size(trimmed, CONTEXT_CAP, 4000)
    helper_cap  = _cap_size(trimmed, HELPER_CAP, 4000)

    guidelines = read_text(
        project_dir / "guidelines.txt",
        max_chars=(GUIDELINES_CAP if GUIDELINES_CAP > 0 else None)
    )
    context    = read_text(project_dir / "context.txt", max_chars=context_cap)

    helpers = [
        ("extractor.py",           helper_cap),
        ("pdf_creation_script.py", helper_cap),
        ("chart_policy.py",        helper_cap),
        ("zones_process.py",       helper_cap),
        ("report_limits.py",       (2000 if trimmed else 4000)),
        ("report_config.json",     (2000 if trimmed else 4000)),
        ("floorplans.json",        (2000 if trimmed else 4000)),
        ("zones.json",             (2000 if trimmed else 4000)),
    ]
    helper_snips: List[str] = []
    for fname, cap in helpers:
        txt = read_text(project_dir / fname, max_chars=cap)
        if txt:
            helper_snips += [f"\n>>> {fname}\n", txt]

    floorplan = next(
        (project_dir / n for n in ("floorplan.jpeg", "floorplan.jpg", "floorplan.png") if (project_dir / n).exists()),
        None
    )
    assets_lines = "\n".join([
        f" - floorplan.(jpeg|jpg|png) : {'present' if floorplan else 'missing'}",
        f" - redpoint_logo.png : {'present' if (project_dir/'redpoint_logo.png').exists() else 'missing'}",
        f" - trackable_objects.json : {'present' if (project_dir/'trackable_objects.json').exists() else 'missing'}",
    ])

    csv_lines = "\n".join(f" - {p}" for p in csv_paths) or "(none)"

    INSTR = (
        "INSTRUCTIONS (MANDATORY — FOLLOW EXACTLY)\n"
        "-----------------------------------------\n"
        "You are InfoZoneBuilder. Generate ONE self-contained Python script that analyzes RTLS position data and writes\n"
        "one branded PDF plus PNGs. Use local helpers; never network or /mnt/data. Return ONE code block only (no fences).\n"
        "\n"
        "PATHS (container-safe):\n"
        "  import sys, os; from pathlib import Path\n"
        "  ROOT = Path(os.environ.get('INFOZONE_ROOT', '')).resolve() if os.environ.get('INFOZONE_ROOT') else Path(__file__).resolve().parent\n"
        "  if not (ROOT / 'guidelines.txt').exists(): ROOT = ROOT.parent\n"
        "  if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\n"
        "  OUT_ENV = os.environ.get('INFOZONE_OUT_DIR', '').strip()\n"
        "  out_dir = Path(OUT_ENV).resolve() if OUT_ENV else (Path(csv_paths[0]).resolve().parent if csv_paths else ROOT)\n"
        "  out_dir.mkdir(parents=True, exist_ok=True)\n"
        "  LOGO = ROOT / 'redpoint_logo.png'; FLOORJSON = ROOT / 'floorplans.json'; ZONES_JSON = ROOT / 'zones.json'\n"
        "\n"
        "MATPLOTLIB ≥3.9 SHIM:\n"
        "  import matplotlib; matplotlib.use('Agg')\n"
        "  from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA; import numpy as _np\n"
        "  _FCA.tostring_rgb = getattr(_FCA,'tostring_rgb', lambda self: _np.asarray(self.buffer_rgba())[..., :3].tobytes())\n"
        "  import matplotlib as _mpl; _get_cmap = getattr(getattr(_mpl,'colormaps',_mpl),'get_cmap',None)\n"
        "\n"
        "## --- DB auto-select from ROOT/db (HYphen filenames; inclusive ranges; ASSUME 2025 when missing) ---\n"
        "from pathlib import Path as _P\n"
        "import re as _re, datetime as _dt\n"
        "DB_DIR = ROOT / 'db'; DEFAULT_YEAR = 2025\n"
        "def _normalize_text(s:str)->str:\n"
        "    for k,v in {'thru':'through','’':'\\'','–':'-','—':'-'}.items(): s=s.replace(k,v)\n"
        "    return s\n"
        "def _parse_dates_from_text(txt: str):\n"
        "    t = _normalize_text(txt).lower()\n"
        "    ymd = [tuple(map(int, m.groups())) for m in _re.finditer(r'(\\d{4})[-/](\\d{1,2})[-/](\\d{1,2})', t)]\n"
        "    for m in _re.finditer(r'(\\d{1,2})[-/](\\d{1,2})[-/](\\d{4})', t): ymd.append((int(m.group(3)), int(m.group(1)), int(m.group(2))))\n"
        "    md  = [tuple(map(int, m.groups())) for m in _re.finditer(r'\\b(\\d{1,2})[-/](\\d{1,2})\\b', t)]\n"
        "    for m in _re.finditer(r'\\b([a-z]{3,9})\\s+(\\d{1,2})(?:st|nd|rd|th)?\\b', t):\n"
        "        _M={'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12,\n"
        "            'january':1,'february':2,'march':3,'april':4,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}\n"
        "        mon=_M.get(m.group(1).lower());\n"
        "        if mon: md.append((mon, int(m.group(2))))\n"
        "    rng = _re.search(r'(?P<a>.+?)\\s+(?:to|through|thru|\\-|–|—|\\.\\.|between)\\s+(?P<b>.+)', t)\n"
        "    return {'ymd': ymd, 'md': md, 'range': rng is not None}\n"
        "R_YYYY = _re.compile(r'(?:positions|postions)_(\\d{4})-(\\d{1,2})-(\\d{1,2})\\.csv$', _re.I)\n"
        "R_MD   = _re.compile(r'(?:positions|postions)_(\\d{1,2})-(\\d{1,2})\\.csv$', _re.I)\n"
        "def _index_db():\n"
        "    files = list(DB_DIR.glob('*.csv')); ymd = {}; md = {}\n"
        "    for f in files:\n"
        "        n=f.name.lower(); m=R_YYYY.match(n)\n"
        "        if m:\n"
        "            yyyy,mm,dd=map(int,m.groups()); ymd[(yyyy,mm,dd)]=f; md.setdefault((mm,dd),[]).append((yyyy,f)); continue\n"
        "        m=R_MD.match(n)\n"
        "        if m:\n"
        "            mm,dd=map(int,m.groups()); md.setdefault((mm,dd),[]).append((None,f))\n"
        "    for k in md: md[k].sort(key=lambda t: (t[0] is None, t[0]), reverse=True)\n"
        "    return ymd, md\n"
        "_user_args = sys.argv[2:]; _dirs = [p for p in _user_args if _P(p).is_dir()]\n"
        "if (not _user_args) or _dirs:\n"
        "    _dates = _parse_dates_from_text(user_prompt); _by_ymd, _by_md = _index_db(); _want = []\n"
        "    if _dates['ymd']:\n"
        "        _ymd = sorted(_dates['ymd'])\n"
        "        if len(_ymd)==1: y,m,d=_ymd[0]; _want.append((y,m,d))\n"
        "        else:\n"
        "            (y1,m1,d1),(y2,m2,d2) = _ymd[0], _ymd[-1]\n"
        "            a=_dt.date(y1,m1,d1); b=_dt.date(y2,m2,d2)\n"
        "            if b<a: a,b=b,a\n"
        "            cur=a\n"
        "            for _ in range(400):\n"
        "                _want.append((cur.year,cur.month,cur.day))\n"
        "                if cur==b: break\n"
        "                cur += _dt.timedelta(days=1)\n"
        "    elif _dates['md']:\n"
        "        if _dates['range'] and len(_dates['md'])>=2:\n"
        "            (m1,d1),(m2,d2) = _dates['md'][0], _dates['md'][1]\n"
        "            y=DEFAULT_YEAR; a=_dt.date(y,m1,d1); b=_dt.date(y,m2,d2)\n"
        "            if b<a: a,b=b,a\n"
        "            cur=a\n"
        "            for _ in range(400):\n"
        "                _want.append((cur.year,cur.month,cur.day))\n"
        "                if cur==b: break\n"
        "                cur += _dt.timedelta(days=1)\n"
        "        else:\n"
        "            for mm,dd in _dates['md']: _want.append((DEFAULT_YEAR,mm,dd))\n"
        "    _chosen=[]\n"
        "    for y,mm,dd in _want:\n"
        "        if (y,mm,dd) in _by_ymd: _chosen.append(_by_ymd[(y,mm,dd)])\n"
        "        elif (mm,dd) in _by_md:\n"
        "            cand=[p for yr,p in _by_md[(mm,dd)] if yr==DEFAULT_YEAR]; _chosen.append(cand[0] if cand else _by_md[(mm,dd)][0][1])\n"
        "    _chosen=[str(_P(p).resolve()) for p in _chosen if p]; _chosen=list(dict.fromkeys(_chosen))\n"
        "    if not _chosen:\n"
        "        print('DB DEBUG — found in /app/db:', ', '.join(f.name for f in DB_DIR.glob('*.csv')) or '(none)')\n"
        "        print('DB DEBUG — wanted days:', _want)\n"
        "        print('Error Report:'); print('No matching CSVs found in db for requested date(s).'); raise SystemExit(1)\n"
        "    _extra=[str(_P(p).resolve()) for p in _user_args if _P(p).is_file()]\n"
        "    csv_paths = _chosen + _extra\n"
        "    print('SELECTED FROM DB:', ', '.join(_P(p).name for p in csv_paths[:20]))\n"
        "\n"
        "INGEST:\n"
        "  from extractor import extract_tracks\n"
        "  raw = extract_tracks(csv_path, mac_map_path=str(ROOT / 'trackable_objects.json'))\n"
        "  audit = raw.get('audit', {}) or {}\n"
        "  if not audit.get('mac_map_loaded', False) or int(audit.get('mac_hits', 0)) == 0:\n"
        "      print('Error Report:'); print('MAC map not loaded or no MACs matched; ensure trackable_objects.json is in the app root and passed explicitly.'); raise SystemExit(1)\n"
        "  print(f\"AUDIT mac_map_loaded={audit.get('mac_map_loaded')} mac_col={audit.get('mac_col_selected')} macs_seen={audit.get('macs_seen')} mac_hits={audit.get('mac_hits')} uids_seen={audit.get('uids_seen')} uid_hits={audit.get('uid_hits')} rows_total={audit.get('rows_total')} trade_rate={audit.get('trade_nonempty_rate')}\")\n"
        "  import pandas as pd\n"
        "  df = pd.DataFrame(raw.get('rows', []))\n"
        "  if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]\n"
        "\n"
        "  # GC: SAFE floor crop — keep x>=12000 & y>=15000; handle empty/missing columns without errors\n"
        "  def _safe_floor_crop(df):\n"
        "      import pandas as pd\n"
        "      if 'x' not in df.columns or 'y' not in df.columns:\n"
        "          return df.iloc[0:0].copy() if len(df)==0 else df.copy()\n"
        "      xn = pd.to_numeric(df['x'], errors='coerce')\n"
        "      yn = pd.to_numeric(df['y'], errors='coerce')\n"
        "      mask = (xn >= 12000) & (yn >= 15000)\n"
        "      if not hasattr(mask, 'index') or mask.shape != df.index.shape:\n"
        "          return df.iloc[0:0].copy()\n"
        "      return df.loc[mask].copy()\n"
        "  df = _safe_floor_crop(df)\n"
        "\n"
        "  # QUICK EMPTY GUARD — if the file contributed no rows, skip it cleanly\n"
        "  if df.shape[0] == 0:\n"
        "      continue\n"
        "\n"
        "  # Ensure required columns EXIST before schema validation (may remain empty)\n"
        "  for _col in ('trackable_uid','trackable','trade','zone_name','x','y'):\n"
        "      if _col not in df.columns:\n"
        "          df[_col] = ''\n"
        "\n"
        "  # Timestamp canon\n"
        "  src = df['ts_iso'] if 'ts_iso' in df.columns else (df['ts'] if 'ts' in df.columns else '')\n"
        "  df['ts_utc'] = pd.to_datetime(src, utc=True, errors='coerce')\n"
        "\n"
        "  # Schema validation (columns, not row counts)\n"
        "  cols = set(df.columns.astype(str))\n"
        "  if not ((('trackable' in cols) or ('trackable_uid' in cols)) and ('trade' in cols) and ('x' in cols) and ('y' in cols)):\n"
        "      print('Error Report:'); print('Missing required columns for analysis.'); print('Columns detected: ' + ','.join(df.columns.astype(str))); raise SystemExit(1)\n"
        "\n"
        "  # Optional date-window filter could go here (skip silently if it empties the frame)\n"
        "  if df.empty:\n"
        "      continue\n"
        "\n"
        "  # From here your aggregation/figures logic continues ...\n"
    )

    parts: List[str] = []
    parts += [
        "USER PROMPT",
        "-----------",
        user_prompt,
        "",
        "CSV INPUTS (absolute paths or empty if using DB auto-select)",
        "------------------------------------------------------------",
        csv_lines or "(none — will auto-select from ROOT/db if dates are present)",
        "",
        "LOCAL ASSETS (present/missing — read from disk)",
        "-----------------------------------------------",
        assets_lines,
        "",
        "MANDATORY RULES (guidelines.txt — full text)",
        "--------------------------------------------",
        (guidelines if GUIDELINES_CAP == 0 else (guidelines[:GUIDELINES_CAP] if guidelines else "")),
        "",
        "BACKGROUND CONTEXT (excerpt)",
        "----------------------------",
        context,
        "",
        "HELPER EXCERPTS (READ-ONLY; use these APIs — do NOT re-implement)",
        "-----------------------------------------------------------------",
    ]
    for s in helper_snips:
        parts.append(s)
    parts.append("")
    parts.append(INSTR)
    return "\n".join(parts)

def build_minimal_user_message(user_prompt: str, csv_paths: List[str]) -> str:
    csv_lines = "\n".join(f" - {p}" for p in csv_paths) or "(none — will auto-select from ROOT/db if dates are present)"
    return f"""
Return ONE Python script in a single code block and nothing else.

Requirements:
- CLI: python generated.py "<USER_PROMPT>" [/abs/csv1 [/abs/csv2 ...]]   # CSVs optional
- Resolve ROOT from INFOZONE_ROOT or __file__; OUT_DIR = INFOZONE_OUT_DIR or first CSV dir (mkdir -p).
- If csv_paths is empty or contains directories, parse dates from the prompt and auto-select from ROOT/db using hyphen filenames with inclusive ranges and year=2025 when missing.
- Import local helpers; save PDF/PNGs to OUT_DIR; print file:/// links exactly (use _print_links).
- Per-file processing; GC safe floor crop; create required columns before schema validation; use dt.floor("h"); tables only if explicitly requested.

CSV INPUTS:
{csv_lines}
""".strip()

# ---------- OpenAI call ----------
def responses_create_text(client: OpenAI, model: str, system_msg: str, user_msg: str) -> str:
    print(f"[{now_ts()}] [DEBUG] Calling Responses.create with model={model}")
    print(f"[{now_ts()}] [DEBUG] System chars: {len(system_msg)} | User chars: {len(user_msg)}")
    kwargs = {
        "model": model,
        "input": [{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if model_supports_reasoning(model):
        kwargs["reasoning"] = {"effort": REASONING_EFFORT}
        print(f"[{now_ts()}] [DEBUG] reasoning.effort={REASONING_EFFORT}")
    resp = client.responses.create(**kwargs)

    raw = getattr(resp, "output_text", "") or ""
    if not raw:
        parts: List[str] = []
        for out in getattr(resp, "output", []) or []:
            for c in getattr(out, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    parts.append(getattr(c, "text", "") or "")
        raw = "\n\n".join([p for p in parts if p])
    print(f"[{now_ts()}] [DEBUG] Raw response length: {len(raw)}")
    return raw

# ---------- Generate / Compile / Run / Robust Repairs ----------
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

                # Validation-level repair
                print(f"[{now_ts()}] [WARN] Code failed validation ({issues}). Retrying with REPAIR prompt.")
                repair_prompt = (
                    msg
                    + "\n\nREPAIR-STRUCTURE (MANDATORY):\n"
                    + "- Keep ROOT/out_dir/imports. Use DB auto-select (hyphen patterns), inclusive ranges, assume 2025; print 'SELECTED FROM DB:'.\n"
                    + "- Include MAC-map audit guard + smoke log; GC safe floor crop via _safe_floor_crop; ensure required columns exist BEFORE schema validation; use ts_utc; dt.floor('h').\n"
                    + "- Floorplan 'selected==1' selection; regex preflights; asset/DB sanity; safe _print_links.\n"
                    + "- The script MUST compile with compile(...,'exec').\n"
                    + f"- Structural issues to fix: {', '.join(issues)}\n"
                )
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

        # Minimal emergency
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

def compile_with_retries(client: OpenAI, model: str, system_msg: str,
                         user_msg_full: str, code_text: str) -> Tuple[bool, str]:
    """Compile; if it fails, run up to COMPILE_RETRIES syntax-repair attempts."""
    for attempt in range(1, COMPILE_RETRIES + 1):
        try:
            compile(code_text, "<generated>", "exec")
            return True, code_text
        except SyntaxError as e:
            err_line = (e.text or "").strip()
            err_msg  = f"{e.msg} at line {e.lineno}: {err_line}"
            print(f"[{now_ts()}] [ERROR] Compile failed (attempt {attempt}/{COMPILE_RETRIES}): {err_msg}", file=sys.stderr)

            broken_capped = _clip(code_text, REPAIR_CODE_CAP)
            repair_prompt = (
                "MAIN PRIORITY: FIX THE PYTHON SCRIPT (COMPILATION MUST SUCCEED)\n"
                "- Repair WITHOUT changing behavior/IO.\n"
                "- Strict PYTHON SYNTAX AUDIT before returning:\n"
                "  • All 'if/elif/else/try/except/finally/for/while/def/class' end with ':'\n"
                "  • Every block is indented; no empty blocks (use 'pass' if needed)\n"
                "  • Balanced (), [], {}, and quotes; no unterminated f-strings\n"
                "  • Every 'try:' has at least one 'except' or 'finally'\n"
                "  • Regex named groups use (?P<a>) / (?P<b>) — NEVER '(?P>)'\n"
                "  • Define & USE _print_links(pdf_path, png_paths) — no inline nested f-strings for links\n"
                "  • Constants exist: ROOT, LOGO, FLOORJSON, ZONES_JSON, out_dir (no misspellings)\n"
                "  • Keep DB auto-select (hyphen), inclusive ranges, ASSUME 2025; MAC-map guard; GC _safe_floor_crop; ensure required columns exist BEFORE schema validation\n"
                "  • Avoid utcnow(); if needed, use timezone-aware now\n"
                "\n--- COMPILER ERROR ---\n"
                f"{err_msg}\n"
                "\n--- BROKEN SCRIPT (capped) ---\n"
                f"{broken_capped}\n"
            )
            raw = responses_create_text(client, model, system_msg, user_msg_full + "\n\n" + repair_prompt)
            code_text = strip_fences(extract_code_block(raw))
    # final compile try
    try:
        compile(code_text, "<generated>", "exec")
        return True, code_text
    except SyntaxError as e2:
        print(f"[{now_ts()}] [ERROR] Compile failed after retries: {e2.msg} at line {e2.lineno}: {e2.text}", file=sys.stderr)
        return False, code_text

def run_script(script_path: Path, user_prompt: str, csv_paths: List[str], project_dir: Path) -> Tuple[int, str, str]:
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
    return proc.returncode, proc.stdout or "", proc.stderr or ""

def runtime_repair_loop(client: OpenAI, model: str, system_msg: str, user_msg_full: str,
                        code_text: str, project_dir: Path, user_prompt: str, csv_paths: List[str]) -> Tuple[int, str, str]:
    runs_dir = project_dir / RUNS_DIR
    ensure_dir(runs_dir)

    for attempt in range(1, RUNTIME_RETRIES + 1):
        script_path = runs_dir / f"rtls_run_{now_stamp()}.py"
        script_path.write_text(code_text, encoding="utf-8")
        print(f"[{now_ts()}] [INFO] Wrote generated script to {script_path}")

        rc, out, err = run_script(script_path, user_prompt, csv_paths, project_dir)
        if rc == 0:
            return rc, out, err

        print(f"[{now_ts()}] [WARN] Script exited rc={rc}. Attempting runtime repair ({attempt}/{RUNTIME_RETRIES}).")
        failed_code = script_path.read_text(encoding="utf-8", errors="ignore")
        failed_code_c = _clip(failed_code, REPAIR_CODE_CAP)
        out_c = _clip(out, REPAIR_LOG_CAP)
        err_c = _clip(err, REPAIR_LOG_CAP)

        repair_msg = (
            user_msg_full
            + "\n\n=== MAIN PRIORITY: FIX THE PYTHON SCRIPT ===\n"
            + "The following Python script failed at runtime. Do NOT change scope/outputs; ONLY fix the code so it runs end-to-end.\n"
            + "- Keep DB auto-select (hyphen), inclusive ranges, assume 2025; MAC-map guard; GC _safe_floor_crop; ensure required columns exist BEFORE schema validation; safe _print_links.\n"
            + "- Fix NameErrors from mis-typed loop variables (replace brittle comprehensions with helpers if needed).\n"
            + "- Ensure constants exist (ROOT, LOGO, FLOORJSON, ZONES_JSON, out_dir) and are correctly referenced.\n"
            + "\n--- EXIT CODE ---\n"
            + f"{rc}\n"
            + "\n--- STDERR (capped) ---\n"
            + f"{err_c}\n"
            + "\n--- STDOUT (capped) ---\n"
            + f"{out_c}\n"
            + "\n--- CURRENT SCRIPT (capped) ---\n"
            + failed_code_c
        )

        raw = responses_create_text(client, model, system_msg, repair_msg)
        code_text = strip_fences(extract_code_block(raw))

        ok, code_text = compile_with_retries(client, model, system_msg, user_msg_full, code_text)
        if not ok:
            continue

    last_path = project_dir / RUNS_DIR / f"rtls_run_{now_stamp()}_last.py"
    last_path.write_text(code_text, encoding="utf-8")
    print(f"[{now_ts()}] [ERROR] Exhausted runtime repair attempts. Last script saved to {last_path}", file=sys.stderr)
    return 1, "", "Runtime repair attempts exhausted."

# ---------- Orchestration ----------
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

    # Model order
    model_list: List[str] = []
    if ENV_MODEL:
        model_list.append(ENV_MODEL)
    if "gpt-5" not in model_list:
        model_list.insert(0, "gpt-5")
    model_list += [m for m in FALLBACK_MODELS if m not in model_list]
    print(f"[{now_ts()}] [INFO] Model preference order: {model_list}")

    # Generate (with validation/repair) -> code
    model_used, code_text = try_models_with_retries(client, model_list, system_msg, user_msg_full, user_msg_trimmed, user_msg_minimal)
    print(f"[{now_ts()}] [INFO] Using model: {model_used}")
    print(f"[{now_ts()}] [INFO] Final code length: {len(code_text)} chars")

    # Compile with retries (pure-syntax repairs allowed)
    ok, code_text = compile_with_retries(client, model_used, system_msg, user_msg_full, code_text)
    if not ok:
        return 4

    # Run; if non-zero exit, runtime repair loop
    rc, out, err = runtime_repair_loop(client, model_used, system_msg, user_msg_full, code_text, project_dir, user_prompt, csv_paths)

    if out:
        print(out, end="")
    if err:
        print(err, file=sys.stderr, end="")
    return rc

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate analysis code and execute locally (container-safe paths) with robust repairs (GC).")
    ap.add_argument("prompt", help="User prompt for the analysis (quoted)")
    ap.add_argument("csv", nargs="*", help="CSV path(s) or directories (optional; DB auto-select parses dates from the prompt)")
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parent
    csv_paths = [str(Path(p).resolve()) for p in (args.csv or [])]

    rc = generate_and_run(args.prompt, csv_paths, project_dir)
    if rc != 0:
        print(f"\nERROR: Generated script exited with code {rc}.", file=sys.stderr)
    sys.exit(rc)

if __name__ == "__main__":
    main()
