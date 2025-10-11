# server.py — FastAPI bridge for InfoZone (Windows-friendly)
from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# ----------------- Project ROOT detection -----------------
# Must be the directory that contains main.py & helpers.
PROJECT_ROOT = Path(os.environ.get("INFOZONE_ROOT") or Path(__file__).resolve().parent).resolve()
os.environ["INFOZONE_ROOT"] = str(PROJECT_ROOT)

MAIN_PY = PROJECT_ROOT / "main.py"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Serve everything under PROJECT_ROOT at /files so the browser can load PDFs/PNGs
app = FastAPI(title="InfoZone Web Bridge", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=str(PROJECT_ROOT)), name="files")

# Matches Markdown links the runner prints, e.g.
# [Download the PDF](file:///C:/path/to/report.pdf)
FILELINK_RE = re.compile(r"\[[^\]]+\]\(file:///([^)\r\n]+)\)")

def _artifact_type(path: str) -> str:
    ext = (path.rsplit(".", 1)[-1] if "." in path else "").lower()
    if ext == "pdf":
        return "pdf"
    if ext in {"png", "jpg", "jpeg", "gif", "webp"}:
        return "image"
    return "file"

def _to_http_url(local_path: str) -> Optional[str]:
    """
    Convert an absolute local path to an HTTP URL under /files, if the path
    resides inside PROJECT_ROOT. Returns None if outside.
    """
    try:
        # Normalize Windows-style backslashes and resolve real path casing
        p = Path(local_path.replace("\\", "/")).resolve()
        root = PROJECT_ROOT.resolve()
        # Case-insensitive check (Windows)
        if str(p).lower().startswith(str(root).lower()):
            rel = p.relative_to(root).as_posix()
            return f"/files/{rel}"
        return None
    except Exception:
        return None

@app.get("/api/ping")
def ping() -> Dict[str, Any]:
    return {"ok": True, "root": str(PROJECT_ROOT), "has_main": MAIN_PY.exists()}

@app.post("/api/run")
def run(prompt: str = Form(...), files: List[UploadFile] = File(default=[])) -> JSONResponse | PlainTextResponse:
    if not MAIN_PY.exists():
        return PlainTextResponse("main.py not found under INFOZONE_ROOT", status_code=500)

    # Save uploads into a per-request directory to keep paths stable
    job_dir = UPLOAD_DIR / f"job_{os.getpid()}_{os.urandom(3).hex()}"
    job_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = []
    for f in files or []:
        dest = job_dir / f.filename
        with dest.open("wb") as w:
            shutil.copyfileobj(f.file, w)
        csv_paths.append(dest.resolve())

    # Build the command: python main.py "<prompt>" <csv1> <csv2> ...
    cmd = [sys.executable, str(MAIN_PY), prompt] + [str(p) for p in csv_paths]

    # Ensure helpers import from ROOT
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + py_path if py_path else "")

    # Run the generator
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Parse file:/// links from stdout
    links = FILELINK_RE.findall(stdout)
    # Also accept bare file:/// lines (fallback)
    for line in stdout.splitlines():
        if "file:///" in line and "](" not in line:
            # Try to extract the path part after file:///
            try:
                part = line.split("file:///", 1)[1]
                # Trim markdown artifacts/spaces
                part = part.split(")")[0].strip()
                links.append(part)
            except Exception:
                pass

    artifacts: List[Dict[str, str]] = []
    seen: set[str] = set()
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        http = _to_http_url(link)
        if not http:
            continue
        artifacts.append(
            {
                "url": http,
                "type": _artifact_type(link),
                "filename": Path(link).name,
            }
        )

    ok = proc.returncode == 0
    summary = "Report created." if ok and artifacts else "Runner finished (check logs)."
    status = 200 if ok else 500

    return JSONResponse(
        {
            "ok": ok,
            "summary": summary,
            "artifacts": artifacts,
            "logs": (stdout + ("\n\n[stderr]\n" + stderr if stderr else "")).strip(),
        },
        status_code=status,
    )
