# extractor.py
"""
Fast extractor for the RTLS positions CSV (positions-first) with compact zone names.

CSV columns (case-insensitive, minimal):
  trackable, trackable_uid, mac, ts (timestamp), x, y, z
Optional zone-like columns (if present in CSV; not required):
  zone_name | zone | area_name | area | location  -> normalized to a short display

Key guarantees (to prevent downstream “missing columns”):
- Output rows ALWAYS include: trackable, trackable_uid, trade, mac, ts, ts_iso, ts_short, x, y, z,
  zone_name, zone_keyword, zone_simple  (present even if blank).
- We DO NOT parse x,y,z to numeric here (kept as strings) — prevents NaT-type issues during ingestion.
- MAC → (trackable, trackable_uid) via /mnt/data/trackable_objects.json (format-insensitive).
- AFTER MAC mapping, derive canonical 'trade' from final `trackable`.
- Timestamp helpers (fast path; no heavy parsing by default):
    ts_iso (ISO UTC)  |  ts_short (MM-DD\\nHH:MM)
- Zone normalization (if a zone-like column exists):
    zone_name   -> 1–2 word display (“Sales Floor”, “Vestibule”, “Receiving”, “FET”, …)
    zone_keyword -> last word of display (Title Case), e.g., “Sales”
    zone_simple  -> “zone - <zone_keyword>”
  (We DO NOT filter/drop any rows based on zones here.)

Public API
  extract_tracks(csv_path: str,
                 mac_map_path: str="/mnt/data/trackable_objects.json",
                 row_limit: int | None = None,
                 strict_ts: bool = False) -> dict
"""

from __future__ import annotations
import io, os, re, json
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

# ---------------------------- Fast CSV read ----------------------------
ENCODING_TRIES = ["utf-8","utf-8-sig","cp1252","latin-1","utf-16","utf-16le"]
CANON_COLS = ["trackable","trackable_uid","mac","ts","x","y","z"]
READ_OPTS = dict(sep=",", header=0, dtype=str, engine="c", quotechar='"',
                 keep_default_na=False, na_filter=False, low_memory=True, memory_map=True)

def _read_bytes(p:str)->bytes:
    with open(p,"rb") as f: return f.read()

def _decode(b:bytes)->Tuple[str,str]:
    for enc in ENCODING_TRIES:
        try: return b.decode(enc), enc
        except Exception: continue
    return b.decode("latin-1", errors="replace"), "latin-1"

def _preamble_len(text:str, scan:int=300)->int:
    pre=0
    for line in text.splitlines()[:scan]:
        s=line.lstrip()
        if s.startswith("#") or s=="": pre+=1
        else: break
    return pre

def _read_csv_fast(path:str, row_limit:Optional[int])->Tuple[pd.DataFrame,str,int,str]:
    raw=_read_bytes(path); text,enc=_decode(raw); pre=_preamble_len(text)
    engine="c"
    try:
        df=pd.read_csv(io.StringIO(text), skiprows=pre, usecols=None, **READ_OPTS)
    except Exception:
        engine="python"
        df=pd.read_csv(io.StringIO(text), skiprows=pre, usecols=None, engine="python",
                       **{k:v for k,v in READ_OPTS.items() if k not in ("engine","memory_map")})
    df=df.copy()

    # Keep canonical + any zone-like columns if present
    zone_like=[c for c in df.columns if c.lower().strip() in ("zone_name","zone","area_name","area","location")]
    keep=list(dict.fromkeys(CANON_COLS+zone_like))
    df=df[[c for c in df.columns if c in keep]]

    # Canonical rename (fuzzy)
    low={c.lower().strip():c for c in df.columns}
    ren={}
    for want in CANON_COLS:
        if want in low: ren[low[want]]=want
        else:
            for c in df.columns:
                lc=c.lower().strip()
                if want=="trackable_uid" and lc in ("trackable_uid","trackableuid","uid","id"): ren[c]="trackable_uid"; break
                if want=="mac" and lc in ("mac","mac_address","tag_mac","device_mac","node_mac","bt_mac"): ren[c]="mac"; break
                if want=="ts" and lc in ("ts","time","timestamp","date_time","datetime","occurred_at"): ren[c]="ts"; break
    if ren: df=df.rename(columns=ren)

    # Ensure canonical columns exist
    for c in CANON_COLS:
        if c not in df.columns: df[c]=""

    if row_limit is not None and len(df)>row_limit:
        df=df.iloc[:row_limit].copy()

    return df, enc, pre, engine

# ---------------------------- MAC mapping & trade ----------------------------
def _normalize_mac(s:str)->str:
    return re.sub(r"[^0-9a-f]","",str(s or "").lower())

def _load_mac_map(path:str="/mnt/data/trackable_objects.json")->Dict[str,Dict[str,str]]:
    try:
        with open(path,"r",encoding="utf-8") as f: data=json.load(f)
        arr=data.get("trackable_objects",[])
        out={}
        for obj in arr:
            mac=(obj.get("mac_address") or (obj.get("node") or {}).get("mac_address") or "")
            k=_normalize_mac(mac)
            if not k: continue
            out[k]={"name":(obj.get("name") or obj.get("display_name") or "").strip(),
                    "uid": (obj.get("uid") or (obj.get("node") or {}).get("uid") or obj.get("assoc_uid") or "").strip()}
        return out
    except Exception:
        return {}

TRADE_PATTERNS=[
    (re.compile(r"\bcarpentr(y|y_\d+)\b|\bcarpenter\b",re.I),"carpentry"),
    (re.compile(r"\bpainter\b|\bpainting\b",re.I),"painter"),
    (re.compile(r"\bsteel\b",re.I),"steel"),
    (re.compile(r"\bmechanical\b|\bhvac\b",re.I),"mechanical"),
    (re.compile(r"\bclean(ing|er)?\b",re.I),"cleaning"),
    (re.compile(r"\brefrig(eration)?\b|\brefrig\b",re.I),"refrigeration"),
    (re.compile(r"\bstrip(ing|e)?\b",re.I),"striping"),
    (re.compile(r"\broof(ing)?\b",re.I),"roofing"),
    (re.compile(r"\bfloor(ing)?\b",re.I),"flooring"),
    (re.compile(r"\bconcrete\b",re.I),"concrete"),
    (re.compile(r"\b(building\s*signage|signage)\b",re.I),"building_signage"),
    (re.compile(r"\bfire\s*sprink(ler)?\b|\bfire\s*spk\b",re.I),"fire_sprinkler"),
    (re.compile(r"\bems\b",re.I),"ems"),
    (re.compile(r"\bplumb(ing|er)?\b",re.I),"plumbing"),
    (re.compile(r"\belectric(ians?|ian)?\b|\belec\b",re.I),"electrician"),
]
def _infer_trade_from_trackable(trackable:str)->str:
    s=str(trackable or "").strip().lower().replace("-"," ").replace("_"," ")
    s=re.sub(r"\s+"," ",s)
    for pat,canon in TRADE_PATTERNS:
        if pat.search(s): return canon
    return ""

# ---------------------------- Timestamp helpers ----------------------------
ISO_Z_RE=re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[ T](?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2})(?:\.\d{1,6})?)?(Z|[+-]\d{2}:?\d{2})?$")
DATE_FALLBACK_RE=re.compile(r"(\d{4})-(\d{2})-(\d{2}).*?(\d{2}):(\d{2})")

def _ts_iso_fast(series:pd.Series)->pd.Series:
    s=series.astype(str)
    base=s.str.slice(0,16)  # YYYY-MM-DDTHH:MM
    out=base.where(base.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"), other="")
    out=out.mask(out=="", s)
    def norm(v:str)->str:
        m=ISO_Z_RE.match(v)
        if not m: return ""
        return f"{m.group('y')}-{m.group('m')}-{m.group('d')}T{m.group('H')}:{m.group('M')}:{m.group('S') or '00'}Z"
    return out.map(norm)

def _ts_short_fast(series:pd.Series)->pd.Series:
    s=series.astype(str)
    short=s.str.slice(5,10) + "\n" + s.str.slice(11,16)
    ok=short.str.match(r"^\d{2}-\d{2}\n\d{2}:\d{2}$")
    if ok.all(): return short
    def fb(v:str)->str:
        m=DATE_FALLBACK_RE.search(v)
        if m: return f"{m.group(2)}-{m.group(3)}\n{m.group(4)}:{m.group(5)}"
        return ""
    return s.map(fb)

# ---------------------------- Zone normalization (optional) ----------------------------
ZONE_MAP=[
    (re.compile(r"\b(100\s+gr\s+)?sales\s+floor\b",re.I),"Sales Floor"),
    (re.compile(r"\bvestibule\b",re.I),"Vestibule"),
    (re.compile(r"\bfet\b",re.I),"FET"),
    (re.compile(r"\bpersonnel\b",re.I),"Personnel"),
    (re.compile(r"\bpharmacy\b",re.I),"Pharmacy"),
    (re.compile(r"\b(restroom|customer\s+restroom)s?\b",re.I),"Restroom"),
    (re.compile(r"\bbreak\s*room\b|\bbreakroom\b",re.I),"Breakroom"),
    (re.compile(r"\bpickup\b",re.I),"Pickup"),
    (re.compile(r"\bdeli\b",re.I),"Deli"),
    (re.compile(r"\breceiving\b",re.I),"Receiving"),
    (re.compile(r"\bback\s*room\b",re.I),"Backroom"),
    (re.compile(r"\bfront\s*end\b|\bfront\b",re.I),"Front"),
]
ZONE_STOP={"zone","area","aisle","hall","hallway","bay","dock","office","department","dept",
           "section","room","floor","gm","gr","store","backroom","front","front-end","pickup",
           "storage","information","active","inactive"}

def _desired_zone_display(raw:str)->str:
    s=str(raw or "").strip()
    if not s: return ""
    for pat,disp in ZONE_MAP:
        if pat.search(s): return disp
    tail=re.sub(r"^\s*zone\s+\d+(\.\d+)?\s*-\s*","",s,flags=re.I)
    tail=re.sub(r"^\s*(\d{1,4}(\.\d+)?(\s+[A-Z]{2,})?\s*)+","",tail).strip()
    if not tail: return ""
    words=re.findall(r"[A-Za-z]+",tail)
    joined=" ".join(words).title()
    if "Sales" in joined and "Floor" in joined: return "Sales Floor"
    keep=[w for w in words if w.lower() not in ZONE_STOP]
    if not keep: return tail.title()
    if len(keep)>=2: return f"{keep[-2].title()} {keep[-1].title()}"
    return keep[-1].title()

def _zone_keyword_from_display(display:str)->str:
    toks=re.findall(r"[A-Za-z]+",str(display or ""))
    return toks[-1].title() if toks else ""

def _apply_zone_normalization(df:pd.DataFrame)->pd.DataFrame:
    src=next((c for c in df.columns if c.lower().strip() in ("zone_name","zone","area_name","area","location")), None)
    if not src:
        for c in ("zone_name","zone_keyword","zone_simple"):
            if c not in df.columns: df[c]=""
        return df
    df["zone_name"]=df[src].map(_desired_zone_display)
    df["zone_keyword"]=df["zone_name"].map(_zone_keyword_from_display)
    df["zone_simple"]=df["zone_keyword"].map(lambda k: f"zone - {k}" if k else "")
    return df

# ---------------------------- Public API ----------------------------
def extract_tracks(csv_path:str,
                   mac_map_path:str="/mnt/data/trackable_objects.json",
                   row_limit:Optional[int]=None,
                   strict_ts:bool=False)->Dict[str,Any]:
    if not os.path.exists(csv_path): raise FileNotFoundError(csv_path)

    df, enc, pre, engine=_read_csv_fast(csv_path, row_limit)

    # MAC mapping → trackable/uid
    df["mac_norm"]=df["mac"].map(_normalize_mac)
    mac_map=_load_mac_map(mac_map_path)
    df["mac_name"]=df["mac_norm"].map(lambda k: mac_map.get(k,{}).get("name",""))
    df["mac_uid"] =df["mac_norm"].map(lambda k: mac_map.get(k,{}).get("uid",""))

    def _coalesce(a:str,b:str)->str:
        a=str(a or "").strip(); b=str(b or "").strip()
        return a if a else b

    df["trackable"]    =[ _coalesce(a,b) for a,b in zip(df["trackable"], df["mac_name"]) ]
    df["trackable_uid"]=[ _coalesce(a,b) for a,b in zip(df["trackable_uid"], df["mac_uid"]) ]

    # Trade from final trackable
    df["trade"]=df["trackable"].map(_infer_trade_from_trackable)

    # Timestamp helpers (fast path)
    if strict_ts:
        dt=pd.to_datetime(df["ts"], errors="coerce", utc=True, infer_datetime_format=True)
        df["ts_iso"]=dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
        df["ts_short"]=dt.dt.strftime("%m-%d\n%H:%M").fillna("")
    else:
        df["ts_iso"]  =_ts_iso_fast(df["ts"])
        df["ts_short"]=_ts_short_fast(df["ts"])

    # Optional zone normalization (safe; never removes rows)
    df=_apply_zone_normalization(df)

    # ---- Build strings-only rows, ensuring required columns exist
    out_cols=["trackable","trackable_uid","trade","mac","ts","ts_iso","ts_short","x","y","z",
              "zone_name","zone_keyword","zone_simple"]
    for c in out_cols:
        if c not in df.columns: df[c]=""

    view=df[out_cols].copy()
    rows=[{c: ("" if pd.isna(v) else str(v)) for c,v in r.items()} for r in view.to_dict(orient="records")]

    audit={
        "encoding_used": enc,
        "engine_used": engine,
        "delimiter_used": ",",
        "preamble_lines_skipped": pre,
        "rows_returned": len(rows),
        "columns_detected": out_cols,
        "mac_map_loaded": bool(mac_map),
        "strict_ts": bool(strict_ts),
        "row_limit": row_limit if row_limit is not None else "all",
        "zone_normalized": True
    }
    return {"columns": out_cols, "rows": rows, "audit": audit}
