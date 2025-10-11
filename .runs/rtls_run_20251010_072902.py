```python
#!/usr/bin/env python
# coding: utf-8
"""
InfoZoneBuilder - Walmart Renovation RTLS Position Summary

CLI:
    python generated.py "<USER_PROMPT>" /abs/csv1 [/abs/csv2 ...]

Behavior:
- Processes each CSV one-by-one using extractor.extract_tracks (memory-safe).
- Applies DataFrame safety rules (duplicate-name guard; UTC timestamp canon ts_utc).
- Builds concise aggregates and optional floorplan overlay (x,y in world-mm).
- Writes PNG charts to the first CSV directory, then composes a branded PDF.
- Prints file:// links to the PDF and the saved plots (Windows-safe).
"""

import sys, os, json, math, traceback, random, gc
from pathlib import Path
from