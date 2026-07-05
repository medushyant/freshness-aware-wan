"""A-WAN Phase 2. All new code lives here; frozen Phase-1 code is reached
only through awan.adapters."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
