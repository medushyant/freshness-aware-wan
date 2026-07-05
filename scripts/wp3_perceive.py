"""Stage-1 driver: run SmolVLM2 perception over the scene set and cache
everything (run once; every WP-3 figure then regenerates cache-only)."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from awan.grounded.pipeline import perceive_scenes

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    stats = perceive_scenes(range(n))
    print("VLM stats:", stats)
