"""Run-directory discipline (playbook §0.5): every experiment writes
runs/<UTCstamp>_<name>/{config.yaml, results.json, log.txt} and its figures
go to figures/awan/. Every run records seeds, config, git commit, wall time."""

import datetime
import json
import pathlib
import subprocess
import time

import numpy as np
import yaml

from . import ROOT

RUNS = ROOT / "runs"
FIGS = ROOT / "figures" / "awan"


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


class Run:

    def __init__(self, name, config):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = RUNS / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        FIGS.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.t0 = time.time()
        self.config = dict(config)
        self.config["git_commit"] = _git_commit()
        self.log_lines = []
        (self.dir / "config.yaml").write_text(yaml.safe_dump(_jsonable(self.config)))

    def log(self, msg):
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line)
        self.log_lines.append(line)

    def finish(self, results):
        results = _jsonable(results)
        results["_meta"] = {"name": self.name, "wall_s": round(time.time() - self.t0, 1),
                            "git_commit": self.config["git_commit"],
                            "config": _jsonable(self.config)}
        (self.dir / "results.json").write_text(json.dumps(results, indent=1))
        (self.dir / "log.txt").write_text("\n".join(self.log_lines) + "\n")
        return self.dir / "results.json"


def latest_results(name):
    """Newest results.json for a run name (figure scripts read these)."""
    hits = sorted(RUNS.glob(f"*_{name}/results.json"))
    if not hits:
        raise FileNotFoundError(f"no runs/*_{name}/results.json — run the experiment first")
    return json.loads(hits[-1].read_text())
