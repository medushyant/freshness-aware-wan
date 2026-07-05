"""Figure registry (playbook §0.5): figure id -> the run that produced it."""

import json

from . import ROOT

REG = ROOT / "awan" / "registry.json"


def record(fig_id, run_name, path):
    reg = {}
    if REG.exists():
        reg = json.loads(REG.read_text())
    reg[fig_id] = {"run": run_name, "figure": str(path)}
    REG.write_text(json.dumps(reg, indent=1, sort_keys=True))


def lookup(fig_id=None):
    reg = json.loads(REG.read_text()) if REG.exists() else {}
    return reg if fig_id is None else reg.get(fig_id)
