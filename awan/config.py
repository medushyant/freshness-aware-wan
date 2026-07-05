"""YAML config loading with deep-merge overrides (playbook §0.5)."""

import yaml

from . import ROOT

CFG_DIR = ROOT / "awan" / "configs"


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(name="defaults", **over):
    cfg = yaml.safe_load((CFG_DIR / f"{name}.yaml").read_text())
    if name != "defaults" and (CFG_DIR / "defaults.yaml").exists():
        cfg = _merge(yaml.safe_load((CFG_DIR / "defaults.yaml").read_text()), cfg)
    return _merge(cfg, over)
