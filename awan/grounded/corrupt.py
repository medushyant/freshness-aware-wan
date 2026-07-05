"""Corruption operators on one leaf agent's facts (playbook §4.5)."""

import numpy as np

from .synth_views import CLASSES, COLORS


def corrupt(facts, op, rng, k=3):
    facts = [dict(f) for f in facts]
    if op == "fabricate":
        for _ in range(k):
            facts.append({"object": CLASSES[rng.integers(0, len(CLASSES))],
                          "attr": list(COLORS)[rng.integers(0, len(COLORS))],
                          "confidence": 0.95, "_corrupt": True})
    elif op == "swap":
        for f in facts:
            f["object"] = CLASSES[rng.integers(0, len(CLASSES))]
            f["_corrupt"] = True
    elif op == "delete":
        keep = max(0, len(facts) - k)
        idx = rng.permutation(len(facts))[:keep]
        facts = [facts[i] for i in idx]
    elif op == "jitter":
        for f in facts:
            f["attr"] = list(COLORS)[rng.integers(0, len(COLORS))]
            f["_corrupt"] = True
    else:
        raise ValueError(op)
    return facts


def root_corruption_rate(root_facts):
    """Fraction of runs where >=1 corrupted fact survives to the root is
    computed by the caller; here: does THIS root contain corruption?"""
    return any(f.get("_corrupt") for f in root_facts)
