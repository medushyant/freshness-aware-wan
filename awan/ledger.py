"""EnergyLedger — every joule in a Phase-2 run is tagged and conserved.

kind   : mobility | compute | comm_payload | comm_control | negotiation_llm
source : MEASURED | MODELED
Invariant (checked): total() == sum of every per-tag slice, no negatives, no NaN.
"""

import math

KINDS = ("mobility", "compute", "comm_payload", "comm_control", "negotiation_llm")
SOURCES = ("MEASURED", "MODELED")


class EnergyLedger:

    def __init__(self):
        self.entries = []

    def add(self, joules, kind, agent=None, rnd=None, source="MODELED", note=""):
        j = float(joules)
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        if math.isnan(j) or j < 0:
            raise ValueError(f"bad energy {j!r} ({kind}, agent={agent}, round={rnd})")
        self.entries.append({"J": j, "kind": kind, "agent": agent,
                             "round": rnd, "source": source, "note": note})

    def total(self):
        return sum(e["J"] for e in self.entries)

    def by_kind(self):
        out = {k: 0.0 for k in KINDS}
        for e in self.entries:
            out[e["kind"]] += e["J"]
        return out

    def by_agent(self):
        out = {}
        for e in self.entries:
            if e["agent"] is not None:
                out[e["agent"]] = out.get(e["agent"], 0.0) + e["J"]
        return out

    def by_source(self):
        out = {s: 0.0 for s in SOURCES}
        for e in self.entries:
            out[e["source"]] += e["J"]
        return out

    def check(self, tol=1e-9):
        t = self.total()
        ok = (abs(t - sum(self.by_kind().values())) <= tol * max(t, 1.0)
              and abs(t - sum(self.by_source().values())) <= tol * max(t, 1.0)
              and all(e["J"] >= 0 and not math.isnan(e["J"]) for e in self.entries))
        return ok

    def to_dict(self):
        return {"total_J": self.total(), "by_kind": self.by_kind(),
                "by_source": self.by_source(), "by_agent": self.by_agent(),
                "n_entries": len(self.entries)}
