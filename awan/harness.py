"""PASS/FAIL harness in the exact Phase-1 style (playbook §0.6).
Runners collect checks and append them to results_awan.txt."""

from . import ROOT

OUT = ROOT / "results_awan.txt"


class Checks:

    def __init__(self, section):
        self.section = section
        self.rows = []

    def check(self, cid, claim, ok, detail=""):
        line = ("PASS  " if ok else "FAIL  ") + f"{cid} {claim}" + (f"  | {detail}" if detail else "")
        print(line)
        self.rows.append((cid, bool(ok), line))
        return bool(ok)

    def flush(self, replace_ids=True):
        """Append to results_awan.txt, replacing any earlier lines with the
        same check ids so re-runs stay idempotent."""
        old = OUT.read_text().splitlines() if OUT.exists() else []
        ids = {r[0] for r in self.rows}
        kept = [l for l in old
                if not any(l.startswith(p + f"{i} ") for i in ids
                           for p in ("PASS  ", "FAIL  "))]
        new = kept + [r[2] for r in self.rows]
        OUT.write_text("\n".join(new) + "\n")
        n_ok = sum(1 for r in self.rows if r[1])
        print(f"\n{self.section}: {n_ok}/{len(self.rows)} PASS -> {OUT.name}")
        return n_ok == len(self.rows)
