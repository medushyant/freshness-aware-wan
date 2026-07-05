"""WP-0 — integration audit & scaffold verification (playbook §1).

W0.1  regression gate green (identifier set vs baseline/, no FAIL lines)
W0.2  adapters reproduce the E1 blossom energy from results.txt
W0.3  ledger invariant + simcore parity on a 3-round sim
W0.4  config -> run -> results.json -> figure round-trip

Run under .venv-awan:  python run_awan_wp0.py
"""

import pathlib
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awan import adapters as A
from awan.config import load
from awan.harness import Checks
from awan.registry import record
from awan.runio import FIGS, Run, latest_results
from awan.simcore import run_episode

ROOT = pathlib.Path(__file__).resolve().parent
C = Checks("WP-0")

# ---------------------------------------------------------------- W0.1
ok, detail = True, []
for base in sorted((ROOT / "baseline").glob("results*.txt")):
    cur = ROOT / base.name
    ids = lambda t: sorted(set(re.findall(r"^(?:PASS|FAIL)\s+(\S+)", t, re.M)))
    same = cur.exists() and ids(cur.read_text()) == ids(base.read_text())
    clean = cur.exists() and not re.search(r"^FAIL", cur.read_text(), re.M)
    ok &= same and clean
    detail.append(f"{base.name}:{'ok' if same and clean else 'DRIFT'}")
n_pass = sum(len(re.findall(r"^PASS", f.read_text(), re.M))
             for f in sorted(ROOT.glob("results_*.txt")) if f.name != "results_awan.txt")
n_pass += len(re.findall(r"^PASS", (ROOT / "results.txt").read_text(), re.M))
C.check("W0.1", f"regression gate green ({n_pass} Phase-1 PASS lines)", ok, " ".join(detail))

# ---------------------------------------------------------------- W0.2
txt = (ROOT / "results.txt").read_text()
m = re.search(r"blossom <= random\s+\|\s+([\d.]+) vs", txt)
printed = float(m.group(1))
vals = [A.run_mission(sd, A.STRESS, mode="paper", allow_motion=True)["E"]
        for sd in (3, 4, 5)]
ours = float(np.mean(vals))
C.check("W0.2", "adapters reproduce the E1 blossom mission energy",
        abs(ours - printed) < 5e-3,
        f"adapter {ours:.4f} J vs results.txt {printed:.3f} J")

# ---------------------------------------------------------------- W0.3
out = run_episode(5, p=A.STRESS, n=8)
led = out["ledger"]
ref = A.phase1_mission(5, A.STRESS, policy="paper", allow_motion=False, n=8)
conserved = led.check() and abs(led.total() - out["E"]) < 1e-6 * max(out["E"], 1.0)
parity = out["feasible"] and abs(out["E"] - ref["E"]) < 1e-9
C.check("W0.3", "ledger invariant holds on a 3-round sim AND simcore == frozen mission",
        conserved and parity and out["rounds"] >= 3,
        f"ledger {led.total():.4f} J == episode {out['E']:.4f} J; frozen {ref['E']:.4f} J; "
        f"{out['rounds']} rounds; kinds " +
        ",".join(f"{k}={v:.3f}" for k, v in led.by_kind().items() if v > 0))

# ---------------------------------------------------------------- W0.4
cfg = load("defaults", experiment="wp0_smoke", seed=5, n=8)
run = Run("wp0_smoke", cfg)
run.log(f"episode E={out['E']:.4f} J rounds={out['rounds']}")
res_path = run.finish({"E_J": out["E"], "rounds": out["rounds"],
                       "ledger": led.to_dict(), "seeds": [5]})
res = latest_results("wp0_smoke")
fig_ok = False
if abs(res["E_J"] - out["E"]) < 1e-12:
    A.use_style()
    kinds = {k: v for k, v in res["ledger"]["by_kind"].items() if v > 0}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(list(kinds), list(kinds.values()), color="#4cc9f0")
    ax.set_ylabel("energy [J]  (MODELED)")
    ax.set_title(f"WP-0 smoke: ledger split, seed 5, N=8 — total {res['E_J']:.2f} J")
    fp = FIGS / "F0_wp0_smoke.png"
    fig.tight_layout(); fig.savefig(fp, dpi=150); plt.close(fig)
    record("F0", "wp0_smoke", fp)
    fig_ok = fp.exists() and fp.stat().st_size > 5_000
C.check("W0.4", "config -> run -> results.json -> figure round-trip works",
        fig_ok, f"{res_path.parent.name} -> figures/awan/F0_wp0_smoke.png")

C.flush()
