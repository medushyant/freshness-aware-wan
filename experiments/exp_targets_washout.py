"""DECISIVE EXPERIMENT — does target motion actually matter in the paper's
aggregation engine, or does it wash out?

The handoff proposes the integration: each round, re-anchor every agent's
patrol centre c_i on its (predicted) target, then run the paper's mission.
This script tests, in the REAL D1 engine (wan/network.run_mission), whether
that integration changes anything.

It answers two questions with measured numbers:

  Q1 (washout?)  With the paper's own geometry (patrol radius R=80-100 m),
                 does total mission energy differ across the three target
                 classes (static / dynamic / time-varying)?  If not, target
                 motion is invisible to the model and the naive integration
                 is scientifically empty.

  Q2 (mechanism) The reason motion washes out is that the coverage radius R
                 dwarfs the per-round target displacement, so the coverage
                 constraint ||p_i - c_i|| <= R stays slack. We decouple a
                 SENSING radius r_track (how tightly the agent must stay on
                 the target) from the coverage radius and sweep it down.
                 We measure the r_track at which target class starts to bite.

Run:  python3 -m experiments.exp_targets_washout
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wan.model import P
from wan.network import run_mission
from wan.targets import spawn_targets
from wan.style import use_style
use_style()

os.makedirs("figures", exist_ok=True)
KINDS = ["static", "dynamic", "time_varying"]
NICE = {"static": "static", "dynamic": "dynamic (NCV)", "time_varying": "time-varying"}
SEEDS = range(12)
N = P["N"]
T_TG = P["Tmax"]          # targets advance one full round-deadline per round


def run_class(kind, r_track, seeds=SEEDS, predict=False):
    """Mean (energy, forced-move) over seeds for one target class + r_track."""
    Es, FMs = [], []
    for s in seeds:
        tgs = spawn_targets(kind, N, np.random.default_rng(1000 + s))
        out = run_mission(s, P, mode="paper", targets=tgs,
                          target_T=T_TG, r_track=r_track, predict=predict)
        if np.isfinite(out["E"]):
            Es.append(out["E"]); FMs.append(out["forced_move"])
    return (np.mean(Es) if Es else np.nan,
            np.std(Es) if Es else np.nan,
            np.mean(FMs) if FMs else np.nan,
            len(Es))


def per_round_displacement():
    """How far a target moves per round vs the patrol radius — the root cause."""
    rows = {}
    for kind in KINDS:
        disp = []
        for s in SEEDS:
            tg = spawn_targets(kind, 1, np.random.default_rng(1000 + s))[0]
            prev = tg.pos().copy()
            for _ in range(6):
                tg.step(T_TG)
                disp.append(np.linalg.norm(tg.pos() - prev)); prev = tg.pos().copy()
        rows[kind] = np.mean(disp)
    return rows


print("=" * 72)
print("CONTEXT — per-round target displacement vs the paper's patrol radius")
print("=" * 72)
disp = per_round_displacement()
print(f"  patrol radius R              : {P['R_cov'][0]:.0f}-{P['R_cov'][1]:.0f} m")
print(f"  round duration (= Tmax)      : {T_TG:.1f} s")
for k in KINDS:
    print(f"  {NICE[k]:16s} moves        : {disp[k]:5.1f} m / round")
print("  => displacement is a small FRACTION of R: the coverage disk already "
      "contains\n     the moved target, so the agent is never forced to chase.")

# ----------------------------------------------------------------- Q1 washout
print("\n" + "=" * 72)
print("Q1 — WASHOUT TEST: paper geometry, c_i re-anchored on target, full R")
print("=" * 72)
base = {}
for kind in KINDS:
    mE, sE, mFM, nfe = run_class(kind, r_track=None)
    base[kind] = (mE, sE, mFM)
    print(f"  {NICE[kind]:16s}: E = {mE:6.3f} +/- {sE:4.3f} J   "
          f"forced-move = {mFM:5.2f} m   (feasible {nfe}/{len(SEEDS)})")

spread = max(v[0] for v in base.values()) - min(v[0] for v in base.values())
rel = 100 * spread / np.mean([v[0] for v in base.values()])
washes_out = rel < 2.0 and max(v[2] for v in base.values()) < 1.0
print(f"\n  energy spread across classes : {spread:.3f} J  ({rel:.1f}% of mean)")
print(f"  max forced-move across classes: {max(v[2] for v in base.values()):.2f} m")
print("  VERDICT:", "WASHES OUT — target class is invisible (naive integration is empty)."
      if washes_out else "classes differ — motion is visible even at full R.")

# ----------------------------------------------------------------- Q2 mechanism
print("\n" + "=" * 72)
print("Q2 — MECHANISM: decouple a SENSING radius r_track and sweep it down")
print("=" * 72)
R_TRACKS = [100, 60, 40, 25, 15, 8]
print("  showing  E[J] (feasible n/12, forced-move m)  per class")
print(f"  {'r_track[m]':>10s} | " + " | ".join(f"{NICE[k]:>26s}" for k in KINDS))
grid_E = {k: [] for k in KINDS}
grid_FM = {k: [] for k in KINDS}
grid_NFE = {k: [] for k in KINDS}
for rt in R_TRACKS:
    cells = []
    for kind in KINDS:
        mE, sE, mFM, nfe = run_class(kind, r_track=rt)
        grid_E[kind].append(mE); grid_FM[kind].append(mFM); grid_NFE[kind].append(nfe)
        cells.append(f"{mE:6.3f} ({nfe:2d}/12, {mFM:4.1f}m)")
    print(f"  {rt:>10d} | " + " | ".join(f"{c:>26s}" for c in cells))

# the r_track where dynamic exceeds static by >5%
bite = None
for rt, es, ed in zip(R_TRACKS, grid_E["static"], grid_E["dynamic"]):
    if np.isfinite(es) and np.isfinite(ed) and ed > 1.05 * es:
        bite = rt
print(f"\n  target motion starts to bite (dynamic > static + 5%) at "
      f"r_track <= {bite} m" if bite else
      "\n  motion never bit in the swept range")

# ----------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k in KINDS:
    ax[0].plot(R_TRACKS, grid_FM[k], "-o", label=NICE[k])
ax[0].axvspan(P["R_cov"][0], P["R_cov"][1], color="grey", alpha=.15,
              label="paper's R range")
ax[0].set_xlabel("sensing radius r_track [m]"); ax[0].set_ylabel("forced tracking move [m/round]")
ax[0].set_title("the coverage constraint only binds once r_track shrinks")
ax[0].invert_xaxis(); ax[0].legend(fontsize=8)
for k in KINDS:
    ax[1].plot(R_TRACKS, grid_E[k], "-o", label=NICE[k])
ax[1].axvspan(P["R_cov"][0], P["R_cov"][1], color="grey", alpha=.15)
ax[1].set_xlabel("sensing radius r_track [m]"); ax[1].set_ylabel("total mission energy [J]")
ax[1].set_title("target class is invisible at large r_track, separates when small")
ax[1].invert_xaxis(); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/W1_washout_and_mechanism.png", dpi=150)
print("\nwrote figures/W1_washout_and_mechanism.png")
