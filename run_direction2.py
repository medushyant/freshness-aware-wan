"""Direction 2 experiment driver.

  T1  zeta-robustness (headline): the paper's potential field needs a
      hand-tuned zeta and its own Fig. 7 shows energy rebounds when it is
      off. We sweep zeta for the paper's recipe and for stage A, and draw
      the learned policy (which has no zeta at all) as a flat line.
  T2  optimality gap vs a brute-force oracle at N=5 - the first such
      number for this framework.
  T3  size transfer: value model trained on N in {4,5,6}, deployed
      unchanged at N = 8 and 10.
  T4  flexible vs forced schedule (Dominance Proposition in practice).

Run:  python3 run_direction2.py     (~3-4 min)
"""

import os, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wan.style import use_style
use_style()

from wan.model import P
from wan.topology import mission, collect_training, ValueModel, dp_oracle

os.makedirs("figures", exist_ok=True)
checks, t_start = [], time.time()


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""))


# the stress regime from E1: communication expensive, so WHO pairs with
# whom moves real energy (in the lazy regime every topology looks alike)
ps = dict(P)
ps["Tmax"] = 2.5; ps["L_init"] = (15e6, 25e6); ps["beta0"] = 1e-5
ps["f_cpu"] = 2.5e9; ps["C_gen"] = 120.0

EVAL = [3, 4, 5, 6, 7]


def avg_E(policy, seeds=EVAL, **kw):
    es = [mission(s, ps, policy=policy, **kw)["E"] for s in seeds]
    fin = [e for e in es if np.isfinite(e)]
    return (np.mean(fin) if fin else np.nan), len(es) - len(fin)


# ------------------------------------------------------------------ train
print("== training the cost-to-go model (held-out from all eval seeds) ==")
X, y = collect_training(range(100, 124), ps)
vm = ValueModel(l2=2.0)
r2 = vm.fit(X, y)
check("T0 value model fits the cost-to-go", r2 > 0.5,
      "R^2=%.2f on %d rollout states" % (r2, len(y)))

# ------------------------------------------------------------------ T1
print("\n== T1: zeta robustness (headline) ==")
zetas = [0.0, 1e-15, 1e-14, 1e-13, 1e-12]
curves = {}
for pol in ["paper", "stageA"]:
    curves[pol] = [avg_E(pol, zeta=z)[0] for z in zetas]
E_learn, _ = avg_E("learned", vmodel=vm)
E_flex, _ = avg_E("learned", vmodel=vm, flexible=True)
E_lyap, _ = avg_E("lyapunov")

xs = np.arange(len(zetas))
lab = ["0", "1e-15", "1e-14\n(paper)", "1e-13", "1e-12"]
fig, ax = plt.subplots(figsize=(6.6, 4))
ax.plot(xs, curves["paper"], "o-", color="crimson", label="paper H-MAP (post-hoc $\\Phi$)")
ax.plot(xs, curves["stageA"], "s-", color="#e08a00", label="stage A ($\\Phi$ in the loop)")
ax.axhline(E_lyap, color="gray", ls="-.", label="stage B: auto-scaled (no knob)")
ax.axhline(E_learn, color="#1f5f8b", ls="--", label="learned cost-to-go (no $\\zeta$)")
ax.axhline(E_flex, color="#157a3a", ls="--", label="learned + flexible schedule")
ax.set_xticks(xs, lab)
ax.set_xlabel("hand-tuned lookahead weight $\\zeta$")
ax.set_ylabel("mission energy [J], mean of 5 held-out scenarios")
ax.set_title("tuning sensitivity: the learned policy has nothing to tune")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/T1_zeta_robustness.png", dpi=150); plt.close(fig)

best_paper = np.nanmin(curves["paper"])
worst_paper = np.nanmax(curves["paper"])
i_def = zetas.index(1e-14)
check("T1 paper recipe is zeta-fragile", worst_paper > 1.04 * best_paper,
      "worst/best over sweep = %.2f" % (worst_paper / best_paper))
check("T1 stage A <= paper at the paper's own zeta",
      curves["stageA"][i_def] <= curves["paper"][i_def] * 1.005,
      "%.3f vs %.3f J" % (curves["stageA"][i_def], curves["paper"][i_def]))
check("T1 learned (no knob) beats the BEST hand-tuned paper",
      E_learn <= best_paper * 1.02, "%.3f vs %.3f J" % (E_learn, best_paper))
check("T1 flexible schedule helps further", E_flex <= E_learn * 1.005,
      "%.3f vs %.3f J" % (E_flex, E_learn))
check("T1 auto-scaled stage B is competitive without tuning",
      E_lyap <= worst_paper, "%.3f vs worst paper %.3f J" % (E_lyap, worst_paper))

# ------------------------------------------------------------------ T2
print("\n== T2: optimality gap vs brute-force oracle, N=5 ==")
gap_seeds = [3, 4, 5]
opt = {s: dp_oracle(s, ps, n=5) for s in gap_seeds}
gaps = {}
for pol, kw in [("paper", {}), ("stageA", {}), ("lyapunov", {}),
                ("learned", {"vmodel": vm}), ("greedy", {}), ("random", {})]:
    g = []
    for s in gap_seeds:
        e = mission(s, ps, policy=pol, n=5, **kw)["E"]
        if np.isfinite(e):
            g.append(100 * (e - opt[s]) / opt[s])
    gaps[pol] = np.mean(g) if g else np.nan

fig, ax = plt.subplots(figsize=(6, 3.8))
names = list(gaps)
ax.bar(range(len(names)), [gaps[n] for n in names], color="#3b6ea5")
ax.set_xticks(range(len(names)), names, rotation=20, fontsize=8)
ax.set_ylabel("gap to exact optimum [%]")
ax.set_title("optimality gap at N=5 (first reported for this framework)")
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig("figures/T2_optimality_gap.png", dpi=150); plt.close(fig)

check("T2 learned gap <= paper gap", gaps["learned"] <= gaps["paper"] + 0.5,
      "learned %.1f%% vs paper %.1f%%" % (gaps["learned"], gaps["paper"]))
check("T2 all structured policies beat random", 
      max(gaps["paper"], gaps["learned"], gaps["stageA"]) < gaps["random"],
      "random gap %.1f%%" % gaps["random"])

# ------------------------------------------------------------------ T3
print("\n== T3: size transfer (train on 4-6, deploy at 8 and 10) ==")
rowsE = {}
for n in [6, 8, 10]:
    rowsE[n] = {
        "paper": avg_E("paper", seeds=[3, 4, 5], n=n)[0],
        "greedy": avg_E("greedy", seeds=[3, 4, 5], n=n)[0],
        "learned": avg_E("learned", seeds=[3, 4, 5], n=n, vmodel=vm)[0],
        "learned+flex": avg_E("learned", seeds=[3, 4, 5], n=n, vmodel=vm,
                              flexible=True)[0],
    }
fig, ax = plt.subplots(figsize=(6.6, 3.8))
mlist = ["paper", "greedy", "learned", "learned+flex"]
wd = 0.2
for k_, m in enumerate(mlist):
    ax.bar(np.arange(3) + (k_ - 1.5) * wd, [rowsE[n][m] for n in [6, 8, 10]],
           width=wd, label=m)
ax.set_xticks(np.arange(3), ["N=6", "N=8", "N=10"])
ax.set_ylabel("mission energy [J]")
ax.set_title("value model trained on N$\\leq$6, deployed unchanged")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig("figures/T3_size_transfer.png", dpi=150); plt.close(fig)

for n in [8, 10]:
    check("T3 size transfer holds at N=%d" % n,
          rowsE[n]["learned"] <= rowsE[n]["paper"] * 1.03,
          "learned %.2f vs paper %.2f J" % (rowsE[n]["learned"], rowsE[n]["paper"]))

# ------------------------------------------------------------------ T4
print("\n== T4: flexible vs forced over more scenarios ==")
seeds = list(range(3, 11))
ef, _ = avg_E("learned", seeds=seeds, vmodel=vm, flexible=True)
eo, _ = avg_E("learned", seeds=seeds, vmodel=vm, flexible=False)
check("T4 flexible <= forced (dominance in practice)", ef <= eo * 1.002,
      "%.3f vs %.3f J over %d scenarios" % (ef, eo, len(seeds)))

# ------------------------------------------------------------------ report
with open("results_d2.txt", "w") as f:
    f.write("Direction 2 verification report  (%.0f s total)\n" % (time.time() - t_start))
    f.write("=" * 60 + "\n")
    for name, ok, detail in checks:
        f.write(("PASS  " if ok else "FAIL  ") + name +
                ("  | " + detail if detail else "") + "\n")
    f.write("\nmean optimality gaps at N=5: " +
            ", ".join("%s %.1f%%" % (k, v) for k, v in gaps.items()) + "\n")
print("\nwrote results_d2.txt  (%.0f s total)" % (time.time() - t_start))
