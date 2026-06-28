"""Rigor + presentation upgrades on top of run_direction1/2.

  U1  headline Pareto frontier redone with 8 scenarios and error bars
      (overwrites figures/E3_pareto_frontier.png)
  U2  zeta-robustness redone with 8 scenarios, shaded bands, and the
      Stage-D decision-focused line (overwrites figures/T1_zeta_robustness.png)
  U3  Stage D: zeroth-order fine-tune of the value weights on realized
      mission energy, evaluated on held-out scenarios
  U4  battery fairness (audit L16): worst single agent's energy
  U5  sensitivity: the conclusions survive moving Dmax and the WZ factor

Run AFTER the two main drivers:  python3 run_upgrades.py   (~5 min)
"""

import os, time, warnings, copy
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wan.style import use_style
use_style()

from wan.model import P, make_agents
from wan.solver import solve_pair
from wan.network import run_mission
from wan.topology import mission, collect_training, ValueModel, \
    finetune_decision_focused

os.makedirs("figures", exist_ok=True)
checks, t0 = [], time.time()


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""))


SEEDS = list(range(3, 11))          # 8 held-out scenarios everywhere below
ps = dict(P)                         # the D2 stress regime
ps["Tmax"] = 2.5; ps["L_init"] = (15e6, 25e6); ps["beta0"] = 1e-5
ps["f_cpu"] = 2.5e9; ps["C_gen"] = 120.0

# ------------------------------------------------------------------ U1
print("== U1: Pareto frontier with error bars (8 scenarios) ==")
lam_grid = [0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30]
mE, sE, mD, sD = [], [], [], []
for lam in lam_grid:
    outs = [run_mission(s, P, mode="fidelity", lam=lam, allow_motion=False)
            for s in SEEDS]
    Es = [o["E"] for o in outs]; Ds = [o["D"] for o in outs]
    mE.append(np.mean(Es)); sE.append(np.std(Es))
    mD.append(np.mean(Ds)); sD.append(np.std(Ds))
ref = [run_mission(s, P, mode="paper", allow_motion=False) for s in SEEDS]
Eref, Dref = np.mean([o["E"] for o in ref]), np.mean([o["D"] for o in ref])
sEr, sDr = np.std([o["E"] for o in ref]), np.std([o["D"] for o in ref])

fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.errorbar(mD, mE, xerr=sD, yerr=sE, fmt="o-", capsize=3, color="#1f5f8b",
            label="ours (fidelity-aware), $\\lambda$ sweep")
for lam, d, e in zip(lam_grid, mD, mE):
    ax.annotate("$\\lambda$=%.2f" % lam, (d, e), fontsize=7,
                textcoords="offset points", xytext=(6, 5))
ax.errorbar([Dref], [Eref], xerr=[sDr], yerr=[sEr], fmt="o", capsize=3,
            color="#c1403d", ms=8, label="reference paper optimum")
ax.axvline(P["Dmax"], color="#e08a00", ls="--", label="fidelity floor $D_{max}$")
ax.axvspan(P["Dmax"], max(max(mD), Dref) * 1.1, color="orange", alpha=.07)
ax.set_xlabel("root-report distortion $D$  (lower = better report)")
ax.set_ylabel("mission energy [J]")
ax.set_title("energy-fidelity frontier, N=6, mean $\\pm$ s.d. over 8 scenarios")
ax.legend(fontsize=8)
fig.savefig("figures/E3_pareto_frontier.png"); plt.close(fig)

dom = any(d <= Dref and e <= Eref for d, e in zip(mD, mE))
check("U1 dominance holds with 8 scenarios", dom,
      "ref (D=%.2f, E=%.3f)" % (Dref, Eref))
check("U1 reference still violates the floor", Dref > P["Dmax"],
      "D_ref=%.2f" % Dref)

# ------------------------------------------------------------------ U2+U3
print("\n== U2/U3: zeta robustness + Stage D (8 scenarios) ==")
X, y = collect_training(range(100, 124), ps)
vm = ValueModel(l2=2.0); r2 = vm.fit(X, y)

def stats(policy, **kw):
    es = [mission(s, ps, policy=policy, **kw)["E"] for s in SEEDS]
    fin = [e for e in es if np.isfinite(e)]
    return np.mean(fin), np.std(fin)

zetas = [0.0, 1e-15, 1e-14, 1e-13, 1e-12]
curves = {pol: [stats(pol, zeta=z) for z in zetas] for pol in ["paper", "stageA"]}
mC, sC = stats("learned", vmodel=vm)
mF, sF = stats("learned", vmodel=vm, flexible=True)
mB, _ = stats("lyapunov")

vmD = copy.deepcopy(vm)
trainJ = finetune_decision_focused(vmD, ps, train_seeds=list(range(200, 212)),
                                   iters=24, flexible=True)
# validation gate (standard model selection): stage D may replace stage C
# only if it wins on scenarios never seen by EITHER training step
def _val(m):
    return np.mean([mission(s, ps, policy="learned", vmodel=m,
                            flexible=True)["E"] for s in range(300, 306)])
vD, vC = _val(vmD), _val(vm)
gated = vD > vC
if gated:
    vmD = vm
mDfl, sDfl = stats("learned", vmodel=vmD, flexible=True)
print("stage D train %.3f J | val D=%.3f C=%.3f%s | eval %.3f J"
      % (trainJ, vD, vC, " (gated->C)" if gated else "", mDfl))

xs = np.arange(len(zetas))
lab = ["0", "1e-15", "1e-14\n(paper)", "1e-13", "1e-12"]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for pol, col, nm in [("paper", "#c1403d", "paper H-MAP (post-hoc $\\Phi$)"),
                     ("stageA", "#e08a00", "stage A ($\\Phi$ in the loop)")]:
    m = np.array([a for a, _ in curves[pol]]); s_ = np.array([b for _, b in curves[pol]])
    ax.plot(xs, m, "o-", color=col, label=nm)
    ax.fill_between(xs, m - s_, m + s_, color=col, alpha=.13)
for val, sd, col, ls, nm in [
        (mB, 0, "#777777", "-.", "stage B: auto-scaled (no knob)"),
        (mC, sC, "#1f5f8b", "--", "stage C: learned cost-to-go (no $\\zeta$)"),
        (mDfl, sDfl, "#157a3a", "--", "stage D: decision-focused (val-gated)")]:
    ax.axhline(val, color=col, ls=ls, label=nm)
    if sd:
        ax.axhspan(val - sd, val + sd, color=col, alpha=.08)
ax.set_xticks(xs, lab)
ax.set_xlabel("hand-tuned lookahead weight $\\zeta$")
ax.set_ylabel("mission energy [J], mean $\\pm$ s.d., 8 scenarios")
ax.set_title("tuning sensitivity: the learned ladder has nothing to tune")
ax.legend(fontsize=7.6)
fig.savefig("figures/T1_zeta_robustness.png"); plt.close(fig)

best_paper = min(a for a, _ in curves["paper"])
worst_paper = max(a for a, _ in curves["paper"])
check("U2 paper fragility persists at 8 scenarios",
      worst_paper > 1.04 * best_paper, "worst/best=%.2f" % (worst_paper / best_paper))
check("U2 stage C (no knob) <= best hand-tuned paper",
      mC <= best_paper * 1.02, "%.3f vs %.3f J" % (mC, best_paper))
check("U3 stage D <= stage C+flex on held-out scenarios",
      mDfl <= mF * 1.01, "D: %.3f vs C+flex: %.3f J" % (mDfl, mF))
check("U3 full ladder beats paper's best by a clear margin",
      mDfl <= best_paper * 0.97,
      "%.3f vs %.3f J (-%.0f%%)" % (mDfl, best_paper, 100 * (1 - mDfl / best_paper)))

# ------------------------------------------------------------------ U4
print("\n== U4: battery fairness (worst single agent) ==")
fair = {}
for nm, kw in [("paper", {}), ("learned", {"vmodel": vm}),
               ("learned+flex", {"vmodel": vm, "flexible": True})]:
    res = [mission(s, ps, policy="learned" if "learned" in nm else nm, **kw)
           for s in SEEDS]
    fair[nm] = (np.mean([r["Emax"] for r in res]), np.mean([r["E"] for r in res]))
fig, ax = plt.subplots(figsize=(5.6, 3.8))
names = list(fair)
ax.bar(np.arange(3) - 0.17, [fair[n][1] for n in names], width=0.34,
       label="total mission energy", color="#1f5f8b")
ax.bar(np.arange(3) + 0.17, [fair[n][0] for n in names], width=0.34,
       label="worst single agent", color="#c1403d")
ax.set_xticks(np.arange(3), names)
ax.set_ylabel("energy [J], mean of 8 scenarios")
ax.set_title("saving total energy without draining one robot (L16)")
ax.legend(fontsize=8)
fig.savefig("figures/U4_fairness.png"); plt.close(fig)
check("U4 learned saves total energy without a worse bottleneck agent",
      fair["learned"][0] <= fair["paper"][0] * 1.05,
      "worst-agent %.2f vs %.2f J" % (fair["learned"][0], fair["paper"][0]))

# ------------------------------------------------------------------ U5
print("\n== U5: sensitivity of the conclusions ==")
sens = []
for dmax in [0.9, 1.2, 1.5]:
    pp = dict(P); pp["Dmax"] = dmax
    ds = [run_mission(s, pp, mode="fidelity", lam=0.0, derived_cap=True,
                      allow_motion=False)["D"] for s in [3, 4, 5]]
    sens.append((dmax, max(ds)))
check("U5 derived bound tracks ANY chosen floor",
      all(d <= dm + 0.05 for dm, d in sens),
      "; ".join("Dmax=%.1f -> worst D=%.2f" % t for t in sens))

p4 = dict(P); p4["Tmax"] = 3.0; p4["L_init"] = (12e6, 12e6)
rngs = np.random.default_rng(21); agS = make_agents(rngs, p4)
xS, yS = agS[0], agS[1]; xS["rho_pred"] = yS["rho_pred"] = 0.5
cen = np.mean([a["pos"] for a in agS], axis=0)
oms = []
for om in [0.5, 0.9]:
    pw = dict(p4); pw["omega_wz"] = om
    a = solve_pair(xS, yS, pw, fidelity=True, lam=0.05, rho_pair=0.8,
                   use_wz=True, eta_cap_j=1.0, centroid=cen, allow_motion=False)
    b = solve_pair(xS, yS, pw, fidelity=True, lam=0.05, rho_pair=0.8,
                   use_wz=False, eta_cap_j=1.0, centroid=cen, allow_motion=False)
    oms.append((om, 100 * (b["E"] - a["E"]) / b["E"]))
check("U5 WZ saving positive across omega",
      all(g > 0 for _, g in oms),
      "; ".join("omega=%.1f -> %.1f%%" % t for t in oms))

with open("results_upgrades.txt", "w") as f:
    f.write("Upgrade verification report  (%.0f s, value R^2=%.2f)\n" % (time.time() - t0, r2))
    f.write("=" * 60 + "\n")
    for name, ok, detail in checks:
        f.write(("PASS  " if ok else "FAIL  ") + name +
                ("  | " + detail if detail else "") + "\n")
print("\nwrote results_upgrades.txt  (%.0f s)" % (time.time() - t0))
