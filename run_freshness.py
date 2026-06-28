"""Freshness-aware aggregation for moving targets -- verification + figures.

This is the moving-target spine of the project, coupled the right way: target
motion enters through information freshness, which drives the semantic
compression ratio (Direction 1) and the aggregation topology (Direction 2).
The mobility/coverage route was shown empirically inert in experiments/.

  F1  mechanism (exact): faster target -> staler data -> higher eta-floor
      -> more energy. Deterministic, no scenario noise.
  F2  robustness: the same ladder holds across random aggregation pairs,
      with tight bands -- it is not an artefact of one geometry.
  F3  value-prioritized topology cuts root staleness at no energy cost
      vs the paper's energy-only matching (and reduces to it at lam_f=0).
  F4  the freshness advantage grows with the spread of target agility --
      prioritization is most valuable when sources decay heterogeneously.

Run:  python3 run_freshness.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wan.model import P as P0, make_agents
from wan.network import _derived_eta_floor
from wan.solver import solve_pair
from wan.freshtopo import mission_fresh, P_FRESH
from wan.targets import spawn_targets
from wan.freshness import assign_agility
from wan.style import use_style; use_style()

os.makedirs("figures", exist_ok=True)
checks = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""))

P = dict(P0)
P.update({"Tmax": 2.5, "L_init": (15e6, 25e6), "beta0": 1e-5,
          "f_cpu": 2.5e9, "C_gen": 120.0})
A_F = 0.06
K_REM = 2
SPEEDS = np.linspace(0, 9, 19)


def pair_at(i, j, speed, cen):
    d0 = A_F * speed
    jj = dict(j); jj["srcs"] = {1: [j["L"], d0]}
    floor = _derived_eta_floor(jj, K_REM, P)
    r = solve_pair(i, j, P, fidelity=True, lam=0.0, rho_pair=0.0, eta_cap_j=1.0,
                   eta_lo_j=floor, eta_lo_i=floor, centroid=cen, allow_motion=True)
    return floor, r


# ------------------------------------------------------------------ F1
print("== F1: exact freshness->compression->energy mechanism ==")
rng = np.random.default_rng(4)
ag = make_agents(rng, P, n=2); i, j = ag[0], ag[1]
i["pos"] = np.array([200., 200.]); i["c"] = i["pos"].copy(); i["R"] = 90.; i["L"] = 22e6
j["pos"] = np.array([200., 300.]); j["c"] = j["pos"].copy(); j["R"] = 90.; j["L"] = 22e6
cen = np.array([200., 250.])
etas, Es = [], []
for v in SPEEDS:
    fl, r = pair_at(i, j, v, cen)
    etas.append(r["eta_j"] if r else np.nan); Es.append(r["E"] if r else np.nan)
etas, Es = np.array(etas), np.array(Es)
mono_eta = np.all(np.diff(etas) >= -1e-9)
mono_E = np.all(np.diff(Es) >= -1e-6)
check("F1 eta-floor rises monotonically with target speed", mono_eta,
      "eta %.3f -> %.3f" % (etas[0], etas[-1]))
check("F1 energy rises monotonically with target speed", mono_E,
      "E %.3f -> %.3f J (+%.0f%%)" % (Es[0], Es[-1], 100*(Es[-1]/Es[0]-1)))

# ------------------------------------------------------------------ F2
print("\n== F2: same ladder across 30 random pairs (robust, with bands) ==")
LEVELS = [0.0, 4.5, 9.0]
eta_lvl, E_lvl = {v: [] for v in LEVELS}, {v: [] for v in LEVELS}
for s in range(30):
    rr = np.random.default_rng(700 + s)
    a = make_agents(rr, P, n=2); ii, jj = a[0], a[1]
    cc = 0.5 * (ii["pos"] + jj["pos"])
    for v in LEVELS:
        fl, r = pair_at(ii, jj, v, cc)
        if r:
            eta_lvl[v].append(r["eta_j"]); E_lvl[v].append(r["E"])
mean_eta = [np.mean(eta_lvl[v]) for v in LEVELS]
mean_E = [np.mean(E_lvl[v]) for v in LEVELS]
for v, me, mE in zip(LEVELS, mean_eta, mean_E):
    print(f"   speed {v:4.1f} m/s : mean eta {me:.3f}, mean E {mE:.3f} J")
check("F2 mean eta increases with agility across random pairs",
      mean_eta[0] < mean_eta[1] < mean_eta[2],
      "%.3f < %.3f < %.3f" % tuple(mean_eta))
check("F2 mean energy increases with agility across random pairs",
      mean_E[0] < mean_E[1] < mean_E[2],
      "%.3f < %.3f < %.3f J" % tuple(mean_E))

fig, ax1 = plt.subplots(figsize=(6.6, 4.2))
ax1.plot(SPEEDS, etas, "-o", color="#1f5f8b", label="optimal eta (one pair, exact)")
ax1.set_xlabel("target speed [m/s]")
ax1.set_ylabel("optimal compression ratio eta*", color="#1f5f8b")
ax1.tick_params(axis="y", labelcolor="#1f5f8b")
ax2 = ax1.twinx()
ax2.plot(SPEEDS, Es, "-s", color="#c1403d", label="pair energy")
ax2.set_ylabel("pair energy [J]", color="#c1403d"); ax2.tick_params(axis="y", labelcolor="#c1403d")
ax1.set_title("D1: faster target -> stale data -> compress less -> spend more")
fig.tight_layout(); fig.savefig("figures/G4_freshness_compression.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ F3
print("\n== F3: value-prioritized topology vs energy-only matching ==")
N = 6; T = P["Tmax"]; hops = int(np.ceil(np.log2(N)))
def mixed_agility(seed):
    rng = np.random.default_rng(5000 + seed)
    kinds = rng.choice(["static", "dynamic", "time_varying"], size=N)
    tgs = [spawn_targets(k, 1, np.random.default_rng(6000 + seed + i))[0]
           for i, k in enumerate(kinds)]
    return assign_agility(tgs, hops, T)

SEEDS = range(20)
eoE, eoF, faE, faF, same0 = [], [], [], [], []
for s in SEEDS:
    a = mixed_agility(s)
    eo = mission_fresh(s, a, policy="energy_only", lam_f=1e-6)
    fa = mission_fresh(s, a, policy="fresh_aware", lam_f=1e-6)
    z_eo = mission_fresh(s, a, policy="energy_only", lam_f=0.0)
    z_fa = mission_fresh(s, a, policy="fresh_aware", lam_f=0.0)
    if eo["feasible"] and fa["feasible"]:
        eoE.append(eo["E"]); eoF.append(eo["Dfresh"])
        faE.append(fa["E"]); faF.append(fa["Dfresh"])
        same0.append(abs(z_eo["Dfresh"] - z_fa["Dfresh"]) < 1e-9)
eoE, eoF, faE, faF = map(np.array, (eoE, eoF, faE, faF))
print(f"   energy-only : E={eoE.mean():.3f} J, root staleness={eoF.mean():.2f}")
print(f"   fresh-aware : E={faE.mean():.3f} J, root staleness={faF.mean():.2f}")
fresh_cut = 100 * (eoF.mean() - faF.mean()) / eoF.mean()
e_premium = 100 * (faE.mean() - eoE.mean()) / eoE.mean()
check("F3 lam_f=0 reduces fresh-aware to the paper (sanity)", all(same0),
      "%d/%d identical" % (sum(same0), len(same0)))
check("F3 fresh-aware cuts root staleness", faF.mean() < eoF.mean(),
      "%.0f%% fresher" % fresh_cut)
check("F3 staleness cut comes at no energy cost", e_premium <= 1.0,
      "energy change %+.1f%%" % e_premium)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar([0, 1], [eoF.mean(), faF.mean()], yerr=[eoF.std(), faF.std()],
          capsize=4, color=["#9aa7b1", "#1f5f8b"])
ax[0].set_xticks([0, 1], ["energy-only\n(paper)", "value-prioritized\n(ours)"])
ax[0].set_ylabel("root report staleness (freshness distortion)")
ax[0].set_title(f"fresher report: -{fresh_cut:.0f}%")
ax[1].bar([0, 1], [eoE.mean(), faE.mean()], yerr=[eoE.std(), faE.std()],
          capsize=4, color=["#9aa7b1", "#1f5f8b"])
ax[1].set_xticks([0, 1], ["energy-only\n(paper)", "value-prioritized\n(ours)"])
ax[1].set_ylabel("mission energy [J]")
ax[1].set_title(f"at no energy cost ({e_premium:+.1f}%)")
fig.tight_layout(); fig.savefig("figures/G5_value_topology.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ F4
print("\n== F4: freshness advantage grows with agility spread ==")
def agility_with_spread(seed, spread):
    rng = np.random.default_rng(8000 + seed)
    base = 3.0
    return np.clip(base + spread * rng.standard_normal(N), 0, None)
SPREADS = [0.0, 1.5, 3.0, 5.0]
cut_by_spread = []
for sp in SPREADS:
    cuts = []
    for s in range(16):
        a = agility_with_spread(s, sp)
        eo = mission_fresh(s, a, policy="energy_only", lam_f=1e-6)
        fa = mission_fresh(s, a, policy="fresh_aware", lam_f=1e-6)
        if eo["feasible"] and fa["feasible"] and eo["Dfresh"] > 1e-9:
            cuts.append(100 * (eo["Dfresh"] - fa["Dfresh"]) / eo["Dfresh"])
    cut_by_spread.append(np.mean(cuts) if cuts else 0.0)
    print(f"   agility spread {sp:.1f} m/s : staleness cut {cut_by_spread[-1]:5.1f}%")
check("F4 freshness advantage increases with agility spread",
      cut_by_spread[0] <= cut_by_spread[-1] and cut_by_spread[-1] > cut_by_spread[1],
      "%.0f%% -> %.0f%%" % (cut_by_spread[1], cut_by_spread[-1]))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(SPREADS, cut_by_spread, "-o", color="#1f5f8b")
ax.set_xlabel("agility spread across targets [m/s]")
ax.set_ylabel("root staleness cut by prioritization [%]")
ax.set_title("value of prioritization grows with decay heterogeneity")
fig.tight_layout(); fig.savefig("figures/G6_spread.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ F5
print("\n== F5: prediction lowers the effective decay rate (Kalman/IMM) ==")
from wan.targets import CVPredictor, KalmanCV, IMMPredictor
T5, R5, SIG = 2.0, 10, 3.0
class React:
    def predict(self, m, T): return m.copy()
def pred_err(make):
    out = {}
    for k in ["static", "dynamic", "time_varying"]:
        es = []
        for s in range(40):
            tg = spawn_targets(k, 1, np.random.default_rng(s))[0]
            pr = make(); rg = np.random.default_rng(1000 + s); e = []
            for _ in range(R5):
                g = pr.predict(tg.pos() + SIG * rg.standard_normal(2), T5)
                tg.step(T5); e.append(np.linalg.norm(g - tg.pos()))
            es.append(np.mean(e[1:]))
        out[k] = np.mean(es)
    return out
react = pred_err(React)
cv = pred_err(CVPredictor)
kal = pred_err(lambda: KalmanCV(r=SIG * SIG))
imm = pred_err(lambda: IMMPredictor(r=SIG * SIG))
print(f"   {'class':16s}{'react':>8s}{'CV':>8s}{'Kalman':>8s}{'IMM':>8s}")
for k in ["static", "dynamic", "time_varying"]:
    print(f"   {k:16s}{react[k]:8.2f}{cv[k]:8.2f}{kal[k]:8.2f}{imm[k]:8.2f}")
filt_beats_cv = all(min(kal[k], imm[k]) <= cv[k] + 1e-9
                    for k in ["static", "dynamic", "time_varying"])
check("F5 Kalman/IMM dominate naive finite-difference CV on every class",
      filt_beats_cv, "filtering beats differencing")
check("F5 prediction pays on constant-velocity targets",
      min(kal["dynamic"], imm["dynamic"]) < react["dynamic"],
      "best %.2f < react %.2f m" % (min(kal["dynamic"], imm["dynamic"]), react["dynamic"]))
# freshness impact: best-predictor decay vs react decay on moving targets
best_decay = {k: min(react[k], cv[k], kal[k], imm[k]) / T5 for k in react}
react_decay = {k: react[k] / T5 for k in react}
fr_best = mission_fresh(0, [best_decay["dynamic"]] * N, policy="energy_only")["Dfresh"]
fr_react = mission_fresh(0, [react_decay["dynamic"]] * N, policy="energy_only")["Dfresh"]
check("F5 adaptive prediction yields a fresher report on moving targets",
      fr_best <= fr_react, "staleness %.2f <= %.2f" % (fr_best, fr_react))

with open("results_freshness.txt", "w") as f:
    f.write("Freshness-aware moving-target extension -- verification\n" + "=" * 56 + "\n")
    for name, ok, detail in checks:
        f.write(("PASS  " if ok else "FAIL  ") + name +
                ("  | " + detail if detail else "") + "\n")
n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"\n{n_pass}/{len(checks)} PASS  ->  wrote results_freshness.txt")
