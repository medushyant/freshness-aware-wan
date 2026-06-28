"""Metaheuristic matchers as comparison baselines (mentor's request).

The honest framing: the per-round pairing is solved EXACTLY by Blossom in
polynomial time, so nature-inspired metaheuristics cannot beat it at this
scale -- they can only approach it. We show exactly that, two ways:

  M1  per-round matching quality vs the exact Blossom optimum. Blossom is 0%
      above optimal by definition; each metaheuristic is a small positive gap
      (none beats exact -- the sanity check); all are far better than random.
  M2  mission-level optimality gap vs the brute-force oracle. The metaheuristics
      cluster at the myopic per-round level; only the LEARNED lookahead policy
      (Direction 2) closes the gap -- so the contribution is the learning, not
      the metaheuristic, which is the point.

Run:  python3 run_metaheuristics.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wan.model import P, make_agents, jaccard_disks
from wan.solver import solve_pair
from wan.topology import mission, dp_oracle, _pick_pairs, ValueModel, collect_training
from wan.metaheuristics import MATCHERS, _matching_cost, _random_matching, _orient
from wan.style import use_style; use_style()

os.makedirs("figures", exist_ok=True)
checks = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""))

PP = dict(P)
PP.update({"Tmax": 2.5, "L_init": (15e6, 25e6), "beta0": 1e-5,
           "f_cpu": 2.5e9, "C_gen": 120.0})


def instance(seed, n):
    rng = np.random.default_rng(seed); geo = np.random.default_rng(seed + 99)
    ag = make_agents(rng, PP, n=n)
    active = list(range(n)); w = {}
    cen = np.mean([a["pos"] for a in ag], axis=0)
    for i in active:
        for j in active:
            if i == j:
                continue
            rp = jaccard_disks(ag[i]["disks"], ag[j]["disks"], geo)
            r = solve_pair(ag[i], ag[j], PP, rho_pair=rp, centroid=cen, allow_motion=False)
            w[(i, j)] = r["E"] if r else float("inf")
    return active, w, ag


# ------------------------------------------------------------------ M1
print("== M1: per-round matching quality vs exact Blossom, scaling with N ==")
import time
def blossom_cost(active, w, ag, rng):
    pairs = _pick_pairs(active, w, {}, ag, "paper", rng)
    return _matching_cost([(i, j) for i, j in pairs], w)

NS = [6, 10, 16, 22, 30]
meth = ["ACO", "ABC", "Cuckoo", "Egyptian Vulture", "Random"]
gap_vs_n = {m: [] for m in meth}
rt_blossom, rt_aco = [], []
for n in NS:
    g = {m: [] for m in meth}; tb, ta = [], []
    for s in range(10):
        active, w, ag = instance(1000 + s, n)
        rng = np.random.default_rng(s)
        t0 = time.perf_counter(); opt = blossom_cost(active, w, ag, rng); tb.append(time.perf_counter()-t0)
        if not np.isfinite(opt) or opt <= 0:
            continue
        for m in MATCHERS:
            if m == "ACO":
                t0 = time.perf_counter(); pairs = MATCHERS[m](active, w, {}, np.random.default_rng(s+1)); ta.append(time.perf_counter()-t0)
            else:
                pairs = MATCHERS[m](active, w, {}, np.random.default_rng(s+1))
            c = _matching_cost(pairs, w)
            if np.isfinite(c):
                g[m].append(100*(c-opt)/opt)
        rc = _matching_cost(_orient(_random_matching(active, np.random.default_rng(s+2)), w), w)
        if np.isfinite(rc):
            g["Random"].append(100*(rc-opt)/opt)
    for m in meth:
        gap_vs_n[m].append(np.mean(g[m]) if g[m] else np.nan)
    rt_blossom.append(1000*np.mean(tb)); rt_aco.append(1000*np.mean(ta))
    print(f"   N={n:2d}: " + "  ".join(f"{m.split()[0]} {gap_vs_n[m][-1]:4.1f}%" for m in meth)
          + f"  | blossom {rt_blossom[-1]:.2f}ms ACO {rt_aco[-1]:.1f}ms")

worst_meta = max(np.nanmax(gap_vs_n[m]) for m in MATCHERS)
check("M1 no metaheuristic beats exact Blossom at any N (sanity)",
      all(np.nanmin(gap_vs_n[m]) >= -0.5 for m in MATCHERS),
      "best metaheuristic still %.1f%% above" % min(np.nanmin(gap_vs_n[m]) for m in MATCHERS))
check("M1 metaheuristics stay far below random",
      worst_meta < np.nanmin(gap_vs_n["Random"]),
      "worst meta %.1f%% < random %.1f%%" % (worst_meta, np.nanmin(gap_vs_n["Random"])))
check("M1 exact Blossom stays fast (no need for metaheuristics)",
      rt_blossom[-1] < 50, "blossom %.1f ms at N=%d" % (rt_blossom[-1], NS[-1]))

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
cols = {"ACO": "#2a9d8f", "ABC": "#e9c46a", "Cuckoo": "#f4a261",
        "Egyptian Vulture": "#e76f51", "Random": "#9aa7b1"}
ax[0].axhline(0, color="#1f5f8b", lw=2, label="Blossom (exact)")
for m in meth:
    ax[0].plot(NS, gap_vs_n[m], "-o", color=cols[m], label=m)
ax[0].set_xlabel("agents N"); ax[0].set_ylabel("% above exact matching optimum")
ax[0].set_title("metaheuristics approach but never beat exact"); ax[0].legend(fontsize=8)
ax[1].plot(NS, rt_blossom, "-o", color="#1f5f8b", label="Blossom (exact)")
ax[1].plot(NS, rt_aco, "-o", color="#2a9d8f", label="ACO")
ax[1].set_xlabel("agents N"); ax[1].set_ylabel("matcher runtime [ms]")
ax[1].set_title("exact stays fast, so metaheuristics are unnecessary"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/M1_matching_quality.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ M2
print("\n== M2: mission optimality gap vs brute-force oracle (N=5) ==")
X, y = collect_training(range(100, 124), p=PP)
vm = ValueModel(l2=2.0); vm.fit(X, y)
SEEDS = range(3, 11)
def gap(run):
    gs = []
    for s in SEEDS:
        opt = dp_oracle(s, p=PP, n=5); e = run(s)
        if np.isfinite(e) and opt > 0:
            gs.append(100 * (e - opt) / opt)
    return np.mean(gs)

bars = {}
bars["Learned (ours)"] = gap(lambda s: mission(s, PP, policy="learned", vmodel=vm, n=5)["E"])
bars["Blossom (paper)"] = gap(lambda s: mission(s, PP, policy="paper", n=5)["E"])
for nm, fn in MATCHERS.items():
    bars[nm] = gap(lambda s, fn=fn: mission(s, PP, policy="paper", n=5,
                   matcher=lambda a, w, E, rng: fn(a, w, E, rng))["E"])
bars["Random"] = gap(lambda s: mission(s, PP, policy="random", n=5)["E"])
for nm, g in bars.items():
    print(f"   {nm:18s}: {g:5.1f}% gap")
check("M2 learned lookahead beats every per-round matcher",
      all(bars["Learned (ours)"] <= bars[nm] + 1e-9 for nm in bars if nm != "Learned (ours)"),
      "learned %.1f%%" % bars["Learned (ours)"])
check("M2 all structured matchers beat random",
      all(bars[nm] < bars["Random"] for nm in MATCHERS),
      "random %.1f%%" % bars["Random"])

order = ["Learned (ours)", "Blossom (paper)", "ACO", "ABC", "Cuckoo",
         "Egyptian Vulture", "Random"]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
cc = ["#1f5f8b", "#577590", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#9aa7b1"]
ax.bar(range(len(order)), [bars[nm] for nm in order], color=cc)
ax.set_xticks(range(len(order)), order, rotation=20, ha="right")
ax.set_ylabel("% above brute-force optimum")
ax.set_title("only the learned lookahead closes the optimality gap")
fig.tight_layout(); fig.savefig("figures/M2_optimality_gap.png", dpi=150); plt.close(fig)

with open("results_metaheuristics.txt", "w") as f:
    f.write("Metaheuristic comparison baselines -- verification\n" + "=" * 52 + "\n")
    for nm, ok, d in checks:
        f.write(("PASS  " if ok else "FAIL  ") + nm + ("  | " + d if d else "") + "\n")
print(f"\n{sum(1 for _,ok,_ in checks if ok)}/{len(checks)} PASS -> results_metaheuristics.txt")
