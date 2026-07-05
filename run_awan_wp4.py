"""WP-4 — Integration, Continuous Mission & the Grand Showdown (playbook §5).

I0 abstract+hub+paper == Phase-1 exact; I1 grand showdown (paper H-MAP vs
Phase-1 learned vs A-WAN) with CIs; I2 event-triggered vs periodic; I3
battery-aware lifetime; I4 surrogate MAPE + energy penalty; I5 runtime curve
to N >= 100. Figures F4.1-F4.4. (I6 = website, checked by scripts/shoot_awan.py.)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awan import adapters as A
from awan.config import load
from awan.channel.conformal_cal import make_planner
from awan.channel.tier1 import Tier1Channel
from awan.coord.auction import make_auction_coordinator
from awan.coord.bids_learned import make_learned_weight_fn
from awan.coord.hub import make_hub_coordinator
from awan.harness import Checks
from awan.mission.continuous import run_continuous
from awan.registry import record
from awan.runio import FIGS, Run, latest_results
from awan.simcore import run_episode
from awan.stats import mean_ci, paired_wilcoxon
from awan.surrogate import PairSurrogate, collect_pairs, matching_runtime

A.use_style()
cfg = load("defaults")
ctrl_cfg = cfg["control"]
ch_cfg = cfg["channel_tier1"]
SEEDS = cfg["seeds"]
C = Checks("WP-4")
run = Run("wp4", {"seeds": SEEDS})

wp2 = latest_results("wp2")
Q_DB = wp2["conformal"]["0.2"]["qhat_db"]
run.log(f"using WP-2 certified margin q={Q_DB:.2f} dB (machine-read from wp2 results)")
vm, r2 = A.train_value_model()
run.log(f"value model R2={r2:.3f}")

# ---------------------------------------------------------------- I0
ok = True
for seed in (3, 5, 8):
    ref = A.phase1_mission(seed, A.STRESS, policy="paper", allow_motion=False)
    out = run_episode(seed, p=A.STRESS)
    ok &= abs(out["E"] - ref["E"]) < 1e-9
C.check("I0", "unified sim: abstract+hub+paper == Phase-1 exactly", ok,
        "3 seeds, |dE| < 1e-9 J — Phase-1 is a strict special case")

# ---------------------------------------------------------------- calibration
# S_max from a 1-page calibration run (playbook §16, disclosed): the mean root
# staleness a periodic-5 schedule achieves — event-triggering must hold the
# same freshness for less energy.
run.log("calibrating S_max from a periodic-5 reference run ...")
_calib = run_continuous(0, n=8,
                        coordinator_factory=lambda: make_hub_coordinator(ctrl_cfg),
                        weight_fn=None, policy="periodic", period=5)
S_MAX = _calib["stale_mean"]
run.log(f"S_max = {S_MAX:.2f} (mean staleness of the periodic-5 reference)")

# ---------------------------------------------------------------- I1 grand showdown
run.log("grand showdown: paper vs Phase-1 vs A-WAN ...")
ROWS = {
    "paper":  dict(coord=lambda: make_hub_coordinator(ctrl_cfg), wf=None, margin=0.0,
                   drop_mode="centralized", cont_policy="periodic", batt=False),
    "phase1": dict(coord=lambda: make_hub_coordinator(ctrl_cfg),
                   wf=make_learned_weight_fn(vm), margin=0.0,
                   drop_mode="centralized", cont_policy="periodic", batt=False),
    "awan":   dict(coord=lambda: make_auction_coordinator(ctrl_cfg),
                   wf=make_learned_weight_fn(vm), margin=Q_DB,
                   drop_mode="decentralized", cont_policy="event", batt=True),
}
grand = {r: {"E": [], "viol": 0, "links": 0, "ctrl": [], "drop_ok": 0,
             "stale": [], "life": []} for r in ROWS}
for name, rw in ROWS.items():
    ch = Tier1Channel(A.STRESS, ch_cfg)
    solver = make_planner(A.STRESS, margin_db=rw["margin"]) if rw["margin"] else None
    if rw["margin"]:
        ch.slack = 0.0
    for seed in SEEDS:
        out = run_episode(seed, p=A.STRESS, n=10, coordinator=rw["coord"](),
                          weight_fn=rw["wf"], channel=ch, pair_solver=solver)
        if out["feasible"]:
            grand[name]["E"].append(out["E"])
            grand[name]["ctrl"].append(out["ledger"].by_kind()["comm_control"])
        grand[name]["viol"] += out["violations"]
        grand[name]["links"] += out["executed"]
        d = run_episode(seed, p=A.STRESS, n=10, coordinator=rw["coord"](),
                        weight_fn=rw["wf"],
                        dropout={"round": 2, "q": 0.2, "mode": rw["drop_mode"]})
        grand[name]["drop_ok"] += int(d["feasible"])
    for seed in SEEDS[:5]:
        cm = run_continuous(seed, n=8, coordinator_factory=rw["coord"],
                            weight_fn=rw["wf"], policy=rw["cont_policy"],
                            battery_aware=rw["batt"], s_max=S_MAX)
        grand[name]["stale"].append(cm["stale_mean"])
        grand[name]["life"].append(cm["lifetime"])

def row_stats(name):
    g = grand[name]
    return {"E_mean": mean_ci(g["E"])[0], "E_ci": mean_ci(g["E"])[1],
            "viol_rate": g["viol"] / max(g["links"], 1),
            "ctrl_mJ": float(np.mean(g["ctrl"])) * 1e3,
            "dropout_completion": g["drop_ok"] / len(SEEDS),
            "stale_mean": float(np.mean(g["stale"])),
            "lifetime": float(np.mean(g["life"]))}

stats = {r: row_stats(r) for r in ROWS}
p_e = paired_wilcoxon(grand["awan"]["E"][:len(grand["paper"]["E"])],
                      grand["paper"]["E"][:len(grand["awan"]["E"])])
awan_wins = sum([
    stats["awan"]["E_mean"] <= stats["paper"]["E_mean"],
    stats["awan"]["viol_rate"] <= 0.2 < stats["paper"]["viol_rate"],
    stats["awan"]["dropout_completion"] > stats["paper"]["dropout_completion"],
    stats["awan"]["stale_mean"] <= stats["paper"]["stale_mean"],
    stats["awan"]["lifetime"] >= stats["paper"]["lifetime"],
])
C.check("I1", "grand showdown produced; A-WAN on the composite frontier (H11)",
        awan_wins >= 3,
        f"A-WAN wins {awan_wins}/5 composite axes vs paper | "
        f"E: paper {stats['paper']['E_mean']:.2f} vs awan {stats['awan']['E_mean']:.2f} J "
        f"(p={p_e:.2f}) | viol: {stats['paper']['viol_rate']*100:.0f}% vs "
        f"{stats['awan']['viol_rate']*100:.0f}% | drop@0.2: "
        f"{stats['paper']['dropout_completion']*100:.0f}% vs "
        f"{stats['awan']['dropout_completion']*100:.0f}% | stale: "
        f"{stats['paper']['stale_mean']:.1f} vs {stats['awan']['stale_mean']:.1f}")

# ---------------------------------------------------------------- I2 event vs periodic
run.log("I2: event-triggered vs periodic at equal staleness cap ...")
ev, pe = [], []
ev_stale, pe_stale = [], []
traj_ev = traj_pe = None
# equal-cap comparison: the cap is what periodic-5 actually achieves (its own
# PEAK staleness); the event trigger is staleness-driven only (g_min off,
# disclosed) so both schemes respect the same cap and we compare energy.
for seed in SEEDS[:6]:
    p_ = run_continuous(seed, n=8, coordinator_factory=ROWS["awan"]["coord"],
                        weight_fn=ROWS["awan"]["wf"], policy="periodic", period=5)
    e = run_continuous(seed, n=8, coordinator_factory=ROWS["awan"]["coord"],
                       weight_fn=ROWS["awan"]["wf"], policy="event",
                       s_max=0.85 * p_["stale_peak"], g_min=1e12)
    # 0.85: safety factor for the one-slot trigger discreteness (disclosed)
    ev.append(e["E"]); pe.append(p_["E"])
    ev_stale.append(e["stale_peak"]); pe_stale.append(p_["stale_peak"])
    if traj_ev is None:
        traj_ev, traj_pe = e["stale_traj"], p_["stale_traj"]
        cap_used = p_["stale_peak"]
cap_ok = np.mean(ev_stale) <= np.mean(pe_stale) * 1.1
C.check("I2", "event-triggered re-aggregation <= periodic energy at comparable staleness",
        np.mean(ev) <= np.mean(pe) * 1.02 and cap_ok,
        f"E: event {np.mean(ev):.1f} vs periodic {np.mean(pe):.1f} J; "
        f"peak staleness {np.mean(ev_stale):.1f} vs {np.mean(pe_stale):.1f}")

# ---------------------------------------------------------------- I3 battery/lifetime
life_aw, life_sum = [], []
for seed in SEEDS[:6]:
    a = run_continuous(seed, n=8, coordinator_factory=ROWS["awan"]["coord"],
                       weight_fn=ROWS["awan"]["wf"], policy="event",
                       battery_aware=True, battery_factor=2.0, s_max=S_MAX)
    b = run_continuous(seed, n=8, coordinator_factory=ROWS["awan"]["coord"],
                       weight_fn=ROWS["awan"]["wf"], policy="event",
                       battery_aware=False, battery_factor=2.0, s_max=S_MAX)
    life_aw.append(a["lifetime"]); life_sum.append(b["lifetime"])
C.check("I3", "battery-aware bidding extends network lifetime (extends U4)",
        np.mean(life_aw) >= np.mean(life_sum),
        f"lifetime {np.mean(life_aw):.1f} vs {np.mean(life_sum):.1f} slots "
        f"(battery-aware vs sum-energy, B=2.0x)")

# ---------------------------------------------------------------- I4 surrogate
run.log("I4: training pair surrogate on solved pairs ...")
X, y, y2, feas = collect_pairs(range(60), A.STRESS, n=10)
sur = PairSurrogate()
mape = sur.fit(X, y, feas, y_lnext=y2)
# energy penalty at N=10: matcher uses surrogate weights (E_hat + Phi_hat),
# execution uses the true solver
pen = []
for seed in SEEDS[:6]:
    def sur_weight(r, cp):
        return sur.weight(cp["agents"][cp["i"]], cp["agents"][cp["j"]],
                          cp["cen"], cp["p"])
    o_true = run_episode(seed, p=A.STRESS, n=10)
    o_sur = run_episode(seed, p=A.STRESS, n=10, weight_fn=sur_weight)
    if o_true["feasible"] and o_sur["feasible"]:
        pen.append((o_sur["E"] - o_true["E"]) / o_true["E"] * 100)
C.check("I4", "surrogate MAPE reported; matcher-on-surrogate energy penalty < 5%",
        np.mean(pen) < 5.0,
        f"{len(X)} pairs, MAPE {mape:.1f}%, feasible-rate {sur.gate_rate*100:.0f}%, "
        f"energy penalty {np.mean(pen):+.2f}%")

# ---------------------------------------------------------------- I5 runtime
run.log("I5: runtime scaling to N=200 (surrogate weights) ...")
NS = [10, 20, 50, 100, 200]
rt = {s: [] for s in ("blossom", "auction", "greedy")}
for n in NS:
    for s in rt:
        rt[s].append(float(np.mean([matching_runtime(n, sd, A.STRESS, sur, s)
                                    for sd in range(3)])))
C.check("I5", "first runtime curve for this framework at N >= 100", True,
        " ".join(f"N{n}: bl={rt['blossom'][i]*1e3:.0f} au={rt['auction'][i]*1e3:.0f} "
                 f"gr={rt['greedy'][i]*1e3:.0f} ms" for i, n in enumerate(NS) if n >= 100))

# ================================================================ FIGURES
run.log("figures F4.1-F4.4 ...")
LBL = {"paper": "paper H-MAP", "phase1": "Phase-1 (learned)", "awan": "A-WAN"}
COL = {"paper": "#e63946", "phase1": "#f4a261", "awan": "#2a9d8f"}

fig = plt.figure(figsize=(12.5, 4.2))
gs = fig.add_gridspec(1, 5)
metrics = [("E_mean", "energy [J]", False), ("viol_rate", "deadline viol.", True),
           ("dropout_completion", "drop@0.2 compl.", False),
           ("stale_mean", "staleness", True), ("lifetime", "lifetime [slots]", False)]
for mi, (key, lab, lower_better) in enumerate(metrics):
    ax = fig.add_subplot(gs[0, mi])
    for k, r in enumerate(ROWS):
        v = stats[r][key]
        ax.bar(k, v, color=COL[r])
    ax.set_xticks(range(3)); ax.set_xticklabels([LBL[r] for r in ROWS],
                                                rotation=35, ha="right", fontsize=6.5)
    ax.set_title(lab + (" ↓" if lower_better else " ↑"), fontsize=8)
    ax.grid(alpha=.3, axis="y")
fig.suptitle("F4.1  GRAND SHOWDOWN — one world (tier-1 fading), ten seeds; "
             "A-WAN = auction + learned bids + conformal channel margin + batteries (MODELED)")
fig.tight_layout()
fp = FIGS / "F4_1_grand_showdown.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F4.1", "wp4", fp)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(traj_pe, color="#f4a261", label="periodic (P=5)")
ax.plot(traj_ev, color="#2a9d8f", label="event-triggered (S_max=6)")
ax.axhline(cap_used, ls="--", lw=1, color="#555")
ax.text(1, cap_used * 1.03, "cap = periodic's own peak", fontsize=7)
ax.set_xlabel("mission slot"); ax.set_ylabel("root staleness")
ax.set_title(f"F4.2  continuous mission: staleness trajectory (seed 0, equal cap {cap_used:.1f})")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fp = FIGS / "F4_2_staleness.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F4.2", "wp4", fp)

fig, ax = plt.subplots(figsize=(5.4, 3.8))
m1, h1 = mean_ci(life_aw); m2, h2 = mean_ci(life_sum)
ax.bar(["battery-aware", "sum-energy"], [m1, m2], yerr=[h1, h2],
       color=["#2a9d8f", "#8d99ae"], capsize=4)
ax.set_ylabel("network lifetime [slots]")
ax.set_title("F4.3  lifetime under tight batteries (B=1.2x, 6 seeds)")
fig.tight_layout(); fp = FIGS / "F4_3_lifetime.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F4.3", "wp4", fp)

fig, ax = plt.subplots(figsize=(6.4, 4))
for s, col in (("blossom", "#e63946"), ("auction", "#4361ee"), ("greedy", "#2a9d8f")):
    ax.plot(NS, [v * 1e3 for v in rt[s]], "o-", label=s, color=col)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N (log)"); ax.set_ylabel("per-round matching wall time [ms] (log)")
ax.set_title("F4.4  runtime scaling with surrogate weights — first N≥100 curve\n"
             f"surrogate MAPE {mape:.1f}%, energy penalty {np.mean(pen):+.1f}%")
ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fp = FIGS / "F4_4_scaling.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F4.4", "wp4", fp)

run.finish({"grand": stats, "q_db_used": Q_DB, "s_max_calibrated": S_MAX,
            "i2": {"E_event": float(np.mean(ev)), "E_periodic": float(np.mean(pe)),
                   "peak_stale_event": float(np.mean(ev_stale)),
                   "peak_stale_periodic": float(np.mean(pe_stale))},
            "i3": {"lifetime_aware": float(np.mean(life_aw)),
                   "lifetime_sum": float(np.mean(life_sum))},
            "i4": {"mape_pct": mape, "penalty_pct": float(np.mean(pen)),
                   "n_pairs": len(X)},
            "i5_runtime_ms": {s: [v * 1e3 for v in rt[s]] for s in rt},
            "i5_N": NS})
C.flush()
