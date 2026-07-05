"""WP-2 — Agentic Dynamic-Channel Intelligence (playbook §3).

C0 paper-channel parity; C1 (H5a) the paper's deterministic plan misses
deadlines under tier-1 fading; C2 (H5b) split-conformal planner restores
Pr(deadline met) >= 1-alpha; C3 (H6) move-to-predicted-channel beats
move-to-shorter-distance; C4 predictor A-vs-B learning curves.
Figures F2.1-F2.4.  Run under .venv-awan:  python run_awan_wp2.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awan import adapters as A
from awan.config import load
from awan.channel.conformal_cal import certified_run, collect_scores, qhat
from awan.channel.mobility import run_policies
from awan.channel.paper import PaperChannel
from awan.channel.predictor import (GPRadioMapB, learning_curve, sample_links)
from awan.channel.tier1 import Tier1Channel
from awan.harness import Checks
from awan.registry import record
from awan.runio import FIGS, Run
from awan.simcore import run_episode
from awan.stats import mean_ci

A.use_style()
cfg = load("defaults")
ch_cfg = cfg["channel_tier1"]
ALPHAS = cfg["conformal"]["alphas"]
C = Checks("WP-2")
run = Run("wp2", {"channel": ch_cfg, "alphas": ALPHAS})

# ---------------------------------------------------------------- C0 parity
ok = True
for seed in (3, 4, 7):
    ref = A.phase1_mission(seed, A.STRESS, policy="paper", allow_motion=False)
    out = run_episode(seed, p=A.STRESS, channel=PaperChannel())
    ok &= abs(out["E"] - ref["E"]) < 1e-9
C.check("C0", "channel: model=paper reproduces Phase-1 exactly", ok,
        "3 seeds, |dE| < 1e-9 J")

# ---------------------------------------------------------------- C1 (H5a)
run.log("C1: deterministic paper plan under tier-1 fading ...")
viol_rates, sev = {}, {}
for sig in (4.0, 8.0):
    ch = Tier1Channel(A.STRESS, {**ch_cfg, "sigma_sh_db": sig})
    v = t = 0
    overruns = []
    for seed in range(12):
        col = []
        out = run_episode(seed, p=A.STRESS, channel=ch, collector=col)
        v += out["violations"]; t += out["executed"]
        overruns += [max(c["s_time"], 0) for c in col]
    viol_rates[sig] = v / t
    sev[sig] = float(np.mean([s for s in overruns if s > 0])) if any(s > 0 for s in overruns) else 0.0
C.check("C1", "paper's deterministic optimum violates deadlines under realistic fading (H5a)",
        all(r >= 0.20 for r in viol_rates.values()),
        f"violation rate sigma4={viol_rates[4.0]*100:.0f}% sigma8={viol_rates[8.0]*100:.0f}% "
        f"(hypothesis >=20%); mean overrun {sev[8.0]:.2f}s at sigma8")

# ---------------------------------------------------------------- C2 (H5b)
run.log("C2: split-conformal dB-margin calibration + held-out coverage ...")
ch = Tier1Channel(A.STRESS, ch_cfg)
cal_scores = collect_scores(ch, A.STRESS, seeds=range(100, 120))
run.log(f"calibration links: {len(cal_scores)}, fade-depth range "
        f"[{cal_scores.min():.1f}, {cal_scores.max():.1f}] dB")
cov = {}
for alpha in ALPHAS:
    q = qhat(cal_scores, alpha)
    cr = certified_run(ch, A.STRESS, seeds=range(200, 220), margin_db=q)
    cov[alpha] = {"qhat_db": q, **cr}
    prem = ((cr["E_certified"] / cr["E_uncertified"]) - 1) * 100 \
        if cr["E_certified"] and cr["E_uncertified"] else float("nan")
    cov[alpha]["energy_premium_pct"] = prem
    run.log(f"alpha={alpha}: qhat={q:.2f} dB -> coverage {cr['coverage']:.3f} "
            f"({cr['n_links']} links), completion {cr['completion']:.2f}, "
            f"certification premium {prem:+.0f}% energy")
C.check("C2", "conformal dB-margin planner meets Pr(deadline) >= 1-alpha held-out (H5b)",
        all(cov[a]["coverage"] >= 1 - a for a in ALPHAS),
        "  ".join(f"alpha={a}: {cov[a]['coverage']:.2f}>={1-a:.2f} "
                  f"(q={cov[a]['qhat_db']:.1f}dB, +{cov[a]['energy_premium_pct']:.0f}%E)"
                  for a in ALPHAS))

# ---------------------------------------------------------------- C3 (H6)
run.log("C3: mobility policies — static/distance/predictive/certified ...")
ch3 = Tier1Channel(A.STRESS, ch_cfg)
q02 = cov[0.2]["qhat_db"]
pol = run_policies(ch3, A.STRESS, seeds=range(300, 340), margin_db=q02)
Es = {k: float(np.mean(v["E"])) if v["E"] else float("inf") for k, v in pol.items()}
vr = {k: v["viol"] / max(v["tot"] - v["infeas"], 1) for k, v in pol.items()}
inf_r = {k: v["infeas"] / v["tot"] for k, v in pol.items()}
dpct = (Es["predictive"] - Es["distance"]) / Es["distance"] * 100
C.check("C3", "move-to-predicted-channel saves energy vs move-to-shorter-distance (H6)",
        Es["predictive"] <= Es["distance"],
        f"static {Es['static']:.3f}J/{vr['static']*100:.0f}%v | "
        f"distance {Es['distance']:.3f}J/{vr['distance']*100:.0f}%v | "
        f"predictive {Es['predictive']:.3f}J/{vr['predictive']*100:.0f}%v "
        f"(Delta {dpct:+.1f}%) | certified(q={q02:.1f}dB) "
        f"{Es['certified']:.3f}J/{vr['certified']*100:.0f}%v/"
        f"{inf_r['certified']*100:.0f}%defer")

# ---------------------------------------------------------------- C4
run.log("C4: predictor A vs B learning curves ...")
ch4 = Tier1Channel(A.STRESS, ch_cfg)
NTR = [30, 100, 300]
lc = learning_curve(ch4, A.STRESS, seeds=[0, 1, 2], n_train_list=NTR)
C.check("C4", "twin+residual GP (B) beats full GP (A) at every sample size",
        all(lc["B"][n] <= lc["A"][n] for n in NTR),
        "  ".join(f"n={n}: A={lc['A'][n]:.2f} B={lc['B'][n]:.2f} dB" for n in NTR))

# ================================================================ FIGURES
run.log("figures F2.1-F2.4 ...")
# F2.1 violation-rate bars
fig, ax = plt.subplots(figsize=(6.4, 4))
labels = ["paper σ=4dB", "paper σ=8dB"] + [f"conformal α={a}" for a in ALPHAS]
vals = [viol_rates[4.0] * 100, viol_rates[8.0] * 100] + \
       [(1 - cov[a]["coverage"]) * 100 for a in ALPHAS]
cols = ["#e63946", "#e63946", "#2a9d8f", "#2a9d8f"]
ax.bar(labels, vals, color=cols)
for a in ALPHAS:
    ax.axhline(a * 100, ls="--", lw=1, color="#888")
    ax.text(3.45, a * 100 + 1, f"α={a}", fontsize=7, color="#555")
ax.set_ylabel("deadline-violation rate [%]")
ax.set_title("F2.1  the paper's channel optimism is unsafe; conformal dB-margin certifies (H5)\n"
             "tier-1 fading, σ_sh as labeled, MODELED energies")
fig.tight_layout(); fp = FIGS / "F2_1_violation_rates.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F2.1", "wp2", fp)

# F2.2 coverage vs target
fig, ax = plt.subplots(figsize=(5.4, 4))
tg = [1 - a for a in ALPHAS]
emp = [cov[a]["coverage"] for a in ALPHAS]
ax.plot([0.7, 1.0], [0.7, 1.0], "--", color="#888", lw=1, label="target")
ax.scatter(tg, emp, s=80, color="#4361ee", zorder=3, label="empirical")
for a, x, y in zip(ALPHAS, tg, emp):
    ax.annotate(f"α={a}\nq̂={cov[a]['qhat_db']:.1f} dB", (x, y), textcoords="offset points",
                xytext=(8, -12), fontsize=7)
ax.set_xlabel("target coverage 1-α"); ax.set_ylabel("empirical coverage")
ax.set_title("F2.2  split-conformal deadline certificate")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fp = FIGS / "F2_2_coverage.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F2.2", "wp2", fp)

# F2.3 mobility policy energies
fig, ax = plt.subplots(figsize=(6.4, 4))
names = ["static", "distance", "predictive", "certified"]
ms = [mean_ci(pol[k]["E"]) for k in names]
ax.bar(names, [m for m, _ in ms], yerr=[h for _, h in ms],
       color=["#8d99ae", "#f4a261", "#2a9d8f", "#4361ee"], capsize=4)
for i, k in enumerate(names):
    tag = f"{vr[k]*100:.0f}% viol" + (f"\n{inf_r[k]*100:.0f}% defer" if inf_r[k] > 0 else "")
    ax.text(i, ms[i][0] * 1.01, tag, ha="center", fontsize=8)
ax.set_ylabel("realized pair energy [J] (MODELED)")
ax.set_title(f"F2.3  mobility policies under tier-1 fading (40 scenarios, vmax-bounded)\n"
             f"predictive vs distance: {dpct:+.1f}%; certified adds the α=0.2 dB margin")
fig.tight_layout(); fp = FIGS / "F2_3_mobility_policies.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F2.3", "wp2", fp)

# F2.4 radio map with trajectory overlay (visual headline)
ch5 = Tier1Channel(A.STRESS, ch_cfg); ch5.new_scenario(7)
agents, _, _ = A.scenario_gen(7, n=2, p=A.STRESS)
rmap = GPRadioMapB(A.STRESS).fit(*sample_links(ch5, A.STRESS, np.random.default_rng(16), 120))
g = 45; xs = np.linspace(0, 500, g)
MU = np.zeros((g, g)); SD = np.zeros((g, g))
aj = agents[1]
for yi, y in enumerate(xs):
    for xi, x in enumerate(xs):
        mu, sd = rmap.predict(np.array([x, y]), aj["pos"])
        MU[yi, xi] = mu; SD[yi, xi] = sd
fig, ax = plt.subplots(figsize=(6.6, 5.2))
im = ax.imshow(MU, origin="lower", extent=[0, 500, 0, 500], cmap="viridis", alpha=.9)
cs = ax.contour(xs, xs, SD, levels=4, colors="w", linewidths=.7, alpha=.8)
ax.clabel(cs, fontsize=6, fmt="σ=%.1f")
th = np.linspace(0, 2 * np.pi, 100)
ai = agents[0]
ax.plot(ai["c"][0] + ai["R"] * np.cos(th), ai["c"][1] + ai["R"] * np.sin(th), "w--", lw=1.2)
ax.scatter(*ai["pos"], c="#f72585", s=60, zorder=5, label="sender start")
ax.scatter(*aj["pos"], c="#4cc9f0", s=60, zorder=5, label="receiver")
from awan.channel.mobility import candidate_grid, _plan_at
best, bestp = None, None
for cand in candidate_grid(ai["c"], ai["R"], 7):
    mu, sd = rmap.predict(cand, aj["pos"])
    det = 10 * np.log10(A.chan_gain(cand, aj["pos"], A.STRESS))
    dm = float(np.linalg.norm(cand - ai["pos"]))
    tm = min(dm / A.STRESS["vmax"] * 1.15 + 1e-3, 1.5) if dm > 0.5 else 0.0
    pl = _plan_at(cand, aj, ai, A.STRESS, factor_db=(mu - sd) - det, t_mob=tm, d_move=dm)
    if pl and (best is None or pl["E_plan"] < best):
        best, bestp = pl["E_plan"], cand
if bestp is not None:
    ax.annotate("", xy=bestp, xytext=ai["pos"],
                arrowprops=dict(arrowstyle="->", color="#f72585", lw=2))
    ax.scatter(*bestp, c="#ffd166", s=90, marker="*", zorder=6, label="predictive endpoint")
ax.set_title("F2.4  learned radio map μ(x) [dB gain to receiver], σ contours,\n"
             "and the predictive move (seed 7)")
ax.legend(fontsize=7, loc="lower right"); fig.colorbar(im, ax=ax, shrink=.85, label="μ dB")
fig.tight_layout(); fp = FIGS / "F2_4_radio_map.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F2.4", "wp2", fp)

run.finish({"viol_rates": viol_rates, "overrun_s": sev, "conformal": cov,
            "mobility": {k: {"E_mean": Es[k], "viol_rate": vr[k],
                             "infeasible": pol[k]["infeas"]} for k in pol},
            "predictor_rmse_db": lc, "cal_links": len(cal_scores),
            "failed_time_margin_variant": {
                "note": "additive time-margin conformal rejected — see DECISIONS.md",
                "qhat_s_alpha01": 19.03, "deadline_s": A.STRESS["Tmax"]}})
C.flush()
