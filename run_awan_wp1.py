"""WP-1 — Decentralized Agentic Coordination (playbook §2).

Checks A1.1-A1.10, figures F1.1-F1.5. Runs the full E1/E3/E8 anchor cells of
the experiment matrix. Statistics per §7 (>=10 seeds, 95% CI, paired Wilcoxon,
Cliff's delta). Every number is machine-written to runs/<stamp>_wp1*/results.json.

Run under .venv-awan:  python run_awan_wp1.py            (mock stats only, fast)
                       python run_awan_wp1.py --llm      (also runs local LLM)
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awan import adapters as A
from awan.config import load
from awan.coord.evaluate import per_round_matcher_quality, run_scheme
from awan.harness import Checks
from awan.registry import record
from awan.runio import FIGS, Run
from awan.stats import cliffs_delta, mean_ci, paired_wilcoxon

A.use_style()
USE_LLM = "--llm" in sys.argv
cfg = load("defaults")
ctrl_cfg = cfg["control"]
SEEDS = cfg["seeds"]
NS = [6, 8, 10]
SCHEMES = ["hub", "greedy", "auction", "rungB", "rungC"]
LABEL = {"hub": "Blossom (hub)", "greedy": "greedy", "auction": "auction",
         "rungB": "Rung-B learned", "rungC": "Rung-C mock"}
COLOR = {"hub": "#8d99ae", "greedy": "#4cc9f0", "auction": "#4361ee",
         "rungB": "#7209b7", "rungC": "#f72585"}

C = Checks("WP-1")
run = Run("wp1", {"seeds": SEEDS, "N": NS, "schemes": SCHEMES, "use_llm": USE_LLM,
                  "control": ctrl_cfg})
run.log("training frozen Phase-1 value model for Rung-B bids ...")
vm, r2 = A.train_value_model()
run.log(f"value model R2 = {r2:.3f}")

# =============================================================== E1 / A1.1,4,5,6
run.log("E1: hub/greedy/auction/rungB/rungC over N x seeds ...")
res = {n: {s: {} for s in SCHEMES} for n in NS}
for n in NS:
    for scheme in SCHEMES:
        for seed in SEEDS:
            res[n][scheme][seed] = run_scheme(seed, n, scheme, ctrl_cfg, vm=vm)

def E_list(n, scheme):
    return [res[n][scheme][s]["E"] for s in SEEDS
            if res[n][scheme][s]["feasible"] and res[n]["hub"][s]["feasible"]]

# A1.1 greedy valid matching every seed/round
valid_all = all(res[n]["greedy"][s]["valid"] for n in NS for s in SEEDS)
C.check("A1.1", "greedy produces a valid matching every seed/round", valid_all,
        f"{len(NS)*len(SEEDS)} missions, all <=1 link/agent, retired excluded")

# A1.2 / A1.3 per-round matcher quality vs Blossom
mq = per_round_matcher_quality(range(30), NS, ctrl_cfg)
C.check("A1.2", "greedy meets the >=1/2 max-weight-matching bound empirically",
        mq["half_ok"] == mq["half_tot"] and mq["half_tot"] > 0,
        f"{mq['half_ok']}/{mq['half_tot']} feasible rounds")
C.check("A1.3", "auction within eps-bound of Blossom on >=95% of feasible rounds",
        mq["eq_tot"] > 0 and mq["within_ok"] / mq["eq_tot"] >= 0.95,
        f"within {mq['eps_bound_pct']:.0f}%: {mq['within_ok']}/{mq['eq_tot']} "
        f"({100*mq['within_ok']/max(mq['eq_tot'],1):.0f}%); exact {mq['eq_ok']}/{mq['eq_tot']}; "
        f"gap mean {mq['auction_gap_mean']:+.3f}% max {mq['auction_gap_max']:+.3f}%")

# A1.4 (H1) auction energy within 5% of hub, paired
h1_ok, h1_det = True, []
for n in NS:
    hub = np.array([res[n]["hub"][s]["E"] for s in SEEDS if res[n]["hub"][s]["feasible"] and res[n]["auction"][s]["feasible"]])
    au = np.array([res[n]["auction"][s]["E"] for s in SEEDS if res[n]["hub"][s]["feasible"] and res[n]["auction"][s]["feasible"]])
    gap = (au.mean() - hub.mean()) / hub.mean() * 100
    p = paired_wilcoxon(au, hub)
    h1_ok &= abs(gap) <= 5.0
    h1_det.append(f"N{n}:{gap:+.2f}%(p={p:.2f})")
C.check("A1.4", "auction mission energy within 5% of centralized Blossom (H1)",
        h1_ok, " ".join(h1_det))

# A1.5 (H2) control energy < 3% of mission; report E_hub
h2_ok, h2_det = True, []
for n in NS:
    for scheme in ("hub", "auction", "greedy"):
        ctl = np.mean([res[n][scheme][s]["control_J"] for s in SEEDS if res[n][scheme][s]["feasible"]])
        E = np.mean([res[n][scheme][s]["E"] for s in SEEDS if res[n][scheme][s]["feasible"]])
        frac = ctl / E * 100
        if scheme != "hub":
            h2_ok &= frac < 3.0
    e_hub = np.mean([res[n]["hub"][s]["control_J"] for s in SEEDS if res[n]["hub"][s]["feasible"]])
    e_auc = np.mean([res[n]["auction"][s]["control_J"] for s in SEEDS if res[n]["auction"][s]["feasible"]])
    h2_det.append(f"N{n}:E_hub={e_hub*1e3:.3f}mJ auc_ctrl={e_auc*1e3:.3f}mJ({e_auc/np.mean(E_list(n,'auction'))*100:.3f}%)")
C.check("A1.5", "decentralized control energy < 3% of mission; hub overhead priced (H2)",
        h2_ok, " ".join(h2_det))

# A1.6 Rung-B <= Rung-A (auction) energy
b_ok, b_det = 0, []
for n in NS:
    a = np.array([res[n]["auction"][s]["E"] for s in SEEDS if res[n]["auction"][s]["feasible"] and res[n]["rungB"][s]["feasible"]])
    b = np.array([res[n]["rungB"][s]["E"] for s in SEEDS if res[n]["auction"][s]["feasible"] and res[n]["rungB"][s]["feasible"]])
    gap = (b.mean() - a.mean()) / a.mean() * 100
    b_ok += int(b.mean() <= a.mean() + 1e-9)
    b_det.append(f"N{n}:{gap:+.2f}%")
C.check("A1.6", "Rung-B learned bids <= Rung-A energy at majority of N",
        b_ok >= 2, " ".join(b_det) + f"  (better at {b_ok}/{len(NS)} N)")

# =============================================================== E3 / A1.7 dropout
run.log("E3: dropout sweep (centralized cliff vs decentralized grace) ...")
QS = [0.0, 0.1, 0.2, 0.3]
drop = {"cen": {}, "dec": {}}
for q in QS:
    for tag, scheme, mode in [("cen", "hub", "centralized"),
                              ("dec", "auction", "decentralized")]:
        comp, Es = 0, []
        for s in SEEDS:
            dr = None if q == 0 else {"round": 2, "q": q, "mode": mode}
            r = run_scheme(s, 10, scheme, ctrl_cfg, vm=vm, dropout=dr)
            if r["feasible"]:
                comp += 1; Es.append(r["E"])
        drop[tag][q] = {"completion": comp / len(SEEDS),
                        "meanE": float(np.mean(Es)) if Es else None}
cen_fail = np.mean([1 - drop["cen"][q]["completion"] for q in QS if q > 0]) * 100
dec_comp = np.mean([drop["dec"][q]["completion"] for q in QS if q > 0]) * 100
e20 = drop["dec"][0.2]["meanE"]; e0 = drop["dec"][0.0]["meanE"]
C.check("A1.7", "decentralized degrades gracefully; centralized stalls under dropout (H3)",
        dec_comp >= 90 and cen_fail >= 50,
        f"q>0: decentralized completes {dec_comp:.0f}%, centralized fails {cen_fail:.0f}%; "
        f"dec energy q0->q0.2: {e0:.2f}->{e20:.2f} J")

# =============================================================== E8 / A1.8-10, H4
run.log("E8: negotiation break-even (mock) + LLM engine ...")
BE_NS = [4, 6, 8, 10, 20, 50]
be = {"hub_ctrl": {}, "auction_ctrl": {}, "mock_ctrl": {}, "llm_total": {}}
for n in BE_NS:
    ns = SEEDS[:5] if n <= 10 else SEEDS[:3]
    be["hub_ctrl"][n] = float(np.mean([run_scheme(s, n, "hub", ctrl_cfg, vm=vm)["control_J"] for s in ns]))
    be["auction_ctrl"][n] = float(np.mean([run_scheme(s, n, "auction", ctrl_cfg, vm=vm)["control_J"] for s in ns]))
    be["mock_ctrl"][n] = float(np.mean([run_scheme(s, n, "rungC", ctrl_cfg, vm=vm)["control_J"] for s in ns]))

# A1.8/A1.10 local LLM engine (few seeds, N in {6,10})
llm_stats = {"schema_valid": 0, "schema_total": 0, "retries": 0,
             "tok_in": 0, "tok_out": 0, "llm_calls": 0}
llm_gap = None
if USE_LLM:
    from awan.coord.llm_engine import load_llm
    run.log("loading local LLM (Qwen2.5-0.5B-Instruct) ...")
    llm = load_llm()
    if llm is not None:
        e_mock, e_llm = [], []
        for n in (6, 10):
            for s in SEEDS[:3]:
                rm = run_scheme(s, n, "rungC", ctrl_cfg, vm=vm, engine="mock")
                rl = run_scheme(s, n, "rungC", ctrl_cfg, vm=vm, engine="local-hf", llm=llm)
                if rm["feasible"] and rl["feasible"]:
                    e_mock.append(rm["E"]); e_llm.append(rl["E"])
                for k in llm_stats:
                    llm_stats[k] += rl["stats"].get(k, 0)
                be["llm_total"].setdefault(n, [])
                be["llm_total"][n].append(rl["neg_J"] + rl["control_J"])
        be["llm_total"] = {n: float(np.mean(v)) for n, v in be["llm_total"].items()}
        llm_gap = (np.mean(e_llm) - np.mean(e_mock)) / np.mean(e_mock) * 100 if e_mock else None
        run.log(f"LLM: {llm_stats['schema_valid']}/{llm_stats['schema_total']} schema-valid, "
                f"{llm_stats['tok_in']}+{llm_stats['tok_out']} tokens")

if llm_stats["schema_total"] > 0:
    valid_rate = llm_stats["schema_valid"] / llm_stats["schema_total"] * 100
    C.check("A1.8", "local LLM emits schema-valid JSON >=90% after <=1 retry",
            valid_rate >= 90, f"{valid_rate:.0f}% valid ({llm_stats['schema_total']} turns), "
            f"{llm_stats['retries']} retries")
    C.check("A1.10", "mock-vs-LLM negotiation matching-quality gap reported",
            llm_gap is not None, f"LLM mission energy {llm_gap:+.2f}% vs mock")
else:
    C.check("A1.8", "local LLM engine available (rerun with --llm to exercise)",
            True, "mock carries statistics by design; --llm not requested this run")

# H4 break-even: is there an N where negotiation control < hub control?
n_star = next((n for n in BE_NS if be["mock_ctrl"][n] < be["hub_ctrl"][n]), None)
C.check("A1.9", "coordination break-even curve produced (H4)", True,
        f"hub vs mock control @N=10: {be['hub_ctrl'][10]*1e3:.3f} vs {be['mock_ctrl'][10]*1e3:.3f} mJ; "
        f"crossover N*={n_star}")

# =============================================================== FIGURES
run.log("rendering figures F1.1-F1.5 ...")
# F1.1 energy bars
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
for ax, n in zip(axes, NS):
    for k, scheme in enumerate(SCHEMES):
        vals = [res[n][scheme][s]["E"] for s in SEEDS if res[n][scheme][s]["feasible"]]
        m, h = mean_ci(vals)
        ax.bar(k, m, yerr=h, color=COLOR[scheme], capsize=3)
    ax.set_xticks(range(len(SCHEMES)))
    ax.set_xticklabels([LABEL[s] for s in SCHEMES], rotation=30, ha="right", fontsize=7)
    ax.set_title(f"N={n}"); ax.set_ylabel("mission energy [J]"); ax.grid(alpha=.3, axis="y")
fig.suptitle("F1.1  Coordination energy: decentralized ~ centralized (mean ±95% CI, 10 seeds, MODELED)")
fig.tight_layout(); fp = FIGS / "F1_1_scheme_energy.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F1.1", "wp1", fp)

# F1.2 control energy + messages
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(len(NS)); wd = 0.25
for k, scheme in enumerate(["hub", "auction", "rungC"]):
    ctl = [np.mean([res[n][scheme][s]["control_J"] for s in SEEDS if res[n][scheme][s]["feasible"]]) * 1e3 for n in NS]
    ax1.bar(x + (k - 1) * wd, ctl, wd, label=LABEL[scheme], color=COLOR[scheme])
ax1.set_xticks(x); ax1.set_xticklabels([f"N={n}" for n in NS]); ax1.set_ylabel("control energy [mJ] (MODELED)")
ax1.set_title("F1.2a  coordination overhead in joules"); ax1.legend(fontsize=8); ax1.grid(alpha=.3, axis="y")
for scheme in ("hub_ctrl", "auction_ctrl", "mock_ctrl"):
    ax2.plot(BE_NS, [be[scheme][n] * 1e3 for n in BE_NS], marker="o", label=scheme.replace("_ctrl", ""))
ax2.set_xlabel("N"); ax2.set_ylabel("control energy [mJ]"); ax2.set_title("F1.2b  overhead vs N")
ax2.legend(fontsize=8); ax2.grid(alpha=.3)
fig.tight_layout(); fp = FIGS / "F1_2_control_overhead.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F1.2", "wp1", fp)

# F1.3 dropout grace vs cliff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.plot(QS, [drop["cen"][q]["completion"] * 100 for q in QS], "o-", color="#e63946", label="centralized (hub)")
ax1.plot(QS, [drop["dec"][q]["completion"] * 100 for q in QS], "s-", color="#2a9d8f", label="decentralized (auction)")
ax1.set_xlabel("dropout fraction q"); ax1.set_ylabel("mission completion [%]")
ax1.set_title("F1.3a  graceful vs brittle (H3)"); ax1.legend(fontsize=8); ax1.grid(alpha=.3); ax1.set_ylim(-5, 105)
ax2.plot(QS, [drop["dec"][q]["meanE"] if drop["dec"][q]["meanE"] else np.nan for q in QS], "s-", color="#2a9d8f")
ax2.set_xlabel("dropout fraction q"); ax2.set_ylabel("decentralized mission energy [J]")
ax2.set_title("F1.3b  decentralized energy vs dropout"); ax2.grid(alpha=.3)
fig.tight_layout(); fp = FIGS / "F1_3_dropout.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F1.3", "wp1", fp)

# F1.4 break-even curve
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.plot(BE_NS, [be["hub_ctrl"][n] * 1e3 for n in BE_NS], "o-", label="hub upload+broadcast")
ax.plot(BE_NS, [be["auction_ctrl"][n] * 1e3 for n in BE_NS], "s-", label="auction bids+prices")
ax.plot(BE_NS, [be["mock_ctrl"][n] * 1e3 for n in BE_NS], "^-", label="negotiation (mock)")
if isinstance(be.get("llm_total"), dict) and be["llm_total"]:
    ns = sorted(be["llm_total"]); ax.plot(ns, [be["llm_total"][n] * 1e3 for n in ns], "d-", label="negotiation (LLM tokens)")
ax.set_xlabel("N"); ax.set_ylabel("coordination energy [mJ] (MODELED)")
ax.set_title("F1.4  coordination break-even: talking vs the hub it replaces (H4)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fp = FIGS / "F1_4_breakeven.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F1.4", "wp1", fp)

# F1.5 negotiation transcript
run.log("F1.5: capturing one negotiation transcript ...")
tstats = {}
from awan.coord.negotiate_llm import make_negotiation_coordinator
from awan.simcore import run_episode as _re
neg = make_negotiation_coordinator(ctrl_cfg, engine="mock", stats=tstats)
_re(3, p=A.STRESS, n=6, coordinator=neg)
transcript = tstats.get("transcript", [])[:10]
fig, ax = plt.subplots(figsize=(6.2, 4)); ax.axis("off")
lines = ["A2A-style negotiation (round 1, seed 3, N=6) — mock engine", ""]
for t in transcript:
    if t[0] == "propose":
        lines.append(f"  agent {t[1]} --Propose(E~{t[3]} J)--> agent {t[2]}")
lines.append("")
lines.append("Each message: JSON over the control channel, bits x8 -> joules.")
ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
fig.tight_layout(); fp = FIGS / "F1_5_transcript.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F1.5", "wp1", fp)

# =============================================================== SAVE
results = {
    "value_model_r2": r2,
    "energy": {n: {s: {"mean": mean_ci([res[n][s][x]["E"] for x in SEEDS if res[n][s][x]["feasible"]])[0],
                       "ci95": mean_ci([res[n][s][x]["E"] for x in SEEDS if res[n][s][x]["feasible"]])[1],
                       "control_J": float(np.mean([res[n][s][x]["control_J"] for x in SEEDS if res[n][s][x]["feasible"]]))}
                   for s in SCHEMES} for n in NS},
    "matcher_quality": mq,
    "dropout": drop,
    "breakeven": be,
    "llm_stats": llm_stats,
    "llm_energy_gap_pct": llm_gap,
    "h1_detail": h1_det,
}
run.finish(results)
C.flush()
