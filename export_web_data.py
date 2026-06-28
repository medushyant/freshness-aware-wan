"""Export every real, verified number into web/data.json for the demo site.

The chart series for the new freshness contribution are recomputed live here
(fast), so the site is driven by real engine output. The heavier verified
headline numbers (oracle gaps, conformal coverage, etc.) are taken from the
passing run_*.py outputs, with the source script named for provenance. Nothing
on the site is invented -- it replays computed results only.
"""
import os, json, warnings, re; warnings.filterwarnings("ignore")
import numpy as np

from wan.model import P as P0, make_agents
from wan.network import _derived_eta_floor
from wan.solver import solve_pair
from wan.freshtopo import mission_fresh, P_FRESH
from wan.targets import spawn_targets
from wan.freshness import assign_agility

P = dict(P0); P.update({"Tmax": 2.5, "L_init": (15e6, 25e6), "beta0": 1e-5,
                        "f_cpu": 2.5e9, "C_gen": 120.0})
A_F = 0.06

# ---- live: D1 freshness ladder (eta + energy vs target speed) --------------
rng = np.random.default_rng(4); ag = make_agents(rng, P, n=2); i, j = ag
i["pos"] = np.array([200., 200.]); i["c"] = i["pos"].copy(); i["R"] = 90.; i["L"] = 22e6
j["pos"] = np.array([200., 300.]); j["c"] = j["pos"].copy(); j["R"] = 90.; j["L"] = 22e6
cen = np.array([200., 250.])
speeds = list(np.round(np.linspace(0, 9, 13), 2))
eta_series, e_series = [], []
for v in speeds:
    jj = dict(j); jj["srcs"] = {1: [j["L"], A_F * v]}
    floor = _derived_eta_floor(jj, 2, P)
    r = solve_pair(i, j, P, fidelity=True, lam=0.0, rho_pair=0.0, eta_cap_j=1.0,
                   eta_lo_j=floor, eta_lo_i=floor, centroid=cen, allow_motion=True)
    eta_series.append(round(r["eta_j"], 3)); e_series.append(round(r["E"], 3))

# ---- live: D2 value topology (energy-only vs fresh-aware) -------------------
N = 6; T = P["Tmax"]; hops = int(np.ceil(np.log2(N)))
def mixed(seed):
    rg = np.random.default_rng(5000 + seed)
    kinds = rg.choice(["static", "dynamic", "time_varying"], size=N)
    tgs = [spawn_targets(k, 1, np.random.default_rng(6000 + seed + i))[0]
           for i, k in enumerate(kinds)]
    return assign_agility(tgs, hops, T)
eoF = eoE = faF = faE = 0.0; cnt = 0
for s in range(20):
    a = mixed(s)
    eo = mission_fresh(s, a, policy="energy_only", lam_f=1e-6)
    fa = mission_fresh(s, a, policy="fresh_aware", lam_f=1e-6)
    if eo["feasible"] and fa["feasible"]:
        eoF += eo["Dfresh"]; eoE += eo["E"]; faF += fa["Dfresh"]; faE += fa["E"]; cnt += 1
topo = {"energy_only": {"stale": round(eoF/cnt, 2), "energy": round(eoE/cnt, 3)},
        "fresh_aware": {"stale": round(faF/cnt, 2), "energy": round(faE/cnt, 3)}}

# spread curve (from run_freshness F4, verified)
spread = {"x": [0.0, 1.5, 3.0, 5.0], "cut_pct": [21, 34, 44, 48]}


def count_checks(path):
    if not os.path.exists(path):
        return None
    t = open(path).read()
    return {"pass": len(re.findall(r"^PASS", t, re.M)),
            "total": len(re.findall(r"^(PASS|FAIL)", t, re.M))}

checks = {m: count_checks(f) for m, f in {
    "Direction 1": "results.txt", "Direction 2": "results_d2.txt",
    "Upgrades": "results_upgrades.txt", "Targets": "results_targets.txt",
    "Freshness": "results_freshness.txt",
    "Metaheuristics": "results_metaheuristics.txt"}.items()}
checks = {k: v for k, v in checks.items() if v}
total_pass = sum(v["pass"] for v in checks.values())
total_all = sum(v["total"] for v in checks.values())

data = {
    "title": "Freshness-Aware Semantic Aggregation for Swarms Tracking Moving Targets",
    "subtitle": "An energy-, fidelity-, and freshness-coupled extension of ILAC-WAN (arXiv:2604.02381)",
    "checks": checks, "total_pass": total_pass, "total_all": total_all,

    # D1 fidelity (verified, run_direction1)
    "d1": {
        "lemma": {"paper": "eta* = eta_req (always compress maximally)",
                  "ours": "eta* interior, set by lambda AND target motion"},
        "eta_range": [0.16, 0.72],
        "pareto_ref": {"D": 2.21, "E": 0.613, "Dmax": 1.2},
        "wz_saving_pct": 5, "conformal_cov": 0.97, "conformal_target": 0.80,
    },
    # D2 topology (verified, run_direction2 / run_metaheuristics)
    "d2": {
        "zeta_fragility": 1.10,
        "gap": {"Learned (ours)": 3.6, "Blossom (paper)": 14.7, "ACO": 14.7,
                "ABC": 14.7, "Cuckoo": 14.7, "Egyptian Vulture": 12.6, "Random": 29.1},
        "size_transfer": {"N=8": {"learned": 12.36, "paper": 12.81},
                          "N=10": {"learned": 15.88, "paper": 16.92}},
        "fairness": {"paper": 3.09, "learned": 2.66, "flex": 2.49},
    },
    # moving-target negative result (verified, experiments/)
    "washout": {"displacement": {"static": 0.0, "dynamic": 30.5, "time_varying": 40.6},
                "R": [80, 100],
                "energy": {"static": 0.587, "dynamic": 0.830, "time_varying": 0.911},
                "r_track8_demand": 40.7, "r_track8_energy": 0.597, "static_energy": 0.601},
    # freshness contribution (live above + verified)
    "freshness": {
        "speeds": speeds, "eta": eta_series, "energy": e_series,
        "energy_rise_pct": round(100*(e_series[-1]/e_series[0]-1)),
        "topo": topo,
        "stale_cut_pct": round(100*(topo["energy_only"]["stale"]-topo["fresh_aware"]["stale"])
                               / topo["energy_only"]["stale"]),
        "energy_premium_pct": round(100*(topo["fresh_aware"]["energy"]-topo["energy_only"]["energy"])
                                    / topo["energy_only"]["energy"], 1),
        "spread": spread,
        "prediction": {"class": ["static", "dynamic", "time_varying"],
                       "react": [3.90, 9.74, 8.75], "CV": [8.46, 8.53, 10.42],
                       "Kalman": [7.37, 7.48, 9.66], "IMM": [7.12, 7.78, 9.69]},
    },
}

# ---- live: a real HMAP pairing trace (topology the animation replays) ------
from wan.network import run_mission
def trace_of(mode):
    out = run_mission(7, P, mode=mode)
    rounds = []
    for entry in out["log"]:
        pairs = [[int(i), (None if j is None else int(j))] for i, j in entry["pairs"]]
        rounds.append({"round": entry["round"], "active": entry["active"], "pairs": pairs})
    return {"rounds": rounds, "E": round(out["E"], 3),
            "D": round(out["D"], 3) if np.isfinite(out["D"]) else None,
            "root": out["root"]}
data["trace"] = {"paper": trace_of("paper"), "ours": trace_of("fidelity")}
data["d2"]["zeta_fragility"] = 1.10
# initial payloads for the six agents (Mb) so the HMAP bars start real
rng2 = np.random.default_rng(7); _ag = make_agents(rng2, P, n=6)
data["payloads0"] = [round(a["L"] / 1e6, 1) for a in _ag]

os.makedirs("web", exist_ok=True)
json.dump(data, open("web/data.json", "w"), indent=2)
print(f"wrote web/data.json  ({total_pass}/{total_all} checks across {len(checks)} modules)")
print("freshness eta:", eta_series[0], "->", eta_series[-1],
      "| topo stale cut", data["freshness"]["stale_cut_pct"], "%")
