"""Freshness-aware aggregation topology (Direction 2, value-prioritized).

Same inner solver and same Blossom matcher as the paper -- only the matching
weight changes. The energy-only policy reproduces the paper; the freshness-
aware policy adds the marginal staleness a source accrues when it takes one
more hop, so the topology learns to give fast-decaying (fast-target) sources
shorter paths to the root. Everything else is identical, so the comparison is
like-for-like.
"""

import numpy as np

from .model import P, make_agents, jaccard_disks
from .solver import solve_pair
from .topology import _pick_pairs

INF = float("inf")

# stress regime: the same one the paper's topology experiments use, so that
# communication actually costs energy and topology choices move real joules.
P_FRESH = dict(P)
P_FRESH.update({"Tmax": 2.5, "L_init": (15e6, 25e6), "beta0": 1e-5,
                "f_cpu": 2.5e9, "C_gen": 120.0})


def _root_freshness(holder, active, a_F, T):
    src = {}
    for a in active:
        src.update(holder[a])
    tot = sum(m["bits"] for m in src.values())
    if tot <= 0:
        return 0.0, src
    fd = sum(m["bits"] * a_F * m["agility"] * (m["hops"] * T)
             for m in src.values()) / tot
    return fd, src


def mission_fresh(seed, agility, p=P_FRESH, policy="fresh_aware", lam_f=3e-7,
                  a_F=1.0, allow_motion=False, n=None):
    """policy in {energy_only, fresh_aware}. agility: per-agent decay speeds.
    Returns total energy, root freshness distortion, and their weighted sum."""
    rng = np.random.default_rng(seed)
    geo = np.random.default_rng(seed + 10_000)
    agents = make_agents(rng, p, n=n)
    n = len(agents)
    T = p["Tmax"]
    holder = {i: {i: {"bits": agents[i]["L"], "agility": float(agility[i]),
                      "hops": 0}} for i in range(n)}

    active = list(range(n))
    E_total = 0.0
    k = 0
    while len(active) > 1:
        k += 1
        cen = np.mean([agents[i]["pos"] for i in active], axis=0)
        rho = {}
        for i in active:
            for j in active:
                if i < j:
                    rho[(i, j)] = jaccard_disks(agents[i]["disks"],
                                                agents[j]["disks"], geo)
        E, w, sol = {}, {}, {}
        for i in active:
            for j in active:
                if i == j:
                    continue
                rp = rho[(min(i, j), max(i, j))]
                r = solve_pair(agents[i], agents[j], p, rho_pair=rp,
                               centroid=cen, allow_motion=allow_motion)
                if r is None:
                    E[(i, j)] = w[(i, j)] = INF
                    continue
                sol[(i, j)] = r
                E[(i, j)] = r["E"]
                if policy == "fresh_aware":
                    # i is the sender: its sources each take one more hop now
                    marg = sum(m["bits"] * m["agility"] for m in holder[i].values())
                    w[(i, j)] = r["E"] + lam_f * a_F * T * marg
                else:
                    w[(i, j)] = r["E"]

        pairs = _pick_pairs(active, w, E, agents, "blossom", rng)

        progressed, new_active = False, []
        for (i, j) in pairs:
            if j is None or (i, j) not in sol:
                new_active.append(i)
                if j is not None:
                    new_active.append(j)
                continue
            r = sol[(i, j)]
            E_total += r["E"]; progressed = True
            for s, m in holder[i].items():        # sender's sources relay
                m = dict(m); m["hops"] += 1
                if s not in holder[j] or m["hops"] < holder[j][s]["hops"]:
                    holder[j][s] = m
            agents[j]["L"] = r["L_next"]
            agents[j]["pos"] = np.asarray(r["pe_j"], float)
            seen = {(round(c[0], 2), round(c[1], 2), round(rd, 2))
                    for c, rd in agents[j]["disks"]}
            for c, rd in agents[i]["disks"]:
                key = (round(c[0], 2), round(c[1], 2), round(rd, 2))
                if key not in seen:
                    agents[j]["disks"].append((c, rd)); seen.add(key)
            new_active.append(j)
        if not progressed:
            return {"E": INF, "Dfresh": INF, "obj": INF, "feasible": False}
        active = new_active
        if k > 14:
            break

    Dfresh, _ = _root_freshness(holder, active, a_F, T)
    return {"E": E_total, "Dfresh": Dfresh,
            "obj": E_total + lam_f * Dfresh, "feasible": True,
            "max_hops": max(m["hops"] for m in holder[active[0]].values())}
