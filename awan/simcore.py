"""Unified Phase-2 round loop (playbook §1.3).

state -> coordination -> pair execution -> state update, with pluggable
Coordinator / Channel / pair-solver backends. With the defaults
(hub coordinator, paper channel, abstract payload) this reproduces the frozen
`wan.topology.mission(policy='paper')` EXACTLY — asserted by check I0/W0.3 —
because it mirrors that loop's rng discipline, solve calls, weights, matcher,
and state updates through the adapters.
"""

import numpy as np

from . import adapters as A
from .ledger import EnergyLedger

INF = float("inf")


def hub_paper_coordinator(ctx):
    """The paper's centralized recipe: Blossom on w = E + zeta*Phi (Eq. 36)."""
    return A._pick_pairs(ctx["active"], ctx["w"], ctx["E"], ctx["agents"],
                         "paper", ctx["rng"])


def run_episode(seed, p=None, n=None, coordinator=hub_paper_coordinator,
                allow_motion=False, channel=None, pair_solver=None,
                ledger=None, weight_fn=None, max_rounds=14, dropout=None,
                collector=None):
    """One aggregation episode. Returns totals, ledger, and the round log.

    coordinator(ctx) -> [(sender, receiver|None)]; may spend control energy
                        through ctx['ledger'] and read only what its scheme
                        is allowed to see.
    channel          -> object with realize_pair(r, ai, aj, rnd) returning
                        {'T_comm_real', 'E_comm_real', 'violated'} — applied at
                        execution time (None = the paper's deterministic radio).
    pair_solver(ai, aj, rho_pair, cen, p) -> solve_pair result (None=infeasible);
                        default is the frozen solver (planner overrides in WP-2).
    weight_fn(r, ctx_pair) -> matching weight; default = paper Eq. (36).
    dropout          -> {'round':k, 'q':fraction, 'mode':'centralized'|
                        'decentralized', 'stall':2}. At round k, remove ceil(q*|active|)
                        agents. Centralized (paper hub, no dropout handling) stalls
                        `stall` rounds; both share a hard budget of ceil(log2 N)+2
                        rounds — exceeding it is a mission failure (§2.5).
    """
    p = p or A.STRESS
    ledger = ledger if ledger is not None else EnergyLedger()
    rng = np.random.default_rng(seed)
    geo = np.random.default_rng(seed + 10_000)
    agents = A.make_agents(rng, p, n=n)
    if channel is not None:
        channel.new_scenario(seed)

    active = list(range(len(agents)))
    n0 = len(active)
    drop_rng = np.random.default_rng(seed + 777)
    round_budget = int(np.ceil(np.log2(max(n0, 2)))) + 2 if dropout else max_rounds
    E_total, agentE = 0.0, {}
    violations, executed, stalls = 0, 0, 0
    log = []
    k = 0
    while len(active) > 1:
        k += 1
        if dropout and k == dropout["round"]:
            n_drop = int(np.ceil(dropout["q"] * len(active)))
            if n_drop > 0:
                dropped = list(drop_rng.choice(active, size=min(n_drop, len(active) - 1),
                                               replace=False))
                active = [a for a in active if a not in dropped]
                if dropout.get("mode") == "centralized":
                    # the paper's synchronous hub waits on every active agent's
                    # upload; a permanently-missing one stalls each retry, so
                    # unless it is granted a recovery timeout it exhausts the
                    # round budget and the mission fails (§2.5, L10).
                    stalls = int(dropout.get("stall", round_budget + 5))
        if stalls > 0:                       # paper hub waits on missing uploads
            stalls -= 1
            k_extra = k
            log.append({"round": k, "active": list(active), "pairs": [], "stall": True})
            if k > round_budget:
                break
            continue
        if k > round_budget:
            break
        cen = np.mean([agents[i]["pos"] for i in active], axis=0)
        rho = A.pair_overlaps(agents, active, geo)

        E, w, sol = {}, {}, {}
        for i in active:
            for j in active:
                if i == j:
                    continue
                rp = rho[(min(i, j), max(i, j))]
                if pair_solver is None:
                    r = A.solve_pair(agents[i], agents[j], p, rho_pair=rp,
                                     centroid=cen, allow_motion=allow_motion)
                else:
                    r = pair_solver(agents[i], agents[j], rp, cen, p)
                if r is None:
                    E[(i, j)] = w[(i, j)] = INF
                    continue
                sol[(i, j)] = r
                E[(i, j)] = r["E"]
                if weight_fn is None:
                    w[(i, j)] = r["E"] + A.potential_phi(r["pe_j"], cen, r["L_next"], p)
                else:
                    w[(i, j)] = weight_fn(r, {"i": i, "j": j, "cen": cen, "p": p,
                                              "agents": agents, "active": active})

        ctx = {"active": active, "agents": agents, "E": E, "w": w, "sol": sol,
               "rng": rng, "cen": cen, "round": k, "ledger": ledger, "p": p,
               "rho": rho, "channel": channel}
        pairs = coordinator(ctx)

        progressed = False
        new_active = []
        for (i, j) in pairs:
            if j is None or (i, j) not in sol:
                new_active.append(i)
                if j is not None:
                    new_active.append(j)
                continue
            r = sol[(i, j)]
            e_pair = r["E"]
            if channel is not None:
                real = channel.realize_pair(r, agents[i], agents[j], k, p)
                violations += int(real["violated"])
                e_pair = e_pair - real["E_comm_planned"] + real["E_comm_real"]
                if collector is not None:
                    collector.append({"s_time": real["T_comm_real"] - r["t1"],
                                      "s_db": -real.get("factor_db", 0.0)})
            executed += 1
            E_total += e_pair
            progressed = True
            agentE[i] = agentE.get(i, 0.0) + r.get("E_i", 0.6 * e_pair)
            agentE[j] = agentE.get(j, 0.0) + r.get("E_j", 0.4 * e_pair)
            em, ec, ecm = A.energy_split(r, agents[i], agents[j], p)
            if channel is not None:
                ecm = real["E_comm_real"]
                em = max(r["E"] - ec - real["E_comm_planned"], 0.0)
            ledger.add(em, "mobility", agent=i, rnd=k)
            ledger.add(ec, "compute", agent=i, rnd=k)
            ledger.add(max(ecm, 0.0), "comm_payload", agent=i, rnd=k)
            _apply(agents[i], agents[j], r, rho[(min(i, j), max(i, j))])
            new_active.append(j)

        log.append({"round": k, "active": list(active), "pairs": pairs})
        if not progressed:
            return {"E": INF, "rounds": k, "feasible": False, "ledger": ledger,
                    "violations": violations, "executed": executed, "log": log}
        active = new_active
        if k > max_rounds:
            break

    done = (len(active) == 1)
    return {"E": E_total if done else INF, "rounds": k, "feasible": done,
            "root": active[0] if done else None,
            "Emax": max(agentE.values()) if agentE else 0.0,
            "agentE": agentE, "ledger": ledger, "violations": violations,
            "executed": executed, "n0": n0, "survivors": len(active), "log": log}


def _apply(ai, aj, r, rho_pair):
    """Identical state update to the frozen wan.topology._apply."""
    aj["rho_pred"] = rho_pair
    seen = {(round(c[0], 2), round(c[1], 2), round(rd, 2)) for c, rd in aj["disks"]}
    for c, rd in ai["disks"]:
        key = (round(c[0], 2), round(c[1], 2), round(rd, 2))
        if key not in seen:
            aj["disks"].append((c, rd))
            seen.add(key)
    aj["L"] = r["L_next"]
    aj["pos"] = np.asarray(r["pe_j"], float)
