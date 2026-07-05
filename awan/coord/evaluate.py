"""Mission-level evaluation helpers for the WP-1 coordinators."""

import numpy as np

from .. import adapters as A
from ..simcore import run_episode
from .auction import make_auction_coordinator
from .bids_learned import make_learned_weight_fn
from .greedy_dist import make_greedy_coordinator
from .hub import make_hub_coordinator
from .negotiate_llm import make_negotiation_coordinator

INF = float("inf")


def make_scheme(scheme, cfg, vm=None, engine="mock", llm=None, stats=None):
    """Return (coordinator, weight_fn) for a named scheme."""
    if scheme == "hub":
        return make_hub_coordinator(cfg), None
    if scheme == "greedy":
        return make_greedy_coordinator(cfg), None
    if scheme == "auction":
        return make_auction_coordinator(cfg), None
    if scheme == "rungB":
        return make_auction_coordinator(cfg), make_learned_weight_fn(vm)
    if scheme in ("rungC", "negotiate"):
        return make_negotiation_coordinator(cfg, engine=engine, llm=llm,
                                            stats=stats), None
    raise ValueError(scheme)


def run_scheme(seed, n, scheme, cfg, vm=None, dropout=None, engine="mock",
               llm=None):
    stats = {}
    coord, wf = make_scheme(scheme, cfg, vm=vm, engine=engine, llm=llm, stats=stats)
    out = run_episode(seed, p=A.STRESS, n=n, coordinator=coord, weight_fn=wf,
                      dropout=dropout)
    kinds = out["ledger"].by_kind()
    valid, n_pairs = _validate_pairs(out["log"], n if dropout is None else None)
    return {"E": out["E"], "feasible": out["feasible"], "rounds": out["rounds"],
            "control_J": kinds["comm_control"], "neg_J": kinds["negotiation_llm"],
            "valid": valid, "survivors": out.get("survivors"),
            "stats": stats, "log": out["log"]}


def per_round_matcher_quality(seeds, n_list, cfg):
    """A1.2/A1.3: on all-feasible rounds, compare the distributed greedy and
    auction to exact Blossom on the IDENTICAL cost matrix. Returns greedy
    >=1/2-benefit rate and auction ==Blossom (within eps) rate."""
    import networkx as nx
    from .auction import auction_match
    from .greedy_dist import _greedy_match
    from .common import symmetric_weights
    from .control import ControlChannel
    from ..ledger import EnergyLedger

    def blossom_on_c(active, c):
        G = nx.Graph(); G.add_nodes_from(active)
        if len(active) % 2:
            G.add_node(-1)
        big = 1 + max(c.values())
        for (i, j), cij in c.items():
            G.add_edge(i, j, weight=big * 100 - cij)
        if -1 in G:
            for i in active:
                G.add_edge(-1, i, weight=big * 100)
        return [(min(u, v), max(u, v))
                for u, v in nx.max_weight_matching(G, maxcardinality=True)
                if -1 not in (u, v)]

    def cost(pairs, c):
        return sum(c[(min(i, j), max(i, j))] for (i, j) in pairs if j is not None)

    half_ok = half_tot = eq_ok = eq_tot = within_ok = 0
    eps_bound_pct = 2.0
    gaps = []
    for n in n_list:
        for seed in seeds:
            agents, rng, geo = A.scenario_gen(seed, n=n)
            active = list(range(n))
            cen = np.mean([agents[i]["pos"] for i in active], axis=0)
            rho = A.pair_overlaps(agents, active, geo)
            w = {}
            feasible = True
            for i in active:
                for j in active:
                    if i == j:
                        continue
                    r = A.solve_pair(agents[i], agents[j], A.STRESS,
                                     rho_pair=rho[(min(i, j), max(i, j))],
                                     centroid=cen, allow_motion=False)
                    w[(i, j)] = (r["E"] + A.potential_phi(r["pe_j"], cen, r["L_next"], A.STRESS)) if r else INF
            for a_ in range(n):
                for b_ in range(a_ + 1, n):
                    if not np.isfinite(min(w.get((active[a_], active[b_]), INF),
                                          w.get((active[b_], active[a_]), INF))):
                        feasible = False
            if not feasible:
                continue
            c = symmetric_weights(active, w, soft=False)
            if len(c) < n // 2:
                continue
            Cmax = max(c.values()) + 1.0
            ben = {k: Cmax - v for k, v in c.items()}
            bl = blossom_on_c(active, c)
            au = [(i, j) for (i, j) in auction_match(active, dict(c)) if j is not None]
            gr = [(i, j) for (i, j) in _greedy_match(active, dict(c), agents,
                  ControlChannel(A.STRESS, ledger=EnergyLedger()), 1, 1e9)
                  if j is not None]
            cb, ca = cost(bl, c), cost(au, c)
            bb, bg = cost(bl, ben), cost(gr, ben)
            eq_tot += 1
            if abs(ca - cb) <= 1e-6 * max(cb, 1) + 1e-9:
                eq_ok += 1
            if cb > 0:
                g = (ca - cb) / cb * 100
                gaps.append(g)
                if g <= eps_bound_pct:
                    within_ok += 1
            half_tot += 1
            if bg >= 0.5 * bb - 1e-9:
                half_ok += 1
    return {"half_ok": half_ok, "half_tot": half_tot,
            "eq_ok": eq_ok, "eq_tot": eq_tot,
            "within_ok": within_ok, "eps_bound_pct": eps_bound_pct,
            "auction_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
            "auction_gap_max": float(np.max(gaps)) if gaps else 0.0}


def _validate_pairs(log, n):
    """Per-round matching validity (playbook §12): each agent appears in <=1
    link, no self-pairs, and every paired agent is in that round's active set
    (retired agents are excluded by construction — simcore removes them). The
    across-round retirement mechanics are simcore's job, not the matcher's."""
    ok = True
    total_pairs = 0
    for rec in log:
        if rec.get("stall"):
            continue
        active = set(rec["active"])
        seen = set()
        for (i, j) in rec["pairs"]:
            if i == j or i not in active or (j is not None and j not in active):
                ok = False
            if i in seen or (j is not None and j in seen):
                ok = False
            seen.add(i)
            if j is not None:
                seen.add(j)
                total_pairs += 1
    return ok, total_pairs
