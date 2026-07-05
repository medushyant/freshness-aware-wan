"""Shared helpers for the decentralized matchers."""

import numpy as np

INF = float("inf")


def symmetric_weights(active, w, agents=None, r_comm=None, soft=True,
                      big_penalty=None):
    """Undirected candidate costs c_ij = min(w_ij, w_ji) over ALL active pairs.

    Infeasible pairs (INF) and, if r_comm is given, pairs whose agents are out
    of radio range are kept but penalized by a big-M so the matcher can still
    reach the H-MAP floor(N/2) cardinality (soft=True). With soft=False such
    pairs are dropped instead (used only for the algorithm-only sanity check).
    """
    raw = {}
    for a_ in range(len(active)):
        for b_ in range(a_ + 1, len(active)):
            i, j = active[a_], active[b_]
            cij = min(w.get((i, j), INF), w.get((j, i), INF))
            out_of_range = (r_comm is not None and agents is not None and
                            np.linalg.norm(agents[i]["pos"] - agents[j]["pos"]) > r_comm)
            raw[(i, j)] = (cij, out_of_range)
    finite = [c for (c, _) in raw.values() if np.isfinite(c)]
    if big_penalty is None:
        big_penalty = 100.0 * (max(finite) if finite else 1.0)
    c = {}
    for (i, j), (cij, oor) in raw.items():
        bad = (not np.isfinite(cij)) or oor
        if bad and not soft:
            continue
        c[(i, j)] = big_penalty if bad else cij
    return c


def order_pair(i, j, w):
    """Sender->receiver orientation: lower directed cost sends."""
    cij, cji = w.get((i, j), INF), w.get((j, i), INF)
    return (i, j) if cij <= cji else (j, i)


def matching_cost(pairs, w):
    tot = 0.0
    for (i, j) in pairs:
        if j is None:
            continue
        tot += min(w.get((i, j), INF), w.get((j, i), INF))
    return tot


def add_leftovers(pairs, active):
    matched = {x for pr in pairs for x in pr if x is not None}
    return pairs + [(i, None) for i in active if i not in matched]
