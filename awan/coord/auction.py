"""Rung A2 — Bertsekas auction with epsilon-scaling (playbook §2.2).

We reduce the per-round max-weight MATCHING to a symmetric assignment on the
benefit graph b_ij = w_max - c_ij (c = symmetrized pair cost). A distributed
forward auction (Jacobi/parallel: all unassigned bidders bid at once, objects
keep the highest bid, prices broadcast locally) solves it. Epsilon-scaling
(eps <- eps/4 from eps0 = w_max/2) drives the eps-complementary-slackness gap
to within M*eps of the optimal assignment (Bertsekas), i.e. effectively exact
at our scale. Agreeing 2-cycles (i->j and j->i) are peeled off as matched
pairs; leftover agents re-enter, then idle. Every bid and price broadcast is a
counted, costed control message.
"""

import numpy as np

from .control import ControlChannel
from .common import add_leftovers, order_pair, symmetric_weights

INF = float("inf")
NEG = -1e18


def make_auction_coordinator(cfg=None):
    cfg = cfg or {}
    r_comm = cfg.get("R_comm_m", 250.0)
    eps0_frac = cfg.get("eps0_frac", 0.5)
    eps_div = cfg.get("eps_div", 4.0)
    ctrl_holder = {}

    def coordinator(ctx):
        ctrl = ControlChannel(ctx["p"], ledger=ctx["ledger"], cfg=cfg)
        ctrl_holder["c"] = ctrl
        agents, active, w = ctx["agents"], ctx["active"], ctx["w"]
        c = symmetric_weights(active, w, agents=agents, r_comm=r_comm)
        pairs = auction_match(active, c, agents, ctrl, ctx["round"], r_comm,
                              eps0_frac, eps_div)
        return [order_pair(i, j, w) if j is not None else (i, None) for (i, j) in pairs]

    coordinator.ctrl_holder = ctrl_holder
    return coordinator


def auction_match(active, c, agents=None, ctrl=None, rnd=None, r_comm=250.0,
                  eps0_frac=0.5, eps_div=4.0):
    """Peel agreeing 2-cycles from repeated symmetric auctions -> matching."""
    if not c:
        return [(i, None) for i in active]
    w_max = max(c.values()) if c else 1.0
    benefit = {}
    for (i, j), cij in c.items():
        b = w_max - cij
        benefit[(i, j)] = b
        benefit[(j, i)] = b

    remaining = list(active)
    pairs = []
    for _ in range(len(active)):
        if len(remaining) < 2:
            break
        assign = _forward_auction(remaining, benefit, w_max, eps0_frac, eps_div,
                                  ctrl, agents, rnd, r_comm)
        got = []
        used = set()
        for i in remaining:
            j = assign.get(i)
            if j is not None and assign.get(j) == i and i < j \
                    and i not in used and j not in used:
                got.append((i, j))
                used.add(i); used.add(j)
        if not got:
            break
        pairs.extend(got)
        remaining = [x for x in remaining if x not in used]
    # completion: force the H-MAP floor(N/2) pairs (Eq. 16b) by matching any
    # agents the auction left over on the cheapest available edges (odd one out
    # idles, matching Phase-1 flexible-schedule semantics).
    if len(remaining) >= 2:
        pool = set(remaining)
        edges = sorted((cij, i, j) for (i, j), cij in c.items()
                       if i in pool and j in pool)
        for cij, i, j in edges:
            if i in pool and j in pool:
                pairs.append((i, j)); pool.discard(i); pool.discard(j)
        remaining = list(pool)
    return add_leftovers(pairs, active)


def _forward_auction(bidders, benefit, w_max, eps0_frac, eps_div,
                     ctrl, agents, rnd, r_comm):
    price = {a: 0.0 for a in bidders}
    eps = max(w_max * eps0_frac, 1e-9)
    eps_min = max(w_max * 1e-4, 1e-9)
    assign, owner = {}, {}
    while eps >= eps_min:
        assign, owner = {}, {}
        unassigned = list(bidders)
        for _ in range(200 * len(bidders)):
            if not unassigned:
                break
            bids = {}
            for i in unassigned:
                vals = sorted(((benefit.get((i, j), NEG) - price[j], j)
                               for j in bidders if j != i), reverse=True)
                vals = [(v, j) for v, j in vals if v > NEG / 2]
                if not vals:
                    continue
                v1, j1 = vals[0]
                v2 = vals[1][0] if len(vals) > 1 else v1 - w_max
                bid = price[j1] + (v1 - v2) + eps
                bids.setdefault(j1, []).append((bid, i))
                if ctrl is not None:
                    _msg(ctrl, agents, i, j1, r_comm, rnd, "BID")
            for j, lst in bids.items():
                lst.sort(reverse=True)
                bp, win = lst[0]
                if j in owner and owner[j] in assign:
                    assign.pop(owner[j], None)
                price[j] = bp
                owner[j] = win
                assign[win] = j
                if ctrl is not None:
                    ctrl.broadcast(ctrl.B_msg, radius=r_comm, rnd=rnd, agent=j,
                                   note="PRICE")
            unassigned = [i for i in bidders if i not in assign]
        eps /= eps_div
    return assign


def _msg(ctrl, agents, i, j, r_comm, rnd, note):
    if agents is None:
        ctrl.send(ctrl.B_msg, r_comm, rnd=rnd, agent=i, note=note)
    else:
        d = float(np.linalg.norm(agents[i]["pos"] - agents[j]["pos"]))
        ctrl.send(ctrl.B_msg, min(d, r_comm), rnd=rnd, agent=i, note=note)
