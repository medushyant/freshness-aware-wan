"""Rung A1 — distributed greedy matching (playbook §2.2).

Locally-heaviest-edge (Preis/Hoepman style): each agent proposes to its
mutually-best feasible neighbor within radio range; a mutual-best edge commits
both endpoints. On a benefit graph b_ij = w_max - c_ij this is the classical
greedy matching whose weight is >= 1/2 of the optimal (Preis 1999) — stated in
the report as the guarantee. Every PROPOSE / COMMIT / REJECT is a counted,
costed control message.
"""

import numpy as np

from .control import ControlChannel
from .common import add_leftovers, order_pair, symmetric_weights

INF = float("inf")


def make_greedy_coordinator(cfg=None):
    cfg = cfg or {}
    r_comm = cfg.get("R_comm_m", 250.0)
    ctrl_holder = {}

    def coordinator(ctx):
        ctrl = ControlChannel(ctx["p"], ledger=ctx["ledger"], cfg=cfg)
        ctrl_holder["c"] = ctrl
        agents, active, w = ctx["agents"], ctx["active"], ctx["w"]
        c = symmetric_weights(active, w, agents=agents, r_comm=r_comm)
        pairs = _greedy_match(active, c, agents, ctrl, ctx["round"], r_comm)
        return [order_pair(i, j, w) if j is not None else (i, None) for (i, j) in pairs]

    coordinator.ctrl_holder = ctrl_holder
    return coordinator


def _greedy_match(active, c, agents, ctrl, rnd, r_comm):
    """Iterated mutual-best-edge matching with per-message accounting."""
    unmatched = set(active)
    pairs = []
    # candidate neighbor lists (cheapest first) per agent
    nbr = {i: [] for i in active}
    for (i, j), cij in c.items():
        nbr[i].append((cij, j))
        nbr[j].append((cij, i))
    for i in nbr:
        nbr[i].sort()

    while True:
        # each unmatched agent proposes to its current best unmatched neighbor
        proposals = {}
        for i in list(unmatched):
            cand = next((j for _, j in nbr[i] if j in unmatched), None)
            if cand is None:
                continue
            d = float(np.linalg.norm(agents[i]["pos"] - agents[cand]["pos"]))
            ctrl.send(ctrl.B_msg, min(d, r_comm), rnd=rnd, agent=i, note="PROPOSE")
            proposals[i] = cand
        if not proposals:
            break
        committed = False
        for i, j in list(proposals.items()):
            if proposals.get(j) == i and i in unmatched and j in unmatched:
                d = float(np.linalg.norm(agents[i]["pos"] - agents[j]["pos"]))
                ctrl.send(ctrl.B_msg, min(d, r_comm), rnd=rnd, agent=i, note="COMMIT")
                ctrl.send(ctrl.B_msg, min(d, r_comm), rnd=rnd, agent=j, note="COMMIT")
                pairs.append((i, j))
                unmatched.discard(i); unmatched.discard(j)
                committed = True
        if not committed:
            # no mutual-best this cycle: accept the single globally-cheapest edge
            live = [(cij, i, j) for (i, j), cij in c.items()
                    if i in unmatched and j in unmatched]
            if not live:
                break
            _, i, j = min(live)
            ctrl.send(ctrl.B_msg, min(float(np.linalg.norm(
                agents[i]["pos"] - agents[j]["pos"])), r_comm), rnd=rnd, agent=i,
                note="COMMIT")
            pairs.append((i, j))
            unmatched.discard(i); unmatched.discard(j)
    return add_leftovers(pairs, active)
