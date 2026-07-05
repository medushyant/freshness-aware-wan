"""Rung B — the learned bid, decentralized (playbook §2.3).

The frozen Phase-1 ridge cost-to-go becomes each agent's LOCAL bid function.
The matching stays decentralized (auction/greedy); only the edge weight changes
from the paper's E + zeta*Phi to the learned E + V_hat(after-state). Global
quantities the value features need (the swarm centroid) are replaced by a
gossip estimate obtained from `gossip_rounds` neighbour-averaging broadcasts,
whose RMSE vs the true centroid is reported.
"""

import numpy as np

from .. import adapters as A


def gossip_centroid(agents, active, r_comm, rounds=3, ctrl=None, rnd=None):
    """Distributed average-consensus estimate of the swarm centroid.

    Each agent initializes with its own position and repeatedly averages with
    in-range neighbours (Metropolis weights). Returns {agent: estimate} and the
    RMSE against the true centroid. Each round is one costed broadcast/agent.
    """
    pos = {i: agents[i]["pos"].astype(float).copy() for i in active}
    est = {i: pos[i].copy() for i in active}
    nbrs = {i: [j for j in active if j != i and
                np.linalg.norm(pos[i] - pos[j]) <= r_comm] for i in active}
    for _ in range(rounds):
        new = {}
        for i in active:
            deg_i = len(nbrs[i])
            acc = est[i].copy()
            wsum = 1.0
            for j in nbrs[i]:
                wij = 1.0 / (1 + max(deg_i, len(nbrs[j])))
                acc = acc + wij * (est[j] - est[i])
                wsum += 0.0
            new[i] = acc
            if ctrl is not None:
                ctrl.broadcast(ctrl.B_msg, radius=r_comm, rnd=rnd, agent=i,
                               note="gossip")
        est = new
    true_c = np.mean([pos[i] for i in active], axis=0)
    rmse = float(np.sqrt(np.mean([np.sum((est[i] - true_c) ** 2) for i in active])))
    return est, rmse


def make_learned_weight_fn(vmodel, centroid_est=None):
    """Edge weight w_ij = E_ij + V_hat(after-state if i->j fires). Uses the
    frozen Phase-1 value model and feature map."""

    def weight_fn(r, cp):
        agents, active, i, j = cp["agents"], cp["active"], cp["i"], cp["j"]
        after = _hypothetical(agents, active, i, j, r, centroid_est)
        return r["E"] + vmodel(A.state_features(after, cp["p"]))

    return weight_fn


def _hypothetical(agents, active, i, j, r, centroid_est=None):
    out = []
    for a in active:
        if a == i:
            continue
        if a == j:
            out.append({"pos": np.asarray(r["pe_j"]), "L": r["L_next"]})
        else:
            out.append({"pos": agents[a]["pos"], "L": agents[a]["L"]})
    return out
