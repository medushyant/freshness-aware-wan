"""Learned inner-solver surrogate for N-scaling (playbook §5.3).

Ridge regression W_hat(features) -> pair energy, trained on solved pairs from
real scenarios. At N >= 20 the O(N^2) solve_pair sweep per round dominates
wall time; the surrogate replaces it for matcher weights so the FIRST runtime
curve for this framework at N >= 100 can be measured. Its energy penalty is
quantified at N=10 against the true solver (target < 5%).
"""

import time

import numpy as np

from . import adapters as A


def pair_features(ai, aj, cen, p):
    d = np.linalg.norm(ai["pos"] - aj["pos"])
    dc = np.linalg.norm(aj["pos"] - cen)
    return np.array([
        d, np.log1p(d), ai["L"] / 1e6, aj["L"] / 1e6,
        (ai["L"] + aj["L"]) / 1e6, ai["rho_pred"], aj["rho_pred"],
        dc, np.log1p(dc), d * (ai["L"] / 1e6),
    ])


class PairSurrogate:
    """Two ridge heads: E_hat (pair energy) and Lnext_hat (fused payload, for
    the potential-field lookahead — with motion off, pe_j == pos_j exactly, so
    the paper's matching weight E + zeta*||pos_j - cen||^delta * L_next is
    fully reconstructible from the two heads)."""

    def __init__(self, l2=1e-3):
        self.l2 = l2
        self.w = self.w2 = self.mu = self.sd = None

    def _solve(self, Z, y):
        return np.linalg.solve(Z.T @ Z + self.l2 * np.eye(Z.shape[1]), Z.T @ y)

    def fit(self, X, y, feas, y_lnext=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        ok = np.asarray(feas, bool)
        self.gate_rate = ok.mean()
        Xo, yo = X[ok], y[ok]
        self.mu, self.sd = Xo.mean(0), Xo.std(0) + 1e-9
        Z = np.c_[np.ones(len(Xo)), (Xo - self.mu) / self.sd]
        self.w = self._solve(Z, yo)
        if y_lnext is not None:
            self.w2 = self._solve(Z, np.asarray(y_lnext, float)[ok])
        pred = Z @ self.w
        return float(np.mean(np.abs(pred - yo) / np.maximum(yo, 1e-9))) * 100

    def energy(self, ai, aj, cen, p):
        z = np.r_[1.0, (pair_features(ai, aj, cen, p) - self.mu) / self.sd]
        return float(max(z @ self.w, 1e-6))

    def weight(self, ai, aj, cen, p):
        """E_hat + the paper's Eq. (36) potential, using Lnext_hat."""
        f = pair_features(ai, aj, cen, p)
        z = np.r_[1.0, (f - self.mu) / self.sd]
        e = float(max(z @ self.w, 1e-6))
        if self.w2 is None:
            return e
        lnext = float(max(z @ self.w2, 0.0))
        phi = p["zeta"] * np.linalg.norm(aj["pos"] - cen) ** p["delta"] * lnext
        return e + phi

    __call__ = weight


def collect_pairs(seeds, p, n=10, per_scene_rounds=1):
    X, y, y2, feas = [], [], [], []
    for seed in seeds:
        agents, rng, geo = A.scenario_gen(seed, n=n, p=p)
        active = list(range(n))
        cen = np.mean([agents[i]["pos"] for i in active], axis=0)
        rho = A.pair_overlaps(agents, active, geo)
        for i in active:
            for j in active:
                if i == j:
                    continue
                r = A.solve_pair(agents[i], agents[j], p,
                                 rho_pair=rho[(min(i, j), max(i, j))],
                                 centroid=cen, allow_motion=False)
                X.append(pair_features(agents[i], agents[j], cen, p))
                y.append(r["E"] if r else 0.0)
                y2.append(r["L_next"] if r else 0.0)
                feas.append(r is not None)
    return np.array(X), np.array(y), np.array(y2), np.array(feas)


def matching_runtime(n, seed, p, surrogate, scheme):
    """Wall-clock of ONE round's coordination at size n using surrogate
    weights (isolates matcher runtime, no inner solves)."""
    import networkx as nx
    from .coord.auction import auction_match
    from .coord.greedy_dist import _greedy_match
    from .coord.control import ControlChannel
    from .ledger import EnergyLedger

    agents, rng, geo = A.scenario_gen(seed, n=n, p=p)
    active = list(range(n))
    cen = np.mean([agents[i]["pos"] for i in active], axis=0)
    c = {}
    for a_ in range(n):
        for b_ in range(a_ + 1, n):
            i, j = active[a_], active[b_]
            c[(i, j)] = surrogate(agents[i], agents[j], cen, p)

    t0 = time.perf_counter()
    if scheme == "blossom":
        G = nx.Graph(); G.add_nodes_from(active)
        big = 1 + max(c.values())
        for (i, j), cij in c.items():
            G.add_edge(i, j, weight=big * 100 - cij)
        nx.max_weight_matching(G, maxcardinality=True)
    elif scheme == "auction":
        auction_match(active, dict(c))
    elif scheme == "greedy":
        _greedy_match(active, dict(c), agents,
                      ControlChannel(p, ledger=EnergyLedger()), 1, 1e9)
    return time.perf_counter() - t0
