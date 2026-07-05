"""Per-agent radio-map predictors (playbook §3.2).

Predictor A : GaussianProcessRegressor (Matern nu=1.5 + WhiteKernel) on
              (mid_x, mid_y, log10 d) -> realized gain dB. Learns the whole map,
              distance law included.
Predictor B : twin + residual — the analytic 10log10(beta0 d^-delta) term is
              known, so the GP only has to learn the smooth shadowing residual
              chi(mid). Expected (and measured) to be more sample-efficient.

Both return mu(x), sigma(x) for any candidate endpoint pair.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from .. import adapters as A


def _feat(pi, pj):
    mid = 0.5 * (np.asarray(pi) + np.asarray(pj))
    d = max(np.linalg.norm(np.asarray(pi) - np.asarray(pj)), 1.0)
    return mid[0], mid[1], np.log10(d)


def _gp():
    k = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=40.0, nu=1.5,
        length_scale_bounds=(5.0, 300.0)) + WhiteKernel(1.0, (1e-3, 1e2))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, alpha=1e-6,
                                    n_restarts_optimizer=0)


class GPRadioMapA:
    kind = "A_full"

    def __init__(self, p):
        self.p = p
        self.gp = _gp()

    def fit(self, links, gains_db):
        X = np.array([list(_feat(a, b)) for a, b in links])
        self.gp.fit(X, np.asarray(gains_db))
        return self

    def predict(self, pi, pj):
        X = np.array([list(_feat(pi, pj))])
        mu, sd = self.gp.predict(X, return_std=True)
        return float(mu[0]), float(sd[0])


class GPRadioMapB:
    kind = "B_twin_residual"

    def __init__(self, p):
        self.p = p
        self.gp = _gp()

    def _analytic_db(self, pi, pj):
        return 10 * np.log10(A.chan_gain(pi, pj, self.p))

    def fit(self, links, gains_db):
        X = np.array([[0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])] for a, b in links])
        resid = np.array([g - self._analytic_db(a, b)
                          for (a, b), g in zip(links, gains_db)])
        self.gp.fit(X, resid)
        return self

    def predict(self, pi, pj):
        mid = 0.5 * (np.asarray(pi) + np.asarray(pj))
        mu, sd = self.gp.predict(mid[None, :], return_std=True)
        return float(self._analytic_db(pi, pj) + mu[0]), float(sd[0])


def sample_links(channel, p, rng, n, max_range=250.0):
    """Random link geometries and their realized (fade-perturbed) gain in dB."""
    links, gains = [], []
    arena = p.get("area", 500.0)
    for _ in range(n):
        pi = rng.uniform(0, arena, 2)
        ang = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(20, max_range)
        pj = np.clip(pi + d * np.array([np.cos(ang), np.sin(ang)]), 0, arena)
        true_db = channel.true_gain_db(pi, pj, p)
        fade_db = 10 * np.log10(channel.gain_factor(pi, pj, link_key=("s", _),
                                                    advance=True)
                                / 10 ** (channel.shadow_db(0.5 * (pi + pj)) / 10))
        links.append((pi, pj)); gains.append(true_db + fade_db)
    return links, np.array(gains)


def learning_curve(channel, p, seeds, n_train_list, n_test=200):
    """RMSE (dB) vs #samples for predictors A and B on held-out links."""
    out = {"A": {}, "B": {}}
    for n in n_train_list:
        errA, errB = [], []
        for seed in seeds:
            channel.new_scenario(seed)
            rng = np.random.default_rng(seed + 55)
            links, gains = sample_links(channel, p, rng, n)
            tl, tg = sample_links(channel, p, rng, n_test)
            truth = np.array([channel.true_gain_db(a, b, p) for a, b in tl])
            for name, Cls, err in (("A", GPRadioMapA, errA), ("B", GPRadioMapB, errB)):
                m = Cls(p).fit(links, gains)
                pred = np.array([m.predict(a, b)[0] for a, b in tl])
                err.append(float(np.sqrt(np.mean((pred - truth) ** 2))))
        out["A"][n] = float(np.mean(errA)); out["B"][n] = float(np.mean(errB))
    return out
