"""Tier-1 realistic stochastic channel (playbook §3.1), pure numpy.

    h(tx,rx,t) = beta0 * d^-delta * 10^(chi(x_mid)/10) * |g(t)|^2

  chi : zero-mean Gaussian random field over the arena, exponential
        (Gudmundson) covariance C(u,v)=sigma_sh^2 exp(-||u-v||/d_c). Sampled
        once per scenario on a coarse grid by Cholesky, bilinear-interpolated;
        link value taken at the pair midpoint (the standard simplification).
  g   : Rician (K dB) small-scale fade, block-constant per transmission, AR(1)
        correlated (rho_t) across an agent's successive transmissions.

`model: paper` (factor 1) must reproduce Phase-1 exactly — the C0 sanity.
"""

import numpy as np

from .. import adapters as A


class Tier1Channel:
    name = "tier1"

    def __init__(self, p, cfg=None):
        cfg = cfg or {}
        self.p = p
        self.arena = p.get("area", 500.0)
        self.sigma_sh = cfg.get("sigma_sh_db", 8.0)
        self.d_c = cfg.get("d_c_m", 25.0)
        self.K_db = cfg.get("rician_K_db", 6.0)
        self.rho_t = cfg.get("ar1_rho", 0.7)
        self.grid_m = cfg.get("grid_m", 12.5)
        self.slack = 0.0          # conformal time margin granted at execution
        self._field = None
        self._fade = {}

    # --- shadowing GRF -------------------------------------------------
    def new_scenario(self, seed):
        rng = np.random.default_rng(seed + 20_000)
        g = int(np.ceil(self.arena / self.grid_m)) + 1
        xs = np.linspace(0, self.arena, g)
        gx, gy = np.meshgrid(xs, xs)
        pts = np.c_[gx.ravel(), gy.ravel()]
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        cov = self.sigma_sh ** 2 * np.exp(-d / self.d_c)
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(len(pts)))
        self._field = (xs, (L @ rng.standard_normal(len(pts))).reshape(g, g))
        self._fade = {}
        self._rng = rng

    def shadow_db(self, pos):
        xs, F = self._field
        step = xs[1] - xs[0]
        x = np.clip(pos[0], 0, self.arena - 1e-6)
        y = np.clip(pos[1], 0, self.arena - 1e-6)
        i = int(x // step); j = int(y // step)
        tx = (x - xs[i]) / step; ty = (y - xs[j]) / step
        return float((1 - tx) * (1 - ty) * F[j, i] + tx * (1 - ty) * F[j, i + 1]
                     + (1 - tx) * ty * F[j + 1, i] + tx * ty * F[j + 1, i + 1])

    # --- small-scale Rician + AR(1) ------------------------------------
    def _fade_power(self, link_key, advance):
        K = 10 ** (self.K_db / 10)
        los = np.sqrt(K / (K + 1))
        sig = np.sqrt(1.0 / (2 * (K + 1)))
        st = self._fade.get(link_key)
        if st is None or advance:
            new = sig * (self._rng.standard_normal() + 1j * self._rng.standard_normal())
            scat = new if st is None else self.rho_t * st + np.sqrt(1 - self.rho_t ** 2) * new
            self._fade[link_key] = scat
            st = scat
        return float(abs(los + st) ** 2)

    def gain_factor(self, pi, pj, link_key=None, advance=False):
        mid = 0.5 * (np.asarray(pi) + np.asarray(pj))
        sh = 10 ** (self.shadow_db(mid) / 10)
        fade = self._fade_power(link_key or (round(mid[0]), round(mid[1])), advance)
        return sh * fade

    def gain(self, pi, pj, p, link_key=None, advance=False):
        return A.chan_gain(pi, pj, p) * self.gain_factor(pi, pj, link_key, advance)

    def true_gain_db(self, pi, pj, p):
        """Deterministic large-scale + shadowing (no fast fade) in dB — the
        quantity the radio-map predictor learns."""
        h0 = A.chan_gain(pi, pj, p)
        return 10 * np.log10(h0) + self.shadow_db(0.5 * (np.asarray(pi) + np.asarray(pj)))

    # --- execution-time realization ------------------------------------
    def realize_pair(self, r, ai, aj, rnd, p):
        # the plan's own comm spend (exact: comm_energy == ptx*t1 when uncapped)
        e_plan = r["ptx"] * r["t1"]
        factor = self.gain_factor(r["pe_i"], r["pe_j"],
                                  link_key=(ai["id"], aj["id"]), advance=True)
        h_real = A.chan_gain(r["pe_i"], r["pe_j"], p) * factor
        real_rate = A.rate(r["ptx"], h_real, p)
        t_real = r["Lout"] / real_rate if real_rate > 0 else np.inf
        e_real = r["ptx"] * t_real if np.isfinite(t_real) else r["ptx"] * r["t1"] * 50
        violated = t_real > r["t1"] * (1 + 1e-6) + self.slack
        return {"E_comm_planned": e_plan, "E_comm_real": float(e_real),
                "T_comm_real": float(t_real), "violated": bool(violated),
                "factor_db": float(10 * np.log10(factor))}
