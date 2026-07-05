"""Ray-traced ground truth from Sionna RT (playbook §3.1 stretch, C5/F2.5).

`assets/sionna_gain_grid.npy` is the RadioMapSolver path-gain grid (dB) of the
built-in Munich scene, exported on Colab (MEASURED geometry: real buildings).
We use it as the SHADOWING texture for our arena: the grid's own distance law
to its transmitter is fitted and removed, leaving the building-induced
residual chi_rt(x); links then see

    h = beta0 * d^-delta * 10^(chi_rt(midpoint)/10) * |g|^2

i.e. the paper's path-loss law with ray-traced (not GRF) spatial correlation,
plus the same Rician small-scale fades as tier-1. Cells the ray tracer marked
unreachable (-300 dB, building interiors) are clamped to the 1st percentile of
reachable residuals — deep but finite shadow.
"""

import numpy as np

from .. import ROOT
from .tier1 import Tier1Channel

GRID = ROOT / "assets" / "sionna_gain_grid.npy"


class SionnaMapChannel(Tier1Channel):
    name = "sionna_map"

    def __init__(self, p, cfg=None):
        super().__init__(p, cfg)
        raw = np.load(GRID).astype(float)
        self._residual = self._detrend(raw)

    def _detrend(self, g):
        """Only ~2% of the exported grid received rays (single low TX, 1 m
        cells): the rest is building interior / out of coverage, not physical
        link space. We (a) fit+remove the distance law on covered cells,
        (b) reconstruct a full field by nearest-covered-cell lookup — links
        inherit the residual of the nearest street the tracer actually lit —
        and (c) center the window sampling on the covered urban core. The
        resulting texture has the MEASURED ray-traced statistics (sd ~6.6 dB,
        heavy left tail) with real-geometry spatial structure."""
        from scipy import ndimage
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w]
        reach = g > -250.0
        ty, tx = np.unravel_index(np.argmax(np.where(reach, g, -np.inf)), g.shape)
        d = np.sqrt((yy - ty) ** 2 + (xx - tx) ** 2) + 1.0
        A = np.c_[np.ones(reach.sum()), np.log10(d[reach])]
        coef, *_ = np.linalg.lstsq(A, g[reach], rcond=None)
        resid = g - (coef[0] + coef[1] * np.log10(d))
        resid[reach] -= resid[reach].mean()
        _, (iy, ix) = ndimage.distance_transform_edt(~reach, return_indices=True)
        filled = resid[iy, ix]
        self._core = (ty, tx)
        return np.clip(filled, -35.0, 35.0)

    def new_scenario(self, seed):
        rng = np.random.default_rng(seed + 20_000)
        self._rng = rng
        self._fade = {}
        h, w = self._residual.shape
        ty, tx = self._core
        # windows stay within the covered urban core around the TX
        self._oy = int(np.clip(ty + rng.integers(-250, 50), 0, max(h - 500, 0)))
        self._ox = int(np.clip(tx + rng.integers(-250, 50), 0, max(w - 500, 0)))

    def shadow_db(self, pos):
        h, w = self._residual.shape
        y = int(np.clip(self._oy + pos[1] / self.arena * min(h - 1, 499), 0, h - 1))
        x = int(np.clip(self._ox + pos[0] / self.arena * min(w - 1, 499), 0, w - 1))
        return float(self._residual[y, x])
