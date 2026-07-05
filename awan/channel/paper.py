"""Paper channel wrapper: h = beta0 * d^-delta, deterministic (Eq. 8).

Realizing a pair under this channel is a no-op (factor 1), so a mission run
with PaperChannel reproduces the frozen numbers exactly — the C0 parity check.
"""

from .. import adapters as A


class PaperChannel:
    name = "paper"
    slack = 0.0

    def new_scenario(self, seed):
        pass

    def gain_factor(self, pi, pj, link_key=None, advance=False):
        return 1.0

    def gain(self, pi, pj, p, link_key=None, advance=False):
        return A.chan_gain(pi, pj, p)

    def realize_pair(self, r, ai, aj, rnd, p):
        e = r["ptx"] * r["t1"]
        return {"E_comm_planned": e, "E_comm_real": e,
                "T_comm_real": r["t1"], "violated": False, "factor_db": 0.0}
