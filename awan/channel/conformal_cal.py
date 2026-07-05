"""Split-conformal deadline certification in the dB domain (playbook §3.3).

Design decision (see docs/DECISIONS.md): the additive time-margin variant is
unusable here — deep fades collapse the rate multiplicatively and its
calibrated margin exceeds the whole deadline. The channel-margin construction
certifies the same event cleanly:

  nonconformity of an executed link:  s = fade depth below the planning
  belief, in dB:  s = belief_dB - realized_dB  (= -factor_dB for the
  deterministic belief).

  q_db = ceil((n+1)(1-alpha))/n empirical quantile of s on calibration
  scenarios. Deployment plans EVERY link as if the channel were q_db worse
  (beta0' = beta0 * 10^(-q_db/10)); the link then violates its deadline iff
  the realized fade is deeper than q_db, which by split conformal happens with
  probability <= alpha on exchangeable test scenarios.

Cost of certainty: planning against a worse channel costs transmit energy —
measured and reported (the honest price of the certificate).
"""

import numpy as np

from .. import adapters as A
from ..simcore import run_episode


def make_planner(p, margin_db=0.0, extra_factor_db_fn=None):
    """pair_solver that plans with beta0 discounted by margin_db (and any
    additional predicted per-link factor)."""

    def pair_solver(ai, aj, rho_pair, cen, pp):
        p2 = dict(pp)
        fdb = -margin_db
        if extra_factor_db_fn is not None:
            fdb += extra_factor_db_fn(ai["pos"], aj["pos"], pp)
        if fdb != 0.0:
            p2["beta0"] = pp["beta0"] * 10 ** (fdb / 10)
        return A.solve_pair(ai, aj, p2, rho_pair=rho_pair, centroid=cen,
                            allow_motion=False)

    return pair_solver


def collect_scores(channel, p, seeds):
    """Fade-depth scores s_db from calibration missions (planned like the
    paper: deterministic belief, no margin)."""
    scores = []
    for seed in seeds:
        col = []
        run_episode(seed, p=p, channel=channel, collector=col)
        scores += [c["s_db"] for c in col]
    return np.array(scores)


def qhat(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k - 1, n - 1)])


def certified_run(channel, p, seeds, margin_db):
    """Held-out missions planned with the certified margin. Returns empirical
    per-link deadline coverage, the energy premium, and completion stats."""
    met = tot = 0
    E_cert, E_base, feas = [], [], 0
    solver = make_planner(p, margin_db=margin_db)
    for seed in seeds:
        out = run_episode(seed, p=p, channel=channel, pair_solver=solver)
        base = run_episode(seed, p=p, channel=channel)
        tot += out["executed"]
        met += out["executed"] - out["violations"]
        if out["feasible"]:
            feas += 1
            E_cert.append(out["E"])
        if base["feasible"]:
            E_base.append(base["E"])
    return {"coverage": met / tot if tot else float("nan"), "n_links": tot,
            "completion": feas / len(list(seeds)),
            "E_certified": float(np.mean(E_cert)) if E_cert else None,
            "E_uncertified": float(np.mean(E_base)) if E_base else None}
