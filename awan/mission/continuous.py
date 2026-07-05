"""Continuous mission on the Phase-1 freshness spine (playbook §5.2).

Discrete slots of length T = Tmax. Targets keep moving (agilities as in the
frozen freshness modules); each agent's local picture of its target is fresh,
but the ROOT's picture ages between aggregations: staleness of source s at
slot t is agility_s * (t - t_last_root_update_s) * T. An aggregation episode
(K = ceil(log2 N) slots, energy from the chosen coordination scheme) resets
every source's root-age to its tree depth in slots.

Re-aggregation policies:
  periodic(P)      : start an episode every P slots.
  event-triggered  : start when PREDICTED next-slot root staleness > S_max
                     OR accumulated new-info bits > G_min.

Batteries: B_i = battery_factor x mean one-shot per-agent energy; an agent
dies at 0 (coverage hole logged); lifetime = first-death slot. Battery-aware
bidding adds a low-residual sender penalty (extends Phase-1 U4).
"""

import numpy as np

from .. import adapters as A
from ..simcore import run_episode

INF = float("inf")


def episode_cost(seed, p, coordinator_factory, weight_fn=None, channel=None):
    out = run_episode(seed, p=p, coordinator=coordinator_factory(),
                      weight_fn=weight_fn, channel=channel)
    return out


def run_continuous(seed, p=None, n=8, horizon=60, policy="event",
                   coordinator_factory=None, weight_fn=None, channel=None,
                   s_max=6.0, g_min=60e6, period=8, battery_factor=3.0,
                   battery_aware=False, agilities=None):
    """One continuous mission. Returns energy, staleness trajectory, AoI,
    lifetime, episode count."""
    p = p or A.STRESS
    rng = np.random.default_rng(seed)
    if agilities is None:
        agilities = rng.uniform(0.5, 6.0, n)          # m/s per-target agility
    K = int(np.ceil(np.log2(n)))
    T = p["Tmax"]

    # calibrate batteries on a one-shot episode
    base = run_episode(seed, p=p, coordinator=coordinator_factory(),
                       weight_fn=weight_fn)
    per_agent = base["agentE"]
    B0 = battery_factor * (np.mean(list(per_agent.values())) if per_agent else 1.0)
    batt = {i: B0 for i in range(n)}

    age = {i: 0.0 for i in range(n)}                  # slots since root saw i
    new_bits = 0.0
    stale_traj, aoi_traj = [], []
    E_total, episodes, first_death = 0.0, 0, None
    ep_left, ep_seed = 0, seed

    for t in range(horizon):
        # sensing accrues novelty; root ages
        for i in range(n):
            age[i] += 1.0
        new_bits += float(np.sum(agilities)) * 1e6 * T / 6.0
        stale = float(np.mean([agilities[i] * age[i] * T for i in range(n)]))
        stale_traj.append(stale)
        aoi_traj.append(float(np.mean(list(age.values()))) * T)

        if ep_left > 0:                               # episode in progress
            ep_left -= 1
            if ep_left == 0:
                episodes += 1
                ep_seed += 1
                out = run_episode(ep_seed, p=p, n=n,
                                  coordinator=coordinator_factory(),
                                  weight_fn=_battery_weight(weight_fn, batt, B0)
                                  if battery_aware else weight_fn,
                                  channel=channel)
                if out["feasible"]:
                    E_total += out["E"]
                    for i, e in out["agentE"].items():
                        batt[i] = batt.get(i, B0) - e
                    depth = {i: 1.0 for i in range(n)}
                    for i in range(n):
                        age[i] = depth[i] * (K / max(K, 1))
                    new_bits = 0.0
                    if first_death is None:
                        dead = [i for i, b in batt.items() if b <= 0]
                        if dead:
                            first_death = t
            continue

        # decide whether to launch a new episode
        pred_stale = float(np.mean([agilities[i] * (age[i] + K + 1) * T
                                    for i in range(n)]))
        if policy == "periodic":
            fire = (t % period == period - 1)
        else:
            fire = (pred_stale > s_max) or (new_bits > g_min)
        if fire:
            ep_left = K

    return {"E": E_total, "episodes": episodes,
            "stale_mean": float(np.mean(stale_traj)),
            "stale_peak": float(np.max(stale_traj)),
            "stale_traj": stale_traj, "aoi_traj": aoi_traj,
            "lifetime": first_death if first_death is not None else horizon,
            "batt_min": float(min(batt.values())) if batt else 0.0}


def _battery_weight(weight_fn, batt, B0):
    """Sender penalty grows as its residual battery shrinks (U4 extension)."""

    def wf(r, cp):
        base = weight_fn(r, cp) if weight_fn else r["E"]
        frac = max(batt.get(cp["i"], B0), 1e-9) / B0
        return base * (1.0 + 1.5 * (1.0 - frac))

    return wf
