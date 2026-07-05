"""Mobility-policy comparison (playbook §3.3 planner + H6, feeds F2.3/F2.4).

One sender->receiver pair; the sender may reposition inside its patrol disc
before transmitting. Three policies choose the endpoint:

  static      stay put (no mobility spend)
  distance    move as close to the receiver as the disc allows (the classical
              heuristic the paper's motion block effectively follows)
  predictive  score a k x k candidate grid with the GP radio map's lower
              confidence gain mu - kappa*sigma plus the mobility cost, pick the
              cheapest candidate that still fits the deadline

Every policy is then executed under the SAME tier-1 realization, so the
comparison isolates the decision rule. Energies are MODELED (paper model).
"""

import numpy as np

from .. import adapters as A
from .predictor import GPRadioMapB, sample_links


def candidate_grid(c, R, k):
    xs = np.linspace(-R, R, k)
    pts = [c + np.array([dx, dy]) for dx in xs for dy in xs
           if dx * dx + dy * dy <= R * R]
    return pts


def _plan_at(pos_i, aj, ai, p, factor_db=0.0, t_mob=0.0, d_move=0.0):
    """Energy and feasibility of transmitting from pos_i given a channel
    belief offset factor_db and mobility (t_mob, d_move)."""
    eta = p["eta_req"]
    Ec_i, Tc_i = A.comp_energy_time(ai["L"], ai["rho_pred"], eta, p)
    Ec_j, Tc_j = A.comp_energy_time(aj["L"], aj["rho_pred"], eta, p)
    t1 = min(p["Tmax"] - max(t_mob, Tc_i), p["Tmax"] - Tc_j)
    if t1 <= 0.05:
        return None
    h_belief = A.chan_gain(pos_i, aj["pos"], p) * 10 ** (factor_db / 10)
    Lout = eta * ai["L"]
    if Lout / A.rate(p["Pmax"], h_belief, p) > t1:
        return None
    ptx = A.ptx_for(Lout, h_belief, t1, p)
    if ptx > p["Pmax"]:
        return None
    E_mob = A.mob_energy(d_move, max(t_mob, 1e-3), p)
    E_comm = A.comm_energy(Lout, h_belief, t1, p)
    return {"pos": np.asarray(pos_i, float), "t1": t1, "ptx": min(ptx, p["Pmax"]),
            "Lout": Lout, "E_plan": E_mob + Ec_i + Ec_j + E_comm,
            "E_mob": E_mob, "E_comp": Ec_i + Ec_j}


def _execute(plan, aj, channel, p, link_key):
    factor = channel.gain_factor(plan["pos"], aj["pos"], link_key=link_key,
                                 advance=True)
    h_real = A.chan_gain(plan["pos"], aj["pos"], p) * factor
    rate = A.rate(plan["ptx"], h_real, p)
    t_real = plan["Lout"] / rate if rate > 0 else np.inf
    E_comm_real = plan["ptx"] * min(t_real, 3 * p["Tmax"])
    violated = t_real > plan["t1"] * (1 + 1e-6)
    return {"E_real": plan["E_mob"] + plan["E_comp"] + E_comm_real,
            "violated": bool(violated)}


def run_policies(channel, p, seeds, k=7, kappa=1.0, n_survey=120,
                 margin_db=0.0):
    """Per-policy realized energy + violation rate. All policies respect the
    physical motion budget reach = vmax * t_budget (t_budget <= 0.6*Tmax), so
    a round's move is ~7.5 m in the stress regime — the predictive policy's
    edge must come from escaping shadow nulls inside d_c, not teleporting.
    `margin_db` > 0 adds the conformal channel margin to the predictive policy
    (the certified-predictive variant)."""
    out = {pol: {"E": [], "viol": 0, "tot": 0, "infeas": 0} for pol in
           ("static", "distance", "predictive", "certified")}
    t_budget = 0.6 * p["Tmax"]
    reach = p["vmax"] * t_budget
    for seed in seeds:
        channel.new_scenario(seed)
        agents, rng, _ = A.scenario_gen(seed, n=2, p=p)
        ai, aj = agents[0], agents[1]
        rmap = GPRadioMapB(p).fit(*sample_links(channel, p,
                                                np.random.default_rng(seed + 9),
                                                n_survey))

        def in_disc(q):
            v = q - ai["c"]
            n = np.linalg.norm(v)
            return q if n <= ai["R"] else ai["c"] + v / n * ai["R"]

        plans = {}
        plans["static"] = _plan_at(ai["pos"], aj, ai, p)
        # distance heuristic: step toward the receiver, within reach and disc
        v = aj["pos"] - ai["pos"]
        step = min(reach, float(np.linalg.norm(v)))
        tgt = in_disc(ai["pos"] + (v / max(np.linalg.norm(v), 1e-9)) * step)
        d = float(np.linalg.norm(tgt - ai["pos"]))
        plans["distance"] = _plan_at(tgt, aj, ai, p,
                                     t_mob=(d / p["vmax"] * 1.05 + 1e-3) if d > 0.5 else 0.0,
                                     d_move=d if d > 0.5 else 0.0)

        def predictive_plan(extra_margin):
            best = None
            for cand in candidate_grid(ai["pos"], reach, k):
                cand = in_disc(cand)
                dm = float(np.linalg.norm(cand - ai["pos"]))
                if dm > reach:
                    continue
                tm = (dm / p["vmax"] * 1.05 + 1e-3) if dm > 0.5 else 0.0
                mu, sd = rmap.predict(cand, aj["pos"])
                det_db = 10 * np.log10(A.chan_gain(cand, aj["pos"], p))
                fdb = (mu - kappa * sd) - det_db - extra_margin
                pl = _plan_at(cand, aj, ai, p, factor_db=fdb, t_mob=tm,
                              d_move=dm if dm > 0.5 else 0.0)
                if pl and (best is None or pl["E_plan"] < best["E_plan"]):
                    best = pl
            return best

        plans["predictive"] = predictive_plan(0.0)
        plans["certified"] = predictive_plan(margin_db)

        for pol, pl in plans.items():
            out[pol]["tot"] += 1
            if pl is None:
                out[pol]["infeas"] += 1
                continue
            r = _execute(pl, aj, channel, p, link_key=(seed, pol))
            out[pol]["E"].append(r["E_real"])
            out[pol]["viol"] += int(r["violated"])
    return out
