"""The ONLY bridge to the frozen Phase-1 engine (playbook §1.3).

Thin delegations, no behavior changes. Anything Phase-2 needs from `wan.*`
is imported and re-exported here, so a grep for `from wan` outside this file
should return nothing under awan/.
"""

import numpy as np

from . import ROOT  # noqa: F401  (ensures repo root is importable)

from wan.model import (P, make_agents, jaccard_disks, chan_gain, rate,
                       mob_energy, comp_load, comp_energy_time, comm_energy,
                       ptx_for)
from wan.solver import solve_pair
from wan.network import run_mission, _match
from wan.topology import (mission as phase1_mission, state_features,
                          ValueModel, collect_training, dp_oracle,
                          _pick_pairs)
from wan.freshtopo import P_FRESH, mission_fresh
from wan.freshness import target_agility, freshness_distortion
from wan import targets as targets_mod
from wan.style import use_style

STRESS = dict(P_FRESH)  # the documented stress regime (== run_direction1 p1)


def params(stress=True, **over):
    p = dict(STRESS if stress else P)
    p.update(over)
    return p


def scenario_gen(seed, n=None, p=None):
    """Same rng discipline as the frozen missions: scenario from
    default_rng(seed), geometry sampling from default_rng(seed+10000)."""
    p = p or STRESS
    rng = np.random.default_rng(seed)
    geo = np.random.default_rng(seed + 10_000)
    agents = make_agents(rng, p, n=n)
    return agents, rng, geo


def blossom_match(active, w, rng):
    """The paper's centralized matcher (Edmonds' Blossom on min(w_ij, w_ji))."""
    return _match(active, w, "blossom", rng)


def potential_phi(pe_j, centroid, L_next, p):
    """Paper Eq. (36)-(38) potential field term."""
    return p["zeta"] * np.linalg.norm(np.asarray(pe_j) - np.asarray(centroid)) ** p["delta"] * L_next


def pair_overlaps(agents, active, geo):
    """rho for every unordered active pair, in the frozen iteration order."""
    rho = {}
    for i in active:
        for j in active:
            if i < j:
                rho[(i, j)] = jaccard_disks(agents[i]["disks"],
                                            agents[j]["disks"], geo)
    return rho


def energy_split(r, ai, aj, p, max_power=False):
    """Decompose a solve_pair result into (mobility, compute, comm) joules
    using the model identity E = E_mob + Ec_i + Ec_j + Ecomm."""
    Ec_i = comp_energy_time(ai["L"], ai["rho_pred"], r["eta_i"], p)[0]
    Ec_j = comp_energy_time(aj["L"], aj["rho_pred"], r["eta_j"], p)[0]
    h = chan_gain(r["pe_i"], r["pe_j"], p)
    if max_power:
        Ecm = p["Pmax"] * r["Lout"] / rate(p["Pmax"], h, p)
    else:
        Ecm = comm_energy(r["Lout"], h, r["t1"], p)
    Emob = max(r["E"] - Ec_i - Ec_j - Ecm, 0.0)
    return Emob, Ec_i + Ec_j, Ecm


def train_value_model(seeds=range(40), p=None, l2=1.0):
    """The frozen Phase-1 ridge cost-to-go, trained exactly as in Direction 2."""
    p = p or STRESS
    X, y = collect_training(list(seeds), p)
    vm = ValueModel(l2=l2)
    r2 = vm.fit(X, y)
    return vm, r2
