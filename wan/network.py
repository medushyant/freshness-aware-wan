"""Outer level: run a whole aggregation mission round by round.

Implements the H-MAP loop of the paper (weights -> blossom matching ->
execute -> retire senders) and the Direction-1 variant on top of it:
  * source-level bookkeeping of how compressed each source's data is
    (the 'telephone game' state, sum of ln(1/eta) along the path),
  * sub-additive fusion and Wyner-Ziv payloads,
  * a fidelity floor Dmax with an *derived* per-hop compression bound,
  * split-conformal calibration so the floor holds with probability 1-alpha.
"""

import numpy as np
import networkx as nx

from .model import P, make_agents, jaccard_disks
from .solver import solve_pair

INF = float("inf")


def _root_distortion(agent, p=P):
    """Payload-weighted mean accumulated log-compression of the sources."""
    tot = sum(b for b, _ in agent["srcs"].values())
    if tot <= 0:
        return 0.0
    return p["a_D"] * sum(b * lc for b, lc in agent["srcs"].values()) / tot


def _derived_eta_floor(agent, k_rem, p):
    """Endogenous replacement for eta_req (audit L23): largest per-hop
    compression that still keeps the worst source under Dmax after the
    remaining k_rem hops."""
    worst = max((lc for _, lc in agent["srcs"].values()), default=0.0)
    budget = max(p["Dmax"] / p["a_D"] - worst, 0.0)
    if k_rem <= 0:
        return 1.0
    return float(np.exp(-budget / k_rem))


def run_mission(seed, p=P, mode="paper", lam=0.0, topology="blossom",
                allow_motion=True, use_wz=None, derived_cap=False,
                force_eta=None, max_power=False, n=None, rng_geo=None,
                targets=None, target_T=None, r_track=None, predict=False,
                src_distortion0=None):
    """One full aggregation episode. Returns totals and the round log.

    mode      : 'paper' or 'fidelity'
    topology  : 'blossom' (paper), 'distance', 'random'

    Moving-target hook (all optional, default off => paper-identical):
    targets   : list[Target] indexed by agent id. When given, each round the
                active agents' patrol centres c_i are re-anchored on their
                target (the paper's fixed-centre assumption, relaxed), and the
                targets advance by target_T seconds.
    target_T  : per-round duration the targets move; default p['Tmax'].
    r_track   : if set, the agent must keep its (now moving) disk within this
                SENSING radius of the target instead of the patrol radius R.
                This is the knob that decides whether motion bites at all.
    predict   : if True, aim c_i at a one-step CV prediction instead of the
                current measurement (predict-ahead vs react-only).
    Returns also 'forced_move': solver-independent geometric tracking demand,
    the mean per-(round,agent) distance the agent MUST move just to bring the
    target back inside its disk (max(0, ||pos - c|| - R)).
    """
    rng = np.random.default_rng(seed)
    geo = rng_geo or np.random.default_rng(seed + 10_000)
    agents = make_agents(rng, p, n=n)
    fid = (mode == "fidelity")
    if use_wz is None:
        use_wz = fid

    # freshness coupling: pre-load each source's accumulated distortion with a
    # staleness offset (e.g. a_F * target_agility). Faster targets => less
    # remaining fidelity budget => the derived eta-floor forces less
    # compression => more bits/energy. This is the on-model way targets bite.
    if src_distortion0 is not None:
        for i, a in enumerate(agents):
            a["srcs"][i][1] += float(src_distortion0[i])

    preds = None
    if targets is not None:
        from .targets import CVPredictor
        preds = [CVPredictor() for _ in agents]
        T_tg = target_T if target_T is not None else p["Tmax"]
        # anchor each agent on its target at the very start, so round 1 is
        # already a tracking round (not the paper's free initial geometry).
        for i, a in enumerate(agents):
            aim = preds[i].predict(targets[i].pos(), T_tg) if predict else targets[i].pos().copy()
            a["c"] = np.asarray(aim, float)
            if r_track is not None:
                a["R"] = float(r_track)
            a["pos"] = a["c"].copy()   # start on target (fair to every class)
            a["disks"] = [(a["c"].copy(), a["R"])]   # coverage anchored on target

    active = list(range(len(agents)))
    E_total, log = 0.0, []
    forced_move, fm_count = 0.0, 0
    K_est = int(np.ceil(np.log2(len(agents))))
    k = 0
    while len(active) > 1:
        k += 1
        if targets is not None:
            # advance every target one round, then re-anchor active agents
            for tg in targets:
                tg.step(T_tg)
            for i in active:
                aim = preds[i].predict(targets[i].pos(), T_tg) if predict else targets[i].pos().copy()
                agents[i]["c"] = np.asarray(aim, float)
                if r_track is not None:
                    agents[i]["R"] = float(r_track)
                # geometric tracking demand BEFORE the solver moves anything
                gap = np.linalg.norm(agents[i]["pos"] - agents[i]["c"]) - agents[i]["R"]
                forced_move += max(0.0, gap); fm_count += 1
        cen = np.mean([agents[i]["pos"] for i in active], axis=0)
        k_rem = max(K_est - (k - 1), 1)

        # pair overlaps (used by WZ + sub-additive fusion + the weights)
        rho = {}
        for i in active:
            for j in active:
                if i < j:
                    rho[(i, j)] = jaccard_disks(agents[i]["disks"],
                                                agents[j]["disks"], geo)

        # candidate pair costs ------------------------------------------------
        sol, w = {}, {}
        for i in active:
            for j in active:
                if i == j:
                    continue
                rp = rho[(min(i, j), max(i, j))]
                cap, lo_j, lo_i = None, None, None
                if fid and derived_cap:
                    lo_j = _derived_eta_floor(agents[j], k_rem, p)
                    lo_i = _derived_eta_floor(agents[i], k_rem, p)
                    cap = 1.0
                elif fid:
                    cap = 1.0   # no exogenous eta_req in our model
                r = solve_pair(agents[i], agents[j], p, fidelity=fid, lam=lam,
                               rho_pair=rp, use_wz=use_wz, eta_cap_j=cap,
                               eta_lo_j=lo_j, eta_lo_i=lo_i, centroid=cen,
                               allow_motion=allow_motion,
                               force_eta=force_eta, max_power=max_power)
                if r is None:
                    w[(i, j)] = INF
                    continue
                sol[(i, j)] = r
                phi = p["zeta"] * np.linalg.norm(r["pe_j"] - cen) ** p["delta"] * r["L_next"]
                # paper adds Phi after the fact; in fidelity mode Phi was
                # already inside the objective, so the weight is just E.
                w[(i, j)] = r["E"] + (phi if not fid else lam * r["dD"])

        if topology == "distance":
            wmatch = {(i, j): float(np.linalg.norm(agents[i]["pos"] - agents[j]["pos"]))
                      for i in active for j in active if i != j}
        else:
            wmatch = w
        pairs = _match(active, wmatch, topology, rng)

        # execute the round ----------------------------------------------------
        new_active = []
        for (i, j) in pairs:
            if j is None:               # idle (odd count)
                new_active.append(i)
                continue
            if (i, j) not in sol:       # infeasible pick: nothing happened
                new_active.append(i); new_active.append(j)
                continue
            r = sol[(i, j)]
            E_total += r["E"]
            rp = rho[(min(i, j), max(i, j))]
            _execute(agents[i], agents[j], r, rp, fid, p)
            new_active.append(j)
        log.append({"round": k, "active": len(active), "pairs": pairs})
        if len(new_active) == len(active):      # zero progress: infeasible run
            return {"E": float("inf"), "D": float("inf"), "root": None,
                    "log": log, "agents": agents, "feasible": False,
                    "forced_move": forced_move / max(fm_count, 1)}
        active = new_active
        if k > 12:
            break
    root = agents[active[0]]
    return {"E": E_total, "D": _root_distortion(root, p), "root": active[0],
            "log": log, "agents": agents, "feasible": True,
            "forced_move": forced_move / max(fm_count, 1)}


def _match(active, w, topology, rng):
    """Pick this round's sender->receiver pairs."""
    ids = list(active)
    if topology == "random":
        rng.shuffle(ids)
        out = [(ids[2 * t], ids[2 * t + 1]) for t in range(len(ids) // 2)]
        if len(ids) % 2:
            out.append((ids[-1], None))
        return out

    G = nx.Graph()
    G.add_nodes_from(ids)
    virt = -1
    if len(ids) % 2:
        G.add_node(virt)
    big = 1.0 + max([v for v in w.values() if np.isfinite(v)] + [1.0])
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            cost = min(w.get((i, j), INF), w.get((j, i), INF))
            if not np.isfinite(cost):
                cost = 50 * big
            G.add_edge(i, j, weight=big * 100 - cost)
    if virt in G:
        for i in ids:
            G.add_edge(virt, i, weight=big * 100)
    mate = nx.max_weight_matching(G, maxcardinality=True)
    out = []
    for (u, v) in mate:
        if virt in (u, v):
            real = v if u == virt else u
            out.append((real, None))
        else:
            cuv, cvu = w.get((u, v), INF), w.get((v, u), INF)
            out.append((u, v) if cuv <= cvu else (v, u))
    return out


def _execute(ai, aj, r, rho_pair, fid, p):
    """State update after i transmits to j (paper Sec. II-B + our fusion)."""
    ei, ej = r["eta_i"], r["eta_j"]
    # the telephone game: everything each side carries gets re-squeezed
    for s in ai["srcs"]:
        ai["srcs"][s][1] += np.log(1.0 / ei)
    for s in aj["srcs"]:
        aj["srcs"][s][1] += np.log(1.0 / ej)
    # receiver's correlation w.r.t. this sender feeds next round's compute
    aj["rho_pred"] = rho_pair
    # merge: shared sources keep their best (least-compressed) copy
    for s, (b, lc) in ai["srcs"].items():
        if s in aj["srcs"]:
            if lc < aj["srcs"][s][1]:
                aj["srcs"][s] = [b, lc]
        else:
            aj["srcs"][s] = [b, lc]
    seen = {(round(c[0], 2), round(c[1], 2), round(rad, 2)) for c, rad in aj["disks"]}
    for c, rad in ai["disks"]:
        key = (round(c[0], 2), round(c[1], 2), round(rad, 2))
        if key not in seen:
            aj["disks"].append((c, rad))
            seen.add(key)
    aj["L"] = r["L_next"]
    aj["pos"] = np.asarray(r["pe_j"], float)


# ----------------------------------------------------------------------
# conformal risk control for the fidelity floor
# ----------------------------------------------------------------------

def simulate_true_D(D_hat, rng, sigma=0.12):
    """Stand-in for the model error of the learned distortion map: on real
    data this noise comes from the residuals of the mAP-vs-eta fit."""
    return D_hat * np.exp(sigma * rng.standard_normal())


def conformal_pick_lambda(alpha, lam_grid, n_cal=25, p=P, seed0=500,
                          fast=True):
    """Fixed-sequence Learn-then-Test over a monotone lambda ladder:
    for each lam (ascending), run calibration missions, take the
    ceil((n+1)(1-alpha))-th order statistic of the *realized* distortions,
    and accept the first lam whose quantile clears the floor. Standard
    split-conformal validity, no asymptotics."""
    rng = np.random.default_rng(123)
    kq = int(np.ceil((n_cal + 1) * (1 - alpha))) - 1
    for lam in lam_grid:
        scores = []
        for t in range(n_cal):
            out = run_mission(seed0 + t, p, mode="fidelity", lam=lam,
                              allow_motion=not fast)
            scores.append(simulate_true_D(out["D"], rng))
        q = np.sort(scores)[min(kq, n_cal - 1)]
        if q <= p["Dmax"]:
            return lam, q
    return lam_grid[-1], q


def conformal_coverage(lam, q, alpha, n_test=40, p=P, seed0=900, fast=True):
    rng = np.random.default_rng(321)
    hits = 0
    for t in range(n_test):
        out = run_mission(seed0 + t, p, mode="fidelity", lam=lam,
                          allow_motion=not fast)
        if simulate_true_D(out["D"], rng) <= p["Dmax"]:
            hits += 1
    return hits / n_test
