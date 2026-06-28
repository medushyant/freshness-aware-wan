"""Direction 2: who should pair with whom, decided by looking ahead properly.

The paper picks pairs each round by minimum-weight matching on
  w_ij = (pair energy) + Phi,  Phi = zeta * ||p_j - centroid||^delta * L_next,
with zeta hand-tuned and Phi only evaluated AFTER the pair was optimized
without it (audit L19). Their own Fig. 7 shows energy rebounds when zeta
is off. This module replaces that recipe step by step:

  stage A   same zeta-potential, but fed INTO the inner problem (fixes L19)
  stage B   auto-scaled lookahead: the weight between energy and the
            spread term is set from round-1 magnitudes, Lyapunov
            drift-plus-penalty style. no knob to tune.
  learned   a value model V(state) trained on rollouts predicts the
            *remaining* mission energy; matching weight = E_now + V(after).
            no zeta anywhere. optional flexible schedule: a pair may sit
            a round out if V says that is cheaper (dominance: the forced
            schedule stays in the candidate set, so we can never do worse
            by more than the value model's error).

Also here: greedy / random / distance baselines and a brute-force DP
oracle for small N (no published optimality gap exists for this model).
"""

import itertools
import numpy as np
import networkx as nx

from .model import P, make_agents, jaccard_disks
from .solver import solve_pair

INF = float("inf")


# ----------------------------------------------------------------------
# light-weight mission state (Direction 2 doesn't need the srcs ledger)
# ----------------------------------------------------------------------

def _snapshot(agents, ids):
    return [{"id": i, "pos": agents[i]["pos"].copy(), "L": agents[i]["L"],
             "c": agents[i]["c"], "R": agents[i]["R"],
             "rho_pred": agents[i]["rho_pred"],
             "disks": list(agents[i]["disks"]),
             "srcs": {k: list(v) for k, v in agents[i]["srcs"].items()}}
            for i in ids]


def state_features(act, p=P):
    """Hand features of an active set; the value model lives on these.
    Deliberately simple - the point is the *pipeline*, a GNN can slot in
    later without touching anything else."""
    n = len(act)
    if n <= 1:
        return np.zeros(7)
    pos = np.array([a["pos"] for a in act])
    L = np.array([a["L"] for a in act])
    cen = pos.mean(axis=0)
    d_cen = np.linalg.norm(pos - cen, axis=1)
    pd = [np.linalg.norm(pos[i] - pos[j]) for i in range(n) for j in range(i + 1, n)]
    return np.array([
        n,
        np.ceil(np.log2(n)),            # rounds still to go
        L.sum() / 1e6,                  # total mass to haul [Mb]
        L.max() / 1e6,
        float((L * d_cen).sum() / L.sum()),   # payload-weighted spread [m]
        float(np.mean(pd)),
        float(np.min(pd)),
    ])


class ValueModel:
    """Ridge regression on standardized features. Closed form, no deps."""

    def __init__(self, l2=1.0):
        self.l2, self.w, self.mu, self.sd = l2, None, None, None

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        A = Z.T @ Z + self.l2 * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ y)
        r2 = 1 - np.sum((Z @ self.w - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
        return r2

    def __call__(self, feats):
        z = np.r_[1.0, (np.asarray(feats) - self.mu) / self.sd]
        return float(max(z @ self.w, 0.0))


# ----------------------------------------------------------------------
# one mission under a chosen pairing policy
# ----------------------------------------------------------------------

def mission(seed, p=P, policy="paper", zeta=None, vmodel=None,
            flexible=False, allow_motion=False, n=None, behavior_eps=0.0,
            record=None, matcher=None):
    """policy in {paper, stageA, lyapunov, learned, greedy, random, distance}.
    zeta overrides p['zeta'] for the potential-field policies.
    record: pass a list to collect (features, future-cost) training rows.
    matcher: optional callable(active, w, E, rng) -> pairs, used instead of the
    Blossom matcher (for the metaheuristic comparison baselines)."""
    pp = dict(p)
    if zeta is not None:
        pp["zeta"] = zeta
    rng = np.random.default_rng(seed)
    geo = np.random.default_rng(seed + 10_000)
    agents = make_agents(rng, pp, n=n)
    active = list(range(len(agents)))
    E_total = 0.0
    agentE = {}
    round_E, round_states = [], []
    lyap_V = None
    k = 0
    while len(active) > 1:
        k += 1
        cen = np.mean([agents[i]["pos"] for i in active], axis=0)
        rho = {}
        for i in active:
            for j in active:
                if i < j:
                    rho[(i, j)] = jaccard_disks(agents[i]["disks"],
                                                agents[j]["disks"], geo)

        E, w, sol = {}, {}, {}
        for i in active:
            for j in active:
                if i == j:
                    continue
                rp = rho[(min(i, j), max(i, j))]
                r = solve_pair(agents[i], agents[j], pp, rho_pair=rp,
                               centroid=cen, allow_motion=allow_motion,
                               phi_in_loop=(policy == "stageA"))
                if r is None:
                    E[(i, j)] = w[(i, j)] = INF
                    continue
                sol[(i, j)] = r
                E[(i, j)] = r["E"]
                spread = np.linalg.norm(r["pe_j"] - cen) ** pp["delta"] * r["L_next"]
                if policy == "paper":
                    w[(i, j)] = r["E"] + pp["zeta"] * spread       # Eq. (36)
                elif policy == "stageA":
                    w[(i, j)] = r["E"] + pp["zeta"] * spread       # same form,
                    # but r was optimized WITH the Phi term inside (L19 fixed)
                elif policy == "lyapunov":
                    w[(i, j)] = (r["E"], spread)   # combined after auto-scale
                elif policy == "learned":
                    after = _hypothetical(agents, active, i, j, r)
                    w[(i, j)] = r["E"] + vmodel(state_features(after, pp))
                else:                                  # greedy/random/distance
                    w[(i, j)] = r["E"]

        if policy == "lyapunov":
            # drift-plus-penalty: scale the spread term so its round-1
            # magnitude matches the energy term. set once, never tuned.
            if lyap_V is None:
                es = [v[0] for v in w.values() if isinstance(v, tuple)]
                ds = [v[1] for v in w.values() if isinstance(v, tuple)]
                lyap_V = (np.mean(es) / max(np.mean(ds), 1e-12)) if es else 0.0
            w = {k_: (v[0] + lyap_V * v[1]) if isinstance(v, tuple) else v
                 for k_, v in w.items()}

        pairs = (matcher(active, w, E, rng) if matcher is not None
                 else _pick_pairs(active, w, E, agents, policy, rng))

        if flexible and policy == "learned" and len(active) % 2 == 0 and len(active) > 2:
            pairs = _maybe_idle_worst(pairs, sol, agents, active, vmodel, pp)

        if record is not None:
            round_states.append(state_features([agents[i] for i in active], pp))

        progressed, E_round = False, 0.0
        new_active = []
        for (i, j) in pairs:
            if j is None or (i, j) not in sol:
                new_active.append(i)
                if j is not None:
                    new_active.append(j)
                continue
            r = sol[(i, j)]
            E_total += r["E"]; E_round += r["E"]; progressed = True
            agentE[i] = agentE.get(i, 0.0) + r.get("E_i", 0.6 * r["E"])
            agentE[j] = agentE.get(j, 0.0) + r.get("E_j", 0.4 * r["E"])
            _apply(agents[i], agents[j], r, rho[(min(i, j), max(i, j))])
            new_active.append(j)
        round_E.append(E_round)
        if not progressed:
            return {"E": INF, "rounds": k, "feasible": False}
        active = new_active
        if k > 14:
            break

    if record is not None:
        for t, f in enumerate(round_states):
            record.append((f, float(sum(round_E[t:]))))
    return {"E": E_total, "rounds": k, "feasible": True,
            "Emax": max(agentE.values()) if agentE else 0.0}


def _hypothetical(agents, active, i, j, r):
    """Cheap copy of the post-round active set if i->j fires and the rest
    pair up *somehow* - we only move/merge this one pair, which is enough
    signal for the value features."""
    out = []
    for a in active:
        if a == i:
            continue
        s = {"pos": agents[a]["pos"], "L": agents[a]["L"]}
        if a == j:
            s = {"pos": np.asarray(r["pe_j"]), "L": r["L_next"]}
        out.append(s)
    return out


def _pick_pairs(active, w, E, agents, policy, rng):
    ids = list(active)
    if policy == "random":
        rng.shuffle(ids)
        out = [(ids[2 * t], ids[2 * t + 1]) for t in range(len(ids) // 2)]
        if len(ids) % 2:
            out.append((ids[-1], None))
        return out
    if policy == "greedy":
        out, used = [], set()
        for (i, j), c in sorted(E.items(), key=lambda kv: kv[1]):
            if np.isfinite(c) and i not in used and j not in used:
                out.append((i, j)); used |= {i, j}
        for i in ids:
            if i not in used:
                out.append((i, None))
        return out

    ww = w
    if policy == "distance":
        ww = {(i, j): float(np.linalg.norm(agents[i]["pos"] - agents[j]["pos"]))
              for i in ids for j in ids if i != j}
    G = nx.Graph(); G.add_nodes_from(ids)
    virt = -1
    if len(ids) % 2:
        G.add_node(virt)
    fin = [v for v in ww.values() if np.isfinite(v)]
    big = 1.0 + (max(fin) if fin else 1.0)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            c = min(ww.get((i, j), INF), ww.get((j, i), INF))
            G.add_edge(i, j, weight=big * 100 - (c if np.isfinite(c) else 50 * big))
    if virt in G:
        for i in ids:
            G.add_edge(virt, i, weight=big * 100)
    out = []
    for (u, v) in nx.max_weight_matching(G, maxcardinality=True):
        if virt in (u, v):
            out.append((v if u == virt else u, None))
        else:
            cuv, cvu = ww.get((u, v), INF), ww.get((v, u), INF)
            out.append((u, v) if cuv <= cvu else (v, u))
    return out


def _maybe_idle_worst(pairs, sol, agents, active, vmodel, p):
    """Flexible schedule: compare 'all pairs fire' with 'worst pair rests'.
    The forced plan is always a candidate, so flexibility cannot lose
    (Dominance Proposition, audit L12)."""
    real = [pr for pr in pairs if pr[1] is not None and pr in sol]
    if len(real) < 2:
        return pairs

    def plan_score(fire):
        aft, e_now = [], 0.0
        firing = {pr for pr in fire}
        merged_to = {i: r for (i, _), r in []}
        for a in active:
            sender = next((pr for pr in firing if pr[0] == a), None)
            recv = next((pr for pr in firing if pr[1] == a), None)
            if sender:
                continue
            if recv:
                r = sol[recv]; aft.append({"pos": np.asarray(r["pe_j"]), "L": r["L_next"]})
            else:
                aft.append({"pos": agents[a]["pos"], "L": agents[a]["L"]})
        e_now = sum(sol[pr]["E"] for pr in fire)
        return e_now + vmodel(state_features(aft, p))

    full = plan_score(real)
    worst = max(real, key=lambda pr: sol[pr]["E"])
    rest = [pr for pr in real if pr != worst]
    alt = plan_score(rest) if rest else INF
    if alt < full:
        idle = [pr for pr in pairs if pr[1] is None]
        return rest + [(worst[0], None), (worst[1], None)] + idle
    return pairs


def _apply(ai, aj, r, rho_pair):
    aj["rho_pred"] = rho_pair
    seen = {(round(c[0], 2), round(c[1], 2), round(rd, 2)) for c, rd in aj["disks"]}
    for c, rd in ai["disks"]:
        key = (round(c[0], 2), round(c[1], 2), round(rd, 2))
        if key not in seen:
            aj["disks"].append((c, rd)); seen.add(key)
    aj["L"] = r["L_next"]
    aj["pos"] = np.asarray(r["pe_j"], float)


# ----------------------------------------------------------------------
# training data + the exact small-N oracle
# ----------------------------------------------------------------------

def collect_training(seeds, p=P, n_list=(4, 5, 6)):
    """Rollouts under a mixed behavior policy so the value model sees
    good and bad states, not just the paper's trajectory."""
    rows = []
    for s in seeds:
        pol = ["paper", "random", "greedy"][s % 3]
        mission(s, p, policy=pol, allow_motion=False,
                n=n_list[s % len(n_list)], record=rows)
    X = np.array([f for f, _ in rows]); y = np.array([c for _, c in rows])
    return X, y


def dp_oracle(seed, p=P, n=5):
    """Brute force over every pairing sequence (directions included),
    motion off. Ground-truth optimum for the optimality-gap figure."""
    rng = np.random.default_rng(seed)
    geo = np.random.default_rng(seed + 10_000)
    agents = make_agents(rng, p, n=n)
    base = _snapshot(agents, range(n))

    def all_matchings(ids):
        if not ids:
            yield []
            return
        if len(ids) % 2 == 1:
            for idle in ids:
                rest = [x for x in ids if x != idle]
                for m in all_matchings(rest):
                    yield m + [(idle, None)]
            return
        a = ids[0]
        for b in ids[1:]:
            rest = [x for x in ids if x not in (a, b)]
            for m in all_matchings(rest):
                yield m + [(a, b)]

    cache = {}

    def pair_cost(st, i, j):
        key = (i, j, round(st[i]["L"] / 1e4), round(st[j]["L"] / 1e4),
               round(st[i]["rho_pred"], 2), round(st[j]["rho_pred"], 2))
        if key in cache:
            return cache[key]
        rp = jaccard_disks(st[i]["disks"], st[j]["disks"], geo)
        r = solve_pair(st[i], st[j], p, rho_pair=rp, centroid=np.zeros(2),
                       allow_motion=False)
        cache[key] = (r, rp)
        return r, rp

    best = [INF]

    def rec(st, ids, acc):
        if acc >= best[0]:
            return
        if len(ids) == 1:
            best[0] = acc
            return
        for m in all_matchings(list(ids)):
            for dirs in itertools.product(*[[0, 1] if b is not None else [0]
                                            for (a, b) in m]):
                st2 = {k: {**v, "disks": list(v["disks"])} for k, v in st.items()}
                cost, ok, nxt = 0.0, True, []
                for (a, b), d in zip(m, dirs):
                    if b is None:
                        nxt.append(a); continue
                    i, j = (a, b) if d == 0 else (b, a)
                    r, rp = pair_cost(st, i, j)
                    if r is None:
                        ok = False; break
                    cost += r["E"]
                    st2[j] = {**st2[j], "L": r["L_next"], "rho_pred": rp,
                              "disks": st2[j]["disks"] + st2[i]["disks"]}
                    nxt.append(j)
                if ok:
                    rec(st2, nxt, acc + cost)

    st0 = {a["id"]: a for a in base}
    rec(st0, list(range(n)), 0.0)
    return best[0]


def finetune_decision_focused(vm, p, train_seeds, iters=16, n=6, flexible=False):
    """Stage D: nudge the value weights to minimize the REALIZED mission
    energy, i.e. the loss is evaluated through the exact Blossom matcher.
    Plain zeroth-order random search (perturbation-gradient spirit:
    the solver stays a black box, no derivatives needed). Keeps the best
    weights seen, so it can never end worse than stage C on the
    training scenarios."""
    rng = np.random.default_rng(7)

    def J(wvec):
        vm.w = wvec
        return np.mean([mission(s, p, policy="learned", vmodel=vm, n=n,
                                flexible=flexible)["E"] for s in train_seeds])

    w_best = vm.w.copy()
    f_best = J(w_best)
    scale = 0.15 * np.linalg.norm(w_best)
    for t_ in range(iters):
        d = rng.standard_normal(w_best.shape)
        d *= scale / (np.linalg.norm(d) * (1 + t_ / 6))
        for sgn in (1.0, -1.0):
            f = J(w_best + sgn * d)
            if f < f_best - 1e-9:
                f_best, w_best = f, w_best + sgn * d
                break
    vm.w = w_best
    return f_best
