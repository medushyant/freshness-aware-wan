"""Nature-inspired matchers for the per-round pairing problem.

These exist for ONE reason: as comparison baselines in the optimality-gap
figure the mentor asked for. The per-round matching is solved exactly by
Blossom in polynomial time at this scale, so a metaheuristic cannot beat it --
the literature is consistent that swarm metaheuristics only pay off on large
instances where exact methods get slow. We implement four standard ones (ant
colony, artificial bee colony, cuckoo search, Egyptian vulture) over the same
pairing objective so the comparison is honest and the implementations are real.

Each has the signature  match(active, w, E, rng) -> list of (sender, receiver)
pairs, identical to the engine's own matcher, so they drop straight into
topology.mission(matcher=...).
"""

import numpy as np

INF = float("inf")


def _pair_cost(i, j, w):
    return min(w.get((i, j), INF), w.get((j, i), INF))


def _matching_cost(matching, w):
    c = 0.0
    for i, j in matching:
        if j is None:
            continue
        pc = _pair_cost(i, j, w)
        if not np.isfinite(pc):
            return INF
        c += pc
    return c


def _random_matching(active, rng):
    ids = list(active)
    rng.shuffle(ids)
    m = [(ids[2 * t], ids[2 * t + 1]) for t in range(len(ids) // 2)]
    if len(ids) % 2:
        m.append((ids[-1], None))
    return m


def _neighbor(matching, active, rng):
    m = [list(p) for p in matching]
    real = [k for k, p in enumerate(m) if p[1] is not None]
    if len(real) >= 2:
        a, b = rng.choice(real, size=2, replace=False)
        if rng.random() < 0.5:
            m[a][1], m[b][1] = m[b][1], m[a][1]
        else:
            m[a][1], m[b][0] = m[b][0], m[a][1]
    elif real and any(p[1] is None for p in m):
        a = real[0]; idle = [k for k, p in enumerate(m) if p[1] is None][0]
        m[a][1], m[idle][0] = m[idle][0], m[a][1]
    return [tuple(p) for p in m]


def _orient(matching, w):
    out = []
    for i, j in matching:
        if j is None:
            out.append((i, None)); continue
        out.append((i, j) if w.get((i, j), INF) <= w.get((j, i), INF) else (j, i))
    return out


def aco(active, w, E, rng, n_ants=10, n_iter=14, evap=0.3):
    ids = list(active)
    tau = {(a, b): 1.0 for a in ids for b in ids if a < b}
    best, best_c = None, INF
    for _ in range(n_iter):
        for _ in range(n_ants):
            free = ids.copy(); rng.shuffle(free); m = []
            while len(free) >= 2:
                i = free.pop()
                probs = np.array([tau[(min(i, j), max(i, j))] ** 1.2 *
                                  (1.0 / (1e-6 + _pair_cost(i, j, w))) ** 2.0
                                  if np.isfinite(_pair_cost(i, j, w)) else 1e-9
                                  for j in free])
                if probs.sum() <= 0:
                    j = free.pop()
                else:
                    j = free.pop(int(np.argmax(rng.random(len(free)) ** (1.0 / probs))))
                m.append((i, j))
            if free:
                m.append((free[0], None))
            c = _matching_cost(m, w)
            if c < best_c:
                best, best_c = m, c
        for k in tau:
            tau[k] *= (1 - evap)
        if best:
            for i, j in best:
                if j is not None:
                    tau[(min(i, j), max(i, j))] += 1.0 / (1e-6 + best_c)
    return _orient(best or _random_matching(active, rng), w)


def abc(active, w, E, rng, n_food=12, n_iter=14, limit=4):
    foods = [_random_matching(active, rng) for _ in range(n_food)]
    costs = [_matching_cost(m, w) for m in foods]
    trials = [0] * n_food
    for _ in range(n_iter):
        for s in range(n_food):
            cand = _neighbor(foods[s], active, rng); cc = _matching_cost(cand, w)
            if cc < costs[s]:
                foods[s], costs[s], trials[s] = cand, cc, 0
            else:
                trials[s] += 1
        fit = np.array([1.0 / (1e-6 + c) if np.isfinite(c) else 0.0 for c in costs])
        if fit.sum() > 0:
            for _ in range(n_food):
                s = int(np.searchsorted(np.cumsum(fit / fit.sum()), rng.random()))
                s = min(s, n_food - 1)
                cand = _neighbor(foods[s], active, rng); cc = _matching_cost(cand, w)
                if cc < costs[s]:
                    foods[s], costs[s], trials[s] = cand, cc, 0
        for s in range(n_food):
            if trials[s] > limit:
                foods[s] = _random_matching(active, rng)
                costs[s], trials[s] = _matching_cost(foods[s], w), 0
    b = int(np.argmin(costs))
    return _orient(foods[b], w)


def cuckoo(active, w, E, rng, n_nest=12, n_iter=14, pa=0.25):
    nests = [_random_matching(active, rng) for _ in range(n_nest)]
    costs = [_matching_cost(m, w) for m in nests]
    for _ in range(n_iter):
        k = rng.integers(n_nest)
        cand = nests[k]
        for _ in range(1 + int(2 * rng.standard_exponential())):   # Levy-like step
            cand = _neighbor(cand, active, rng)
        cc = _matching_cost(cand, w); j = rng.integers(n_nest)
        if cc < costs[j]:
            nests[j], costs[j] = cand, cc
        order = np.argsort([-(c if np.isfinite(c) else 1e18) for c in costs])
        for idx in order[:int(pa * n_nest)]:
            nests[idx] = _random_matching(active, rng)
            costs[idx] = _matching_cost(nests[idx], w)
    b = int(np.argmin(costs))
    return _orient(nests[b], w)


def egyptian_vulture(active, w, E, rng, n_iter=40):
    cur = _random_matching(active, rng); cc = _matching_cost(cur, w)
    best, best_c = cur, cc
    for _ in range(n_iter):
        cand = cur
        for _ in range(1 + rng.integers(2)):       # rolling + tossing of pebbles
            cand = _neighbor(cand, active, rng)
        c = _matching_cost(cand, w)
        if c < cc or rng.random() < 0.1:           # knock force = occasional accept
            cur, cc = cand, c
        if c < best_c:
            best, best_c = cand, c
    return _orient(best, w)


MATCHERS = {"ACO": aco, "ABC": abc, "Cuckoo": cuckoo, "Egyptian Vulture": egyptian_vulture}
