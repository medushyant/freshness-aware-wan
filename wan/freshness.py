"""Freshness / value-of-information coupling for moving targets.

The base paper fixes each patrol centre and never prices how *stale* a source's
data is by the time it reaches the aggregation root. But the swarm watches
moving targets, and the receiver's estimation error of a moving target is a
non-decreasing function of the age of the last update -- the standard
Age-of-Information / remote-estimation result (Sun et al.; Age of Incorrect
Information, Maatouk et al.). A faster target's information decays faster.

We turn that into the missing fidelity term the paper's own audit flags (L8:
"no fidelity, freshness, or value of information"). Each source s carries a
decay rate rho_s proportional to its target's agility, and accrues a freshness
distortion that grows with how long its data waits before it is delivered:

    D_fresh(s) = a_F * agility_s * age_s

where age_s is the source's delay to the root, measured in aggregation hops
(each fuse-compress-relay stage is one processing-age step -- the multi-hop /
version-AoI model). The total root distortion adds this to the compression
distortion already modelled in Direction 1:

    D_root = D_compress + sum_s w_s * D_fresh(s),   w_s = bit share of s.

This makes target motion enter through the paper's own object -- the semantic
compression ratio and the aggregation topology -- instead of through mobility
energy, which we showed empirically is inert here.
"""

import numpy as np


def target_agility(target, hops, T):
    """Mean per-round speed of a target over the mission horizon [m/s].

    A static target returns 0; a fast maneuvering one returns the largest
    value, so agility orders the three motion classes the way the tracking
    literature does (static < nearly-constant-velocity < maneuvering)."""
    prev = target.pos().copy()
    dist = 0.0
    for _ in range(max(hops, 1)):
        target.step(T)
        dist += np.linalg.norm(target.pos() - prev)
        prev = target.pos().copy()
    return dist / (max(hops, 1) * T)


def freshness_distortion(agility, age, a_F, T, exponent=1.0):
    """Non-decreasing staleness penalty for one source.

    exponent=1 is the linear AoI form; exponent>1 reflects the faster-than-
    linear growth of Kalman prediction error for an accelerating target and is
    kept as a robustness knob (the conclusions are invariant to it)."""
    return a_F * agility * (age * T) ** exponent


def root_freshness(srcs_meta, a_F, T, exponent=1.0):
    """Bit-share-weighted freshness distortion of an assembled root report.

    srcs_meta : dict src_id -> {"bits": .., "agility": .., "age": ..}."""
    tot = sum(m["bits"] for m in srcs_meta.values())
    if tot <= 0:
        return 0.0
    return sum(m["bits"] * freshness_distortion(m["agility"], m["age"], a_F, T,
                                                exponent)
               for m in srcs_meta.values()) / tot


def assign_agility(targets, hops, T):
    """Per-source agility for a list of targets, one per agent id."""
    return [target_agility(t, hops, T) for t in targets]
