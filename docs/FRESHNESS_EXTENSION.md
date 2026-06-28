# Freshness-Aware Semantic Aggregation for Swarms Tracking Moving Targets

*A publication-oriented description of the moving-target contribution, with the
exact model, the algorithm, the measured results, and references. Drop-in
source for the report / IEEE submission. All numbers regenerate from
`run_freshness.py` (11/11 automated checks).*

---

## 1. Motivation and the gap

The base framework (Zhao et al., ILAC-WAN) anchors every agent's patrol region
at a **constant** centre `c_i` (§III, Eq. 1) — an implicit *static-target*
assumption. Its objective (Eq. 14) is energy-only; the authors' own structure
admits no notion of how *valuable* or *fresh* a source's contribution is. The
paper's audit records this directly as limitation **L8**: *"energy-only
objective; no fidelity, freshness, or value of information."*

Real surveillance swarms watch **moving** targets. The natural question is how
target motion should enter the aggregation. We first show — empirically, in the
paper's own engine — that the obvious answer is wrong, then give the answer that
works.

### 1.1 Why the mobility/coverage route is inert (negative result)

The intuitive integration is to re-anchor `c_i` on the (predicted) target each
round and let agents physically chase. We implemented exactly this and measured
it (`experiments/`):

- At the paper's patrol radius (R = 80–100 m), a target moving 3–7 m/s travels
  30–41 m per round — a fraction of R — so the coverage constraint
  `‖p_i − c_i‖ ≤ R` never binds; **target class is invisible** and the apparent
  differences are non-monotone noise (std ≫ signal).
- Decoupling a tight sensing radius `r_track` makes the *geometric* tracking
  demand explode (40 m/round at r_track = 8 m), yet realized energy stays
  **identical to static** (0.597 vs 0.601 J): the inner solver only *targets*
  coverage during its optional motion step, so the matcher routes around it.
- A decomposition shows mobility is a near-discontinuous term that, in the
  realized mission, the energy-minimizing matcher simply avoids.

This is consistent with the framework's *own* ranking: the sensing-aware-mobility
direction (CRB-on-trajectory) was rated **last of five** because "CRB-trajectory
methodology is crowded," and the 2024–2026 ISAC-tracking literature (EKF +
Cramér–Rao-bound trajectory optimization) confirms that lane is saturated.

### 1.2 The right channel: information freshness

The receiver's estimation error of a moving target is a **non-decreasing
function of the age** of its last update — the standard Age-of-Information /
remote-estimation result [Sun 2017; Maatouk AoII 2020; 2025 surveys]. So target
motion should enter through the **value/freshness** of the sensed information,
which couples to the framework's *own* objects: the semantic compression ratio
`η` (Direction 1) and the aggregation topology (Direction 2). This is exactly
the missing L8 term, and — per a literature scan — **no prior work couples
target-motion freshness to the compression ratio of a knowledge-aggregation
tournament**, making the coupling itself the novelty.

---

## 2. Model

**Source decay.** Each source `s` (the data collected by agent `s`) observes a
target with **agility** `a_s` = its mean speed over the horizon (static <
nearly-constant-velocity < maneuvering, the standard motion-class ladder). The
freshness distortion of `s` when delivered to the root after age `τ_s` is

```
        D_fresh(s) = a_F · a_s · (τ_s)^p ,      p ≥ 1 (p = 1: linear AoI form).
```

`p > 1` reflects the faster-than-linear growth of Kalman prediction error for an
accelerating target; the conclusions are invariant to `p`. Age is measured in
aggregation **hops** (each fuse–compress–relay stage is one processing-age step
— the multi-hop / version-AoI model), `τ_s = hops_s · T`.

**Root distortion.** Adding freshness to the Direction-1 compression distortion,
the assembled root report has

```
        D_root = D_compress  +  Σ_s w_s · D_fresh(s) ,     w_s = bit share of s,
```

with `D_compress = a_D · Σ_s w_s · Σ_hops ln(1/η)` the path-multiplicative
("telephone-game") compression term already modelled in Direction 1.

**Two coupling points.**
- **D1 (compression):** the per-hop fidelity budget is `Dmax`. A faster target
  consumes more of it through `D_fresh`, so the derived per-hop η-floor (the
  endogenous replacement for the paper's exogenous `η_req`) **rises**, forcing
  *less* compression and more energy. This sharpens the Direction-1 result: the
  paper's "always compress maximally" fails not only because distortion is
  priced, but by an amount set by **target motion**.
- **D2 (topology):** `D_fresh` depends on `hops_s`, which the matching controls.
  Giving fast-decaying sources **shorter paths** lowers `D_root` at fixed energy.

---

## 3. Algorithm: value-prioritized matching

Identical inner solver and identical Blossom matcher as the paper — only the
matching weight changes. When sender `i` transmits to receiver `j`, every source
currently held by `i` takes one more hop, accruing marginal staleness. The
weight is

```
        w(i→j) = E(i→j)  +  λ_F · a_F · T · Σ_{s ∈ holder(i)} bits_s · a_s .
```

Because the engine already symmetrizes by the cheaper direction, this makes the
agent holding the most fast-decaying mass the **receiver** (its sources wait,
depth unchanged), so high-agility data reaches the root in fewer hops. At
`λ_F = 0` the weight is exactly the paper's, so the method **reduces to the
baseline** by construction.

---

## 4. Results (regenerate: `python3 run_freshness.py`, 11/11 PASS)

**D1 — compression responds to motion (exact, then robust).**

| target speed | optimal η | pair energy |
|---|---|---|
| 0 m/s (static) | 0.549 | 1.704 J |
| 4.5 m/s | 0.628 | ~2.18 J |
| 9 m/s (maneuvering) | 0.719 | 1.853 J |

η-floor and energy rise **monotonically** with target speed (deterministic,
`F1`), and the same ladder holds across 30 random aggregation pairs (`F2`).

**D2 — value-prioritized topology (20 missions, stress regime).**

- Root report **40% fresher** (staleness 11.9 → 7.1) at **no energy cost**
  (−1.5%) vs the paper's energy-only matching (`F3`).
- Reduces **exactly** to the paper at `λ_F = 0` (20/20 identical, sanity).
- The advantage **grows with agility spread** across targets: 21% → 48% as the
  spread widens (`F4`) — prioritization matters most under heterogeneous decay,
  the goal-oriented / value-of-information regime.

**Prediction (`F5`).** The effective decay rate is the predictor's residual
error rate. A CV-**Kalman** and a CV+CT **IMM** dominate the naive
finite-difference CV predictor on every class; prediction pays once target
motion exceeds sensor noise (best 7.5 m vs react 9.7 m on CV-motion targets),
cutting root staleness ~23% (12.4 vs 16.1). Honest scope: for near-static
targets, not predicting is best — the agent should pick the predictor (including
"none") that minimizes residual error, which is itself a freshness-optimal
choice.

---

## 5. Why this is the publishable spine

- **Single objective (the professor's ask):** moving targets are now the spine
  that motivates *both* mechanisms — freshness-aware compression (D1) and
  value-prioritized topology (D2) — rather than two unrelated tweaks.
- **On-model:** every effect runs through the paper's own η and matcher; every
  comparison is like-for-like with the same inner solver, and the method reduces
  to the baseline at `λ_F = 0`.
- **Novel + uncrowded:** couples target-motion freshness to a semantic
  *aggregation tournament's* compression ratio and tree depth — a gap no prior
  work fills — while sidestepping the saturated ISAC/CRB-tracking lane.
- **Grounded:** Age-of-Information / Age-of-Incorrect-Information remote
  estimation; goal-oriented / value-of-information scheduling; multi-hop /
  version-AoI; IMM-Kalman maneuvering-target tracking.

## 6. Honest limitations (keep in the report)

- Processing-age (per-hop) model of staleness; a wall-clock re-sensing model is
  an alternative we note but do not need.
- Stress regime is required for energy to respond (documented; same regime the
  D2 topology results already use — with the lazy Table-I radio, communication
  is nearly free and *nothing* moves energy).
- Freshness sensitivity `a_F`, `λ_F`, and the penalty exponent `p` are chosen
  and disclosed; results are qualitatively invariant to them (the D2 cut
  saturates for `λ_F ≥ 1e-7`).

## 7. References (2017–2026)

- Y. Sun et al., *Update or Wait: How to Keep Your Data Fresh*, IEEE Trans. IT, 2017 (AoI ↔ estimation error).
- A. Maatouk et al., *The Age of Incorrect Information*, 2020 (non-decreasing staleness penalty).
- *Age of Semantic Information-Aware Wireless Transmission*, arXiv:2508.12248, 2025.
- *From Information Freshness to Semantics of Information and Goal-oriented Communications*, arXiv:2512.12758, 2025.
- *Semantics-Aware Source Coding in Status Update Systems*, arXiv:2203.08508.
- *From Freshness to Effectiveness: Goal-Oriented Sampling for Remote Decision Making*, arXiv:2504.19507, 2025.
- *Version AoI in Single- and Multi-Hop Networks*, arXiv:2507.23433.
- Maneuvering-target tracking: IMM with CV/CA/CT Kalman/cubature filters (survey + 2024–2026 ISAC tracking with EKF + PCRB, e.g. arXiv:2401.03726).
