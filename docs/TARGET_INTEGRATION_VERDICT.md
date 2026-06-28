# Moving-Target Integration — Verdict, Evidence, and Recommended Design

**Date:** 2026-06-27  ·  **Engine:** `wan/` (D1 + D2), verified 41/41 checks pass on this machine.
**Question asked:** *Is it worth adding the "moving targets" the handoff describes, and what is the
single best, most defensible way to do it?*

---

## TL;DR

- **The handoff's way of adding targets (re-anchor each agent's patrol centre `c_i` on its target,
  and let the agent physically "chase" it) is NOT worth implementing.** It produces *no measurable,
  interpretable signal* in the paper's own engine. Three experiments show it either washes out or is
  silently dropped by the solver. It is also the project's *own* Direction 4 — which the v2 proposal
  ranked **last (4/5)** because "CRB-trajectory methodology is crowded." The handoff's suggested
  "add a Kalman filter + Cramér–Rao bound" upgrade lands squarely in the single most saturated
  sub-area of the ISAC literature (many 2024–2026 papers). A B.Tech extension cannot win there.

- **There IS a version of moving targets that is worth it, and it is strong:** couple target motion
  to the paper's *own object* — the semantic **compression ratio η** and the Direction-1 **fidelity**
  model — through **information freshness / value of information**. A faster target's data goes stale
  faster (Age-of-Information / remote-estimation), which consumes the fidelity budget, forces *less*
  compression, and changes both η\* (Direction 1) and which sources to aggregate first (Direction 2).
  This is **on-model, monotone, novel, and aligned with the project's own audit item L8**
  ("energy-only objective; no fidelity, **freshness, or value of information**"). It makes moving
  targets the **single spine** that motivates *both* D1 and D2 — exactly the unification the
  professor asked for.

---

## 1. What was verified first (ground truth, not the handoff's summary)

- Extracted the archive, merged both code trees into `BTP_ILAC_WAN/`, installed numpy/scipy/mpl/nx in
  a venv, and **re-ran all four verifiers on this machine: 41/41 PASS** (targets 5, D1 15, D2 11,
  upgrades 10). Numbers match the handoff within seed/version noise (e.g. `D_ref` 2.21 vs 2.16).
- Read the base paper directly: §III fixes `c_i` as a **constant** patrol-region centre (the static-
  target assumption is real); "dynamic" in the title refers to the *topology* tournament, not target
  motion. Mobility serves *only* radio geometry, within a generous patrol disk.

## 2. Why the handoff's mobility/coverage approach fails (3 experiments)

Added a backward-compatible target hook to `run_mission` (`targets=`, `r_track=`, `predict=`) — the
exact integration the handoff proposed (`c_i ← target each round`). All 41 baseline checks still pass.

**(a) Washout — `experiments/exp_targets_washout.py`.** With the paper's geometry (R = 80–100 m),
targets move 30–41 m/round, a small fraction of R, so the coverage constraint stays slack:
the agent is *never forced to move*. Apparent energy differences across target classes are
**non-monotone and noise-dominated** (e.g. dynamic 0.803 J > time-varying 0.599 J — backwards;
std ±0.56–1.15 J ≫ the 0.32 J spread). forced-move ≈ 0–5 m.

**(b) Shrinking the sensing radius doesn't rescue it.** Sweeping a decoupled `r_track` down to 8 m,
the *geometric* tracking demand explodes to **40 m/round**, yet realized energy stays **0.597 J,
identical to static (0.601 J)**, all feasible. The solver/matcher silently routes *around* the
coverage constraint rather than paying to chase.

**(c) Why — `probe_mobility_share.py` + `probe_silent_violation.py`.** Mobility energy is real when
forced (a 22 m move costs ~3.4 J, 93% of pair energy), but in a full mission the matcher simply
selects pairings that avoid it, and the inner solver only *targets* coverage during the optional
motion step (it is not enforced on the agent's starting position). Net: motion is **inert** in the
realized objective.

> Conclusion: the paper's energy is dominated by **computation + communication**; mobility is a
> near-discontinuous fine-tuning term for the radio. Forcing "tracking" through it is working against
> the grain of the model — fragile and, in the realized mission, invisible.

## 3. Why the information-value/freshness approach works (clean, on-model)

**Mechanism — `exp_value_mechanism_clean.py` (deterministic, no scenario noise).**
Charge target staleness as a **freshness distortion** `d0 = a_F · speed`, consumed from the fidelity
budget `Dmax` *before* compression. Direction 1's derived per-hop η-floor then forces less
compression for faster targets. In the stress regime (where bits cost energy — the same regime the
D2 topology results already use):

| target speed | η-floor | pair energy |
|---|---|---|
| 0 m/s (static) | 0.549 | 1.704 J |
| 4.5 m/s | 0.628 | ~1.78 J |
| 9 m/s (fast maneuver) | 0.719 | 1.853 J |

**Both η-floor and energy rise perfectly monotonically (+9%).** Exact, reproducible, on-model — the
clean contrast to the inert mobility channel.

**Full-mission prototype — `exp_value_coupling.py`.** Mean energy rises with agility (0.893 →
~1.03 J) and the *baseline (no coupling) is perfectly flat* across classes, confirming the coupling
is the only driver. The per-seed result is still noisy with the *hard* floor (cliff effects near the
threshold — a known issue, cf. proposal D5's "cliff effect"). Making it presentation-clean needs the
**smooth** design below; the mechanism is already proven by 3(a).

## 4. Literature check (heavy web search) — the framing that wins

- **"Tracking + topology" alone is not novel** — e.g. *Cooperative Digital-Twin UAV Topology
  Optimization for Multi-Target Tracking* (2025), MA-MESAC, IADM-JCPS. Mature.
- **ISAC tracking with EKF + (P)CRB trajectory optimization is the most crowded sub-area** — *UAV-
  enabled ISAC: Tracking Design and Optimization* (2024), *SE(3)-Based Trajectory Optimization*
  (2025), *Integrated Sensing, Communication and Control for UAV Target Tracking* (2026). **This is
  exactly the "add a Kalman + CRB" route the handoff suggested — avoid it as the contribution.**
- **The defensible, less-crowded home is Age-of-Semantic-Information / Value-of-Information**:
  *Age of Semantic Information-Aware Wireless Transmission* (2025), *Semantics-Aware Source Coding in
  Status Update Systems* (2022), *From Information Freshness to Semantics of Information and
  Goal-oriented Communications* (2025). No prior work couples **target-motion freshness → the
  compression ratio of a knowledge-aggregation tournament** — that specific coupling is open.
- **The project's own proposal already points here:** audit **L8** lists "freshness, value of
  information" as a missing piece of the objective (targeted by Direction 1), and demotes the
  mobility/CRB Direction 4 to rank 4 *because the methodology is crowded.*

## 5. Recommended design (the single best way, in priority order)

**Spine:** *Energy-optimal semantic knowledge aggregation for swarms observing **moving** targets,
where target motion enters through the information's freshness/value — making compression
fidelity-and-freshness-aware (D1) and the aggregation topology value-prioritized (D2).*

1. **Freshness term in the fidelity model (D1).** Total source distortion =
   compression distortion (already implemented) **+ freshness distortion `a_F · agility · age`**,
   where `age` = number of hops the source has waited × round time. Implement as a *smooth* penalty in
   the objective (`min E + λ·D_root` with freshness inside `D_root`), not only the hard floor, to kill
   cliff effects. Targets (`wan/targets.py`) supply `agility` per source (speed / CV-prediction-error
   rate — both already computed). **Headline:** faster targets push η\* up and energy up, *measurably
   and monotonically*; the fidelity floor still holds (new PASS check).
2. **Value-prioritized topology (D2).** Add a per-source **value/decay weight** to the matching so the
   learned value model aggregates fast-decaying (fast-target) sources *sooner* (fewer hops = fresher
   at the root). **Headline:** value-aware topology beats value-blind under moving targets, and the gap
   *widens* with agility — a clean new figure.
3. **Predict-ahead vs react-only (keep targets.py's `CVPredictor`).** Now it has a *real* job: better
   prediction → fresher anchoring → lower freshness distortion. The CV→Kalman upgrade becomes a
   genuine ablation (small, defensible) instead of competing in the saturated ISAC-CRB lane.

**Honest scope note.** This *reframes* moving targets from a (failed) mobility-energy story into a
freshness/value story. The motion models in `wan/targets.py` are reused as-is; only the *coupling
channel* changes — from `c_i` (mobility) to `a_F·agility` (fidelity/value).

## 5b. IMPLEMENTED & VERIFIED (update)

The recommended design is now built and passing (`run_freshness.py`, 11/11):
- **D1:** freshness raises the derived η-floor and energy monotonically with
  target speed — exact (η 0.549→0.719, energy +9%) and robust over 30 random
  pairs. `wan/freshness.py`.
- **D2:** value-prioritized matching (`wan/freshtopo.py`) makes the root report
  **40% fresher at no energy cost**, reduces to the paper at λ_F=0, and its
  advantage grows with agility spread (21%→48%).
- **Prediction:** CV-Kalman + CV/CT IMM added to `wan/targets.py`; they dominate
  naive CV, and adaptive prediction yields a ~23% fresher report on moving
  targets.
- Full project still green: **51 automated checks, 0 fail** (15+11+9+5+11).
- Publication-grade writeup with equations + 2025–2026 references:
  `docs/FRESHNESS_EXTENSION.md`.

## 6. Files produced for this verdict

```
wan/network.py                         + backward-compatible target & freshness hooks (41/41 still pass)
experiments/exp_targets_washout.py     washout + r_track sweep + W1 figure
experiments/probe_mobility_share.py    mobility-vs-comm/compute energy share
experiments/probe_silent_violation.py  coverage constraint silently dropped under hard tracking
experiments/exp_value_coupling.py      full-mission freshness coupling (+ W2 figure)
experiments/exp_value_mechanism_clean.py  DETERMINISTIC clean ladder (+ W3 figure)  <-- the proof
figures/W1_washout_and_mechanism.png, W2_value_coupling.png, W3_value_mechanism_clean.png
```
