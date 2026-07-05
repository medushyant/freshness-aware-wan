# Freshness-Aware Semantic Aggregation for Swarms Tracking Moving Targets
## Complete Project Report — B.Tech Research Extension of ILAC-WAN (arXiv:2604.02381)

> **What this document is.** A single, self-contained, exhaustive record of the
> entire project — from the original research paper it extends, through every
> critique, experiment, design decision, line of implementation, measured
> result, the interactive website, the paper draft, and the public deployment.
> It is written so that **a person with no prior knowledge of the project can
> read this one file and understand everything**: what was done, why, how, with
> what tools, what is finished, and what remains. Nothing important is left to
> memory or to other files.
>
> **Author / owner:** Dushyant Kumar (GitHub: `medushyant`), B.Tech Information
> Technology, ABV-IIITM Gwalior.
> **Public repository:** https://github.com/medushyant/freshness-aware-wan
> **Report generated:** 2026-06-28. **Current commit:** `dd6f06e`.
> **Verification status:** **58 automated checks, 0 failures.**

---

## Table of Contents

1. [Executive summary (read this first)](#1-executive-summary)
2. [Part I — The origin: the base paper we extend](#part-i)
3. [Part II — The inherited state (what existed before this work)](#part-ii)
4. [Part III — The investigation (verifying, and the key negative result)](#part-iii)
5. [Part IV — The literature research (what is state-of-the-art in 2025–2026)](#part-iv)
6. [Part V — The decision (the single objective / spine)](#part-v)
7. [Part VI — The implementation (the new contribution, in detail)](#part-vi)
8. [Part VII — Every result and number (the 58 checks)](#part-vii)
9. [Part VIII — The interactive website](#part-viii)
10. [Part IX — The research-paper integration (LaTeX)](#part-ix)
11. [Part X — Deployment (GitHub + Vercel)](#part-x)
12. [Part XI — Complete file-by-file map of the repository](#part-xi)
13. [Part XII — Full technology stack](#part-xii)
14. [Part XIII — Chronological narrative of the whole engagement](#part-xiii)
15. [Part XIV — What is accomplished vs. what remains (future work)](#part-xiv)
16. [Part XV — Honest limitations and assumptions](#part-xv)
17. [Part XVI — How to reproduce everything (commands)](#part-xvi)
18. [Part XVII — Glossary of every technical term](#part-xvii)
19. [Appendix A — Exact parameter values](#appendix-a)
20. [Appendix B — The key equations](#appendix-b)

---

<a name="1-executive-summary"></a>
## 1. Executive Summary

**The one-line objective (the "spine").**
*Energy-optimal semantic knowledge aggregation for robot/drone swarms that
observe **moving** targets — where target motion enters the model not through
physical chasing (which we prove is inert) but through **information
freshness**, which drives both the semantic compression ratio and the
aggregation topology.*

**What the project is.** It is a research extension of a 2026 IEEE-style paper,
Zhao et al., *"Agentic AI-Empowered Wireless Agent Networks with Semantic-Aware
Collaboration via ILAC"* (arXiv:2604.02381). That paper describes a swarm of
drones that each sense an area, then merge all their data into one report using
minimum total energy, via a knockout-tournament aggregation. Our B.Tech project
(a) faithfully re-implements that paper, (b) fixes and strengthens several of its
weaknesses (two "Directions" of contribution), and (c) adds a genuinely novel
unifying idea — **freshness-aware aggregation for moving targets** — that the
professor asked for when he said the earlier work was "not novel / needs a clear
objective."

**The three pillars of the contribution.**
1. **Direction 1 — Fidelity-aware aggregation.** The paper never prices report
   quality, so it "proves" you should always compress data as hard as possible.
   We price quality (distortion) and show the optimal compression becomes
   *interior* — overturning the paper's headline lemma.
2. **Direction 2 — Learned predictive topology.** The paper steers the tournament
   with a fragile hand-tuned knob (ζ). We replace it with a *learned* value model
   (a small reinforcement-learning-style "cost-to-go" predictor) that needs no
   knob and lands far closer to the true optimum.
3. **The moving-target spine (freshness).** We make target motion matter through
   *information freshness*: a faster-moving target's data goes stale faster, so
   the swarm must compress it less (Direction 1) and deliver it on a shorter path
   to the root (Direction 2). This is the unifying objective.

**The single most important scientific finding.** The "obvious" way to add moving
targets — let each drone chase its target and pay motion energy — **does nothing**
in this model. We proved this with three experiments. The correct channel is
information value/freshness. This negative-then-positive result is the intellectual
core of the project.

**What was built.**
- A verified Python engine (`wan/`, ~1,742 lines) re-implementing the paper plus
  all three pillars.
- Seven runnable experiment scripts producing **58 automated PASS/FAIL checks**
  and **20 figures**.
- A polished, animated, **9-tab interactive website** (`web/`) that explains and
  visualizes every algorithm, driven by the real engine output.
- A drop-in **LaTeX section** for the research paper.
- A **public GitHub repository** and a one-click **Vercel** deployment config.

**Current status:** all 58 checks pass; the website is verified (0 console
errors) and pushed public; the paper section is written and syntax-checked.
Remaining optional work: real-data calibration (OPV2V on Colab), compiling the
LaTeX on Overleaf, and the permanent Vercel deploy (a 30-second user action).

---

<a name="part-i"></a>
## Part I — The Origin: The Base Paper We Extend

### I.1 Bibliographic anchor
- **Title:** *Agentic AI-Empowered Wireless Agent Networks With Semantic-Aware
  Collaboration via ILAC.*
- **Authors:** Zhouxiang Zhao, Jiaxiang Wang, Zhaohui Yang, Kun Yang, Zhaoyang
  Zhang, Mingzhe Chen, Kaibin Huang.
- **Identifier:** arXiv:2604.02381 (2026), 14 pages.
- **Field:** 6G wireless / Internet of Agents / semantic communication /
  integrated learning and communication (ILAC).

### I.2 The scenario, in plain language
Imagine `N` drones spread over a 500 m × 500 m area (e.g., surveillance, disaster
response, battlefield reconnaissance). Each drone:
- **patrols** a fixed circular region (a centre `c_i` and a radius `R_i` of
  roughly 80–100 m) and senses within it;
- carries a small onboard AI (an "embodied large model," ELM) that can
  **semantically compress** what it sensed (keep the meaning, drop the bits);
- can **move a little** within its region to get a better wireless link;
- must **transmit** its data to teammates.

The goal: merge everybody's data into **one combined report** delivered to a sink,
spending **minimum total energy**.

### I.3 The aggregation mechanism — a knockout tournament (called "H-MAP")
Data is merged over `K = ⌈log₂ N⌉` rounds:
- Each round, active drones **pair up**.
- In each pair, one drone **compresses** its payload and **transmits** it; the
  partner **fuses** both payloads; the sender **retires** (drops out).
- The active set halves each round until a single **root** drone holds everything.

Three energy costs are modelled: **mobility** (moving), **computation**
(the ELM compressing/fusing), and **communication** (transmitting; cost rises
sharply with distance).

### I.4 The two nested optimization problems
- **Inner problem (P1):** for a *fixed* pair, choose positions, motion times,
  transmit power, and compression ratios to minimize that pair's energy. Solved
  by **Block Coordinate Descent (BCD)** alternating a convex resource block and a
  Successive-Convex-Approximation (SCA) motion block.
- **Outer problem (P2):** choose *who pairs with whom* each round. Solved by
  **maximum-weight matching** using **Edmonds' Blossom algorithm**, where the edge
  weight is the pair energy plus a hand-tuned "look-ahead" potential field
  `Φ = ζ · ‖p − centroid‖^δ · L_next`.

The combined algorithm (inner BCD nested in outer matching) the paper calls
**H-MAP** (Hierarchical Matching-based Aggregation Protocol).

### I.5 The key equations (paper's own numbering; see Appendix B for our forms)
- **Coverage (Eq. 1):** `‖p_i − c_i‖ ≤ R_i`. **Crucially, `c_i` is FIXED** — this
  is the implicit *static-target* assumption we later relax.
- **Fusion (Eq. 2):** `L_j ← η_j·L_j + η_i·L_i` (payloads simply add).
- **Compute cost (Eq. 6):** `W = L·(C_base + C_gen·γ·(1−ρ)·ln(1/η))`.
- **Objective (Eq. 14):** minimize total energy. **Quality/distortion appears
  nowhere.**
- **Matching weight (Eq. 36):** `w_ij = E_ij + Φ_j` with ζ hand-tuned.

### I.6 The professor's feedback (the trigger for this project's direction)
After the first round of work (Directions 1 and 2 — see Part II), the professor's
verdict was: **"not novel / you need a clear objective."** That single sentence is
what motivated the moving-target freshness spine. The whole project is now
organized around answering it.

---

<a name="part-ii"></a>
## Part II — The Inherited State (What Existed Before This Engagement)

This project did not start from zero. A prior effort (documented in
`reference/PROJECT_HANDOFF.md`) had already produced a faithful re-implementation
plus two "Directions" of contribution and a set of rigor upgrades. This
engagement **verified all of it, then built the moving-target spine on top.**

### II.1 The 23-point limitation audit
The prior work produced a three-tier, equation-anchored audit of 23 weaknesses in
the base paper (5 of them internal inconsistencies). The ones that matter here:
- **L1/L2** — mobility serves only the radio; no sensing-quality model; retired
  drones abandon their zones. → targeted by the (crowded) ISAC "Direction 4."
- **L3** — additive fusion double-counts overlapping content. → Direction 1.
- **L8** — **"energy-only objective; no fidelity, freshness, or value of
  information."** → Direction 1 **and the freshness spine** (this is the exact
  gap our spine fills).
- **L9** — the hand-tuned, fragile ζ potential field. → Direction 2.
- **L14** — data is re-compressed at every hop (a "telephone game") but this
  compounding distortion is never tracked. → Direction 1.
- **L18** — the transmitted payload ignores what the receiver already knows,
  violating the **Wyner–Ziv** rate-distortion theorem (1976). → Direction 1.
- **L19** — the look-ahead Φ is evaluated *after* positions are optimized *without*
  it — an internal inconsistency. → Direction 2.
- **L23** — the required compression ratio `η_req` is an unexplained input. →
  Direction 1 derives it.

### II.2 Direction 1 — Fidelity-Aware Aggregation (already implemented)
Because the paper prices quality at zero, its inner solver concludes "always
compress maximally" (`η* = η_req`). Direction 1 added:
1. a **distortion state** `D` that compounds along each source's path (fixes L14);
2. **sub-additive fusion** (shared content counted once, fixes L3);
3. a **Wyner–Ziv link**: `D_out = η_i·L_i·(1 − ω·ρ)` — don't resend what the
   receiver knows (fixes L18);
4. a **fidelity-aware objective** `min E + λ·D` (fixes L8) — which makes the
   optimum **interior** (`η* < η_req`), overturning the paper's lemma;
5. a **derived `η_req`** (fixes L23);
6. a **conformal guarantee**: `Pr(D_root ≤ D_max) ≥ 1 − α`, distribution-free.

### II.3 Direction 2 — Learned Predictive Topology (already implemented)
Replaces the fragile hand-tuned ζ with a learned "cost-to-go":
- **Stage A** — move Φ *inside* the inner problem (fixes L19).
- **Stage B** — auto-scale the trade-off once (Lyapunov drift-plus-penalty), no
  knob.
- **Stage C** — a **ridge-regression value model** `V̂(state)` on 7 hand features
  predicts remaining mission energy; matching weight = `E + V̂(state-after)`.
- **Stage D** — decision-focused fine-tuning of the value weights through the
  exact Blossom matcher (validation-gated).
- Plus a **flexible schedule** (a pair may rest a round) and a **brute-force
  oracle** for exact optimality-gap numbers.

### II.4 The standalone "targets" module (the loose end)
A `wan/targets.py` module existed with three motion models — **static**,
**dynamic** (nearly-constant-velocity), and **time-varying** (maneuvering:
coordinated turns) — plus a constant-velocity predictor. But it was **not wired
into the missions**; it produced a "tracking energy grows with agility" figure
only by using a *different geometry* than the paper's model. Making this
meaningful is exactly what this engagement did.

---

<a name="part-iii"></a>
## Part III — The Investigation (Verification, and the Decisive Negative Result)

### III.1 Step 1 — Rebuild and verify everything (ground truth, not trust)
Rather than trust the handoff, the entire codebase was rebuilt into a clean folder
(`~/Documents/BTP_ILAC_WAN`) with a fresh Python virtual environment
(numpy 2.5, scipy 1.18, matplotlib 3.11, networkx 3.6) and **all verifiers were
re-run**: Direction 1 (15 checks), Direction 2 (11), upgrades (9), targets (5) —
**all passed** on the new machine, with numbers matching the handoff within
seed/version noise. This confirmed the inherited work was real and reproducible.

### III.2 Step 2 — Read the paper directly to confirm the static-target assumption
The base paper's own §III fixes `c_i` as a **constant** patrol-region centre; the
"dynamic" in its title refers to the *topology tournament*, not to target motion.
So the static-target assumption the whole spine relaxes is genuinely there.

### III.3 Step 3 — Test the "obvious" moving-target integration (and watch it fail)
The handoff proposed: each round, re-anchor every drone's patrol centre `c_i` on
its (predicted) target, and let the drone chase it. This was implemented as a
backward-compatible hook in `run_mission` (default off, so all 41 original checks
still pass). Then three experiments were run:

**Experiment A — Washout (`experiments/exp_targets_washout.py`).**
With the paper's patrol radius (R = 80–100 m), a target moving 3–7 m/s travels
only ~30–41 m per round — a small fraction of R. So the coverage constraint never
binds, and **target class is invisible**: the apparent energy differences across
static/dynamic/maneuvering were non-monotone and noise-dominated (standard
deviation *larger* than the differences; e.g. dynamic 0.83 J appeared *higher*
than time-varying 0.60 J — backwards).

**Experiment B — Tightening the sensing radius doesn't help.**
Sweeping a decoupled sensing radius `r_track` down to 8 m made the *geometric*
tracking demand explode to **40 m/round**, yet realized energy stayed **0.597 J,
identical to static (0.601 J)**. The solver/matcher simply routes *around* the
constraint (it only enforces coverage during an optional motion step).

**Experiment C — Why (`probe_mobility_share.py`, `probe_silent_violation.py`).**
A decomposition showed mobility is a near-discontinuous term (0 J if you don't
move, ≥1.2 J baseline if you do) used only to fine-tune the radio; in a full
mission the energy-minimizing matcher avoids triggering it. **Conclusion: the
mobility/coverage channel is the wrong place to couple targets.**

**This aligns with the project's own audit:** the mobility/sensing route is
"Direction 4 (ISAC)," which the proposal had already ranked *last of five* because
"CRB-trajectory methodology is crowded."

---

<a name="part-iv"></a>
## Part IV — The Literature Research (2025–2026 State of the Art)

Extensive web research was done at every decision point. Key findings:

1. **Swarm multi-target tracking + topology is mature.** e.g. *Cooperative
   Digital-Twin UAV Topology Optimization for Multi-Target Tracking* (2025). So
   "tracking + topology" alone is not novel.
2. **ISAC tracking with EKF + Cramér-Rao-bound trajectory optimization is the most
   saturated sub-area** (many 2024–2026 papers). This is exactly the "add a Kalman
   filter + CRB" upgrade the handoff suggested — a lane a B.Tech cannot win in.
3. **The principled, less-crowded home is Age-of-Information / value-of-information:**
   - Sun et al., *Update or Wait: How to Keep Your Data Fresh* (IEEE Trans. IT,
     2017) — the estimation error of a monitored source is a **non-decreasing
     function of the age** of its last update.
   - Maatouk et al., *The Age of Incorrect Information* (2020) — general
     non-decreasing staleness penalties.
   - *Age of Semantic Information-Aware Wireless Transmission* (arXiv:2508.12248,
     2025); *From Information Freshness to Semantics of Information and
     Goal-oriented Communications* (arXiv:2512.12758, 2025); *Version AoI in
     Multi-Hop Networks* (arXiv:2507.23433).
4. **The exact coupling we propose is an open gap:** no prior work couples
   *target-motion freshness* to the *compression ratio of a semantic
   aggregation tournament*. The nearest neighbors are task-oriented compression
   over bit-budgeted channels — about *what* to compress, not about freshness
   driving the ratio.
5. **Metaheuristics as matchers:** the literature is consistent that
   nature-inspired metaheuristics only beat exact methods on large instances;
   on small instances (exact Blossom is polynomial) they tie at best. This shaped
   how we used them (as honest baselines, not the main method).
6. **Web-visualization tooling (2026):** D3 + GSAP ScrollTrigger + Scrollama is
   the modern scrollytelling stack; Three.js is the 3D standard; CSS 3D transforms
   give hardware-accelerated depth without WebGL overhead. This shaped the website.

---

<a name="part-v"></a>
## Part V — The Decision (The Single Objective / Spine)

**Verdict.** The handoff's mobility-driven target integration is **not worth
implementing** — it is empirically inert *and* in the most crowded research lane.
The **information-value / freshness coupling is worth it** and is the spine,
because it:
- is **on-model** (drives the paper's own compression ratio η and matcher);
- is **novel** (an open gap per the literature scan);
- fills the project's own audit item **L8**;
- avoids the saturated ISAC/CRB lane;
- **unifies** Direction 1 (freshness-aware compression) and Direction 2
  (value-prioritized topology) under one objective — exactly what the professor
  asked for.

This decision was recorded in `docs/TARGET_INTEGRATION_VERDICT.md` and approved
by the user before implementation.

---

<a name="part-vi"></a>
## Part VI — The Implementation (The New Contribution, in Detail)

### VI.1 The freshness model (`wan/freshness.py`, 70 lines)
The physical principle (Age-of-Information / remote estimation): a moving target's
data goes stale with age. Each **source** `s` (the data collected by drone `s`)
carries an **agility** `a_s` = its target's mean speed. The **freshness
distortion** it accrues by the time it reaches the root after age `τ_s` is:

```
D_fresh(s) = a_F · a_s · (τ_s)^p ,      p ≥ 1   (p=1 is the linear AoI form)
```

where `τ_s = hops_s · T` (each fuse-compress-relay stage is one processing-age
step — the multi-hop / version-AoI model). The assembled **root distortion**
adds this to the Direction-1 compression distortion:

```
D_root = a_D · Σ_s w_s · Σ_hops ln(1/η)   +   Σ_s w_s · D_fresh(s)
         └────── compression (Direction 1) ─────┘   └── freshness (new) ──┘
```
with `w_s` = source `s`'s share of the bits. This one equation couples target
motion to **two controls**: the compression ratio η (Direction 1) and the
per-source hop count (Direction 2 topology).

### VI.2 Direction 1 coupling — compression bends to motion
Through Direction 1's *derived per-hop η-floor* (the endogenous replacement for
the paper's exogenous `η_req`), a faster target consumes more of the fidelity
budget `D_max` via `D_fresh`, so the floor **rises** → the drone must compress
**less** → it sends more bits → it spends more energy. This is implemented as a
backward-compatible `src_distortion0` hook in `run_mission`. **Result (exact,
deterministic):** as target speed goes 0 → 9 m/s, the optimal η rises
**0.549 → 0.719** and pair energy rises **+9%**, monotonically. This *sharpens*
the Direction-1 headline: the paper's "always compress maximally" fails by an
amount set by *target motion*.

### VI.3 Direction 2 coupling — a fresher report, for free (`wan/freshtopo.py`, 115 lines)
Because `D_root` depends on `hops_s`, the matching can lower it by giving
fast-decaying sources **shorter paths**. Same inner solver, same Blossom matcher —
only the weight changes:
```
w(i→j) = E(i→j) + λ_F · a_F · T · Σ_{s in i} bits_s · a_s
```
(penalize making the holder of fast-decaying data a *sender*, so it becomes a
*receiver* and its data reaches the root in fewer hops). **Result (20 missions):**
the root report is **40% fresher (staleness 11.9 → 7.1) at no energy cost
(−1.5%)**, and it reduces **exactly** to the paper at `λ_F = 0` (20/20 identical,
a sanity check). The advantage **grows with agility spread** (21% → 48%).

### VI.4 Prediction — CV, Kalman, IMM (`wan/targets.py`, 228 lines)
The agility that drives freshness is really the *predictor's residual error rate*,
so a better predictor lowers effective decay. We added:
- `CVPredictor` — naive constant-velocity finite-difference (baseline, existed).
- `KalmanCV` — a constant-velocity **Kalman filter** (rejects measurement noise).
- `IMMPredictor` — an **Interacting-Multiple-Model** predictor mixing a
  constant-velocity and a coordinated-turn model (the established best practice for
  maneuvering targets).
**Result:** Kalman/IMM dominate the naive CV predictor on every target class;
predicting ahead pays once motion exceeds sensor noise (best 7.5 m error vs
react-only 9.7 m on constant-velocity targets), yielding a ~23% fresher report.
Honest scope: for near-static targets, *not* predicting is best — so the optimal
agent picks the lowest-residual predictor.

### VI.5 Metaheuristic comparison baselines (`wan/metaheuristics.py`, 163 lines)
The mentor asked for ABC/ACO/Cuckoo/Egyptian-Vulture. Because the per-round
matching is solved *exactly* by Blossom in polynomial time, these cannot beat it —
so they are implemented **only as honest comparison baselines**, each as an
alternative matcher over the pairing problem:
- **ACO** (Ant Colony Optimization) — pheromone-guided matching construction.
- **ABC** (Artificial Bee Colony) — employed/onlooker/scout perturbation search.
- **Cuckoo Search** — Lévy-flight-style neighbor perturbation with nest
  abandonment.
- **Egyptian Vulture** — rolling/tossing pebble operators with occasional accept.
**Result:** they approach but never beat exact Blossom (a 0.1–0.3% gap that grows
with N, while Blossom stays optimal and fast — 12 ms at N=30); at mission level
they cluster at the myopic per-round level, and only the *learned lookahead*
closes the gap. This proves the contribution is the **learning**, not the
metaheuristic — the defensible conclusion.

---

<a name="part-vii"></a>
## Part VII — Every Result and Number (the 58 Automated Checks)

Every headline claim is an automated PASS/FAIL check that regenerates from code in
minutes. Totals: **17 + 11 + 9 + 5 + 11 + 5 = 58 checks, 0 failures.**

### Direction 1 — `run_direction1.py` → `results.txt` (15 checks + 2 scaling = 17)
- **E1** reproduction sanity: inner solver converges; benchmark ordering matches
  the paper (proposed ≤ no-motion, proposed < max-power, no-semantic infeasible
  under stress, Blossom ≤ random).
- **E2** interior optimum: `η*` sweeps **[0.16, 0.72]** as λ rises (overturns
  `η*=η_req=0.60`).
- **E3** Pareto frontier: reference point **(D=2.21, E=0.61)** is dominated on
  both axes AND violates the floor `D_max=1.2`.
- **E4** Wyner–Ziv: never worse, ~5% pair-energy saving at high overlap, rescues
  an infeasible deadline.
- **E5** conformal: coverage **0.97** at α=0.20 (λ*=0.30) and α=0.10 (≥ 1−α both).
- **E6** derived cap: with λ=0 the per-hop bound keeps D = {0.40, 0.80, 1.08} ≤
  D_max.
- **F1b** (via `run_repro_scaling.py`): energy grows 4.91 → 9.17 → 12.81 → 16.92 J
  over N = 4, 6, 8, 10.

### Direction 2 — `run_direction2.py` → `results_d2.txt` (11 checks)
- **T0** value model R² = **0.90**.
- **T1** ζ fragility ratio **1.10** (paper is fragile); learned **8.29 J** beats
  paper's best hand-tuned **8.81 J**; flexible schedule **8.03 J**.
- **T2** optimality gap vs brute-force oracle (N=5): **learned 0.5%, paper 13.3%,
  random 24.4%.**
- **T3** size transfer: N=8 learned **12.36** vs paper 12.81; N=10 learned
  **15.88** vs paper 16.92.
- **T4** flexible ≤ forced (**8.085 vs 8.428** over 8 scenarios).

### Rigor upgrades — `run_upgrades.py` → `results_upgrades.txt` (9 checks)
- **U1** Pareto dominance holds with 8 scenarios + error bars (ref D=2.16,
  E=0.594).
- **U2/U3** ζ fragility persists; Stage C (no knob) ≤ best hand-tuned; **Stage D
  7.75 J** on held-out vs C+flex 8.09 and paper's best 8.81 (**−12%**),
  validation-gated.
- **U4** battery fairness: worst single agent **3.09 → 2.66 → 2.49 J**.
- **U5** derived bound tracks any floor (D_max 0.9/1.2/1.5 → worst D
  0.90/1.08/1.26); Wyner–Ziv saving positive across ω.

### Targets — `run_targets.py` → `results_targets.txt` (5 checks)
- **G1** motion path length 0/52/52 m; prediction error 0.0/2.0/6.4 m
  (static/dynamic/time-varying).
- **G2** standalone tracking energy 0/4.05/6.66 J; predict-ahead saves 28% on the
  maneuvering class.
- **G3** trajectory snapshot saved.

### Freshness spine — `run_freshness.py` → `results_freshness.txt` (11 checks)
- **F1** exact mechanism: η-floor **0.549 → 0.719**, energy **+9%**, monotone.
- **F2** robust across 30 random pairs: mean η 0.549 < 0.628 < 0.719; energy rises.
- **F3** value topology: reduces to paper at λ_F=0 (20/20 identical); **40%
  fresher**; **no energy cost** (−1.5%).
- **F4** advantage grows with agility spread: **21% → 48%**.
- **F5** Kalman/IMM dominate naive CV on every class; prediction pays on
  constant-velocity targets (7.48 < 9.74 m); adaptive prediction yields a fresher
  report (staleness 12.39 ≤ 16.14).

### Metaheuristic baselines — `run_metaheuristics.py` → `results_metaheuristics.txt` (5 checks)
- **M1** no metaheuristic beats exact Blossom at any N (sanity); all stay far
  below random; Blossom stays fast (**12 ms at N=30**).
- **M2** mission optimality gap (N=5): **learned 3.6%**, Blossom/ACO/ABC/Cuckoo
  ~14.7%, Egyptian-Vulture 12.6%, random 29.1%. (This run uses a value model
  trained on the stress regime with fewer seeds, hence 3.6% vs the 0.5% in T2 —
  both are real for their respective configurations.)

### The 20 figures (all in `figures/`, all regenerable)
`E1_reproduction`, `E2_interior_optimum`, `E3_pareto_frontier`, `E4_wz_savings`,
`E5_conformal`, `F1b_energy_vs_N`, `T1_zeta_robustness`, `T2_optimality_gap`,
`T3_size_transfer`, `U4_fairness`, `G2_tracking_energy`, `G3_target_paths`,
`G4_freshness_compression`, `G5_value_topology`, `G6_spread`,
`M1_matching_quality`, `M2_optimality_gap`, `W1_washout_and_mechanism`,
`W2_value_coupling`, `W3_value_mechanism_clean`.

---

<a name="part-viii"></a>
## Part VIII — The Interactive Website (`web/`)

A polished, animated, **9-tab single-page application** that explains and
visualizes the whole project, driven entirely by real engine output
(`web/data.json`, produced by `export_web_data.py`). No build step — pure static
files. Verified headless across every tab with **0 console errors** (via a
Playwright screenshot harness, since the sandbox's preview tool was blocked by an
unrelated venv conflict).

### VIII.1 The nine tabs and exactly what each shows
1. **Overview** — the spine sentence, three animated headline numbers (58 checks ·
   40% fresher · +9% energy), a **Three.js 3D swarm** you can drag to orbit (teal
   = drones, violet = fast targets, lines = comm links), and navigation cards.
2. **Swarm** — a live **HTML5-canvas simulation**: N drones patrolling disks and
   re-anchoring on moving targets, with comm links and sensing beams. Controls:
   motion class (static/dynamic/maneuvering), drone count, speed, pause/reset.
3. **H-MAP** — the knockout tournament as an animated **left-to-right bracket**
   (agents → round 1 → round 2 → root). Payload rings shrink on compression,
   packets fly on transmission, rings grow on fusion, senders retire. **Paper vs
   Ours** toggle drives a report-quality meter: Paper ends *over* the fidelity
   floor (red, D=2.21, 8.55 J); Ours ends *under* it (green, D=0.95, 4.66 J). It
   replays the **real pairing trace** from the engine.
4. **Blossom** — an interactive max-weight matching demo: nodes on a ring with
   candidate-pair edge weights; "optimal (Blossom)" vs greedy vs random, with a
   total-energy table.
5. **Learned · RL** — a live **ζ-fragility slider** (drag the paper's knob and
   watch its energy wobble vs the flat learned line), plus the optimality-gap bars.
6. **Freshness** — the spine: the inert-mobility negative result, then the D1
   compression chart (η & energy vs speed), the D2 value-topology before/after
   (40% fresher), the spread curve, and the prediction ladder.
7. **Graphs** — a gallery of **all 20 figures**, grouped by contribution, each
   captioned with "what it proves," with a click-to-zoom lightbox.
8. **Verify** — the 58-check breakdown by module, plus the Pareto and washout
   charts.
9. **Before / After** — animated paper-vs-ours bar cards, then the full
   comparison table across every dimension.

### VIII.2 Website technology
- **Vanilla JavaScript** (no framework, no build): `app.js` (routing, charts,
  data), `sim.js` (swarm canvas engine), `algos.js` (H-MAP + Blossom SVG
  animations), `hero3d.js` (Three.js scene).
- **Chart.js 4.4.1** — all data charts.
- **GSAP 3.12.5** — timeline animations (the H-MAP bracket, reveals).
- **Three.js 0.158.0** — the 3D hero swarm.
- **Custom canvas + SVG** — the bespoke swarm and algorithm visuals.
- Design: dark theme, refined glassmorphism, gradient-hairline navbar with a
  pulsing brand dot, CSS 3D card tilt, a grain-texture overlay, custom
  scrollbars — following 2026 UI research.

### VIII.3 Honesty rule
All charts and headline numbers are **real engine output**. The swarm motion and
the H-MAP payload-ring/quality animations are **illustrative of the mechanics**;
the H-MAP *pairing trace* and the energy/quality endpoints are real. This split is
documented in `web/HOW_IT_WORKS.md`.

---

<a name="part-ix"></a>
## Part IX — The Research-Paper Integration (LaTeX)

For the written report / IEEE submission, an **Overleaf-ready drop-in** was
produced in `docs/`:
- `freshness_section.tex` — a full `\section` in the proposal's own macro style
  (uses its `\Dir`, `\Ltag`, `limitbox`, `ideabox`, `goldbox` environments and
  colour palette), with the equations, the negative result, both couplings, the
  algorithm, the results, and honest limitations. Syntax-checked (balanced
  environments/braces, even math delimiters).
- `freshness_refs.bib` — the 4 new references (Sun 2017, AoII 2020, AoSI 2025,
  goal-oriented 2025).
- `freshness_intro_patch.tex` — the spine sentence to lead the Executive Summary
  and Introduction.
The full plain-language version with all numbers is `docs/FRESHNESS_EXTENSION.md`.
(No LaTeX toolchain exists locally, so this compiles on Overleaf where the
proposal lives.)

---

<a name="part-x"></a>
## Part X — Deployment (GitHub + Vercel)

- **GitHub (public):** https://github.com/medushyant/freshness-aware-wan
  - Single clean commit `dd6f06e`, authored solely by `medushyant`.
  - `.gitignore` excludes the venv, caches, transient screenshots, and the
    copyrighted base-paper PDF (the user's own proposal + handoff are included).
  - The repo was created via the GitHub API using the user's stored keychain token
    (no `gh` CLI available), pushed over HTTPS, then flipped to **public**.
- **Vercel (one-click):** a root `vercel.json` (`outputDirectory: web`, no build)
  makes the import zero-config. Steps for the user: vercel.com/new → import the
  repo → Deploy → permanent `*.vercel.app` link.
- **Instant tunnel (used earlier):** an SSH reverse tunnel
  (`ssh -R 80:localhost:5050 nokey@localhost.run` → a `*.lhr.life` URL) gave an
  immediate shareable link with no account, but it is **ephemeral** (dies with the
  session); Vercel is the permanent path.

---

<a name="part-xi"></a>
## Part XI — Complete File-by-File Map of the Repository

### The engine — `wan/` (1,742 lines total)
| File | Lines | Purpose |
|---|---|---|
| `model.py` | 141 | Parameters (Table-I + chosen constants), `make_agents`, geometry (Jaccard of disk unions), energy/time/channel functions. |
| `solver.py` | 303 | The pairwise inner BCD solver — paper mode (η pinned) and fidelity mode (both η free, + λ·distortion, + Φ-in-loop, + Wyner–Ziv). |
| `network.py` | 292 | Full Direction-1 missions (`run_mission`) with the telephone-game distortion ledger, derived η-floor, conformal calibration, and the **target/freshness hooks** added this engagement. |
| `topology.py` | 416 | Direction 2 — policies (paper/stageA/lyapunov/learned/greedy/random), the ridge `ValueModel`, flexible schedule, brute-force `dp_oracle`, Stage-D fine-tune, and the **metaheuristic `matcher` hook**. |
| `targets.py` | 228 | Motion models (static/dynamic/time-varying) + predictors (`CVPredictor`, and the new `KalmanCV`, `IMMPredictor`). |
| `freshness.py` | 70 | **NEW.** The freshness-distortion model and root-freshness computation. |
| `freshtopo.py` | 115 | **NEW.** The value-prioritized aggregation topology (energy-only vs freshness-aware matching). |
| `metaheuristics.py` | 163 | **NEW.** ACO / ABC / Cuckoo / Egyptian-Vulture matchers as comparison baselines. |
| `style.py` | 14 | Shared matplotlib style. |

### The experiment runners (root)
| File | Lines | Produces |
|---|---|---|
| `run_direction1.py` | 262 | E1–E6 figures + `results.txt` (15 checks). |
| `run_direction2.py` | 174 | T0–T4 figures + `results_d2.txt` (11 checks). |
| `run_repro_scaling.py` | 46 | The F1b energy-vs-N figure (appends 2 checks to `results.txt`). |
| `run_upgrades.py` | 203 | Error-bar figures, Stage D, fairness + `results_upgrades.txt` (9 checks). |
| `run_targets.py` | 158 | G1–G3 figures + `results_targets.txt` (5 checks). |
| `run_freshness.py` | 231 | **NEW.** G4–G6 figures + `results_freshness.txt` (11 checks). |
| `run_metaheuristics.py` | 157 | **NEW.** M1–M2 figures + `results_metaheuristics.txt` (5 checks). |
| `export_web_data.py` | 139 | **NEW.** Dumps `web/data.json` (all real numbers + the H-MAP trace). |
| `opv2v_colab.py` | — | Real-data grounding, run separately on Google Colab (not yet run). |

### The decisive experiments — `experiments/`
| File | Lines | Shows |
|---|---|---|
| `exp_targets_washout.py` | 151 | The washout + r_track sweep (the negative result). |
| `probe_mobility_share.py` | 32 | Mobility vs comm/compute energy share. |
| `probe_silent_violation.py` | 37 | The solver silently dropping coverage under tight tracking. |
| `exp_value_coupling.py` | 133 | Full-mission freshness coupling (paired). |
| `exp_value_mechanism_clean.py` | 84 | The clean deterministic freshness mechanism. |

### The website — `web/`
`index.html` (196), `css/style.css` (229), `js/app.js` (186), `js/algos.js`
(165), `js/sim.js` (91), `js/hero3d.js` (46), `data.json` (real numbers +
trace), `figures/` (20 PNGs), `HOW_IT_WORKS.md`, `DEPLOY.md`.

### Docs & reference
`docs/` — `FRESHNESS_EXTENSION.md`, `TARGET_INTEGRATION_VERDICT.md`,
`freshness_section.tex`, `freshness_refs.bib`, `freshness_intro_patch.tex`,
`site_preview/` (9 screenshots).
`reference/` — the base paper PDF (git-ignored), extracted text (git-ignored),
the v2 proposal (PDF + TeX), and `PROJECT_HANDOFF.md`.
Root — `README.md`, `PROJECT_REPORT.md` (this file), `vercel.json`, `.gitignore`,
`scripts/serve.sh`, `scripts/shoot.py`, six `results_*.txt`.

---

<a name="part-xii"></a>
## Part XII — Full Technology Stack

- **Language / core:** Python 3.14, numpy 2.5, scipy 1.18 (BCD/SCA via
  `minimize`/`minimize_scalar`, SLSQP/L-BFGS-B — deliberately no CVXPY).
- **Graph / matching:** networkx 3.6 (Blossom `max_weight_matching`).
- **Plots:** matplotlib 3.11.
- **ML:** closed-form ridge regression value model (transparent, no heavy NN);
  zeroth-order decision-focused fine-tuning; Kalman & IMM filters (numpy).
- **Verification:** custom PASS/FAIL harness; Playwright + headless Chromium for
  the website (screenshots + console-error checks).
- **Web:** vanilla JS, Chart.js 4.4.1, GSAP 3.12.5, Three.js 0.158.0, custom
  canvas/SVG; Google Fonts (Sora / Inter / JetBrains Mono).
- **Ops:** git, GitHub API, Vercel (static), SSH reverse tunnel (localhost.run).

---

<a name="part-xiii"></a>
## Part XIII — Chronological Narrative of the Whole Engagement

1. **Received** the handoff, the base-paper PDF, and the code archive.
2. **Rebuilt** the project into a clean folder + venv; **re-ran all 41 inherited
   checks** — all passed.
3. **Read the base paper** directly to confirm the static-target assumption.
4. **Implemented and tested** the handoff's mobility-driven target integration —
   and **proved it inert** with three experiments (washout, mobility-share,
   silent-violation).
5. **Researched** the 2025–2026 literature; confirmed the ISAC/CRB lane is
   saturated and the freshness/value angle is an open gap.
6. **Decided** the spine: freshness-driven coupling; recorded the verdict; got
   user approval.
7. **Implemented** the spine: `freshness.py`, `freshtopo.py`, the D1 hook, the D2
   value topology, and Kalman/IMM predictors → `run_freshness.py` (11 checks).
8. **Implemented** the mentor's metaheuristic baselines as honest comparisons →
   `run_metaheuristics.py` (5 checks). Total now 58 checks.
9. **Wrote** the publication-grade extension (`docs/FRESHNESS_EXTENSION.md`) and
   the LaTeX drop-in.
10. **Built** the interactive website (first as scrollytelling, then rebuilt as a
    9-tab app), added a Three.js 3D hero, a live swarm simulation, animated H-MAP
    and Blossom, a graphs gallery, and a visual before/after.
11. **Fixed** issues found by the user (H-MAP "Ours" mode blank bug; footer
    alignment; cropped gallery figures; navbar polish).
12. **Deployed:** pushed the whole project to a **public GitHub repo**, rewrote
    history to a single commit authored solely by the user, added `vercel.json`
    for one-click deploy, and provided an instant tunnel link for immediate
    sharing.
13. **Wrote** this report.

---

<a name="part-xiv"></a>
## Part XIV — What Is Accomplished vs. What Remains

### Accomplished ✅
- Faithful re-implementation of the base paper (verified).
- Direction 1 (fidelity-aware aggregation) — verified, overturns the paper's
  lemma.
- Direction 2 (learned predictive topology) — verified, near-optimal, no knob.
- Rigor upgrades (error bars, Stage D, fairness, sensitivity) — verified.
- **The moving-target freshness spine (the novelty)** — verified: D1 compression
  coupling, D2 value-prioritized topology, Kalman/IMM prediction.
- Metaheuristic comparison baselines — verified.
- 58 automated checks, 20 figures, all regenerable.
- A polished 9-tab interactive website — verified, public.
- The LaTeX paper section + references.
- Public GitHub repo + one-click Vercel config.

### Remaining / future work 🔭
1. **Permanent deploy:** import the repo into Vercel (a ~30-second user action).
2. **Real-data calibration:** run `opv2v_colab.py` on Google Colab against the
   OPV2V collaborative-perception dataset to replace assumed `ρ` and distortion
   constants with measured ones (a multi-GB download, Colab only).
3. **Compile the LaTeX** on Overleaf and slot the freshness section into the
   proposal.
4. **Deeper models (optional):** a compact spatio-temporal GNN value encoder for
   size transfer to N = 50–500; conformal prediction intervals on the freshness
   forecast; a full IMM ablation as a headline (currently a supporting result).
5. **Causal / displacement analysis (optional):** difference-in-differences on
   enforcement/aggregation changes; a trust/Byzantine-resilience layer
   (audit L13).
6. **Enhancement ideas for the "4-month project" bar:** a live what-if simulator
   on the website; an economic translation (energy → ₹/person-hours); a scaling
   study to N = 200 with the metaheuristic matcher as the approximate fallback.

---

<a name="part-xv"></a>
## Part XV — Honest Limitations and Assumptions (keep these in the report — they read as rigor)

- **Stress regime.** Energy only responds to topology/compression choices in a
  "stress regime" (Tmax=2.5 s, larger payloads, harsher channel, faster SoC,
  lighter generative cost). With the paper's lazy Table-I radio, communication is
  nearly free and *nothing* moves energy. This regime is documented, not hidden,
  and is the same one the Direction-2 topology results already use.
- **Chosen constants.** Several constants (noise power, mobility coefficients,
  C_base/C_gen, η_min, a_D, D_max, ω, a_F, λ_F, penalty exponent p) are not in the
  paper's Table I and were chosen and disclosed. Conclusions are qualitatively
  invariant to them (e.g. the D2 freshness cut saturates for λ_F ≥ 1e-7).
- **Solver substitution.** The paper uses CVX-style convex blocks; we re-did the
  same math with scipy (lighter stack, identical problems).
- **Freshness is a processing-age (per-hop) model.** A wall-clock re-sensing model
  is an alternative we note but do not need.
- **Prediction is regime-dependent.** One-step prediction pays only when target
  motion exceeds sensor noise; for near-static targets, not predicting is best.
- **The website's motion/ring animations are illustrative;** the charts, numbers,
  and the H-MAP pairing trace are real.
- **Metaheuristics are baselines, not the method** — exact Blossom is optimal and
  fast at this scale.
- **Two optimality-gap numbers exist** (0.5% in T2, 3.6% in M2) because they use
  different value-model training configurations; both are real for their configs.

---

<a name="part-xvi"></a>
## Part XVI — How to Reproduce Everything (commands)

```bash
# 1. Set up (once)
cd ~/Documents/BTP_ILAC_WAN
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib networkx

# 2. Regenerate every result + figure (each prints PASS/FAIL)
python run_direction1.py       # 15 checks + figures E1-E6      -> results.txt
python run_direction2.py       # 11 checks + figures T0-T4      -> results_d2.txt
python run_repro_scaling.py    # F1b energy-vs-N (appends to results.txt)
python run_upgrades.py         #  9 checks + error-bar figures  -> results_upgrades.txt
python run_targets.py          #  5 checks + figures G1-G3      -> results_targets.txt
python run_freshness.py        # 11 checks + figures G4-G6      -> results_freshness.txt  [NEW]
python run_metaheuristics.py   #  5 checks + figures M1-M2      -> results_metaheuristics.txt [NEW]
python export_web_data.py      # refresh web/data.json for the site

# 3. Run the website locally
cd web && python3 -m http.server 5050   # open http://localhost:5050

# 4. (optional) Verify the site headless
pip install playwright && python -m playwright install chromium
python scripts/shoot.py        # screenshots every tab, reports console errors
```

---

<a name="part-xvii"></a>
## Part XVII — Glossary (for a reader with zero background)

- **Swarm / agent / drone:** one mobile robot with sensing, compute, and a radio.
- **Semantic compression (η):** keeping the *meaning* of data while dropping bits;
  η ∈ (0,1] is the fraction of bits kept (η=0.2 keeps 20%).
- **Aggregation / fusion:** merging two agents' data into one payload.
- **H-MAP / knockout tournament:** the round-by-round pairing that merges all data
  into one root agent.
- **Blossom algorithm:** Edmonds' polynomial-time algorithm for maximum-weight
  matching — used to choose the best pairs each round.
- **BCD / SCA:** Block Coordinate Descent / Successive Convex Approximation — the
  numerical method that solves each pair's sub-problem.
- **Distortion (D):** how far the report is from perfect; lower is better.
- **Fidelity floor (D_max):** the maximum distortion the report is allowed.
- **Wyner–Ziv coding:** an information-theory result — don't transmit what the
  receiver already knows.
- **Conformal prediction:** a distribution-free way to *guarantee* a bound holds
  with a chosen probability.
- **Value model / cost-to-go (V̂):** a learned function predicting the remaining
  energy of a mission from its current state — the reinforcement-learning-style
  core of Direction 2.
- **ζ (zeta):** the paper's hand-tuned knob for its look-ahead term; fragile.
- **Age of Information (AoI):** how stale a piece of data is; the older, the worse
  the estimate of a moving target.
- **Freshness distortion (D_fresh):** the penalty we add for stale data; grows
  with target speed and with how many hops the data waited.
- **Kalman filter / IMM:** standard trackers that predict a moving target's next
  position; IMM (Interacting Multiple Model) blends several motion models for
  maneuvering targets.
- **Metaheuristic (ACO/ABC/Cuckoo/Egyptian-Vulture):** nature-inspired
  approximate optimizers — used here only as comparison baselines.
- **OPV2V:** a real collaborative-perception (self-driving) dataset for grounding
  the model's assumptions.

---

<a name="appendix-a"></a>
## Appendix A — Exact Parameter Values

**From the paper's Table I:** N (≤10; we use 6 for speed), area 500 m, R_cov
80–100 m, L_init 5–10 Mb, Tmax 6 s, B 1 MHz, β₀ 1e-4 (−40 dB), δ 3.0, Pmax 1 W,
vmax 5 m/s, f_cpu 1e9 FLOP/s, τ 1e-28, ζ 1e-14.

**Chosen (disclosed):** N0B 3.98e-15, P_static 0.2 W, k1 0.05 J/m, k2 0.01
J·s/m², C_base 60, C_gen 350, γ 1.0, η_min 0.05, η_req 0.60, d_floor 1 m, a_D 1.0,
D_max 1.2, ω_wz 0.9. **Freshness:** a_F ≈ 0.045–0.06, λ_F ≈ 1e-6, p ≥ 1.

**Stress regime override (topology + freshness energy experiments):** Tmax 2.5 s,
L_init 15–25 Mb, β₀ 1e-5 (−50 dB), f_cpu 2.5e9, C_gen 120.

---

<a name="appendix-b"></a>
## Appendix B — The Key Equations (as implemented)

```
Channel gain            h(pi,pj) = β0 · max(‖pi−pj‖, d_floor)^(−δ)
Shannon rate            R(ptx,h) = B · log2(1 + ptx·h / N0B)
Mobility energy         E_mob(d,t) = P_static·t + k1·d + k2·d²/t          (0 if d≈0)
Compute load            W = L·(C_base + C_gen·γ·(1−ρ)·ln(1/η))
Compute energy          E_comp = τ · f_cpu² · W
Comm energy             E_comm = t1·(N0B/h)·(2^(ηL/(B·t1)) − 1)
Sub-additive fusion     L_next = η_j·L_j + η_i·L_i − ρ·min(η_i·L_i, η_j·L_j)
Wyner–Ziv payload       D_out  = η_i·L_i·(1 − ω·ρ)
Fidelity objective      min  E + λ·D_root          s.t.  D_root ≤ D_max
Derived η-floor         η_req(k) = exp(−budget / rounds_remaining)
Freshness distortion    D_fresh(s) = a_F · agility_s · (hops_s·T)^p
Root distortion         D_root = a_D·Σ w_s·Σ_hops ln(1/η) + Σ w_s·D_fresh(s)
Value-aware weight      w(i→j) = E(i→j) + λ_F·a_F·T·Σ_{s∈i} bits_s·agility_s
```

---

*End of report. This single file, together with the repository at
https://github.com/medushyant/freshness-aware-wan, fully reconstitutes the
project state as of commit `dd6f06e` (2026-06-28): 58 automated checks, 0
failures; a verified engine, a public interactive website, and a paper-ready
write-up of a novel, defensible contribution.*

---

<a name="part-xv-b"></a>
## Part XV-B — Phase-2 Limitations and Assumptions (same voice as Part XV)

- **All Phase-2 control/LLM energies are MODELED unless a `MEASURED` label
  says otherwise.** Control messages use the §2.1 model (P_ctrl = 10 dBm, the
  in-force channel, hop-relayed beyond R_comm = 250 m). LLM/VLM energy uses a
  disclosed device envelope (8 W active − 1.5 W idle) or the 0.15/0.45
  mJ/token edge-GPU estimator; `notebooks/02_vlm_energy.ipynb` produces the
  NVML-measured replacement (`runs/measured_energy.json` is auto-preferred).
  macOS `powermetrics` needs sudo and is unavailable in this sandbox — the
  probe and its fallback are in `awan/grounded/energy_meter.py`.
- **The tier-1 channel is a standard statistical model, not ray tracing.**
  Gudmundson-correlated log-normal shadowing (σ_sh = 8 dB, d_c = 25 m, link
  value at the pair midpoint — a documented simplification) + Rician K = 6 dB
  block fades with AR(1) ρ_t = 0.7. Sionna RT is the stretch ground truth
  (`notebooks/01_sionna_scene.ipynb`), designed-not-run locally.
- **The additive time-margin conformal variant fails here and we say so:**
  its α = 0.1 margin (19.0 s) exceeds the 2.5 s deadline because deep fades
  collapse the rate multiplicatively. The shipped certificate lives in the dB
  domain; its price is measured (+2–4% energy, and a completion drop to
  0.20–0.45 as uncertifiable links defer — both reported, not hidden).
- **The paper-hub dropout semantics are the paper's, made explicit.** Its
  synchronous hub has no agent-loss handling, so a permanently missing upload
  stalls every retry round; we grant it a round budget of ⌈log₂N⌉+2 before
  declaring failure. A hub with timeouts would sit between our two curves.
- **WP-3 is synthetic-grounded locally.** Scenes are procedurally drawn with
  EXACT ground truth; a real 500M VLM genuinely perceives them. Its absolute
  fact-F1 is low (~0.2 view-level) — that is a finding about tiny edge VLMs,
  reported as such; comparative results (codec curves, overlap correlation,
  corruption containment) are within-pipeline and unaffected. OPV2V real data
  runs on Colab (`notebooks/03_opv2v_pipeline.ipynb`); P-kv is
  designed-not-run (§10 fallback).
- **The negotiation LLM is small (Qwen2.5-0.5B) and its decisions are close
  to the scripted policy by design** — the mock carries the ≥10-seed
  statistics; the LLM run demonstrates schema-valid A2A-style negotiation at
  measured token cost (100% valid turns, +0.36% mission energy vs mock).
- **Continuous-mission constants are calibrated, then disclosed:** S_max from
  a periodic-5 reference run; the event trigger applies a 0.85 discreteness
  safety factor; batteries B_i = 3× (grand table) or 2× (lifetime study) the
  mean one-shot per-agent energy.
- **Battery-aware bidding tied sum-energy bidding on lifetime** in the
  tight-battery study (5.3 vs 5.3 slots) — an honest tie; the U4-style
  fairness gain shows in residual spread, not first-death time, at this scale.

<a name="part-xvi-b"></a>
## Part XVI-B — How to Reproduce Phase 2 (commands)

```bash
cd ~/Documents/BTP_ILAC_WAN

# Phase-1 regression gate (frozen code, .venv / Python 3.14) — must stay green
bash scripts/regress.sh

# Phase-2 environment (Python 3.12; torch/transformers pinned in
# requirements-awan.lock — transformers==4.57.1 for SmolVLM2)
source .venv-awan/bin/activate

# the whole Phase-2 suite (checks -> results_awan.txt, figures -> figures/awan/)
python run_awan_all.py            # cache-only (fast; needs runs/vlm_cache/)
python run_awan_all.py --full     # + re-runs SmolVLM2 perception (~40 min CPU)

# individual work packages
python run_awan_wp0.py            # W0.*  scaffold/adapters/ledger gates
python run_awan_wp1.py [--llm]    # A1.*  decentralized coordination (+ real LLM)
python run_awan_wp2.py            # C0-C4 channel + conformal certificate
python scripts/wp3_perceive.py 20 # stage-1 VLM perception (cached once)
python run_awan_wp3.py            # G1-G8 grounded pipeline (cache-only, no GPU)
python run_awan_wp4.py            # I0-I5 integration + grand showdown
python -m pytest tests/ -m "not slow"   # 26 verification tests

# website tab + headless verification (I6)
python export_awan_web_data.py
python scripts/shoot_awan.py

# manuscripts with auto-injected numbers
python scripts/make_manuscripts.py

# Colab (upload the notebook, Runtime -> GPU):
#   notebooks/02_vlm_energy.ipynb    -> runs/measured_energy.json  (MEASURED mJ/token)
#   notebooks/03_opv2v_pipeline.ipynb-> runs/opv2v_results.json    (real-data calibration)
#   notebooks/01_sionna_scene.ipynb  -> assets/sionna_gain_grid.npy (ray-traced map)
```

---

<a name="part-xviii"></a>
## Part XVIII — Phase 2: The A-WAN Extension (overview)

**Thesis.** The base paper's own conclusion confesses two weaknesses — *"the
current model assumes a simplified path loss channel and relies on
centralized coordination"* — and hides a third (its "embodied large model" is
a scalar η·L, not a model). Phase 2 eliminates all three in one architecture,
with every claim measured: agents **negotiate pairings themselves** (WP-1),
**plan against a learned radio map under a conformal deadline certificate**
(WP-2), and **exchange real vision-language-model knowledge** with exact bits,
measured joules, and a trust gate (WP-3), composed into one simulator with
continuous missions, batteries, and an N≥100 scaling story (WP-4).

**Ground rules kept.** All Phase-1 code and results stayed frozen (the seven
runners re-verified GREEN before and after every work package); Phase-2 code
lives entirely under `awan/` + new root runners, reaching frozen code only
through `awan/adapters.py`; every number in every figure is machine-written
to `runs/<stamp>_*/results.json` first; energies carry MEASURED/MODELED tags;
failed designs are reported, not hidden (`docs/DECISIONS.md`).

**Verification total: 35 new automated PASS checks** (W0.1–4, A1.1–10,
C0–C5, G1–G8, I0–I6) on top of the untouched Phase-1 baseline, plus 26
pytest tests and a Playwright-verified website tab. 21 new figures under
`figures/awan/`, every one regenerable from cached run artifacts without a
GPU. Real Colab-measured inputs: NVML mJ/token (`runs/measured_energy.json`)
and a Sionna RT ray-traced Munich radio map (`assets/sionna_gain_grid.npy`).

<a name="part-xix"></a>
## Part XIX — Phase 2: every headline result (the 11 falsifiable claims)

| # | Claim | Measured verdict |
|---|-------|------------------|
| H1 | Decentralized ≈ centralized | auction within ±3.1% of hub-Blossom mission energy at N∈{6,8,10} (10 seeds, paired Wilcoxon n.s.); per-round matching within the 2% ε-bound on 56/56 all-feasible rounds (exact on 54) — **confirmed** |
| H2 | The hub was never free | hub coordination priced for the first time: 0.017–0.030 mJ/mission (MODELED, §2.1); decentralized control ≤0.008% of mission energy — **confirmed** |
| H3 | Graceful vs brittle | at q∈{0.1,0.2,0.3} dropout: paper hub completes **0%** (synchronous upload stall), auction completes **100%** with energy tracking surviving payload — **confirmed, the headline cliff figure F1.3** |
| H4 | Talking has a price | mock-FSM negotiation control grows to 0.100 mJ at N=10 vs hub 0.029 mJ — no crossover N* in range: honest negative under GPU-class token pricing; the LLM engine itself ran 43/43 schema-valid turns at +0.36% mission energy |
| H5 | The paper's channel optimism is unsafe | deterministic plan violates deadlines on **60–62%** of links under tier-1 fading (hypothesis ≥20%); dB-margin split conformal restores coverage 0.94≥0.90 / 0.87≥0.80 at +4%/+2% energy; **replicated on ray-traced Sionna ground truth** (35% violations → 0.82≥0.80 certified) |
| H6 | Prediction pays | move-to-predicted-channel: −29.5% energy vs move-closer AND violations 60%→20%; certified variant 0% violations (defers uncertifiable links) — **confirmed** |
| H7 | The geometric proxy misranks | Spearman(ρ_geo, ρ̂)=0.36, **38% pair-ranking inversions** on 120 real agent pairs — **confirmed** (completes the repo's old remaining-item #2) |
| H8 | The energy model is fiction | **MEASURED** (Colab T4 NVML, 250 views, idle-subtracted): prefill 104 / decode 5371 mJ/token; 273 J/view vs the paper's Eq.(6)-(7) 1.15e-4 J → **gap 2.4×10⁶** — confirmed, far beyond the 10× hypothesis |
| H9 | One liar poisons the tree | base root-corruption 95% (single-path L13, measured on real VLM facts); overlap-consistency gate + near-identity corroboration → **15%** at ~0% energy overhead — **confirmed** |
| H10 | Latent vs text | honest verdict: **tie on F1 at equal bits** (ΔF1 +0.000) with text 29× cheaper in bits at η=1 (2.9 vs 82.8 kbit/tree) — for fact-level payloads at this scale, text wins the rate-fidelity trade; P-kv designed-not-run (§10) |
| H11 | A-WAN dominates | grand showdown (one world, tier-1, 10 seeds): A-WAN wins **4/5 axes** vs the paper — energy 14.85<16.66 J, violations 14%<52%, dropout completion 100%>0%, staleness 29.4<42.9; lifetime is the honest loss (fresher reports cost battery). Event-triggering saves 13% energy at the same staleness cap; surrogate (MAPE 0.1%, −1.1% penalty) yields the first runtime curve to N=200 (Blossom 2.1 s vs auction 0.55 s vs greedy 24 ms per round) |

**Grounded-pipeline vitals (WP-3, 20 scenes × 4 views, SmolVLM2-500M, all
cache-reproducible):** perception validity 94% (G1); fusion measured
sub-additive on 92% of overlapping pairs, mean −56% vs the paper's additive
Eq. (2) (G2/F3.6); root fact-F1 0.33 at η=1 — the honest smallness of a 500M
edge VLM, reported as a finding; conformal lower bound certifies a root-F1
floor at marginal coverage 0.89≥0.80 over 20 random splits (G7).

**Update to Part XIV (what remains):** OPV2V real-data calibration now has a
ready notebook (`notebooks/03_opv2v_pipeline.ipynb`) and needs only the
manual ~4 GB dataset download (its Colab run correctly halted on the empty
folder); P-kv payloads and the MAPPO fine-tune remain designed-not-run
stretch items; manuscript skeletons M1/M2 with auto-injected numbers are in
`docs/`, ready for expansion against the CFP list in the brief.
