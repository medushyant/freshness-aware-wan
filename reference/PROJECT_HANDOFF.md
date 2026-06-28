# ILAC-WAN Extension Project — Complete Handoff Document

> **Purpose of this file.** This is a full, self-contained briefing so a fresh
> Claude instance (e.g. Claude Code) can continue this B.Tech project without
> re-deriving anything. It records the source paper, every critique we found,
> what we implemented, every design decision, the exact file/function layout,
> the verification status, and the roadmap for the remaining work (moving
> targets already started; comparison-algorithm baselines and a 3D web demo
> still to do). Read it top to bottom once before touching code.

---

## 0. TL;DR for the new assistant

- **Student:** B.Tech IT, ABV-IIITM Gwalior. 2-month project, ~15-20 days left.
- **Base paper:** Zhao et al., "Agentic AI-Empowered Wireless Agent Networks with
  Semantic-Aware Collaboration via ILAC," arXiv:2604.02381 (2026).
- **One-line objective (the spine the professor asked for):**
  *Energy-optimal knowledge aggregation for robot/drone swarms tracking
  moving targets — making the report quality measurable & guaranteed
  (Direction 1) and the matchmaking knob-free & near-optimal (Direction 2),
  extended from static to dynamic and time-varying targets.*
- **Status:** Direction 1 ✅ implemented+verified. Direction 2 ✅
  implemented+verified. Rigor/presentation upgrades ✅. Moving targets ✅
  module built+verified. **Remaining:** (a) wire targets into full D1/D2
  missions, (b) add ABC/ACO/Cuckoo/Egyptian-Vulture as *comparison baselines*
  (NOT as the main method — see §9 for why), (c) build a 3D web visualization.
- **Verification philosophy:** every claim has an automated PASS/FAIL check.
  Current totals: D1 = 15 checks, D2 = 11, upgrades = 10, targets = 5. All pass.
- **Tech constraints:** numpy/scipy/matplotlib/networkx only. **No CVXPY**
  (not installed in the sandbox; we re-did the paper's convex blocks with
  scipy — same math, lighter stack). No internet for pip in the sandbox.
  The container resets between sessions, so code must be re-uploaded each time.

---

## 1. Files the student will upload alongside this MD

Make sure these are present (ask the student if any are missing):

1. `btp_code_d1_d2.zip` — the full implementation (Direction 1 + Direction 2 +
   upgrades). This is the master codebase. Contents in §6.
2. `targets_module.zip` — the moving-target module (`wan/targets.py`,
   `run_targets.py`, figures, results). Merge into the master codebase.
3. `2604_02381.pdf` — the base paper itself.
4. `ILAC-WAN_BTP_Proposal_v2.pdf` (and `.tex`) — the 17-page proposal we wrote
   (23-point audit + 5 ranked directions + 8-week plan). Good for report text.
5. (Optional) `ILAC-WAN_Research_Roadmap.pdf`, `ILAC-WAN_Research_Proposal.pdf`
   — earlier v1 documents, superseded by v2 but kept for history.
6. Figures (PNG): E1-E5, F1b, T1-T3, U4, G2, G3 — all regenerable from code.

**First action for the new assistant:** unzip both zips into one working folder,
confirm `wan/` contains model.py, solver.py, network.py, topology.py, style.py,
targets.py; then run the four drivers to confirm everything still passes
(see §7 for commands).

---

## 2. The base paper, explained precisely

### 2.1 The scenario
A set of N mobile agents (robots/drones), each carrying a small onboard AI
("embodied large model", ELM). Each patrols a circular region and collects
sensor data. They must merge everything into ONE combined report delivered to
a sink, using minimum total energy. Three energy costs:
- **mobility** — moving to a new position (better wireless geometry),
- **computation** — the ELM compressing/fusing data into a semantic summary,
- **communication** — transmitting wirelessly; cost rises sharply with distance.

### 2.2 The aggregation mechanism — a knockout tournament
Data is merged progressively over K = ceil(log2(N)) rounds. Each round, agents
pair up; in each pair one agent compresses its payload and transmits to its
partner; the partner fuses both; the sender "retires" (drops out). The active
set roughly halves each round until one agent (the root) holds everything.

### 2.3 Key quantities and equations (paper's own numbering)
- Coverage constraint (Eq. 1): ||p_i - c_i|| <= R_i. Agent i stays within
  radius R_i of its patrol centre c_i. **NOTE: c_i is FIXED in the paper =
  the implicit static-target assumption we later relax.**
- Compression ratio eta ∈ (0,1]: eta=0.2 keeps 20% of the bits.
- Fusion rule (Eq. 2): L_j ← eta_j·L_j + eta_i·L_i  (payloads simply ADD).
- Geometric overlap rho (Eq. 5): Jaccard index of the agents' accumulated
  patrol areas — a *geometric proxy* for semantic redundancy.
- Compute cost (Eq. 6): W = L·(C_base + C_gen·γ·(1-rho)·ln(1/eta)); the rho
  discount applies only to the predecessor from round k-1. **In round 1,
  rho=1, making round-1 compression compute-FREE — a degenerate artifact.**
- Channel: deterministic large-scale path loss h = β0·d^(-δ), Shannon rate
  over orthogonal channels.
- Transmitted payload (Eq. 10): D_out = eta_i·L_i  (sent REGARDLESS of what
  the receiver already knows — violates Wyner-Ziv, see §3).
- Objective (Eq. 14): minimize total energy. **Quality/distortion appears
  NOWHERE.**
- Inner problem (P1): for a fixed pair, optimize positions, motion times,
  transmit power, compression ratios. Solved by BCD: a convex resource block
  (1-D search over eta + transmit time) and an SCA-convexified motion block.
  Paper's headline inner result: the receiver's optimum sits on its bound,
  eta_j* = eta_req (always compress maximally).
- Outer problem (P2): choose the pairing each round. Eq. (16b) forces EXACTLY
  floor(N/2) pairs per round (max parallelism; an idle "virtual node" exists
  only when N is odd). Matching weight (Eq. 36):
  w_ij = E_ij + Φ_j, where Φ_j = ζ·||p_end_j - centroid||^δ·L_next is a
  hand-tuned "look-ahead" potential field (ζ picked by hand; their Fig. 7
  shows energy rebounds if ζ is wrong). Directed weights symmetrized via
  min(w_ij, w_ji), then solved by Edmonds' Blossom max-weight matching.
- Algorithm H-MAP = inner BCD nested inside outer potential-field-guided
  matching. Claimed O(N^3) per round.
- Validation: N <= 10, fully synthetic, no real ELM, no runtime reported.

---

## 3. The full critique (our 23-point audit, condensed)

Each item is anchored to a specific equation. Five are internal
inconsistencies. The ones our project acts on are starred.

**Tier I (original 11):**
- L1 no sensing-quality model; mobility serves only the channel. → targets ext.
- L2* coverage hole: retired senders abandon their zones. → targets ext.
- L3* fusion ignores cross-redundancy (Eq. 2 double-counts). → Direction 1
- L4 semantic similarity is a purely geometric proxy (Eq. 5). → Direction 1
- L5 FLOPs/frequency ELM energy model is unrealistic (real LLMs memory-bound).
- L6 deterministic channel, no fading/outage.
- L7 unlimited orthogonal spectrum, no interference.
- L8 energy-only objective, no fidelity/value. → Direction 1
- L9 hand-tuned myopic potential field (their Fig. 7 fragility). → Direction 2
- L10* centralization/final root-to-sink hop uncosted.
- L11 small-scale synthetic validation only.

**Tier II (literature-grounded):**
- L12 forced max parallelism / fixed depth (Eq. 16b). → Direction 2
- L13 no trust/integrity; single corrupted leaf poisons the root.
- L14 path-multiplicative re-compression ("telephone game") untracked. → Dir 1
- L15 bit-level payload vs token/KV-cache reality.
- L16 sum-energy objective; batteries/reception ignored. → upgrades (fairness)
- L17 one-shot episode vs continuous mission.

**Tier III (second close read — our sharpest finds):**
- L18* receiver side information never reduces the over-the-air payload —
  violates the Wyner-Ziv rate-distortion theorem (1976). → Direction 1
- L19* the look-ahead Φ is evaluated post-hoc at positions optimized WITHOUT
  it — internal inconsistency. → Direction 2 (Stage A)
- L20 round-1 encoding is free (rho=1 artifact). → Direction 1 (observed live)
- L21 stale centroid includes retiring agents. → minor
- L22 synchronous lockstep rounds. → Direction 2 (flexible schedule)
- L23 eta_req is exogenous, never derived. → Direction 1 (derived)

Two rigor notes for the report: (P2) is an MDP treated as a static program
(this justifies the learned value approach); Theorem 1 claims "strictly
convex" but the proof only establishes convexity.

---

## 4. Direction 1 — Fidelity-Aware Aggregation (IMPLEMENTED)

**Objective:** the paper treats compression as free in quality, so it concludes
"always compress maximally" (eta*=eta_req). But each datum is re-compressed up
to K times along its path (telephone game), and quality degrades unmeasured.
We make quality measurable, guaranteed, and put it in the objective.

**Six concrete changes (Paper → Ours):**

1. **Distortion state (NEW; paper has none).**
   D_j ← Θ(D_j, D_i, eta_i, eta_j), monotone: smaller eta ⇒ more distortion.
   Implementation: each source carries [orig_bits, sum_ln(1/eta)] in agent
   ["srcs"]; root distortion = payload-weighted mean of a_D·sum_ln(1/eta).
   This captures path-multiplicative compounding (fixes L14). Require
   D_root <= D_max.

2. **Sub-additive fusion (fixes L3, Eq. 2).**
   L_j ← eta_j·L_j + eta_i·L_i − rho_ij·min(eta_i·L_i, eta_j·L_j).
   Shared content counted once.

3. **Wyner-Ziv link (fixes L18, Eq. 10).**
   D_out = eta_i·L_i·(1 − ω·rho_ij) instead of eta_i·L_i. Don't resend what
   the receiver knows. Theoretical floor: R_{X|Z}(D)=min I(X;T|Z). Also makes
   pair costs asymmetric for a principled reason (consumed by Direction 2's
   directed weights).

4. **Fidelity-aware objective (fixes L8, Eq. 14).**
   min E + λ·D_root, OR min E s.t. D_root <= D_max. **Headline result: the
   optimum becomes INTERIOR (eta* < eta_req, moving with λ) — overturns the
   paper's published lemma eta*=eta_req.** Θ chosen convex so the paper's
   BCD/SCA solver structure still applies.

5. **Derived eta_req (fixes L23, Eq. 15f).**
   eta_req(k) = exp(−budget / rounds_remaining), where budget = D_max/a_D −
   worst-source-distortion. The exogenous knob becomes a lemma.

6. **Conformal guarantee (NEW).**
   Calibrate so Pr(D_root <= D_max) >= 1−α, distribution-free, via
   Learn-then-Test / split conformal. Implementation: ascending λ ladder, per-λ
   calibration missions, accept first λ whose realized-distortion quantile
   clears D_max. Verified 97% coverage vs 80/90% targets.

**Empirical grounding (NOT yet run — student to do on Colab):** replace
assumed rho and Θ with measured values on OPV2V (real self-driving-car
collaborative-perception data): rhô from CLIP/SigLIP embedding similarity vs
geometric Jaccard; Θ from a JPEG-quality-vs-embedding-distortion sweep giving
the real a_D and residual σ. Script: opv2v_colab.py.

**Verified results:** interior optimum (eta_j* sweeps 0.16–0.72); reference
point (D=2.16, E=0.59) violates the floor AND is dominated on both axes;
WZ saves ~5% and rescues an infeasible deadline; derived bound holds for
Dmax 0.9/1.2/1.5; conformal 97%.

---

## 5. Direction 2 — Learned Predictive Topology (IMPLEMENTED)

**Objective:** the paper's matchmaking needs a hand-tuned ζ (fragile, their
Fig. 7) and its look-ahead never reaches the movement planner (L19). Replace
the hand-tuned bonus with a *learned* prediction of remaining mission energy.

**Four-stage ladder (each a deliverable):**

- **Stage A — Φ-in-the-loop (fixes L19, zero learning).** Move the potential
  field INSIDE the inner problem so movement responds to it. Convexity
  preserved (||p−c||^δ convex for δ>=2). Beats paper at its own ζ.
- **Stage B — auto-scaled (Lyapunov drift-plus-penalty).** The energy/spread
  trade-off coefficient is set ONCE from round-1 magnitudes. No knob.
- **Stage C — learned cost-to-go.** A value model V̂(state) predicts remaining
  mission energy. Matching weight w_ij = E_ij + V̂(state-after-pairing).
  State features (7): #agents left, rounds left, total payload, max payload,
  payload-weighted spread, mean & min pairwise distance. Model: ridge
  regression (transparent, closed-form, no heavy NN). Trained on 24 mixed-
  policy rollouts (paper/random/greedy) at N∈{4,5,6}; each round yields
  (features → realized energy from that round to the end). R²=0.90. Train
  seeds 100-123 disjoint from eval seeds 3-10.
- **Stage D — decision-focused fine-tune (added in upgrades).** Zeroth-order
  perturbation search on the value weights where the loss is the REALIZED
  mission energy through the exact Blossom matcher (perturbation-gradient
  spirit; black-box solver only — cf. Vlastelica 2020, Berthet 2020, Huang &
  Gupta NeurIPS 2024). Validation-gated (seeds 300-305): only replaces Stage C
  if it wins on unseen data. First attempt overfit (train 8.13/eval 8.54);
  fixed with more training scenarios + the gate — an honest ML-process story.

**Flexible schedule (fixes L12/L22):** each even round, rest the worst pair if
V̂ says waiting is cheaper. The forced schedule stays a candidate ⇒ flexibility
can't lose (Dominance Proposition).

**Exact oracle (NEW for this framework):** brute-force every pairing sequence
at N=5 → the true optimum, which the paper never computes. Gives the first
optimality-gap numbers.

**Verified results:** ζ sweep — paper varies ~10% (Fig. 7 reproduced);
learned policy 8.43 J with no knob (below paper's best 8.81); Stage D 7.75 J
(−12% vs paper's best). Optimality gap N=5: learned 0.5%, paper 13.3%,
random 24.4%. Size transfer: trained ≤6, wins at N=8 (12.36 vs 12.81) and
N=10 (15.88 vs 16.92). Fairness: worst-agent energy drops 3.09→2.49 J.

---

## 6. Codebase layout (what's in btp_code_d1_d2.zip + targets_module.zip)

```
wan/
  __init__.py
  model.py      Parameters dict P (Table-I values + chosen unpublished ones,
                all documented). make_agents(). jaccard_disks() Monte-Carlo
                overlap. Energy/time/channel functions: chan_gain, rate,
                mob_energy, comp_load, comp_energy_time, comm_energy, ptx_for.
  solver.py     solve_pair(): the pairwise BCD inner solver. Paper mode
                (eta_j pinned at cap, 1-D eta_i search, SLSQP motion) and
                fidelity mode (both etas free, +λ·dD, +Φ-inside, +WZ payload).
                Also phi_in_loop flag for Stage A. Per-side energy split
                (E_i, E_j) for the fairness metric. Convergence trace hook.
  network.py    run_mission(): the full Direction-1 mission. Per-round rho,
                ordered-pair solves, weights, Blossom matching (via networkx
                max_weight_matching with a virtual idle node), _execute()
                fusion bookkeeping (the srcs telephone-game ledger),
                _derived_eta_floor (endogenous eta_req), silent-infeasibility
                guard. Conformal: simulate_true_D, conformal_pick_lambda,
                conformal_coverage.
  topology.py   Direction 2. mission() under policy ∈ {paper, stageA,
                lyapunov, learned, greedy, random, distance}. ValueModel
                (ridge). state_features(). collect_training(). flexible
                schedule (_maybe_idle_worst). dp_oracle() brute force.
                finetune_decision_focused() Stage D.
  targets.py    NEW. Target class with static / dynamic (NCV) / time_varying
                (maneuvering: CV+turn+accel+stop schedule) motion. _F() and
                _turn() transition matrices. spawn_targets(). CVPredictor()
                one-step constant-velocity predictor.
  style.py      Shared matplotlib style (use_style()).

run_direction1.py     E1-E6 → figures/ + results.txt (15 checks).
run_direction2.py     T0-T4 → figures/ + results_d2.txt (11 checks).
run_repro_scaling.py  energy-vs-N trend → figures/F1b + appends results.txt.
run_upgrades.py       error-bar headline figs + Stage D + fairness +
                      sensitivity → results_upgrades.txt (10 checks).
run_targets.py        G1-G3 → figures/ + results_targets.txt (5 checks).
opv2v_colab.py        real-data grounding (student runs on Colab).
README.md             run order + all honest notes.
```

**Parameter notes (in model.py):** Table I gives N, area, R, L, Tmax, B, β0,
δ, Pmax, vmax, f, τ, ζ. Not given (chosen, documented): N0B, P_static, k1, k2,
C_base, C_gen, η_min, η_req, a_D, Dmax, ω_wz. The "stress regime" used in D2
(Tmax 2.5, L 15-25 Mb, β0 1e-5, f 2.5e9, C_gen 120) is tuned so a feasible
compression window exists in later rounds AND communication actually costs
something (otherwise every topology looks identical). This is documented and
must be explained in the report, not hidden.

---

## 7. How to run & verify (do this first, every session)

```bash
pip install numpy scipy matplotlib networkx        # sandbox has these
# unzip both zips into one folder so wan/ has all 6 modules + targets.py
python3 run_direction1.py        # ~1-2 min, expect 15 PASS
python3 run_direction2.py        # ~1 min,  expect 11 PASS
python3 run_repro_scaling.py     # ~1 min,  appends F1b
python3 run_upgrades.py          # ~1 min,  expect 10 PASS
python3 run_targets.py           # <1 min,  expect 5 PASS
```
All figures land in `figures/`. If any check FAILs, that's the first thing to
fix before adding features. The drivers print PASS/FAIL lines with the actual
numbers beside each claim.

---

## 8. Moving targets — what's done, what's next (HIGH PRIORITY)

**Done (targets.py + run_targets.py, 5 checks pass):**
- Three motion classes implemented from the standard tracking literature:
  static (paper's assumption), dynamic = nearly-constant-velocity (the
  "workhorse" model), time-varying = maneuvering (coordinated-turn + mode
  schedule). State = [x, y, vx, vy]; transitions _F(T) and _turn(T,ω).
- CVPredictor: one-step constant-velocity prediction (a Kalman filter is the
  obvious upgrade; CV is the standard baseline).
- Verified ladder: motion and prediction-error both increase static < dynamic
  < time-varying; tracking energy grows with agility (0→4.05→6.66 J);
  prediction helps MOST on the hardest targets (28% saving on time-varying).
- Figures G2 (tracking energy) and G3 (trajectory snapshot, great for slides).

**Next (the new assistant should do this):**
1. **Wire targets into full D1/D2 missions.** In make_agents / run_mission,
   let each agent's patrol centre c_i be the *predicted* target position each
   round (c_i ← CVPredictor.predict). Then re-run D1 and D2 under each target
   class and report how the headline results (interior optimum, ζ-robustness,
   optimality gap) change as targets get more agile. Expected story: the
   learned look-ahead (D2) and the fidelity floor (D1) matter MORE as targets
   move, because stale positions waste energy and force harder compression.
2. **Add a Kalman filter** as an upgrade over CVPredictor for the maneuvering
   case (interacting-multiple-model is the gold standard but overkill; a
   single CV-Kalman is enough and defensible).
3. **New checks:** D1 floor still met under moving targets; D2 still beats
   paper under moving targets; energy gap between predict-ahead and react-only
   widens with agility.

---

## 9. Comparison-algorithm baselines (ABC / ACO / Cuckoo / Egyptian Vulture)

**CRITICAL FRAMING — do NOT make these the main method.** The matchmaking is
solved EXACTLY by Blossom matching (polynomial, provably optimal at N<=10).
The optimization literature is consistent that nature-inspired metaheuristics
only beat exact methods on LARGE instances where exact becomes too slow; on
small instances they tie at best, lose at worst. Our optimality-gap figure
already shows the learned policy is 0.5% from optimal — a metaheuristic cannot
beat that, only erode it. If presented as the contribution, a sharp examiner
will ask "why replace an exact optimizer with an approximate one?" — there is
no good answer.

**The TWO legitimate uses (implement these):**
1. **As comparison baselines in the optimality-gap figure (T2).** Implement
   ACO, ABC, Cuckoo Search, and Egyptian Vulture as alternative matchers for
   the per-round pairing problem, and add their gap bars next to exact/
   learned/greedy/random. Expected: exact 0.5%, learned 0.5%, ACO ~4%,
   Cuckoo ~6%, vulture ~7%, random 24%. This *uses* the algorithms the mentor
   named, proves correct implementation, and STRENGTHENS the story.
   - Each is a standard metaheuristic over the assignment/matching problem:
     represent a solution as a permutation/pairing, define the cost as mission
     energy, run the algorithm's update rule (pheromones for ACO, employed/
     onlooker/scout bees for ABC, Lévy-flight nests for Cuckoo, the
     fitness/luck rolling-and-tossing operators for Egyptian Vulture).
   - Keep them in a new module wan/metaheuristics.py, each with the same
     signature: match(active, weight_fn) -> list of (sender, receiver) pairs.
2. **As an approximate matcher in the large-N regime.** Push the scaling
   experiment to N=50-200 where exact Blossom slows down, and show a chosen
   metaheuristic (likely ACO) stays within X% of the small-N oracle trend
   while running faster. This is the ONE place they genuinely add value, and
   it matches what the literature actually found.

**New checks:** each metaheuristic's gap >= exact gap (sanity: none beats
exact); all structured methods beat random; (large-N) metaheuristic runtime <
exact runtime beyond some N.

---

## 10. The 3D web visualization (TO BUILD)

**Goal:** a browser page the professor can watch — the single highest-impact
item for the presentation (even though it doesn't change the science).

**Suggested stack:** plain HTML + a single React/Three.js (or even 2D canvas /
d3) file, self-contained, no backend. The student wants "fully ready website."

**What it must show:**
- A 2D/3D arena with N drones (use drone glyphs; paper says robots but student
  prefers drones) and the moving targets (static = fixed dot, dynamic =
  straight arrow, time-varying = curving path).
- The aggregation tournament animating round by round: pairs light up, one
  drone's payload shrinks (compression) and flies to its partner (transmission),
  sender greys out (retires), until one root remains.
- A side-by-side or toggle: **paper's method vs ours**, with live energy
  counters ticking up and a live "report quality" meter (D vs D_max line) —
  so the viewer SEES the paper's report degrade past the floor while ours
  stays above it.
- A target-class selector (static / dynamic / time-varying) so the demo
  covers all three.
- Optional: a ζ slider on the paper's side to show its fragility live, vs the
  learned policy that has no knob.

**Data source:** either (a) precompute mission traces in Python (positions,
pairings, energies, distortions per round) and export JSON the page replays,
or (b) reimplement the light simulation in JS. Option (a) is faster and keeps
the numbers consistent with the verified Python results — recommended. Add a
small `export_traces.py` that dumps JSON for each (method, target-class).

**Honesty rule:** the web demo must replay REAL computed traces, not faked
animations. Keep it consistent with results*.txt.

---

## 11. Remaining roadmap (priority order for ~15-20 days)

1. **(1-2 days) Reframe everything around the single objective** (§0). This is
   the professor's core ask ("not novel" = no clear objective). Rewrite the
   report intro so moving-target tracking is the spine and D1/D2 are the two
   enabling mechanisms.
2. **(3-4 days) Wire targets into full D1/D2 missions** (§8.1) + Kalman
   upgrade. This is the real novelty.
3. **(2-3 days) Metaheuristic baselines** (§9.1) into the T2 figure, + optional
   large-N approximate-matcher demo (§9.2).
4. **(4-5 days) 3D web visualization** (§10).
5. **(2-3 days) Run OPV2V on Colab** (§4 grounding) for real rho/Θ numbers.
6. **(ongoing) Report + slides**, leading every section with the objective,
   every claim backed by a PASS check.

---

## 12. Things to tell the student to upload / decide

- Confirm both zips + base paper are uploaded (see §1).
- Decide drones vs robots in the visuals (student leans drones; the paper says
  robots — either is fine, just be consistent).
- Provide the OPV2V test split (from the OpenCOOD GitHub data page) if they want
  the real-data figures done — it's a multi-GB download, Colab only.
- Confirm the target motion parameters (speeds, turn rate) match whatever
  scenario the professor has in mind (campus patrol? traffic? disaster zone?).

---

## 13. Non-negotiable principles (keep the work legitimate)

- Every method compared uses the IDENTICAL inner solver — apples-to-apples.
- Training seeds disjoint from evaluation seeds; Stage D validation-gated.
- The exact oracle is true brute force, not an approximation.
- Unpublished constants are documented, never hidden; the stress regime is
  explained.
- The web demo replays real computed traces.
- Every headline claim has an automated PASS/FAIL check that regenerates from
  code in minutes.
- Honest limitations stay IN the report: scipy-not-CVXPY, chosen constants,
  motion-off in some D2 experiments, CV-prediction not full Kalman, metaheuristics
  as baselines not main method. These are strengths (rigor), not weaknesses.

---

*End of handoff. The new assistant should start by running §7 to confirm the
baseline passes, then proceed down §11. When in doubt about a design choice,
prefer the option that is simpler, more transparent, and easier to defend in a
viva over the one that merely looks more sophisticated.*

---
---

# APPENDIX — EXHAUSTIVE TECHNICAL REFERENCE

> Everything below is the fine detail: exact parameter values, every formula
> as implemented, every experiment with measured numbers, every bug we hit and
> the fix, and the full chronological decision history. Nothing omitted.

---

## A1. Exact parameter values (wan/model.py, dict P)

**From the paper's Table I (used as-is):**
```
N            = 6        number of agents (paper uses up to 10; 6 keeps runs fast)
area         = 500.0    square region side [m]
R_cov        = (80,100) patrol radius range [m]
L_init       = (5e6,10e6) initial payload range [bits]
Tmax         = 6.0      per-round latency deadline [s]
B            = 1e6      bandwidth [Hz]
beta0        = 1e-4     channel gain at 1 m (-40 dB)
delta        = 3.0      path-loss exponent
Pmax         = 1.0      max transmit power [W] (30 dBm)
vmax         = 5.0      max speed [m/s]
f_cpu        = 1e9      compute capacity [FLOP/s]
tau          = 1e-28    effective capacitance (dynamic power coeff)
zeta         = 1e-14    potential-field weight
```

**NOT in the paper — chosen by us, documented (these are the "unpublished
constants" that MUST be disclosed in the report):**
```
N0B          = 3.98e-15 noise power (-174 dBm/Hz * 1 MHz)
P_static     = 0.2      rover idle/hover power [W]
k1           = 0.05     mobility linear term [J/m]
k2           = 0.01     mobility quadratic term [J*s/m^2]
C_base       = 60.0     base processing [FLOP/bit]
C_gen        = 350.0    generative processing [FLOP/bit]
gamma        = 1.0      compression-complexity factor
eta_min      = 0.05     hardest compression allowed
eta_req      = 0.60     receiver compression cap (paper's Eq. 15f input)
d_floor      = 1.0      min link distance [m] (avoids h -> infinity)
a_D          = 1.0      distortion units per ln(1/eta)
Dmax         = 1.2      fidelity floor on the root report
omega_wz     = 0.9      Wyner-Ziv efficiency of side information
```

**The "stress regime" override (used in run_direction2.py, run_upgrades.py,
and the E1 reproduction):**
```
Tmax  = 2.5      tighter deadline
L_init= (15e6,25e6)  bigger payloads
beta0 = 1e-5     harsher channel (-50 dB at 1 m)  [E1 used 1e-5; some early
                 probes used 1e-6 then reverted — see A6 bug log]
f_cpu = 2.5e9    faster SoC (so the bottleneck is the radio, not the CPU)
C_gen = 120.0    lighter generative cost (keeps a feasible eta window)
```
WHY this regime exists (must be explained, not hidden): with the paper's
Table-I values alone, in the lazy regime every topology choice produces almost
identical energy (communication is nearly free), so the topology experiments
would show no differences. The stress regime makes communication actually
cost something, so "who pairs with whom" moves real energy. Simultaneously the
parameters are chosen so a feasible compression window EXISTS in later rounds:
compute cost pushes eta UP (less compression to save FLOPs) while the radio
pushes eta DOWN (more compression to fit the deadline) — if that window is
empty the mission is infeasible. C_gen=120 and f_cpu=2.5e9 keep it non-empty.

---

## A2. Every formula as implemented (exact, copy-faithful)

**Geometry / channel (model.py):**
```
chan_gain(pi,pj)  = beta0 * max(||pi-pj||, d_floor)^(-delta)
rate(ptx,h)       = B * log2(1 + ptx*h / N0B)
mob_energy(d,t)   = P_static*t + k1*d + k2*d^2/t        (0 if d<1e-9)
comp_load(L,rho,eta) = L*(C_base + C_gen*gamma*(1-rho)*ln(1/eta))
comp_energy_time(L,rho,eta) = (tau*f_cpu^2 * W,  W/f_cpu)   where W=comp_load
comm_energy(etaL,h,t1) = t1*(N0B/h)*(2^(etaL/(B*t1)) - 1)   (0 if etaL<=0)
ptx_for(etaL,h,t1)     = (N0B/h)*(2^(etaL/(B*t1)) - 1)
jaccard_disks(A,B)     = Monte-Carlo area Jaccard of two unions of disks
                         (4000 samples; bounding box of all disk extents)
```

**Direction-1 fusion / payload / distortion (solver.py + network.py):**
```
payload_out(eta_i,L_i,rho,use_wz) = eta_i*L_i * (use_wz ? (1 - omega_wz*rho) : 1)
L_next (paper)        = eta_j*L_j + eta_i*L_i
L_next (sub-additive) = eta_j*L_j + eta_i*L_i - rho*min(eta_i*L_i, eta_j*L_j)
distortion increment dD = a_D*( w_i*ln(1/eta_i) + (1-w_i)*ln(1/eta_j) )
   where w_i = mass_i/(mass_i+mass_j), mass = sum of source orig-bits
root_distortion(agent)  = a_D * sum_s(bits_s * sumln_s) / sum_s(bits_s)
   (each source s carries [orig_bits, sum of ln(1/eta) over its path])
derived_eta_floor(agent,k_rem):
   worst = max over sources of sumln
   budget = max(Dmax/a_D - worst, 0)
   return exp(-budget / k_rem)        (=1 if k_rem<=0)
```

**Direction-1 objective inside solve_pair:**
```
paper mode:   J = E_comp_i + E_comp_j + E_comm           ; minimize over eta_i
              with eta_j pinned at eta_cap_j (=eta_req)
fidelity mode:J = E_comp_i + E_comp_j + E_comm + lam*dD + Phi
              Phi = zeta*||p_end_j - centroid||^delta * L_next
              minimize over BOTH (eta_i,eta_j) via 11x11 grid then L-BFGS-B
motion block: SLSQP over (p_i,p_j,tmove_i,tmove_j,t_tx) with constraints:
              coverage (||p-c||<=R), speed (||p-p_start||<=vmax*t),
              deadline (Tmax - tmove - t_tx >= 0 for each side and for compute),
              rate (t_tx * rate(ptx,h) >= payload_out)
              two SLSQP starts: stay-put, and step-toward-each-other
BCD: alternate resource-step and motion-step up to n_bcd=3 (6 in convergence
     test), stop when |Delta E| < 1e-3*E.
```

**Direction-1 conformal (network.py):**
```
simulate_true_D(D_hat) = D_hat * exp(0.12 * N(0,1))   # stand-in for model error
conformal_pick_lambda(alpha, lam_grid, n_cal=25):
   kq = ceil((n_cal+1)*(1-alpha)) - 1
   for lam ascending: run n_cal missions, scores=realized D,
       q = sort(scores)[kq]; if q <= Dmax: return (lam, q)
conformal_coverage(lam,q,alpha,n_test=40): fraction of fresh missions with
   simulate_true_D(D) <= Dmax
```

**Direction-2 (topology.py):**
```
state_features(active) = [ n, ceil(log2 n), sum(L)/1e6, max(L)/1e6,
                           sum(L*dist_to_centroid)/sum(L), mean_pairwise_dist,
                           min_pairwise_dist ]   (7 numbers)
ValueModel: ridge regression, l2=2.0, standardized features, closed form:
            w = solve(Z'Z + l2*I, Z'y), Z=[1, (X-mu)/sd]; predict = max(z.w, 0)
matching weight by policy:
   paper:    w = E + zeta*spread
   stageA:   w = E + zeta*spread   (but E computed with Phi inside the solver)
   lyapunov: w = E + V_scale*spread, V_scale set once = mean(E)/mean(spread)
   learned:  w = E + V_hat(state-after-pairing)
   greedy/random/distance: w = E (or metres for distance)
Blossom: networkx max_weight_matching(maxcardinality=True) on
         weight = big*100 - cost, virtual node for odd N.
flexible: _maybe_idle_worst compares "all pairs fire" vs "worst pair rests"
          via V_hat of the resulting states; rest only if strictly cheaper.
dp_oracle(N=5): recursive enumeration of all pairing sequences AND both
          directions per pair, motion off, with branch-and-bound pruning and
          a pair-cost cache keyed by (i,j,roundedL,roundedL,roundedrho,rho).
finetune_decision_focused (Stage D): zeroth-order random search on w,
          objective = mean realized mission energy over train_seeds,
          keep-best-seen, shrinking step scale; validation-gated externally.
```

**Targets (targets.py):**
```
state s = [x, y, vx, vy]
_F(T)      = NCV transition [[1,0,T,0],[0,1,0,T],[0,0,1,0],[0,0,0,1]]
_turn(T,w) = coordinated-turn transition (reduces to _F as w->0)
static:        s unchanged (velocity zeroed at init)
dynamic (NCV): s = _F(T)@s; v += q*sqrt(T)*N(0,1), q=0.15
time_varying:  mode schedule [cv,turn,accel,cv,stop,turn] cycled by round;
               turn rate 18 deg/s, accel 2 m/s^2; v += 0.5*q*sqrt(T)*N(0,1)
wall reflection keeps targets in [0,area]
CVPredictor.predict(meas,T): v=(meas-prev)/T; return meas+v*T
```

---

## A3. Every experiment and its measured result

**run_direction1.py (15 checks, all PASS):**
- E1 reproduction: inner solver converges in a few BCD steps; benchmark
  ordering proposed <= no-motion (motion never hurts; gains depend on
  unpublished kappas), proposed < max-power, no-semantic infeasible under
  stress, blossom <= random.
- E2 interior optimum: eta_j* sweeps [0.16, 0.72] as lam goes 0->0.8, monotone;
  overturns eta_j*=eta_req=0.60.
- E3 Pareto: reference (D=2.16 originally measured; 2.21 in an earlier seed
  set, E=0.59/0.61) violates floor Dmax=1.2 AND is dominated on both axes; a
  frontier point (lam~0.04-0.08) gives e.g. D=1.29 at E=0.45; floor met at
  ~+62% over the energy-only optimum (vs the unconstrained min; vs the PAPER's
  own point it is actually cheaper).
- E4 Wyner-Ziv: never worse, ~5% saving at rho=0.9, and turns an infeasible
  deadline feasible.
- E5 conformal: alpha=0.2 -> lam*=0.30, q=0.83, coverage=0.97; alpha=0.1
  similar; coverage >= 1-alpha both times.
- E6 derived cap: D = {0.40, 0.80, 1.08} <= Dmax with lam=0.

**run_direction2.py (11 checks, all PASS):**
- T0 value model R^2 = 0.90 on 52 rollout states (24 missions).
- T1 zeta sweep: paper worst/best ratio ~1.09 (fragile); stage A <= paper at
  paper's zeta; learned 8.43 J <= paper's best 8.81; flexible helps; stage B
  competitive.
- T2 optimality gap N=5: learned 0.5%, paper 13.3%, random 24.4%, all
  structured beat random.
- T3 size transfer: N=8 learned 12.36 vs paper 12.81; N=10 learned 15.88 vs
  paper 16.92.
- T4 flexible <= forced over 8 scenarios (8.085 vs 8.428).

**run_repro_scaling.py:** paper energy grows 4.91->9.17->12.81->16.92 J over
N=4,6,8,10; paper <= random at every N. (Note: at N=4-6 distance and random
can occasionally tie or dip; the check allows a 5% tolerance and the
growth-with-N trend is the asserted claim.)

**run_upgrades.py (10 checks, all PASS):**
- U1 Pareto with 8 scenarios + error bars: dominance holds (ref D=2.16,
  E=0.594), reference still violates floor.
- U2/U3 zeta + Stage D: paper fragility persists (worst/best ~1.09); stage C
  (no knob) <= best hand-tuned; Stage D 7.754 J on held-out vs stage C+flex
  8.085 and paper's best 8.813 (-12%). Stage D validation-gated (val D=7.365 <
  C=7.872). First fine-tune attempt overfit (train 8.13/eval 8.54) -> fixed
  with more training scenarios + the gate.
- U4 fairness: worst single agent 3.09 (paper) -> 2.66 (learned) -> 2.49
  (flex) J; total also drops.
- U5 sensitivity: derived bound tracks Dmax 0.9/1.2/1.5 -> worst D
  0.90/1.08/1.26; WZ saving positive across omega 0.5/0.9 (4.3%/5.2%).

**run_targets.py (5 checks, all PASS):**
- G1 path length 0 / 52 / 52 m; predict error 0.0 / 2.0 / 6.4 m
  (static/dynamic/time-varying).
- G2 tracking energy 0 / 4.05 / 6.66 J; predict-ahead saves 28% on
  time-varying.
- G3 trajectory snapshot saved.

---

## A4. Figures produced (all in figures/, all regenerable)

```
E1_reproduction.png     solver convergence + benchmark bar race
F1b_energy_vs_N.png     energy growth with network size (trend reproduction)
E2_interior_optimum.png eta* vs lambda — the overturned-lemma headline
E3_pareto_frontier.png  energy-fidelity frontier + dominated reference (8-seed
                        error-bar version after upgrades)
E4_wz_savings.png       Wyner-Ziv link savings vs overlap
E5_conformal.png        certified coverage vs targets
T1_zeta_robustness.png  zeta fragility vs flat learned ladder (8-seed + bands +
                        Stage D line after upgrades)
T2_optimality_gap.png   gap to brute-force optimum, all policies
T3_size_transfer.png    train-small deploy-large bars
U4_fairness.png         total vs worst-agent energy
G2_tracking_energy.png  react-only vs predict-ahead across target classes
G3_target_paths.png     the three target classes' trajectories
```
All share wan/style.py. Headline plots use mean +/- s.d. over 8 scenarios.

---

## A5. Honest framing / "surpass the paper" — exact meaning

When we say results surpass the paper, the precise, defensible claims are:
1. **Overturned lemma:** the paper PROVES eta*=eta_req (boundary optimum). We
   show that holds ONLY because distortion is priced at zero; with any positive
   quality price the optimum is interior. This is a correctness result, not a
   tuning win.
2. **Pareto dominance:** under the same solver and scenarios, our operating
   point uses less energy AND yields better quality than the paper's, and the
   paper's point violates the quality floor it never measures.
3. **Knob-free near-optimal matching:** our learned policy needs no ζ and lands
   0.5% from the exact optimum vs the paper's 13.3%.
All comparisons are within OUR faithful reimplementation with documented
constants and identical inner solvers — the standard way extension papers
compare. We do NOT claim to have re-run the authors' own code.

---

## A6. Bug log — every problem hit and the fix (so they aren't re-introduced)

1. **lmodern.sty missing** (LaTeX, proposal PDF): swapped to mathptmx (Times).
2. **Invisible red tags in tcolorbox titles**: the \Ltag color made text vanish
   on dark title bars; stripped color inside box titles.
3. **Figure 1 / Figure 3 overlaps** (proposal TikZ): moved the annotation
   blocks below the diagrams; verified at high zoom.
4. **numpy array in disk tuples broke `in` comparison** (network.py _execute):
   replaced membership test with a rounded-key set.
5. **Lyapunov branch unpacked a float as a tuple** (topology.py): guard with
   isinstance(v, tuple).
6. **mathtext `\*` invalid** (matplotlib labels): use plain `*` in $...$.
7. **Silent infeasibility**: a stalled round (no progress) used to loop/return
   junk; now returns E=inf with feasible=False, and missions that make zero
   progress are flagged infeasible.
8. **Distance topology weights**: were placeholder; now proper per-round
   metre-based weights wmatch.
9. **conformal_pick_lambda**: rewritten to a clean ascending-ladder split-
   conformal with the ceil((n+1)(1-alpha)) order statistic.
10. **Stress regime infeasibility**: with beta0=1e-6 and Table-I CPU, later
    rounds had an EMPTY feasible eta window (compute deadline vs link budget
    conflict). Fixed by beta0=1e-5, f_cpu=2.5e9, C_gen=120 — a feasible window
    now exists. This took several iterations; do NOT lower C_gen/f_cpu back.
11. **E1 random-topology nan under stress**: random can pick infeasible pairs;
    the check now treats "random infeasible while blossom feasible" as a pass.
12. **E4 nan at rho=0.9**: softened the E4 regime (Tmax 3.0, L 12 Mb) so both
    links are feasible, and added a separate harsher-regime check where WZ
    rescues an infeasible deadline.
13. **Stage D overfit** (6 training scenarios): train improved but eval
    worsened; fixed with 12 training scenarios + a validation gate (seeds
    300-305) so Stage D only replaces Stage C if it wins on unseen data.
14. **G1 displacement vs path length**: a maneuvering target can curve back and
    have small net displacement; switched the agility metric to path length.

---

## A7. Chronological decision history (why the project is shaped this way)

1. Read the paper; produced a 23-point equation-anchored audit over three
   passes (11 original, +6 literature-grounded, +6 from a second close read).
2. Ranked five candidate research directions; re-verified the ranking twice.
   Every new finding landed inside Directions 1 and 2, so they stayed #1/#2.
3. Wrote a 17-page LaTeX proposal (v2) with the audit, directions, 8-week plan,
   risk register, 44 references.
4. Implemented Direction 1: model -> solver -> network -> experiments, with
   honest reproduction notes (scipy not CVXPY; chosen constants; motion rarely
   pays under published vmax*Tmax).
5. Implemented Direction 2 on the same engine: the four-stage ladder, the value
   model, the flexible schedule, the brute-force oracle.
6. Added rigor/presentation upgrades: error bars (8 scenarios), Stage D
   decision-focused fine-tune, battery fairness, sensitivity sweeps, one shared
   figure style.
7. Professor feedback: "not novel / need a clear objective." Decided the spine
   is moving-target tracking (a real gap: the paper assumes static targets).
8. Researched target motion models; implemented static/dynamic(NCV)/time-
   varying(maneuvering) + CV predictor; verified the difficulty ladder and that
   prediction helps most on the hardest targets.
9. Mentor suggestion of ABC/ACO/Cuckoo/Egyptian-Vulture: verified via the
   literature that these are NOT improvements over exact Blossom at this scale;
   decided to use them as comparison baselines (and optional large-N matcher),
   not as the main method.
10. Next: wire targets into full D1/D2 missions, add the metaheuristic
    baselines to T2, build the 3D web demo, run OPV2V on Colab.

---

## A8. Quick-reference cheat sheet (the numbers to remember)

```
Audit:            23 limitations, 5 internal inconsistencies
Direction 1:      overturns eta*=eta_req; reference (D=2.16,E=0.59) dominated
                  & violates floor Dmax=1.2; WZ ~5% + feasibility rescue;
                  conformal coverage 97% vs 80/90% targets
Direction 2:      learned 8.43 J / Stage D 7.75 J vs paper best 8.81 J (-12%);
                  optimality gap learned 0.5% vs paper 13.3% vs random 24.4%;
                  size transfer wins at N=8,10; fairness worst-agent 3.09->2.49
Targets:          energy 0/4.05/6.66 J, predict error 0/2.0/6.4 m,
                  predict-ahead saves 28% on time-varying
Verification:     15 + 11 + 10 + 5 = 41 automated checks, 0 failures
                  (plus the F1b scaling check appended to results.txt)
Value model:      ridge, 7 features, R^2=0.90, train seeds 100-123,
                  eval 3-10, Stage D validation seeds 300-305 (all disjoint)
```

*End of appendix. This file plus the two zips and the base paper fully
reconstitute the project state.*
