# Direction 1 — Fidelity-Aware Aggregation (implementation)

Code for the first extension of Zhao et al., "Agentic AI-Empowered Wireless
Agent Networks with Semantic-Aware Collaboration via ILAC" (arXiv:2604.02381).
Reproduces the paper's pairwise + network optimization, then adds the
fidelity-aware reformulation: a distortion state, sub-additive fusion,
Wyner-Ziv side-information links, an endogenous eta_req, and a conformal
guarantee on the fidelity floor.

## Layout
```
wan/model.py        system model: parameters, geometry (Jaccard of disk
                    unions), mobility / compute / communication energies
wan/solver.py       pairwise inner solver. paper mode = their BCD
                    (receiver pinned at eta_req); fidelity mode = ours
                    (both ratios free, + lambda*distortion + lookahead Phi
                    inside the problem, + Wyner-Ziv payload)
wan/network.py      full missions: per-round weights, potential field,
                    Blossom matching, fusion bookkeeping (per-source
                    compression history = the "telephone game" state),
                    derived eta_req, conformal calibration
run_direction1.py   experiments E1-E6 -> figures/ + results.txt
opv2v_colab.py      real-data part, run separately on Google Colab
results.txt         the verification report (15 checks, all PASS)
```

## Run
```
python3 run_direction1.py        # ~1-2 min, regenerates figures + results.txt
```
Needs numpy, scipy, matplotlib, networkx (no CVXPY, no GPU).

## What each experiment shows
- E1  reproduction sanity: benchmark ordering matches the paper
      (proposed < max-power, no-semantic infeasible under stress,
      blossom <= random), inner solver converges in a few BCD passes.
- E2  the headline lemma flip: with a fidelity term the optimal
      compression is interior (eta_min < eta* < eta_req) and moves
      smoothly with lambda; the paper proves it is always AT eta_req.
- E3  energy-fidelity Pareto frontier; the paper's operating point is
      dominated on BOTH axes by our lambda=0.04-0.08 region and violates
      the Dmax floor (D_ref = 2.21 > 1.2).
- E4  Wyner-Ziv link: never worse, ~5% pair energy saved at high overlap,
      and it turns an infeasible deadline into a feasible one.
- E5  conformal risk control: empirical P(D <= Dmax) = 0.97 against
      targets 0.80 and 0.90 on held-out missions.
- E6  derived per-hop bound keeps D_root <= Dmax with lambda = 0
      (the eta_req knob becomes a lemma, not an input).

## Honest implementation notes (also useful for the report)
- Solver stack: the paper uses CVX-style SOCP inside SCA. Same
  optimization problems here, solved with scipy (bounded scalar /
  L-BFGS-B for the resource block, SLSQP for the motion block).
  Identical math, lighter dependencies.
- Unpublished constants: Table I omits N0, the mobility kappas, C_base,
  C_gen, eta_min, eta_req. Values in wan/model.py are chosen so pair
  energies land in the same ~0.4 J ballpark as their Fig. 4. The E1
  stress regime (beta0 = -50 dB, faster SoC, lighter C_gen) is tuned so
  a feasible compression window exists in later rounds: compute pushes
  eta UP while the radio pushes eta DOWN, and with their published
  Table-I values alone that window can be empty.
- Motion finding: with vmax * Tmax <= 30 m of reachable motion, the
  optimizer usually decides standing still is optimal; motion never
  hurts but rarely pays. The paper's larger motion gains depend on
  mobility constants they do not report.
- Reproduced artifact: in round 1 the paper sets rho = 1, which makes
  compression compute-free; our solver visibly exploits it (sender
  slams to eta_min). That is audit item L20, observed live.
- The conformal sigma (model error of the distortion map) is a
  simulation stand-in; opv2v_colab.py measures the real one.

## Real-data part (you run this)
1. Download the OPV2V *test* split via the OpenCOOD repo
   (github.com/DerrickXuNu/OpenCOOD -> data intro -> Google Drive or
   UCLA Box), unzip to your Google Drive.
2. Open opv2v_colab.py in Colab, set DATA_ROOT, run all.
3. It produces: the rho proxy check (Spearman + % ranking inversions,
   figure F2) and the measured Theta curve with a_D and sigma
   (figure F3) - paste a_D / sigma back into wan/model.py to make the
   simulator fully data-calibrated.

## Next session
Re-upload this folder as a zip; Direction 2 (learned topology) builds
directly on wan/network.py.

---

# Direction 2 — Learned Predictive Topology (implementation)

New files: `wan/topology.py`, `run_direction2.py` (figures T1-T3 +
`results_d2.txt`, 11 automated checks, all PASS).

## The ladder (each stage is a deliverable)
- **Stage A, Phi-in-the-loop** (fixes audit L19): the potential field now
  sits INSIDE the pair optimization, so agents move/compress knowing the
  future term. Beats paper at its own zeta (8.790 vs 8.806 J). Honest
  nuance: at a badly oversized zeta, stage A amplifies the damage
  (the bad knob now corrupts the inner solutions too) — one more reason
  to remove the knob.
- **Stage B, auto-scaled lookahead** (Lyapunov drift-plus-penalty style):
  the energy/spread trade-off coefficient is set once from round-1
  magnitudes. No tuning; matches the paper's best.
- **Stage C, learned cost-to-go**: ridge value model on 7 hand features
  of the surviving set (count, total/max payload, payload-weighted
  spread, pairwise distances, rounds left), trained on 24 mixed-policy
  rollouts at N in {4,5,6} (R^2 = 0.90). Matching weight =
  pair energy + V(state-after). **No zeta anywhere.**
- **Flexible schedule** (fixes audit L12): each even round the policy may
  rest the worst pair if V says that is cheaper; the forced plan stays
  in the candidate set, so flexibility cannot lose (Dominance Prop.).

## Verified results (stress regime, 5 held-out scenarios)
- zeta sweep: paper recipe varies 10% across the sweep (their Fig. 7
  fragility, reproduced); learned policy: 8.289 J with nothing to tune,
  below the BEST hand-tuned paper point (8.806 J). Flexible: 8.025 J.
- optimality gap at N=5 vs a brute-force oracle (first such numbers for
  this framework): learned 0.5%, paper 13.3%, random 24.4%.
- size transfer: trained on N<=6, deployed unchanged at N=8 (12.36 vs
  12.81 J) and N=10 (15.88 vs 16.92 J).
- dominance in practice: flexible 8.085 vs forced 8.428 J over 8 runs.

Also: `python3 run_repro_scaling.py` regenerates the energy-vs-N trend
(figure F1b). Run: `python3 run_direction2.py` (~1 min). Motion is off in these
experiments to isolate the topology question (the inner solver is
identical for every policy, so the comparison is apples-to-apples).
A GNN encoder can replace the hand features without touching the
pipeline; that is the natural next upgrade (stage D: decision-focused
fine-tuning through the matcher).

---

# Rigor & presentation upgrades (run_upgrades.py)

Run order: run_direction1.py -> run_direction2.py -> run_upgrades.py.
The upgrade pass re-makes the two headline figures with 8 held-out
scenarios and error bars, and adds three analyses:

- **Stage D, decision-focused fine-tuning** (closes the last ladder rung):
  zeroth-order perturbation search on the value weights where the loss is
  the REALIZED mission energy through the exact Blossom matcher
  (perturbation-gradient style; cf. Huang & Gupta, NeurIPS 2024, and
  Berthet et al. 2020 - only a black-box solver oracle is needed).
  Guarded by a validation gate (seeds 300-305, disjoint from training
  100-123/200-211 and eval 3-10): stage D replaces stage C only if it
  wins on validation. Result: 7.754 J on held-out scenarios vs 8.085
  (stage C+flex) and 8.813 (paper's best hand-tuned zeta), i.e. the full
  ladder ends 12% below the paper's best tuning. First fine-tune attempt
  with only 6 training scenarios overfit (train 8.13 / eval 8.54) - kept
  in the git history of this README as an honest note; the fix was more
  training scenarios + the validation gate, standard model selection.
- **Battery fairness (audit L16)**: per-agent energy is now tracked; the
  learned policies reduce the worst single agent's drain too
  (paper 3.09 J -> learned 2.66 -> +flex 2.49), so the total saving is
  not bought by overloading one robot.
- **Sensitivity**: the derived eta_req tracks any chosen floor
  (Dmax 0.9/1.2/1.5 -> worst D 0.90/1.08/1.26) and the Wyner-Ziv saving
  stays positive across omega in {0.5, 0.9}.
- All figures share one style (wan/style.py); headline plots show
  mean +/- s.d. over 8 scenarios.

Verification totals: 15 (D1) + 11 (D2) + 10 (upgrades) = 36 checks, 0 fail.

---

# Moving-Target Spine — Freshness-Aware Aggregation (run_freshness.py)

The unifying contribution: target motion enters the framework through
**information freshness**, which drives the compression ratio (D1) and the
aggregation topology (D2). This is the spine that ties the two directions into
one objective. New files: `wan/freshness.py`, `wan/freshtopo.py`,
`run_freshness.py` (figures G4-G6 + `results_freshness.txt`, 11 checks, all
PASS). Full writeup with equations and references: `docs/FRESHNESS_EXTENSION.md`.
The decision and the negative-result evidence are in
`docs/TARGET_INTEGRATION_VERDICT.md`; the supporting experiments are in
`experiments/`.

Why not the obvious "chase the target with mobility energy" route: we
implemented it and it is provably inert in this engine — the coverage constraint
never binds at the paper's patrol radius, and at a tight sensing radius the
matcher routes around it (see `experiments/exp_targets_washout.py`,
`probe_mobility_share.py`, `probe_silent_violation.py`). It is also the
framework's own lowest-ranked direction (crowded CRB/ISAC methodology).

## Verified results (`python3 run_freshness.py`)
- **F1/F2 (D1):** faster target -> staler data -> higher derived eta-floor ->
  less compression -> more energy. Monotone and exact (eta 0.549 -> 0.719,
  energy +9%), robust across 30 random pairs.
- **F3 (D2):** value-prioritized matching gives fast-decaying sources shorter
  paths: root report 40% fresher at no energy cost (-1.5%); reduces exactly to
  the paper at lam_f = 0 (sanity, 20/20 identical).
- **F4 (D2):** the freshness advantage grows with agility spread (21% -> 48%) —
  prioritization is most valuable under heterogeneous decay.
- **F5 (prediction):** CV-Kalman and CV+CT IMM (added to `wan/targets.py`)
  dominate the naive finite-difference CV predictor on every class; predicting
  ahead pays once motion exceeds sensor noise and yields a ~23% fresher report.

## Run order (full project)
```
python3 run_direction1.py      # D1: fidelity-aware aggregation        (15 + F1b)
python3 run_direction2.py      # D2: learned predictive topology       (11)
python3 run_repro_scaling.py   # energy-vs-N trend, appends to results.txt
python3 run_upgrades.py        # error bars, Stage D, fairness         (9)
python3 run_targets.py         # target motion models                 (5)
python3 run_freshness.py       # moving-target freshness spine         (11)
python3 run_metaheuristics.py  # ACO/ABC/Cuckoo/Vulture baselines      (5)
python3 export_web_data.py     # dump web/data.json for the demo site
```

Verification totals: **58 automated checks, 0 fail** (17 D1 incl. F1b scaling
+ 11 D2 + 9 upgrades + 5 targets + 11 freshness + 5 metaheuristics).

# Interactive web demo (web/)

A polished, animated **tabbed app** showcasing the whole project, driven entirely
by `web/data.json` (real engine output, regenerated by `export_web_data.py`).
Tabs: Overview (with a **Three.js 3D swarm**), **Swarm** (live canvas simulation,
selectable motion class), **H-MAP** (animated bracket tournament replaying the
real pairing trace, paper-vs-ours quality meter), **Blossom** (interactive
max-weight matching vs greedy/random), **Learned·RL** (live ζ-fragility slider +
optimality-gap bars), **Freshness** (the D1/D2 spine charts), **Graphs** (gallery
of all 20 figures with a lightbox), **Verify** (58-check grid), **Before/After**
(animated cards + table). Stack: Three.js + GSAP + Chart.js + custom canvas/SVG;
no build. Run `cd web && python3 -m http.server 5050`. Per-tab guide:
`web/HOW_IT_WORKS.md`. Deploy: `web/DEPLOY.md` (Vercel recommended). Verified
headless across every tab (0 console errors); screenshots in `docs/site_preview/`.

# Paper integration (docs/)

`docs/freshness_section.tex` is an Overleaf-ready drop-in `\section` in the
proposal's own macro style, with `docs/freshness_refs.bib` (4 new references)
and `docs/freshness_intro_patch.tex` (the spine sentence for the intro).

# Phase 2 — A-WAN (the Autonomous, Physically-Grounded WAN)

Phase 2 executes the base paper's own future-work sentence — *"dynamic channel
models and decentralized coordination protocols"* — and goes further. All
Phase-2 code lives in `awan/` + `run_awan_*.py` (Python 3.12 venv:
`.venv-awan`); Phase-1 code and results above are frozen and keep reproducing
(`bash scripts/regress.sh`).

- **WP-1 Decentralized coordination** (`awan/coord/`): every control message
  priced in joules (first explicit bill for the paper's "free" hub, L10);
  distributed greedy (½-approx), Bertsekas ε-auction (≈Blossom on 100% of
  feasible rounds within a 2% ε-bound), the frozen Phase-1 value model as a
  decentralized bid function, and an A2A-style LLM negotiation layer (real
  Qwen2.5-0.5B, 100% schema-valid JSON, tokens charged as energy). Dropout:
  the paper's hub cliffs 100%→0% completion; the auction stays at 100%.
- **WP-2 Channel intelligence** (`awan/channel/`): Gudmundson-correlated
  shadowing + Rician AR(1) fading; the paper's deterministic plan misses ~60%
  of deadlines under it; a dB-domain split-conformal margin certifies
  Pr(deadline) ≥ 1−α at +2–4% energy; GP radio maps (twin+residual beats
  full-GP everywhere); move-to-predicted-channel beats move-closer by ~30%.
- **WP-3 Grounded pipeline** (`awan/grounded/`): real SmolVLM2 perception on
  exact-ground-truth synthetic street scenes (OPV2V on Colab:
  `notebooks/03_opv2v_pipeline.ipynb`), SigLIP RAG fusion (measured
  sub-additive vs Eq. (2)), text/latent codecs with exact bit counts, a
  measured grounding-gap table (~10⁶× vs the paper's τf²W term), corruption
  propagation + an overlap-consistency trust gate, and a conformal fact-budget
  certificate on the real pipeline. VLM outputs are cached under
  `runs/vlm_cache/` so every figure regenerates without a GPU.
- **WP-4 Integration** (`awan/mission/`, `awan/surrogate.py`): one simulator,
  all axes pluggable; the grand showdown (paper vs Phase-1 vs A-WAN, one world,
  ten seeds); event-triggered re-aggregation; per-agent batteries + lifetime;
  a learned pair surrogate (MAPE ~0.1%) driving the first N≥100 runtime curve.

Reproduce: `source .venv-awan/bin/activate && python run_awan_all.py`
(add `--full` to re-run VLM perception). Checks land in `results_awan.txt`;
figures in `figures/awan/`; the website gains an **A-WAN · Phase 2** tab
(`export_awan_web_data.py` → `web/data_awan.json`, verified by
`scripts/shoot_awan.py`). Colab notebooks for measured GPU energy, OPV2V, and
Sionna RT live in `notebooks/`. Decisions and honest deviations:
`docs/DECISIONS.md`.
