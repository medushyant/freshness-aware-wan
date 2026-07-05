# A-WAN — IMPLEMENTATION PLAYBOOK (THE EXECUTION SPEC)
## Read `01_AWAN_PROJECT_BRIEF.md` first. Then execute this file top-to-bottom.

This is the complete, self-contained execution specification for Phase 2. It assumes you (Claude Code) are running inside the repo root `BTP_ILAC_WAN/` with full file access. Web access is NOT required to execute this plan — everything is pinned here; §15 lists the only things worth re-verifying online if you have access.

---

## §0. PRIME DIRECTIVES (non-negotiable; re-read at the start of every session)

1. **FROZEN PHASE-1 CODE.** Never edit these existing files: everything currently under `wan/`, `experiments/`, `web/`, `scripts/`, `docs/`, `reference/`, and the root runners `run_direction1.py`, `run_direction2.py`, `run_repro_scaling.py`, `run_upgrades.py`, `run_targets.py`, `run_freshness.py`, `run_metaheuristics.py`, `export_web_data.py`, `opv2v_colab.py`, plus `PROJECT_REPORT.md`'s existing Parts I–XVII (you may APPEND new parts) and all `results*.txt`. If Phase-2 needs different behavior from an existing function: **wrap, subclass, or copy into `awan/` — never modify in place.** The only permitted edits to existing files: appending new sections to `PROJECT_REPORT.md` and `README.md`, and adding an A-WAN tab to the website by ADDING files under `web/` (plus the minimal nav hook, kept in a clearly marked `<!-- AWAN -->` block).
2. **REGRESSION GATE.** Before writing any new code, and again after completing each WP: run all seven Phase-1 runners in `.venv` and verify every PASS line still passes (§1.2). If any regression appears, stop and fix the cause (it will be an environment issue, since code is frozen).
3. **LEGITIMACY OF RESULTS.** No fabricated, hard-coded, or copy-pasted numbers — ever. Every number in every figure/table/README/report line must be produced by a run script reading a machine-written `results.json`. Every run logs its seed(s), config, git commit, and wall time. Every energy figure/caption labels each quantity **MEASURED** or **MODELED**. If a hypothesis fails (e.g., latent payloads do NOT beat text), report the honest negative — the PASS/FAIL harness may assert "a number was measured and reported", never "the number is favorable".
4. **TWO ENVIRONMENTS.** `.venv` (Python 3.14, numpy/scipy/networkx/matplotlib) stays exactly as-is for Phase-1 runners. Create `.venv-awan` on **Python 3.11 or 3.12** for everything new (torch/transformers/sionna wheels are not guaranteed on 3.14). Heavy GPU work (WP-3 VLM at scale, Sionna scenes) runs in Colab notebooks under `notebooks/` that `pip install -e .` this repo. Never install new packages into `.venv`.
5. **CONFIG-DRIVEN RUNS.** All Phase-2 experiments are driven by YAML in `awan/configs/`; each run writes `runs/<UTCstamp>_<name>/{config.yaml, results.json, log.txt}` and figures to `figures/awan/`. A tiny `awan/registry.py` maps figure IDs → the run that produced them.
6. **HARNESS STYLE PARITY.** Phase-2 runners (`run_awan_wp0.py` … `run_awan_wp4.py`, `run_awan_all.py`) print the same `PASS/FAIL  <id> <one-line claim>  | <numbers>` style as Phase 1 and append to `results_awan.txt`. Target: **+40–60 new automated checks**.
7. **FLOOR-FIRST ORDERING.** Within every WP, implement the floor tier completely (code + checks + figure) before touching expected/stretch tiers. Gates in §11 define go/no-go.
8. **STATISTICS.** Core scheme comparisons: ≥ 10 seeds, report mean ± 95% CI (t-interval or 10k-bootstrap), paired Wilcoxon signed-rank for A-vs-B on identical scenario seeds, and Cliff's delta or Cohen's d as effect size; N-sweeps: ≥ 5 seeds/point. Seeds fixed in configs (`seed_list: [0..9]`).
9. **ASK-THE-USER checkpoints (otherwise fully autonomous):** (a) any download > 5 GB; (b) anything requiring a paid API key; (c) anything requiring `sudo` (macOS `powermetrics`) — offer the MODELED fallback if declined; (d) deleting anything. Log every such decision in `docs/DECISIONS.md`.
10. **HONESTY SECTION.** Every WP contributes bullet(s) to a new "Part XV-B — Phase-2 Limitations and Assumptions" appended to `PROJECT_REPORT.md`, in the same voice as the existing Part XV (chosen constants disclosed, regimes documented, illustrative-vs-real distinctions stated).

---

## §1. WP-0 — INTEGRATION AUDIT & SCAFFOLD (Week 1; floor for everything)

### 1.1 Read and map
- Read `PROJECT_REPORT.md` end-to-end; read every file in `wan/` and the seven runners; read `reference/PROJECT_HANDOFF.md` and `reference/base_paper_text.txt` (the paper itself — re-derive Eqs. (1)–(38) references you'll need).
- Produce `docs/CODEBASE_MAP.md`: for each `wan/*.py`, list public functions/classes, their signatures, which paper equation each implements, and which runner exercises it. This map is your API contract.

### 1.2 Baseline freeze + regression test
```bash
source .venv/bin/activate
for r in run_direction1 run_direction2 run_repro_scaling run_upgrades run_targets run_freshness run_metaheuristics; do python $r.py; done
mkdir -p baseline && cp results*.txt baseline/
```
- New file `tests/test_regression_baseline.py` (runs under `.venv`): re-executes each runner via subprocess, diffs the set of `PASS`-line identifiers against `baseline/` (identifiers must match exactly; numeric tails may drift within a stated tolerance if any RNG is time-seeded — if so, prefer pinning the seed via env var WITHOUT editing frozen files; if impossible, compare identifiers only and record why in DECISIONS.md).
- Add `make regress` (or `scripts/regress.sh`) that runs this gate.

### 1.3 New scaffold (all Phase-2 code lives here)
```
awan/
  __init__.py
  adapters.py        # ONLY bridge to frozen code: solve_pair(), blossom_match(),
                     # potential_phi(), value_model_load/predict(), scenario_gen(seed),
                     # freshness_state(), imm_predict() — thin wrappers importing wan.*
  ledger.py          # EnergyLedger: every joule tagged {agent, round, kind∈
                     # {mobility, compute, comm_payload, comm_control, negotiation_llm},
                     # source∈{MEASURED, MODELED}}; invariant: totals == sum(parts)
  simcore.py         # unified round loop: state → coordination → pair execution →
                     # state update; pluggable Coordinator / Channel / Payload backends
  coord/
    hub.py           # paper baseline: hub-Blossom via adapters + EXPLICIT hub costing (§2.1)
    greedy_dist.py   # Rung A1: locally-heaviest-edge distributed matching
    auction.py       # Rung A2: Bertsekas auction w/ ε-scaling
    bids_learned.py  # Rung B: local-feature ridge bid function (+ optional fine-tune)
    negotiate_llm.py # Rung C: A2A-style FSM; engines: mock | local-hf | (api)
    messages.py      # message dataclasses, byte-accounting, JSON schema (§2.4)
  channel/
    paper.py         # h = β0 d^−δ (wraps existing constants)
    tier1.py         # shadowing GRF + Rician block fading + AR(1) (§3.1)
    sionna_map.py    # loads a precomputed path-gain grid (.npy) exported from Sionna RT
    predictor.py     # GP radio map (Matérn), twin+residual; μ(x), σ(x); online update
    conformal_cal.py # split-conformal κ calibration for deadline risk (§3.3)
  grounded/
    data_opv2v.py    # subset loader (reuses logic/paths from opv2v_colab.py); synthetic
    synth_views.py   #   overlapping-crops generator with ground-truth overlap
    vlm_agent.py     # SmolVLM2 wrapper → JSON facts; retry/parse guard
    memory_rag.py    # FAISS + SigLIP embeddings; dedup merge (cosine>τ)
    payloads.py      # codecs: text-tokens | latent-emb | kv-slice (stretch); bits counting
    energy_meter.py  # powermetrics/NVML samplers → mJ/token (prefill vs decode); MODELED fb
    trust.py         # overlap-consistency score, reputation EMA, gated fusion
    corrupt.py       # corruption operators (delete/swap/fabricate/jitter)
  mission/
    continuous.py    # event-triggered re-aggregation on the freshness spine; batteries
    metrics.py       # energy, F1/fidelity, staleness/AoI, lifetime, violation-rate
  surrogate.py       # ridge/MLP Ŵ(features)→W_ij for N-scaling
  configs/*.yaml     # one per experiment (E-IDs below)
  registry.py
run_awan_wp0.py … run_awan_wp4.py, run_awan_all.py
tests/  (pytest; §12 lists required tests)
notebooks/ (Colab mirrors: 01_sionna_scene.ipynb, 02_vlm_energy.ipynb, 03_opv2v_pipeline.ipynb)
```

### 1.4 Environment setup
```bash
# Phase-2 env (Python 3.11/3.12 via pyenv/uv/conda — pick what's available)
python3.11 -m venv .venv-awan && source .venv-awan/bin/activate
pip install -U pip
pip install numpy scipy networkx matplotlib pyyaml pytest pandas scikit-learn faiss-cpu \
            torch --index-url https://download.pytorch.org/whl/cpu   # local CPU default
pip install transformers accelerate pillow sentencepiece num2words   # VLM stack
# optional/stretch (guard imports; fine if absent locally, use Colab):
pip install sionna-rt            # v2.x; CPU needs LLVM (brew install llvm on macOS)
pip install pynvml codecarbon    # energy meters (NVML only meaningful on NVIDIA)
pip freeze > requirements-awan.lock
```
WP-0 checks (→ `results_awan.txt`): `PASS W0.1 regression gate green (58/58)`, `W0.2 adapters reproduce a known pair energy from results.txt within 1e-6`, `W0.3 ledger invariant holds on a 3-round sim`, `W0.4 config→run→results.json→figure round-trip works`.

---

## §2. WP-1 — DECENTRALIZED AGENTIC COORDINATION (Weeks 2–5)

### 2.1 Control-message & hub cost model (the L10 repair; define ONCE, use everywhere)
- A control message of `B_msg` payload bits (default 256; Rung-C JSON messages use their true UTF-8 byte length ×8) plus `B_hdr = 128` header bits, sent at fixed control power `P_ctrl = 10 dBm` over the SAME channel model in force, at rate `R_ctrl` from Eq. (9) with that power → per-message energy `E_msg = P_ctrl · (B_msg+B_hdr)/R_ctrl` and time `T_msg`. Broadcast within radius `R_comm` (default 250 m) = one transmission; unicast beyond = relayed hop-by-hop (count each hop).
- **Hub baseline costing (new, for fairness):** per round, each active agent uploads a state message to the hub (position, L, |Ω|, battery: 256 bits) and the hub broadcasts the pairing (⌈log2 N⌉·N bits). Hub compute assumed free (conservative — favors the paper). This yields `E_hub(N, rounds)` — the first explicit number for the paper's "free" coordination.
- Ledger kind: `comm_control` (all MODELED — say so).

### 2.2 Rung A — classical floor (two algorithms, both required)
**A1. Distributed greedy (locally-heaviest-edge; Preis/Hoepman-style ½-approximation):**
```
each agent i: candidates = feasible neighbors within R_comm (weights w̃_ij from adapters, or ∞→skip)
repeat until matched or no candidates:
  target(i) = argmin_j w̃_ij among unmatched candidates
  send PROPOSE(i→target(i))                       # 1 msg
  if PROPOSE received from j == target(i): both COMMIT (mutual-best edge)  # +2 msgs
  else on REJECT/timeout: remove j, retry
odd leftover agent → idle (matches Phase-1 flexible-schedule semantics)
```
Guarantee to state in report: greedy on mutually-heaviest edges achieves ≥ ½ of the optimal matching weight (equivalently ≤ 2× on min-cost after weight negation with offset — implement on benefit = w_max − w̃_ij and document the transformation). Track: rounds-to-converge, total messages, E_control.
**A2. Bertsekas auction (ε-scaling) for near-exact decentralization:**
- Build the standard assignment reduction: duplicate each agent as bidder and object on benefit b_ij = w_max − w̃_ij (i≠j), forbid self and already-retired; run distributed auction (each unassigned bidder bids best-minus-second-best + ε; objects accept highest bid; prices broadcast locally). ε-scaling: ε ← ε/4 from ε0 = w_max/2 down to ε < 1/(2M) ⇒ within M·ε of optimal ⇒ effectively exact on our scale. De-duplicate the symmetric assignment into a matching (keep pair if both directions agree; resolve conflicts by lower agent-id keeps, other re-enters — document). Count every bid/price message.
- Sanity oracle: at N ≤ 10 compare auction matching cost to `adapters.blossom_match` — must match within ε-bound on ≥ 95% of 50 seeds.

### 2.3 Rung B — the learned bid, decentralized
- Local observation o_i: own (p, L, |Ω|, battery) + same for neighbors within R_comm + centroid estimate. Centroid via **3 rounds of neighbor-averaging gossip** (each round = 1 broadcast/agent, costed); report centroid-estimate RMSE vs true.
- Bid weight: `w_ij^B = W_ij + V̂(s'_local | i→j)` where `W_ij` comes from the pairwise solve (each candidate pair runs `adapters.solve_pair` locally — allowed: it needs only the two agents' states) and `V̂` is the **frozen Phase-1 ridge value model** re-featurized on local quantities (write the explicit feature vector in `bids_learned.py`; if a Phase-1 feature is inherently global, substitute the gossip estimate and note it).
- Plug `w^B` into A1 and A2. Optional stretch: 200-episode zeroth-order fine-tune of the ridge weights on decentralized rollouts (reuse the Phase-1 Stage-D style perturbation trainer — do NOT import heavy RL frameworks; if you do want PPO, a self-contained ~200-line MAPPO on 2×64 MLPs is the ceiling).

### 2.4 Rung C — the LLM negotiation layer (the agentic centerpiece)
- **Message schema** (`coord/messages.py`; A2A-inspired, versioned `awan/neg/v1`): `AgentCard{agent_id, pos, payload_bits, omega_size, battery_frac, capabilities}`; `Propose{from, to, offered_role∈{sender,receiver}, est_pair_energy_J, est_value}`; `Counter{...}`; `Accept{pair:[i,j], role_map}`; `Decline{reason}`. Serialized JSON; byte length ×8 → bits → `E_msg` via §2.1.
- **FSM (deterministic envelope, bounded):** round-robin over unmatched agents by id; each may issue ≤ 1 Propose per cycle; recipient must Accept/Counter/Decline within the cycle; ≤ 3 cycles, then leftovers fall back to Rung-A greedy (guarantees termination and a valid matching — state this).
- **Engines:** (i) `mock` — a scripted policy (propose current best w^B; accept if within 10% of own best): default for all ≥10-seed statistics; (ii) `local-hf` — Qwen2.5-1.5B-Instruct (or 3B / Llama-3.2-3B-Instruct) via transformers, prompt contains the AgentCards in-range + "reply ONLY with one JSON action matching this schema"; JSON-parse with one retry then fall back to mock for that turn (count retries); run on ≥ 3 seeds × N ∈ {6,10} (compute-bounded); (iii) `api` — optional, only if the user supplies a key (§0.9b).
- **Token→energy:** count prompt+completion tokens per turn; `E_neg = tokens × e_tok` with `e_tok` = the WP-3 measured mJ/token for the local engine (until measured, use a clearly-labeled MODELED placeholder from the energy_meter's calibration run). Ledger kind `negotiation_llm`.
- **Break-even analysis (headline H4):** plot `E_hub(N)` vs `E_control^auction(N)` vs `E_neg^mock(N)` vs `E_neg^llm(N)` for N ∈ {4,6,8,10,20,50 (mock only beyond 10)}; report crossover N* (or its absence — honest either way).

### 2.5 Dropout & asynchrony
- Dropout: at the start of round k ∈ {2}, remove ⌈q·N⌉ random active agents, q ∈ {0, .1, .2, .3}. **Centralized semantics:** hub requires a fresh state message from every active agent; missing ⇒ hub stalls that round (retry next round; mission fails if it cannot complete within K+2 rounds) — this is the paper's implicit assumption made explicit; document it. **Decentralized semantics:** neighbors time-out and re-run local matching among survivors.
- Async stretch: per-agent clock jitter ±20% of T_max; event-driven matching among currently-idle agents.
- Metrics: completion rate, extra energy vs q=0, extra rounds.

### 2.6 WP-1 acceptance checks (ids → `results_awan.txt`; targets are hypotheses — report measured values regardless)
`A1.1` greedy valid matching every seed; `A1.2` greedy ≥ ½-bound holds empirically; `A1.3` auction within ε-bound of Blossom (≥95% of seeds); `A1.4` auction energy gap ≤ 5% vs Blossom at N∈{6,8,10} (H1); `A1.5` control energy < 3% of mission energy, and `E_hub` reported (H2); `A1.6` Rung-B ≤ Rung-A energy (learned bids help); `A1.7` dropout: decentralized completes at q=0.2 with ≤10% extra energy while centralized fails in a measured fraction of runs (H3); `A1.8` LLM engine: ≥90% schema-valid turns after ≤1 retry; `A1.9` break-even curve produced (H4); `A1.10` mock-vs-LLM matching quality gap reported.
Figures: **F1.1** energy: {Blossom, greedy, auction, Rung-B, Rung-C-mock} bars ± CI at N=6/8/10; **F1.2** messages & control-J per scheme; **F1.3** completion/energy vs dropout q (centralized cliff vs decentralized grace — a headline); **F1.4** break-even curve; **F1.5** one annotated negotiation transcript (figure or boxed listing).

---

## §3. WP-2 — DYNAMIC-CHANNEL INTELLIGENCE (Weeks 3–6, parallel track)

### 3.1 Tier-1 realistic channel (pure numpy; the floor)
`h(tx,rx,t) = β0 · d^−δ · 10^(χ(x_tx,x_rx)/10) · |g(t)|²` with:
- Shadowing χ: zero-mean Gaussian random field over the 500×500 m arena, exponential kernel `C(u,v)=σ_sh² exp(−||u−v||/d_c)`, σ_sh = 8 dB, d_c = 25 m. Implement: sample on a 5 m grid via Cholesky/FFT once per scenario seed, bilinear-interpolate; link value = field at midpoint (document this standard simplification).
- Small-scale g: Rician K = 6 dB (Rayleigh option), block-constant per transmission, AR(1) across an agent's successive transmissions with ρ_t = 0.7.
- Config: `channel: {model: tier1, sigma_sh_db, d_c_m, rician_K_db, ar1_rho}`; `model: paper` must reproduce Phase-1 numbers exactly (sanity check C0).

### 3.2 Per-agent radio-map predictor
- Each agent logs (position-pair midpoint, realized gain dB) samples from every transmission/overheard control message. Predictor A (primary): `sklearn.gaussian_process.GaussianProcessRegressor`, Matérn ν=1.5 + WhiteKernel, fit on own+shared samples (sharing costs messages — count them), predict μ(x), σ(x) on candidate endpoints. Predictor B: twin+residual — known β0·d^−δ term analytic, GP only on the shadowing residual (expected better sample-efficiency — measure it). Predictor C (stretch, notebook): LWM 1.1 embeddings (Hugging Face `wi-lab`) + linear head on DeepMIMO channels, as a forward-looking probe; document, don't gate anything on it.
- Report predictor RMSE (dB) vs #samples; A-vs-B learning curves.

### 3.3 Risk-aware planning + conformal deadline certificate
- Planning interface: since the frozen solver optimizes p_end continuously against a deterministic h, wrap it: sample a 7×7 candidate grid of endpoints inside each agent's patrol disc; for each candidate use `ĥ_lo = μ − κσ` (dB domain) as the channel handed to `adapters.solve_pair`; pick the feasible candidate with least energy (SAA-style; document as such — no frozen-code edits).
- κ calibration (split conformal): on a calibration set of 200 executed links (planned-with-μ, realized-under-tier1), nonconformity `s = realized_T_comm − planned_T_comm`; choose κ as the ⌈(1−α)(n+1)⌉/n empirical quantile mapped back through the rate function (implement the simpler direct version: calibrate a time-margin quantile q̂ and require planned_T + q̂ ≤ T_max; equivalent guarantee, cleaner math — pick ONE and document). Certificate: Pr(deadline met) ≥ 1−α on exchangeable test scenarios.
### 3.4 Experiments & checks
`C0` paper-channel parity with Phase-1 (exact); `C1` (H5a) deterministic paper plan executed under tier1: deadline-violation rate at σ_sh∈{4,8} dB — hypothesis ≥20%, report measured; `C2` (H5b) conformal planner empirical coverage ≥ 1−α at α=0.1/0.2 on held-out seeds; `C3` (H6) energy: predictive-planner vs move-to-shorter-distance vs static — Δ% reported; `C4` predictor A-vs-B RMSE curves; `C5` (stretch) Sionna: run `notebooks/01_sionna_scene.ipynb` (sionna-rt v2.x, built-in Munich scene, RadioMapSolver, cell 1 m) → export `assets/sionna_gain_grid.npy` → rerun C1–C3 with `channel: sionna_map`.
Figures: **F2.1** violation-rate bars (paper-plan vs conformal-plan, two α, two σ_sh); **F2.2** coverage vs target 1−α; **F2.3** energy bars of the three mobility policies; **F2.4** a radio map with agent trajectory overlay (μ heat + σ contours) — the visual headline; **F2.5** (stretch) same on the Sionna Munich map.

---

## §4. WP-3 — THE PHYSICALLY-GROUNDED PIPELINE (Weeks 4–9, parallel track; Colab-heavy)

### 4.1 Data
- Primary: **OPV2V** via the OpenCOOD layout (73 CARLA scenarios; agents 2–7/frame; 4 RGB cams + LiDAR + YAML ground-truth boxes per agent; full dataset ≈ 249 GB). **Download ONLY** `test_culvercity` and/or 2–4 scenarios from the chunked archives (UCLA Box / Google Drive links in the OpenCOOD docs; ≈ a few GB) — this is an §0.9a checkpoint. Loader in `grounded/data_opv2v.py`; start from and cite the repo's existing `opv2v_colab.py`.
- Fallback (always implemented, CPU-friendly): `grounded/synth_views.py` — take any wide image (bundle 20 CC0 street photos under `assets/`), cut K=4 overlapping crops with programmed pairwise IoU ∈ [0,0.9] → ground-truth geometric overlap known exactly; per-crop ground-truth "facts" from crop metadata.
- Frame→agent mapping: one OPV2V CAV = one WAN agent; its camera FoV polygon = its "patrol region" for the geometric-ρ comparison.

### 4.2 Agent perception + memory
- VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` primary (Apache-2.0; runs CPU/MPS/GPU; 256M variant for smoke tests; `Qwen/Qwen2.5-VL-3B-Instruct` as the quality alternative on Colab). Prompt: fixed template demanding ONLY JSON `{facts:[{object, attr, approx_location, confidence}]}`; strict-parse with one retry then regex-rescue; log parse-failure rate (must report; target <10%).
- RAG memory (`memory_rag.py`): SigLIP (or open_clip ViT-B) text+image embeddings in FAISS; **fusion op** replacing Eq. (2): incoming facts merged; duplicates = cosine > τ (τ=0.88 default, sweep) → keep higher-confidence copy. This yields a MEASURED fused size (sub-additive in overlap) vs the paper's additive rule — plot both.

### 4.3 Payload codecs (`payloads.py`) — bits counted exactly
- `P-text`: the JSON facts string; bits = UTF-8 bytes×8; compression η = keep top-⌈η·|facts|⌉ facts by confidence.
- `P-latent`: one 512-d fp16 embedding per fact (+8-byte tag) ⇒ 8,256 bits/fact; η = fact-count truncation; receiver matches embeddings into memory (no text needed).
- `P-kv` (stretch): C2C-style — sender's fact-summary prefix KV, layers truncated to r% by a recency/attention-mass heuristic; receiver conditions generation on injected KV (implement only for the SmolVLM2 text decoder; cite Cache-to-Cache, arXiv 2510.03215, code `thu-nics/C2C`, as the paradigm; ours is a simplified instantiation — say so). If engineering cost explodes, deliver `P-latent` results and document P-kv as designed-not-run (floor still met).

### 4.4 Measured quantities (the grounding of the whole paper)
- **ρ̂ vs ρ_geo (H7):** ρ̂_ij = |matched fact pairs|/|union| (embedding-matched at τ) or scene-embedding cosine; ρ_geo = FoV-polygon Jaccard. Deliver scatter, Spearman ρ_s, and % of agent-pair *ranking inversions* — this also completes the repo's own "remaining item #2".
- **Rate–fidelity (H10 input):** root-report fact-F1 vs ground-truth boxes (class+coarse location match), after full tree aggregation, as a function of payload bits, per codec ⇒ the MEASURED analogue of Phase-1's analytic D(η).
- **Energy (H8):** `energy_meter.py` — macOS: `sudo powermetrics --samplers cpu_power,gpu_power -i 200` sidecar, integrate over inference window, subtract 30 s idle baseline (needs §0.9c consent); Colab T4/L4: pynvml `nvmlDeviceGetPowerUsage` sampled at 10 Hz, same baseline subtraction; report **mJ/token, prefill vs decode separately**, ≥ 200 inferences. MODELED fallback (if no consent/GPU): time × device-TDP, labeled MODELED everywhere.
- **Grounding-gap table (H8):** per energy term (compute, comm@bits, mobility n/a): paper-model value with Table-I-style constants vs measured value for the same nominal task; report the ratio (hypothesis ≥10× on compute; print the measured ratio whatever it is).

### 4.5 Hallucination propagation + trust gate (absorbs old D3; H9)
- `corrupt.py` operators on one leaf agent's facts: `fabricate` (inject k=3 false objects), `swap` (class flips), `delete`, `jitter` (location noise). Base experiment: corrupted leaf → run tree → root corruption rate = fraction of runs where ≥1 corrupted fact survives to root (hypothesis ≈100% — the single-path L13 claim, now measured on a real pipeline).
- `trust.py`: for each communicating pair with FoV overlap, consistency c_ij = mean embedding agreement of facts inside the overlap region; reputation EMA τ_i ← 0.7τ_i + 0.3·c̄_i (init 1.0); **gated fusion**: facts from τ<0.5 sources are quarantined unless corroborated by an overlapping honest agent. Measure: root corruption rate and energy overhead (extra verification compute+messages) vs #corrupted leaves f ∈ {0,1,2}.
- **Conformal certificate on the real pipeline:** reuse the Phase-1 CRC recipe to pick the operating η (or fact-budget) s.t. Pr(root-F1 ≥ F_min) ≥ 1−α on held-out frames; report empirical coverage. This is the D1 machinery, re-validated on reality — a strong composition claim.

### 4.6 WP-3 checks
`G1` VLM JSON validity ≥90%; `G2` measured fused size sub-additive (vs Eq. (2) additive) — plot; `G3` (H7) Spearman + inversion% reported; `G4` (H10) per-codec rate–fidelity curves produced; latent-vs-text verdict at equal bits reported (either direction); `G5` (H8) mJ/token table (prefill/decode, ≥2 platforms or 1+MODELED) + grounding-gap ratios; `G6` (H9) base root-corruption ≈ measured X% and gated ≤20% target with ≤10% overhead — report measured; `G7` conformal coverage on real pipeline ≥1−α; `G8` all WP-3 figures regenerate from cached `runs/` artifacts WITHOUT re-running the VLM (cache inference outputs as JSON — mandatory for reproducibility on no-GPU machines).
Figures: **F3.1** ρ̂-vs-ρ_geo scatter + inversions; **F3.2** rate–fidelity per codec; **F3.3** mJ/token bars (prefill/decode × platform); **F3.4** grounding-gap table (rendered); **F3.5** corruption propagation: base vs gated vs f; **F3.6** measured-vs-Eq.(2) fused-size; **F3.7** pipeline diagram (schematic — label illustrative).

---

## §5. WP-4 — INTEGRATION, CONTINUOUS MISSION, GRAND SHOWDOWN (Weeks 8–11)

### 5.1 Unified simulator axes (all pluggable via one YAML)
`coordination ∈ {hub, greedy, auction, rungB, rungC_mock}` × `channel ∈ {paper, tier1[, sionna]}` × `payload ∈ {abstract, text, latent}` × `mission ∈ {oneshot, continuous}`. `abstract` payload = Phase-1 scalar model (so Phase-1 is a strict special case — assert equality sanity `I0`).
### 5.2 Continuous mission (builds ON `wan/freshness.py` + `wan/targets.py` via adapters — do not duplicate)
- Targets keep moving (IMM predictor); **event-triggered re-aggregation**: launch a new aggregation episode when predicted root staleness > S_max OR accumulated new-info bits > G_min (both configurable); between episodes agents sense locally.
- Per-agent battery B_i (J); agent dies at 0 (coverage hole logged); objectives reported: total energy, min residual (fairness, extends Phase-1 U4), **lifetime** = time-to-first-death.
- Metrics module: mission energy, root fact-F1 (grounded mode) or D (abstract), peak/mean staleness (reuse Phase-1 definition), AoI trajectory, deadline-violation rate, lifetime, completion under dropout.
### 5.3 Grand showdown + scaling
- **Grand table/figure (H11):** rows = {Paper H-MAP (hub, paper-channel, abstract), Phase-1 best (hub, paper, abstract+D1/D2/freshness), A-WAN (auction/rungB, tier1+conformal, latent, continuous, trust on)}; columns = the §5.2 metrics; ≥10 seeds, CIs, paired tests. This is THE headline figure of the whole project (F4.1).
- **Scaling:** N ∈ {10, 20, 50, 100(, 200)}: train `surrogate.py` (ridge, then small MLP if needed) on ≥5k solved pairs (features: distances, L's, ρ, battery) predicting W_ij; report surrogate MAPE, end-to-end energy penalty (<5% target), and wall-clock vs N for hub-Blossom / auction / greedy (Blossom O(N³) vs distributed) — first runtime curve for this framework at N≥100.
### 5.4 Website + report integration
- Add tab "A-WAN (Phase 2)" to `web/` (additive files + marked nav hook): the grand figure, dropout-grace animation-style chart, break-even curve, radio-map picture, corruption-propagation chart — all reading a new `web/data_awan.json` written by `export_awan_web_data.py` (new root script). Honesty rule identical to Phase 1.
- Append to `PROJECT_REPORT.md`: Part XVIII (Phase-2 overview), XIX (per-WP results with every number), XV-B (limitations), XVI-B (repro commands), update Part XIV. Never edit existing parts.
### 5.5 WP-4 checks
`I0` abstract+hub+paper == Phase-1 numbers (exact); `I1` grand table produced, A-WAN on the composite frontier (report honestly per metric); `I2` event-triggered ≤ periodic re-aggregation on energy at equal staleness cap; `I3` battery: lifetime under min-residual bidding ≥ sum-energy bidding (extends U4); `I4` surrogate MAPE + ≤5% energy penalty; `I5` runtime curve to N≥100; `I6` website tab renders (extend `scripts/shoot.py`-style Playwright check via a NEW script `scripts/shoot_awan.py`).
Figures: **F4.1** grand showdown; **F4.2** staleness/AoI trajectories one-shot vs event-triggered; **F4.3** lifetime/fairness bars; **F4.4** runtime & energy vs N (log-x).

---

## §6. EXPERIMENT MATRIX (anchor cells only — do NOT run the full factorial)
| ID | coord | channel | payload | mission | N | seeds | feeds |
|----|-------|---------|---------|---------|---|-------|-------|
| E1 | all 5 | paper | abstract | oneshot | 6/8/10 | 10 | F1.1–2, A1.* |
| E2 | hub, auction | tier1 | abstract | oneshot | 6/8/10 | 10 | F2.1–3, C1–C3 |
| E3 | auction | tier1 | abstract | oneshot, dropout q | 10 | 10 | F1.3 |
| E4 | hub | paper | text/latent(/kv) | oneshot | OPV2V frames | 5 | F3.2, G4 |
| E5 | auction+trust | tier1 | latent | oneshot, f corrupted | OPV2V | 5 | F3.5 |
| E6 | grand rows | as defined | as defined | continuous | 10 | 10 | F4.1–3 |
| E7 | hub/auction/greedy | tier1 | abstract(surrogate) | oneshot | 10–200 | 5 | F4.4 |
| E8 | rungC mock vs llm | paper | abstract | oneshot | 6/10 | 3(llm)/10(mock) | F1.4–5 |

## §7. STATISTICS PROTOCOL (binding)
Seeds 0–9 (0–4 where stated); scenario generation seeded through `adapters.scenario_gen`. Report mean ± 95% CI everywhere; paired Wilcoxon (identical seeds across schemes) with Holm correction inside each figure's comparison family; effect size (Cliff's δ) for headline pairs; never report a bare mean for a headline claim. `awan/stats.py` implements all of this once; every runner uses it.

## §8. MASTER FIGURE LIST (16) — each caption must name its E-ID, seed count, MEASURED/MODELED tags
F1.1 scheme energy bars · F1.2 control-overhead bars · F1.3 dropout grace-vs-cliff · F1.4 break-even curve · F1.5 negotiation transcript · F2.1 violation rates · F2.2 conformal coverage · F2.3 mobility-policy energy · F2.4 radio-map trajectory · (F2.5 Sionna) · F3.1 proxy scatter · F3.2 rate–fidelity · F3.3 mJ/token · F3.4 grounding-gap · F3.5 corruption propagation · F3.6 fused-size · F4.1 grand showdown · F4.2 staleness trajectories · F4.3 lifetime/fairness · F4.4 scaling. Style: reuse `wan/style.py` via adapters for visual continuity.

## §9. TWELVE-WEEK SCHEDULE (3 parallel tracks; weekly output in **bold**)
| Wk | Track-1 (WP-1) | Track-2 (WP-2) | Track-3 (WP-3) |
|----|----------------|----------------|----------------|
| 1 | — WP-0 for everyone: **regression green, CODEBASE_MAP, scaffold, adapters, ledger** — | | |
| 2 | msg model + hub costing + greedy → **A1.1–2** | tier1 channel + C0 → **parity** | synth_views + VLM smoke (256M) → **G1 draft** |
| 3 | auction + oracle check → **A1.3–4** | GP predictor + RMSE → **C4** | OPV2V subset decision/download; loader → **frames loading** |
| 4 | control-energy study → **A1.5, F1.2** | candidate-grid planner → **works e2e** | facts+RAG fusion → **G2, F3.6** |
| 5 | Rung-B bids + gossip → **A1.6, F1.1** | conformal κ + C2 → **F2.2** | codecs text/latent; rate–fidelity v1 → **F3.2 draft** |
| 6 | **GATE-1 (§11)** dropout study → **A1.7, F1.3** | C1 violation + C3 energy → **F2.1, F2.3** | energy meter Colab+mac → **G5, F3.3** |
| 7 | Rung-C mock FSM → **valid matchings** | radio-map figure → **F2.4** | ρ̂-vs-ρ_geo → **G3, F3.1** |
| 8 | Rung-C local-LLM + tokens → **A1.8–10, F1.4–5** | (stretch Sionna notebook) → **F2.5?** | corruption base + trust gate → **G6, F3.5** |
| 9 | **GATE-2** — WP-4 integration starts: **I0 parity**; continuous mission → **F4.2** | | conformal-on-real → **G7**; (P-kv stretch) |
| 10 | grand showdown runs → **F4.1, I1–I3** ; scaling+surrogate → **I4–5, F4.4** | | |
| 11 | **GATE-3** website tab + report Parts XVIII/XIX/XV-B → **I6**; full `run_awan_all.py` green | | |
| 12 | Buffer: ablations (τ sweep, σ_sh sweep, κ sweep), manuscript skeletons M1/M2 with real numbers, `results_awan.txt` final, tag release | | |

## §10. RISK REGISTER & FALLBACKS
| Risk | Trigger | Fallback (floor preserved) |
|------|---------|---------------------------|
| OPV2V download too big/slow | >1 day or >5 GB budget | synth_views only; label WP-3 "synthetic-grounded"; opv2v deferred to notebook |
| Sionna install pain (LLVM/GPU) | >0.5 day | Tier-1 only; Sionna = designed-not-run appendix |
| VLM JSON flakiness | parse-fail >10% | tighten template / switch 500M→2.2B on Colab / constrain with regex grammar; report rate |
| Local LLM negotiation slow | >2 s/turn | fewer seeds (3), N≤10 for llm engine; mock carries statistics (already the design) |
| powermetrics sudo declined | user says no | NVML-on-Colab only; else MODELED labeled |
| P-kv engineering blowup | >3 days | ship text+latent; P-kv = design appendix |
| Time slip | any gate red | drop stretch tiers in this order: P-kv → Sionna → MAPPO fine-tune → async → N=200 |
Top-2 risks overall: OPV2V logistics and Rung-C LLM engineering — both have zero-dependency fallbacks above; floors never depend on them.

## §11. GATES (go/no-go; log verdicts in DECISIONS.md)
GATE-1 (end wk 6): A1.1–A1.7 + C0–C3 + G1–G2 green ⇒ proceed; else cut stretch tiers now. GATE-2 (end wk 9): F1.*, F2.1–2.4, F3.1–3.5 exist ⇒ start showdown; else showdown uses whatever axes are green (document). GATE-3 (end wk 11): report+website integrated; wk 12 is polish only.

## §12. VERIFICATION & LEGITIMACY PROTOCOL (pytest under `.venv-awan`, plus the §1.2 gate under `.venv`)
Required tests: matching validity (every scheme, every seed: ≤1 link/agent, retired excluded); constraint feasibility of every executed pair (coverage disc, T_max with realized channel, power bound) — a `probe_silent_violation`-style sweep for Phase 2; ledger conservation (component sums == totals, no negatives/NaN); auction ε-bound vs Blossom; conformal coverage recomputation from raw calibration files; DP-oracle gap at N≤6 for every NEW matcher (extend the Phase-1 T2 oracle through adapters); channel parity C0; abstract-mode parity I0; figure-regeneration test (`make figures-awan` from cached runs/, no network, no GPU); schema validation of every stored results.json. Anti-fabrication: grep-guard test asserting no numeric literals in figure scripts that aren't read from results.json (whitelist axis limits).

## §13. NOVELTY-SAFETY CHECKLIST (re-run before EACH manuscript submission; ~30 min)
Search (Google Scholar + arXiv listing): citations of arXiv:2604.02381; the author group's (Zhao/Yang/Zhang/Chen/Huang) newest listings; phrases: "decentralized semantic aggregation", "distributed matching semantic communication", "auction agent pairing wireless", "LLM negotiation network resource", "wireless agent network LLM energy measured", "token communication multi-agent aggregation", "KV cache communication wireless", "hallucination propagation multi-agent aggregation", "conformal deadline wireless", "radio map Gaussian process trajectory aggregation". For each hit: note delta in `docs/NOVELTY_LOG.md`. Firsts we claim only if still clear: (1) costed decentralized coordination for progressive knowledge aggregation; (2) real-VLM instantiation of this WAN with measured energy + corruption propagation; (3) conformal deadline certification for this framework.

## §14. MANUSCRIPT SKELETONS (draft in `docs/` in week 12, numbers auto-injected from results.json)
**M1** "Decentralized, Channel-Predictive Coordination for Wireless Agent Networks" — I. Intro (paper's own future-work sentence as the hook) II. System (Phase-1 recap 1 page) III. Costed decentralized matching (½-approx + auction + learned bids) IV. Channel intelligence + conformal certificate V. Results (F1.1–4, F2.1–4, F4.4) VI. Related VII. Conclusion. Target: IEEE TCCN / IoT-J.
**M2** "Grounding the Wireless Agent Network: Measured Semantics, Energy, and Hallucination Propagation" — hooks: H7/H8/H9 tables + F3.*; conformal-on-real as the constructive fix. Target: IEEE Networking Letters or ICC/GLOBECOM workshop.

## §15. ONLY-IF-ONLINE VERIFY LIST (everything else is pinned; do not block on this)
pip latest of: sionna-rt (was 2.x, PyPI, Mar 2026), transformers, faiss-cpu; HF ids exist: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`, SigLIP checkpoint, `wi-lab` LWM; OpenCOOD chunked-download links live; §13 novelty sweep.

## §16. DEFAULT HYPERPARAMETERS (single source of truth = `awan/configs/defaults.yaml`; disclosed constants, Phase-1 Part-XV style)
Arena 500×500 m; stress regime constants inherited from Phase-1; R_comm 250 m; B_msg 256+128 bits; P_ctrl 10 dBm; auction ε0 = w_max/2, ÷4 scaling; gossip rounds 3; σ_sh 8 dB (sweep 4/8); d_c 25 m; Rician K 6 dB; AR1 ρ_t 0.7; GP Matérn ν 1.5; α ∈ {0.1, 0.2}; candidate grid 7×7; τ_dedup 0.88 (sweep 0.8–0.95); trust EMA 0.7/threshold 0.5; corrupt k 3 facts; S_max/G_min from a 1-page calibration run (disclose); battery B_i = 3× mean one-shot per-agent energy; seeds 0–9.

— END. Work floor-first, keep the ledger honest, keep Phase 1 green, and print PASS/FAIL for everything. —
