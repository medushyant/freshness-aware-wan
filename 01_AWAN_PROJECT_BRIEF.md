# A-WAN — PROJECT BRIEF (READ THIS FIRST)
## The Autonomous, Physically-Grounded Wireless Agent Network
### Phase-2 extension of the B.Tech project on arXiv:2604.02381 — complete context + mission definition

> **Who you are:** Claude Code, working inside the existing repository `BTP_ILAC_WAN/`.
> **What this file is:** the complete context of a ~5–6 month research project — everything decided across the whole planning engagement, distilled. Read it fully before touching anything.
> **What the companion file is:** `02_AWAN_IMPLEMENTATION_PLAYBOOK.md` — the exhaustive, step-by-step execution spec. This brief = WHAT and WHY. The playbook = HOW, exactly.
> **Prime rule (repeated in the playbook):** ALL existing code and results in this repo are FROZEN and must keep reproducing bit-for-bit. Phase 2 is purely additive.

---

## 1. The reference paper we extend and critique

**"Agentic AI-Empowered Wireless Agent Networks With Semantic-Aware Collaboration via ILAC"** — Zhouxiang Zhao, Jiaxiang Wang, Zhaohui Yang, Kun Yang, Zhaoyang Zhang, Mingzhe Chen, Kaibin Huang. arXiv:2604.02381 [cs.NI], April 2026. Full text lives at `reference/base_paper_text.txt` in this repo.

**In one paragraph.** N mobile agents, each carrying an "embodied large model" (ELM), patrol circular regions (position constraint Eq. (1): ||p_i − c_i|| ≤ R_i). A global situation report is built by *progressive knowledge aggregation*: a knockout tournament over K = ⌈log2 N⌉ synchronous rounds in which agents pair up, the sender's knowledge is semantically fused into the receiver (information-source state Ω evolves by set union; payload by Eq. (2): L_j⁺ = η_j L_j + η_i L_i, where η ∈ (0,1] is the semantic compression ratio), the sender retires, and the surviving half continues until one root agent holds everything. For each pair the paper jointly optimizes: **where to move** (polynomial mobility power, Eqs. (3)–(4)), **how hard to compress** (compute cost Eq. (6): W = L[C_base + C_gen·γ·(1−ρ)·ln(1/η)], energy E = τ f² W, where ρ is a *geometric* Jaccard overlap of patrol-area unions, Eq. (5)), and **how much transmit power** (deterministic path-loss channel h = β0·d^(−δ), Shannon rate, orthogonal channels, Eqs. (8)–(11)), all under a per-round latency deadline T_max (Eq. (15b)).

**The solver (H-MAP).** Inner level (P1): per-pair energy minimization via BCD — a provably convex resource block (bisection over η; receiver's optimum sits on its bound η*_j = η_req) + an SCA/SOCP motion block. Outer level (P2): per-round pairing by minimum-weight perfect matching (Edmonds' Blossom) on *symmetrized* weights min(w_ij, w_ji), augmented by a hand-tuned potential field Φ = ζ·||p_end − centroid||^δ · L (Eqs. (34)–(38)); exactly ⌊N/2⌋ pairs forced per round (Eq. (16b)). Complexity O(N³). Validated only at N ≤ 10, fully synthetic; ~19% energy gain vs a distance-based topology at N = 10.

**The paper's OWN confessed limitations (its conclusion, near-verbatim — this is the professor's priority target):** the model assumes a **simplified path-loss channel** and relies on **centralized coordination**; future work = *"incorporation of agentic intelligence into dynamic channel models and decentralized coordination protocols."* Phase 2 executes that sentence, literally, and goes further.

---

## 2. Everything already accomplished (Phase 1 — FROZEN, the baseline to beat)

The authoritative record is `PROJECT_REPORT.md` in this repo (read it in full during WP-0). Summary:

### 2.1 The 23-point limitation audit (L1–L23)
Anchored, equation-level limitations, incl. five internal inconsistencies. Highlights relevant to Phase 2: L6 deterministic channel; L7 unlimited orthogonal spectrum; **L10 the central hub and the final root-to-sink hop are never energy-costed**; L11 N ≤ 10 synthetic validation; L13 no trust model — a single corrupted leaf poisons the root through the single-path tree; L15 payloads are scalar bit counts while 2025–26 agents exchange tokens/KV-cache; L16 sum-energy ignores per-agent batteries; L17 one-shot episode vs continuous mission.

### 2.2 Direction 1 — Fidelity-aware aggregation (IMPLEMENTED, verified: `run_direction1.py` → `results.txt`, 15+2 checks)
CEO/Wyner–Ziv reformulation + conformal risk control. Key results: the paper's boundary optimum η* = η_req becomes **interior** under a fidelity floor (η*_j ∈ [0.16, 0.72], monotone in λ); the reference optimum **violates** the fidelity floor (D_ref = 2.21 > D_max = 1.20) while a frontier point dominates it on both axes; Wyner–Ziv side-information payloads turn an infeasible deadline feasible and save ~5% pair energy at overlap ρ = 0.9; conformal coverage 0.97 ≥ 1−α at α = 0.20 and 0.10; a derived per-hop bound enforces D_root ≤ D_max with λ = 0.

### 2.3 Direction 2 — Learned predictive topology (IMPLEMENTED, verified: `run_direction2.py` → `results_d2.txt`, 11 checks; + `run_upgrades.py`, 9 checks)
Stage A (Φ-in-the-loop), Stage B (Lyapunov, auto-scaled, no knob), Stage C (learned ridge-regression cost-to-go, R² = 0.90), Stage D (zeroth-order decision-focused fine-tune), flexible schedule (idling allowed). Key results: paper recipe is ζ-fragile (worst/best 1.10); learned with **no knob** beats the *best* hand-tuned paper (8.289 vs 8.806 J); flexible schedule helps further (8.025 J); **first optimality-gap report** at N = 5: learned 0.5% vs paper 13.3%, greedy 18.9%, random 24.4%; size transfer holds unchanged at N = 8 and N = 10; Stage D full ladder −12% vs paper's best; fairness: learned also improves the worst-agent bottleneck (2.66 vs 3.09 J).

### 2.4 The freshness / moving-target spine (IMPLEMENTED — the Phase-1 headline novelty: `run_targets.py`, `run_freshness.py`, `wan/freshness.py`, `wan/freshtopo.py`, `wan/targets.py`)
The paper assumes a static world; Phase 1 added moving targets and a processing-age freshness model coupled into BOTH directions: the η-floor rises with target speed; freshness-aware topology cuts root staleness **40% at −1.5% energy**; Kalman/IMM prediction dominates naive differencing and saves 28% tracking energy on time-varying targets; λ_F = 0 exactly recovers the paper (20/20 sanity).

### 2.5 Also done
Metaheuristic matcher baselines (`run_metaheuristics.py`: exact Blossom remains optimal and fast, 12.3 ms at N = 30 — metaheuristics are baselines only); rigor upgrades (error bars, sensitivity); a **9-tab interactive website** (`web/`, Chart.js/GSAP/Three.js, Playwright-verified, Vercel-ready); a LaTeX paper section; **58 automated PASS/FAIL checks and 20 regenerable figures**; `opv2v_colab.py` (a prepared but NOT-yet-run OPV2V calibration script — Phase 2 WP-3 completes this thread).

### 2.6 The existing tech stack (do not break it)
Python **3.14** venv at `.venv/`; numpy 2.5, scipy 1.18 (BCD/SCA via `minimize`/`minimize_scalar`, SLSQP/L-BFGS-B — **deliberately no CVXPY**), networkx 3.6 (`max_weight_matching`), matplotlib 3.11; ridge-regression value model (transparent, no heavy NN); custom PASS/FAIL harness; documented "stress regime" (T_max = 2.5 s, harsher channel etc.) in which energy actually responds to decisions — keep using it, and keep the Part-XV honesty disclosures pattern.

---

## 3. Phase 2 — the A-WAN mega-project (what you will now build)

**Thesis, one sentence:** *The paper's own two confessed weaknesses — a central boss and a toy radio — plus a third we exposed — a fake AI brain (a scalar η·L instead of any real model) — are all eliminated by one agentic architecture: agents negotiate pairings themselves, predict a realistic channel before moving, and exchange real vision-language-model knowledge, with every claim measured, certified, and benchmarked against both the paper and our own Phase-1 system.*

Four work packages. Each has a **guaranteed floor** (defensible even if the ambitious tier slips), an **expected** tier, and a **stretch** tier. Full specs, pseudocode, acceptance criteria, and schedules are in the playbook.

### WP-1 — Decentralized Agentic Coordination *(kills the central hub; answers future-work clause #2; repairs L10)*
Replace Algorithm 2's hub + centralized Blossom with agent-side decision-making, in three rungs:
- **Rung A (floor):** distributed greedy matching (locally-heaviest-edge, ½-approximation guarantee) and a Bertsekas-style auction with ε-scaling — with EVERY control message counted in bits and joules, including, for the first time in this line, the hub's own uncosted overhead for comparison.
- **Rung B (expected):** the existing Phase-1 ridge cost-to-go becomes each agent's **local bid function** over local observations only (positions/payloads within radio range + a gossip-estimated centroid) — decentralized execution of the already-proven learned lookahead.
- **Rung C (agentic centerpiece — the professor's favorite):** an **LLM-agent negotiation layer**: agents exchange A2A-protocol-style JSON messages (Agent Card → propose → counter → accept) to form pairs under partial information; the negotiation's **token cost is charged to the mission energy budget** (mJ/token from WP-3 measurements), yielding the first protocol-overhead-in-joules accounting for semantic aggregation, plus a break-even analysis: at what N does talking cost more than the hub it replaces? Runs with a deterministic mock policy (for statistics), a local small LLM (Qwen2.5-1.5B/3B-Instruct class), and optionally an API model. (Context: Google's A2A is now a Linux Foundation standard with 150+ member orgs and a stable spec as of April 2026; MCP is its complement — our schema mirrors it, which is the market-alignment hook.)
- **Robustness:** agent-dropout (10–30%) and asynchrony experiments — decentralized degrades gracefully; the hub design stalls.

### WP-2 — Agentic Dynamic-Channel Intelligence *(kills Eq. (8)'s static channel; answers future-work clause #1; supersedes old D5)*
- **Tier-1 (floor, pure numpy):** a realistic stochastic channel — path loss + spatially-correlated log-normal shadowing (Gaussian random field, decorrelation ~25 m) + Rician/Rayleigh block fading with AR(1) time correlation.
- **Predictor (expected):** each agent maintains a learned **radio map** — Gaussian-process (Matérn) on visited-location measurements, and a digital-twin+residual variant — giving μ(x) and uncertainty σ(x) for any candidate position.
- **Risk-aware planning (expected):** the motion step plans against ĥ_lo = μ − κσ, with κ **calibrated by split conformal prediction** so the deadline-violation probability is certified ≤ α (extends Phase-1's conformal machinery from fidelity to the channel).
- **Stretch:** ground truth from **Sionna RT v2.0.1** (NVIDIA's differentiable ray tracer; standalone `sionna-rt` on PyPI since Mar 2026, Python ≥ 3.10, CPU-capable via LLVM; built-in Munich scene; RadioMapSolver exports a path-gain grid our simulator consumes), and a forward-looking probe of **wireless foundation-model** embeddings (LWM 1.1 / the Jan–Apr 2026 WiFo/LWM-Temporal wave) as the channel-prediction backbone.
- **Headline experiment:** the paper's deterministic optimum, executed under the realistic channel, **misses deadlines at a measured rate** (hypothesis: 20–40%+ at σ_sh = 8 dB — report the real number); the certified predictive planner meets Pr(violation) ≤ α, and "move-to-predicted-good-channel" beats "move-to-shorter-distance" on energy.

### WP-3 — The Physically-Grounded Pipeline *(kills the scalar η·L payload and the τf²W energy fiction; completes the repo's own `opv2v_colab.py` thread; absorbs old D3 trust)*
- **Real data:** OPV2V multi-view driving scenes via OpenCOOD (73 CARLA scenarios, 2–7 agents/frame, cameras+LiDAR; full set ≈ 249 GB — we use the small `test_culvercity` split / 2–4 chunked scenarios only), plus a synthetic overlapping-crops fallback generator with known ground-truth overlap.
- **Real brain:** each agent = **SmolVLM2-500M** (Apache-2.0; 256M for CPU smoke tests; Qwen2.5-VL-3B alternative) producing structured JSON scene facts + a FAISS/SigLIP RAG memory with embedding-based dedup.
- **Real payloads:** three codecs — text tokens, latent embeddings, and (stretch) a **Cache-to-Cache-style KV payload** (C2C, arXiv 2510.03215, open code `thu-nics/C2C`, shows latent messaging beats text by 3–5% accuracy at 2.5× lower latency — the exact 2026 frontier the paper's scalar abstraction ignores).
- **Real measurements:** ρ̂ (measured semantic overlap) vs the paper's geometric Jaccard (scatter, Spearman, % ranking inversions); a measured **rate–fidelity curve** (report-F1 vs payload bits per codec); **measured mJ/token** energy (macOS `powermetrics` sidecar and/or Colab NVML integration; prefill vs decode split; any modeled fallback explicitly labeled MODELED) → the **grounding-gap table**: paper's analytic energy terms vs reality, term by term.
- **Trust & hallucination (absorbed D3):** inject corruption at one leaf → measure root-corruption rate (hypothesis ≈ 100% in the paper's tree); add an overlap-consistency trust gate + reputation-weighted fusion → target ≤ 20% root corruption at ≤ 10% energy overhead; certify the root-fidelity floor with the Phase-1 conformal layer, now on a *real* pipeline.

### WP-4 — Integration, Continuous Mission & the Grand Showdown *(builds ON the Phase-1 freshness spine; adds L16/L17 repairs)*
One simulator, all axes composable: coordination {hub-Blossom, auction, Rung-B, Rung-C} × channel {paper, Tier-1, (Sionna)} × payload {abstract, text, latent} × mission {one-shot, continuous}. Continuous mode = moving targets (reuse `wan/targets.py` IMM) + **event-triggered re-aggregation** (trigger on predicted staleness or information gain) + **per-agent batteries** with a network-lifetime metric. Final grand comparison: **paper H-MAP vs Phase-1 (D1+D2+freshness) vs A-WAN** on energy, fidelity/F1, staleness/AoI, lifetime, dropout-robustness, and runtime-scalability to N ∈ {10,…,100(,200)} using a learned inner-solver surrogate.

---

## 4. Headline falsifiable claims (each is PASS/FAIL — the playbook binds each to an experiment)

| # | Claim | Confirming number |
|---|-------|-------------------|
| H1 | Decentralized matching ≈ centralized | auction/greedy within ≤ 5% of Blossom energy, N ∈ {6,8,10}, ≥ 10 seeds |
| H2 | The hub was never free | hub coordination overhead measured in J; decentralized control energy < 3% of mission energy |
| H3 | Graceful vs brittle | at 20% dropout, decentralized finishes with ≤ 10% extra energy; centralized stalls/fails in a measured ≥ X% of runs |
| H4 | Talking has a price and a break-even | LLM-negotiation joules/round measured; break-even N* reported where negotiation < hub cost |
| H5 | The paper's channel optimism is unsafe | deterministic plan violates deadlines at measured ≥ 20% under realistic fading; certified planner ≤ α (α = 0.1, 0.2) |
| H6 | Prediction pays | move-to-predicted-channel saves a measured Δ% energy vs distance heuristic |
| H7 | The geometric proxy misranks | Spearman(ρ_geo, ρ̂) reported + % of pair-ranking inversions on real data |
| H8 | The energy model is fiction | grounding-gap table shows ≥ 10× (measured) discrepancy in the compute term |
| H9 | One liar poisons the tree | base root-corruption ≈ 100%; trust gate ≤ 20% at ≤ 10% energy overhead |
| H10 | Latent beats text | latent/KV payload ≥ text-payload F1 at equal bits (or the honest negative result) |
| H11 | A-WAN dominates | grand table: A-WAN ≤ Phase-1 ≤ paper on the composite frontier; scaling curve to N ≥ 100 |

---

## 5. Novelty position (verified July 2026; re-verify per playbook §13 before any submission)
- No published work does **decentralized coordination for progressive/hierarchical semantic knowledge aggregation** with costed control overhead. Closest neighbors: distributed matching theory (Preis/Hoepman ½-approx; Bertsekas auctions), CBBA task allocation, and 2025–26 LLM-negotiation papers — none in an energy-optimized semantic-aggregation WAN.
- No published work **instantiates this WAN line with a real VLM+RAG pipeline and measured energy**. Closest: token-communication frameworks (e.g., TokenCom, Feb 2026) and C2C-style inter-LLM latent messaging — none inside a mobility+energy-optimized aggregation tree with hallucination-propagation measurement.
- Channel-realism + conformal deadline certification for THIS framework is unclaimed; wireless foundation models (LWM/WiFo wave, Nov 2024–Apr 2026) have not been used for aggregation-driven mobility planning.
- The reference authors' group is active (ILAC/SFMA follow-ups); the window for both "firsts" is open but measured in months — hence one integrated push now.

## 6. Publication mapping
- **M1 (journal: IEEE TCCN or IoT-J):** "Decentralized, Channel-Predictive Coordination for Wireless Agent Networks" = WP-1 + WP-2 (+ Phase-1 D2 as the centralized upper baseline).
- **M2 (letter/workshop → journal: IEEE Networking Letters, ICC/GLOBECOM/INFOCOM-W):** "Grounding the Wireless Agent Network: Measured Semantics, Energy, and Hallucination Propagation" = WP-3 (+ Phase-1 D1 conformal layer).
- Phase-1 material (D1+D2+freshness) is the "prior phase" section of both. Skeletons in playbook §14.

## 7. Market/standards alignment (professor-facing framing, cite in the report)
WP-1 ↔ the 2025–26 agent-interoperability wave (A2A under the Linux Foundation, 150+ orgs, cloud-platform integration, IBM ACP merged in; MCP as the tool-side complement). WP-2 ↔ AI-native 6G / digital-twin channels (NVIDIA Sionna RT; the wireless-foundation-model literature explosion, Jan–Apr 2026). WP-3 ↔ on-device multimodal AI (SmolVLM2/MLX-class edge VLMs) and latent inter-LLM communication (C2C). WP-4 ↔ persistent-autonomy/AoI metrics now standard in goal-oriented communications.

## 8. Success definition
Phase 2 succeeds when: (a) all 58 Phase-1 checks still pass untouched; (b) every WP's **floor** deliverable is green; (c) ≥ 9 of the 11 headline claims have a real measured number (pass OR honestly-reported fail); (d) the new figures (~16) regenerate from scripts; (e) `PROJECT_REPORT.md` is extended (not rewritten) with Phase-2 parts mirroring its existing honesty style; (f) the website gains an A-WAN tab; (g) both manuscript skeletons are drafted with real numbers.

**Now open `02_AWAN_IMPLEMENTATION_PLAYBOOK.md` and follow it top to bottom.**
