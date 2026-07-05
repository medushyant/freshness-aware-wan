# Phase-2 decision log (per playbook §0.9 / §1)

Format: date — decision — why.

- 2026-07-05 — **Colab MEASURED artifacts integrated.** The user ran
  `02_vlm_energy.ipynb` (T4, 250 views, NVML @10 Hz, idle 9.7 W subtracted →
  `runs/measured_energy.json`; prefill 104 / decode 5371 mJ/token) and
  `01_sionna_scene.ipynb` (Munich RadioMapSolver grid →
  `assets/sionna_gain_grid.npy`). H8 and C5 now rest on MEASURED inputs.
  `03_opv2v_pipeline.ipynb` halted correctly on the missing ~4 GB manual
  dataset download — deferred per the §10 risk-register fallback (WP-3 stays
  synthetic-grounded locally; the notebook is ready when the dataset is).
- 2026-07-05 — **Sionna grid used as a ray-traced shadowing texture.** Only
  ~2% of the exported 1 m grid received rays (single low TX), the rest being
  building interiors/out-of-coverage — not physical link space. We fit and
  remove the covered cells' distance law, reconstruct a full field by
  nearest-covered-cell lookup, and sample arena windows inside the covered
  urban core. The texture keeps the MEASURED ray-traced statistics (sd
  6.6 dB, heavy left tail — convergently close to tier-1's assumed 8 dB
  GRF). First naive attempt (interiors clamped to −35 dB) put most links
  inside buildings: 95% violations and a broken margin — rejected as
  unphysical and documented here.

- 2026-07-03 — **Baseline PASS-line count is 57, not 58.** The historical "58
  checks" counted the `F1b` line twice because `run_repro_scaling.py` had been
  run twice (it appends). A fresh single pass of all seven runners yields 57
  PASS lines with the identical identifier set {E1..E6, F1b, T0..T4, U1..U5,
  G1..G3, F1..F5, M1..M2}. The regression gate therefore compares the
  *identifier set*, which is stable, and treats the count 57 as the canonical
  fresh-run total. No frozen file was touched.
- 2026-07-03 — **W0.2 tolerance.** `results.txt` prints energies rounded to
  3 decimals (e.g. `9.173 J`), so "within 1e-6 of results.txt" is applied as:
  adapter-path result == direct frozen-call result within 1e-9 (bit-level
  determinism), and both match the printed value within 5e-4 (rounding).
- 2026-07-03 — **Python for `.venv-awan` is 3.12** (3.11 not installed;
  3.12 available via uv shim at `~/.local/bin/python3.12`). Torch CPU wheels +
  transformers verified to install on 3.12.
- 2026-07-03 — **Channel injection without touching frozen code.** The frozen
  `solve_pair` reads the channel through `p["beta0"]`/`p["delta"]`. Phase-2
  hands it a per-candidate effective `beta0' = beta0 · 10^(X_dB/10)` (X = local
  shadowing+fading factor at the candidate endpoints, treated as locally flat)
  with `allow_motion=False` on a candidate grid — the SAA-style wrapper the
  playbook §3.3 prescribes. Realized-channel execution then re-prices the
  planned (η, p_tx, t1) under the true tier-1 draw.
- 2026-07-03 — **Hub compute assumed free** in the §2.1 hub costing
  (conservative, favors the paper baseline), as the playbook specifies.
- 2026-07-04 — **Conformal deadline margin lives in the dB domain, not the
  time domain.** The playbook offered two constructions (κ·σ channel margin vs
  additive time margin q̂) and said pick one. Measured verdict: the time margin
  FAILS here — deep Rician+shadow fades collapse the rate multiplicatively, so
  the α=0.1 calibrated overrun quantile (19.0 s) exceeds the whole 2.5 s
  deadline and the planner goes infeasible. The dB-domain construction
  (nonconformity s = fade depth in dB below the planning belief; plan every
  link with beta0·10^(−q̂_db/10)) certifies the same guarantee and stays
  feasible. The failed variant is reported in the honesty section, not hidden.
- 2026-07-04 — **Two mJ/token figures coexist, both labeled.** The Rung-C
  negotiation charges tokens at the config placeholder (0.15/0.45 mJ/token —
  the Samsi et al. edge-GPU-class interim estimator, MODELED), while WP-3's
  meter reports this laptop's CPU envelope (~38/113 mJ/token, MODELED — CPU
  decode of a VLM is ~100× less efficient than a served GPU). The Colab NVML
  notebook (`notebooks/02_vlm_energy.ipynb`) supplies the MEASURED number that
  replaces both when its `runs/measured_energy.json` is present. Break-even
  H4 conclusions are reported under the cheap (GPU-class) assumption, which is
  the assumption *least* favorable to our "talking is affordable" finding.
- 2026-07-04 — **WP-3 perception prompt locked by pilot.** The 500M SmolVLM2
  echoes any in-vocabulary example and ignores rigid format instructions; the
  plain describe-then-parse design (free-form description + constrained
  'color type' grammar, a §10-sanctioned fallback) measured best (F1 0.21,
  100% parseable) over 4 piloted prompt designs. Absolute F1 is honestly low
  and reported as a finding, not hidden; the 2.2B Colab path is the upgrade.
- 2026-07-04 — **Root corruption = corrupted AND factually false.** A
  fabricated "blue bus" in a scene that really contains a blue bus is not a
  hallucination; H9's metric counts a corrupted-tagged root fact only when its
  (class, color) is absent from the exact ground truth. Base rate barely
  changes (100→95%); the gated rate stops being polluted by truthful
  coincidences.
- 2026-07-04 — **Corroboration threshold ≠ dedup threshold.** SigLIP text
  embeddings of near-identical short phrases ("blue bus" vs "blue truck") can
  exceed the 0.88 dedup tau, letting fabrications be false-corroborated. The
  trust gate corroborates at tau_corr = 0.95 (near-identity); dedup keeps
  0.88. Result: gated root corruption 45% -> 15% with the honest metric.
- 2026-07-04 — **G7 uses the exact one-sided conformal bound** (certified
  floor = k-th smallest calibration F1, k = floor((n+1)·alpha)), reported as
  the MARGINAL coverage over 20 random cal/test splits — the quantity the
  theory actually guarantees. The earlier LTT-with-default-on-abstain variant
  broke the guarantee and was discarded (its defaulting to eta*=1 on
  non-acceptance is unsound).
- 2026-07-04 — **Mobility policies respect vmax.** First C3 draft let the
  move-toward-receiver heuristic traverse arbitrary distances inside a capped
  motion time (implied speeds >100 m/s). Fixed: all policies choose endpoints
  within reach = vmax × motion budget (≤0.6·Tmax), i.e. ≤ ~7.5 m per round in
  the stress regime — consistent with Phase-1's mobility findings; the
  predictive policy's edge now comes from escaping shadow nulls within the
  fade correlation length d_c = 25 m, which is the physically honest story.
