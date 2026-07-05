# M2 — Grounding the Wireless Agent Network: Measured Semantics, Energy, and Hallucination Propagation
*Target: IEEE Networking Letters or ICC/GLOBECOM workshop. Numbers auto-injected (wp3 results.json).*

## Hooks (each a measured table/figure)
- H7 the geometric overlap proxy misranks: Spearman(ρ_geo, ρ̂) = 0.36,
  38% pair-ranking inversions (F3.1).
- H8 the energy model is fiction: measured(MODELED) 273.0 J/view vs
  Eq. (6)-(7) 1.15e-04 J → 2.4e+06× gap (F3.4); 104.0/5371.1 mJ/token prefill/decode.
- H9 one liar poisons the tree: base 95% root corruption →
  15% with the overlap-consistency gate at 0.00% overhead (F3.5).
- H10 latent-vs-text at equal bits: ΔF1 = +0.000 (honest verdict either way, F3.2).
- Fusion is sub-additive (vs Eq. (2)): F3.6. VLM validity 94% (G1).
- Constructive fix: conformal fact-budget certifies Pr(F1 ≥ 0.04722222222222222) ≥ 0.8
  — held-out coverage 0.89 (G7).

## Honesty
Absolute F1 of a 500M edge VLM on abstract scenes is low — reported as the
finding "the paper's implicit perfect-perception assumption is ~2e+06× off in
energy and far off in fidelity"; OPV2V + 2.2B Colab path in notebooks/.
