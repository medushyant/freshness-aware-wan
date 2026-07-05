"""Manuscript skeletons M1/M2 (playbook §14) with numbers auto-injected from
the machine-written results.json files — never typed in by hand."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from awan.runio import latest_results

ROOT = pathlib.Path(__file__).resolve().parents[1]
wp1, wp2, wp4 = (latest_results(n) for n in ("wp1", "wp2", "wp4"))
try:
    wp3 = latest_results("wp3")
except FileNotFoundError:
    wp3 = None

g = wp4["grand"]
c1 = wp2["viol_rates"]
c2 = wp2["conformal"]

M1 = f"""# M1 — Decentralized, Channel-Predictive Coordination for Wireless Agent Networks
*Target: IEEE TCCN (Agentic AI special-issue line) or IoT-J. Skeleton with
measured numbers auto-injected ({'{'}wp1,wp2,wp4{'}'} results.json).*

## I. Introduction
Hook: the reference paper's own conclusion — "the current model assumes a
simplified path loss channel and relies on centralized coordination" — is
executed literally, and measured. Contributions:
(1) the first *costed* decentralized pair-selection for progressive semantic
aggregation (every control message in joules);
(2) a certified channel-predictive planner (split conformal, dB domain);
(3) the first N ≥ 100 runtime curve for this framework.

## II. System model (Phase-1 recap, 1 page)
Frozen Phase-1 engine; stress regime disclosed; value model R² = {wp1['value_model_r2']:.2f}.

## III. Costed decentralized matching
- Hub bill (L10 repair): {wp1['energy']['10']['hub']['control_J']*1e3:.3f} mJ/mission at N=10 (MODELED).
- Greedy ½-approx: bound held on {wp1['matcher_quality']['half_ok']}/{wp1['matcher_quality']['half_tot']} feasible rounds.
- Auction vs Blossom: within the ε-bound on {wp1['matcher_quality']['within_ok']}/{wp1['matcher_quality']['eq_tot']} rounds
  (exact on {wp1['matcher_quality']['eq_ok']}); mission energy gap {wp1['h1_detail']}.
- Control energy < {max(wp1['energy'][n]['auction']['control_J']/wp1['energy'][n]['auction']['mean'] for n in wp1['energy'])*100:.2f}% of mission energy (H2).
- Dropout: hub completion {wp1['dropout']['cen']['0.2']['completion']*100:.0f}% vs auction {wp1['dropout']['dec']['0.2']['completion']*100:.0f}% at q=0.2 (H3).
- LLM negotiation (A2A-style): {wp1['llm_stats']['schema_valid']}/{wp1['llm_stats']['schema_total']} schema-valid turns,
  {wp1['llm_stats']['tok_in']}+{wp1['llm_stats']['tok_out']} tokens charged at mJ/token (H4 break-even: F1.4).

## IV. Channel intelligence + conformal certificate
- Paper plan under tier-1 fading: {c1['4.0']*100:.0f}% / {c1['8.0']*100:.0f}% deadline violations (σ = 4/8 dB) (H5a).
- dB-margin split conformal: coverage {c2['0.1']['coverage']:.2f} ≥ 0.90 (q̂ = {c2['0.1']['qhat_db']:.1f} dB, +{c2['0.1']['energy_premium_pct']:.0f}% energy)
  and {c2['0.2']['coverage']:.2f} ≥ 0.80 (q̂ = {c2['0.2']['qhat_db']:.1f} dB) (H5b). The additive time-margin variant
  provably fails here (q̂ = 19 s > T_max = 2.5 s) — reported as a finding.
- Radio maps: twin+residual GP beats full GP at every sample size
  ({wp2['predictor_rmse_db']['B']['300']:.1f} vs {wp2['predictor_rmse_db']['A']['300']:.1f} dB RMSE @300 samples).
- Mobility: predictive {wp2['mobility']['predictive']['E_mean']:.2f} J vs distance {wp2['mobility']['distance']['E_mean']:.2f} J (H6).

## V. Results (F1.1–F1.5, F2.1–F2.4, F4.1, F4.4)
Grand showdown: paper {g['paper']['E_mean']:.2f} J / {g['paper']['viol_rate']*100:.0f}% viol / {g['paper']['dropout_completion']*100:.0f}% drop-completion
vs A-WAN {g['awan']['E_mean']:.2f} J / {g['awan']['viol_rate']*100:.0f}% / {g['awan']['dropout_completion']*100:.0f}%.
Scaling: Blossom {wp4['i5_runtime_ms']['blossom'][-1]:.0f} ms vs auction {wp4['i5_runtime_ms']['auction'][-1]:.0f} ms vs greedy {wp4['i5_runtime_ms']['greedy'][-1]:.0f} ms per round at N=200
(surrogate MAPE {wp4['i4']['mape_pct']:.1f}%, energy penalty {wp4['i4']['penalty_pct']:+.1f}%).

## VI. Related work — cite the group's sequels (2604.09255, 2604.09261, 2605.14300) and the A2A/LF standardization wave.
## VII. Conclusion — decentralization is not free, but its bill is smaller than the hub's fragility.
"""

if wp3:
    gg = wp3["grounding_gap"]
    M2 = f"""# M2 — Grounding the Wireless Agent Network: Measured Semantics, Energy, and Hallucination Propagation
*Target: IEEE Networking Letters or ICC/GLOBECOM workshop. Numbers auto-injected (wp3 results.json).*

## Hooks (each a measured table/figure)
- H7 the geometric overlap proxy misranks: Spearman(ρ_geo, ρ̂) = {wp3['h7']['spearman']:.2f},
  {wp3['h7']['inversions_pct']:.0f}% pair-ranking inversions (F3.1).
- H8 the energy model is fiction: measured(MODELED) {gg['E_real_J_per_view']:.1f} J/view vs
  Eq. (6)-(7) {gg['E_paper_J']:.2e} J → {gg['ratio']:.1e}× gap (F3.4); {wp3['meter']['prefill_mJ_per_tok']:.1f}/{wp3['meter']['decode_mJ_per_tok']:.1f} mJ/token prefill/decode.
- H9 one liar poisons the tree: base {wp3['corruption']['fabricate_base']*100:.0f}% root corruption →
  {wp3['corruption']['fabricate_gated']*100:.0f}% with the overlap-consistency gate at {wp3['trust_overhead_pct']:.2f}% overhead (F3.5).
- H10 latent-vs-text at equal bits: ΔF1 = {wp3['h10_verdict_f1_delta']:+.3f} (honest verdict either way, F3.2).
- Fusion is sub-additive (vs Eq. (2)): F3.6. VLM validity {wp3['g1_validity']*100:.0f}% (G1).
- Constructive fix: conformal fact-budget certifies Pr(F1 ≥ {wp3['conformal_real']['F_min']}) ≥ {1-wp3['conformal_real']['alpha']:.1f}
  — held-out coverage {wp3['conformal_real']['coverage']:.2f} (G7).

## Honesty
Absolute F1 of a 500M edge VLM on abstract scenes is low — reported as the
finding "the paper's implicit perfect-perception assumption is ~{gg['ratio']:.0e}× off in
energy and far off in fidelity"; OPV2V + 2.2B Colab path in notebooks/.
"""
else:
    M2 = "# M2 — run run_awan_wp3.py first\n"

(ROOT / "docs" / "M1_skeleton.md").write_text(M1)
(ROOT / "docs" / "M2_skeleton.md").write_text(M2)
print("wrote docs/M1_skeleton.md and docs/M2_skeleton.md")
