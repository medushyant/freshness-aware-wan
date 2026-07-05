# M1 — Decentralized, Channel-Predictive Coordination for Wireless Agent Networks
*Target: IEEE TCCN (Agentic AI special-issue line) or IoT-J. Skeleton with
measured numbers auto-injected ({wp1,wp2,wp4} results.json).*

## I. Introduction
Hook: the reference paper's own conclusion — "the current model assumes a
simplified path loss channel and relies on centralized coordination" — is
executed literally, and measured. Contributions:
(1) the first *costed* decentralized pair-selection for progressive semantic
aggregation (every control message in joules);
(2) a certified channel-predictive planner (split conformal, dB domain);
(3) the first N ≥ 100 runtime curve for this framework.

## II. System model (Phase-1 recap, 1 page)
Frozen Phase-1 engine; stress regime disclosed; value model R² = 0.89.

## III. Costed decentralized matching
- Hub bill (L10 repair): 0.030 mJ/mission at N=10 (MODELED).
- Greedy ½-approx: bound held on 56/56 feasible rounds.
- Auction vs Blossom: within the ε-bound on 56/56 rounds
  (exact on 54); mission energy gap ['N6:-1.15%(p=1.00)', 'N8:+3.04%(p=0.23)', 'N10:-2.07%(p=0.43)'].
- Control energy < 0.01% of mission energy (H2).
- Dropout: hub completion 0% vs auction 100% at q=0.2 (H3).
- LLM negotiation (A2A-style): 0/0 schema-valid turns,
  0+0 tokens charged at mJ/token (H4 break-even: F1.4).

## IV. Channel intelligence + conformal certificate
- Paper plan under tier-1 fading: 62% / 60% deadline violations (σ = 4/8 dB) (H5a).
- dB-margin split conformal: coverage 0.94 ≥ 0.90 (q̂ = 13.3 dB, +4% energy)
  and 0.87 ≥ 0.80 (q̂ = 9.9 dB) (H5b). The additive time-margin variant
  provably fails here (q̂ = 19 s > T_max = 2.5 s) — reported as a finding.
- Radio maps: twin+residual GP beats full GP at every sample size
  (5.5 vs 10.3 dB RMSE @300 samples).
- Mobility: predictive 2.17 J vs distance 3.08 J (H6).

## V. Results (F1.1–F1.5, F2.1–F2.4, F4.1, F4.4)
Grand showdown: paper 16.66 J / 52% viol / 0% drop-completion
vs A-WAN 14.85 J / 14% / 100%.
Scaling: Blossom 2105 ms vs auction 553 ms vs greedy 24 ms per round at N=200
(surrogate MAPE 0.1%, energy penalty -1.1%).

## VI. Related work — cite the group's sequels (2604.09255, 2604.09261, 2605.14300) and the A2A/LF standardization wave.
## VII. Conclusion — decentralization is not free, but its bill is smaller than the hub's fragility.
