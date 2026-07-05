# CODEBASE_MAP — the frozen Phase-1 API contract (playbook §1.1)

Every public symbol Phase-2 is allowed to touch, what paper equation it
implements, and which runner exercises it. Phase-2 reaches these ONLY through
`awan/adapters.py`.

## `wan/model.py` — system model (paper §II, Table I)
| Symbol | Signature | Paper eq. | Exercised by |
|---|---|---|---|
| `P` | dict of all constants | Table I + disclosed extras | everything |
| `make_agents` | `(rng, p=P, n=None) -> list[dict]` | §V scenario | all runners |
| `jaccard_disks` | `(disks_a, disks_b, rng, n_samp=4000) -> float` | Eq. (5) ρ proxy | all missions |
| `chan_gain` | `(pi, pj, p) -> h` | Eq. (8) h=β0·d^−δ | solver |
| `rate` | `(ptx, h, p) -> bit/s` | Eq. (9) Shannon | solver |
| `mob_energy` | `(dist, t_m, p) -> J` | Eqs. (3)–(4) | solver |
| `comp_load` | `(L, rho, eta, p) -> FLOPs` | Eq. (6) | solver |
| `comp_energy_time` | `(L, rho, eta, p) -> (J, s)` | Eq. (7) E=τf²W | solver |
| `comm_energy` | `(eta_L, h, t1, p) -> J` | Eq. (10)/(18) | solver |
| `ptx_for` | `(eta_L, h, t1, p) -> W` | Eq. (23) | solver |

Agent dict fields: `id, c, R, pos, L, disks, srcs {src_id: [bits, Σln(1/η)]}, rho_pred`.

## `wan/solver.py` — inner problem P1 (paper §IV-A)
| Symbol | Signature | Notes |
|---|---|---|
| `solve_pair` | `(ai, aj, p=P, *, fidelity, lam, rho_pair, use_wz, eta_cap_j, eta_lo_j, eta_lo_i, centroid, allow_motion, force_eta, max_power, phi_in_loop, n_bcd, trace) -> dict|None` | BCD: bounded 1-D η search (paper) or 2-D grid+polish (fidelity) + SLSQP motion block. Returns `{E, E_i, E_j, eta_i, eta_j, t1, ptx, pe_i, pe_j, dD, L_next, Lout, feasible}`; `None` = infeasible. |

Energy decomposition identity used by the Phase-2 ledger:
`E = E_mob_total + Ec_i + Ec_j + Ecomm`, where `Ec_* = comp_energy_time(L_*, rho_pred_*, eta_*)[0]`
and `Ecomm = comm_energy(Lout, chan_gain(pe_i, pe_j), t1)` (or Pmax·Lout/R at max_power).

## `wan/network.py` — Direction-1 missions (outer loop, H-MAP)
| Symbol | Signature | Notes |
|---|---|---|
| `run_mission` | `(seed, p, mode, lam, topology, allow_motion, use_wz, derived_cap, force_eta, max_power, n, rng_geo, targets, target_T, r_track, predict, src_distortion0) -> dict` | rng=default_rng(seed), geo=default_rng(seed+10000). Full D1 mission with srcs ledger, derived η-floor, freshness hook. |
| `_match` | `(active, w, topology, rng) -> [(sender, receiver|None)]` | Blossom on symmetrized min(w_ij,w_ji); virtual node for odd N. Private but stable; adapters delegate to it. |
| `conformal_pick_lambda` / `conformal_coverage` | see file | split-conformal / LTT recipe reused by WP-2/WP-3. |

## `wan/topology.py` — Direction-2 policies
| Symbol | Signature | Notes |
|---|---|---|
| `mission` | `(seed, p, policy, zeta, vmodel, flexible, allow_motion, n, behavior_eps, record, matcher) -> {E, rounds, feasible, Emax}` | policies: paper/stageA/lyapunov/learned/greedy/random/distance. `matcher(active, w, E, rng)` hook = Phase-2's entry point for new matchers WITHOUT editing frozen code. |
| `state_features` | `(act, p) -> 7-vector` | value-model features. |
| `ValueModel` | ridge, `.fit(X,y)->R²`, `__call__(feats)->J` | Rung-B bid brain. |
| `collect_training` | `(seeds, p, n_list) -> X, y` | mixed-policy rollouts. |
| `dp_oracle` | `(seed, p, n=5) -> J*` | exact optimum, N≤6. |
| `_pick_pairs` | `(active, w, E, agents, policy, rng)` | Blossom + baselines. |
| `finetune_decision_focused` | Stage-D zeroth-order trainer. |

## `wan/freshtopo.py`, `wan/freshness.py`, `wan/targets.py` — the freshness spine
| Symbol | Notes |
|---|---|
| `P_FRESH` | THE stress regime dict (Tmax 2.5, L 15–25 Mb, β0 1e-5, f_cpu 2.5e9, C_gen 120) — same numbers as run_direction1's p1. |
| `mission_fresh` | `(seed, agility, p, policy, lam_f, a_F, allow_motion, n)` value-prioritized topology. |
| `targets.Target/...` | static/dynamic(NCV)/time-varying(CT) motion models. |
| `targets.CVPredictor/KalmanCV/IMMPredictor` | predictors for WP-4 continuous mission. |
| `freshness.target_agility`, `freshness.freshness_distortion` | staleness pricing. |

## `wan/style.py`
`use_style()` — one matplotlib style for visual continuity (§8).

## Runner → check-id map (the frozen baseline)
| Runner | Ids | File |
|---|---|---|
| run_direction1.py | E1–E6 (15) | results.txt |
| run_repro_scaling.py | F1b (1/run, appends) | results.txt |
| run_direction2.py | T0–T4 (11) | results_d2.txt |
| run_upgrades.py | U1–U5 (9) | results_upgrades.txt |
| run_targets.py | G1–G3 (5) | results_targets.txt |
| run_freshness.py | F1–F5 (11) | results_freshness.txt |
| run_metaheuristics.py | M1–M2 (5) | results_metaheuristics.txt |

Fresh-run total: 57 PASS lines (see DECISIONS.md on the historical 58).
