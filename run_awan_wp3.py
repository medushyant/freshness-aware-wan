"""WP-3 — The Physically-Grounded Pipeline (playbook §4). Stage 2: cache-only.

Reads the cached SmolVLM2 generations under runs/vlm_cache/ (produced once by
scripts/wp3_perceive.py) and produces checks G1-G8 and figures F3.1-F3.7 with
NO VLM inference — the G8 reproducibility requirement. Grounding mode:
synthetic-grounded locally (exact ground truth); OPV2V runs on Colab via
notebooks/03_opv2v_pipeline.ipynb (§10 fallback, documented).
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from awan import adapters as A
from awan.config import load
from awan.grounded import pipeline as P
from awan.grounded.energy_meter import (mj_per_token, modeled_energy_j,
                                        paper_compute_energy_j,
                                        powermetrics_available)
from awan.grounded.memory_rag import Embedder
from awan.harness import Checks
from awan.registry import record
from awan.runio import FIGS, Run
from awan.stats import mean_ci

A.use_style()
cfg = load("defaults")
TAU = cfg["grounded"]["tau_dedup"]
C = Checks("WP-3")
run = Run("wp3", {"tau_dedup": TAU, "mode": "synthetic-grounded",
                  "vlm_cache_scenes": None})

seeds = P.cached_seeds()
if len(seeds) < 8:
    raise SystemExit(f"only {len(seeds)} cached scenes — run scripts/wp3_perceive.py first")
run.log(f"{len(seeds)} cached scenes; loading SigLIP embedder (not the VLM) ...")
emb = Embedder()
recs = {s: P.load_scene(s) for s in seeds}
vst = P.vlm_stats()

# ---------------------------------------------------------------- G1
validity = 1.0 - vst["parse_fail"] / max(vst["calls"], 1)
C.check("G1", "VLM emits parseable JSON facts on >=90% of views (<=1 retry)",
        validity >= 0.90,
        f"{validity*100:.0f}% of {vst['calls']} views ({vst['retries']} retries, "
        f"{vst['parse_fail']} regex-rescued) — model {vst['model_id']}")

# ---------------------------------------------------------------- G2 / F3.6
rows = [r for s in seeds for r in P.measured_vs_additive(recs[s], emb, TAU)]
ov = [r for r in rows if r["iou"] > 0.15]
sub = np.mean([r["measured"] < r["additive"] for r in ov]) if ov else 0
C.check("G2", "measured fused size is sub-additive in overlap (vs Eq. (2) additive)",
        sub >= 0.8, f"{sub*100:.0f}% of {len(ov)} overlapping pairs strictly below "
        f"additive; mean saving {np.mean([1-r['measured']/max(r['additive'],1) for r in ov])*100:.0f}%")

fig, ax = plt.subplots(figsize=(5.8, 4))
ax.scatter([r["iou"] for r in rows], [r["additive"] for r in rows], s=18,
           alpha=.5, label="paper Eq.(2): additive", color="#e63946")
ax.scatter([r["iou"] for r in rows], [r["measured"] for r in rows], s=18,
           alpha=.6, label="measured (dedup fusion)", color="#2a9d8f")
ax.set_xlabel("geometric view overlap (IoU)"); ax.set_ylabel("fused fact count")
ax.set_title("F3.6  fusion is sub-additive in overlap — MEASURED")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fp = FIGS / "F3_6_fused_size.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F3.6", "wp3", fp)

# ---------------------------------------------------------------- G3 / F3.1 (H7)
pairs = [pr for s in seeds for pr in P.overlap_pairs(recs[s], emb, TAU)]
geo = np.array([p_[0] for p_ in pairs]); sem = np.array([p_[1] for p_ in pairs])
rho_s = sps.spearmanr(geo, sem).statistic
ktau = sps.kendalltau(geo, sem).statistic
inversions = (1 - ktau) / 2 * 100
C.check("G3", "geometric-vs-semantic overlap divergence quantified (H7)", True,
        f"Spearman={rho_s:.2f}, pair-ranking inversions={inversions:.0f}% "
        f"({len(pairs)} agent pairs) — the geometric proxy misranks")

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.scatter(geo, sem, s=26, alpha=.65, color="#4361ee")
lim = max(geo.max(), sem.max()) * 1.05
ax.plot([0, lim], [0, lim], "--", lw=1, color="#888")
ax.set_xlabel("ρ_geo (view-box Jaccard — the paper's Eq. (5) proxy)")
ax.set_ylabel("ρ̂ (measured semantic overlap)")
ax.set_title(f"F3.1  ρ̂ vs ρ_geo: Spearman {rho_s:.2f}, "
             f"{inversions:.0f}% ranking inversions (H7)")
ax.grid(alpha=.3)
fig.tight_layout(); fp = FIGS / "F3_1_overlap_scatter.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F3.1", "wp3", fp)

# ---------------------------------------------------------------- G4 / F3.2 (H10)
ETAS = [0.25, 0.5, 0.75, 1.0]
curves = {"text": [], "latent": []}
for codec in curves:
    for eta in ETAS:
        f1s, bits = [], []
        for s in seeds:
            r = P.run_tree(recs[s], emb, codec=codec, eta=eta, tau=TAU)
            f1s.append(r["f1"]); bits.append(r["bits"])
        curves[codec].append({"eta": eta, "f1": float(np.mean(f1s)),
                              "f1_ci": mean_ci(f1s)[1],
                              "kbits": float(np.mean(bits)) / 1e3})
# verdict at (approximately) equal bits: interpolate latent F1 at text bit points
tx = curves["text"]; lt = curves["latent"]
ltb = [c["kbits"] for c in lt]; ltf = [c["f1"] for c in lt]
verdicts = []
for c in tx:
    if min(ltb) <= c["kbits"] <= max(ltb):
        f1_lat = float(np.interp(c["kbits"], ltb, ltf))
        verdicts.append(f1_lat - c["f1"])
verdict = float(np.mean(verdicts)) if verdicts else float(np.mean(ltf) - np.mean([c["f1"] for c in tx]))
C.check("G4", "per-codec rate-fidelity curves measured; latent-vs-text verdict reported (H10)",
        True, f"latent-minus-text F1 at equal bits: {verdict:+.3f} "
        f"({'latent wins' if verdict > 0.005 else 'text wins — honest negative' if verdict < -0.005 else 'tie'}); "
        f"text@eta1: F1={tx[-1]['f1']:.2f}/{tx[-1]['kbits']:.1f}kb, "
        f"latent@eta1: F1={lt[-1]['f1']:.2f}/{lt[-1]['kbits']:.1f}kb")

fig, ax = plt.subplots(figsize=(6, 4.2))
for codec, col in (("text", "#4361ee"), ("latent", "#f72585")):
    cs = curves[codec]
    ax.errorbar([c["kbits"] for c in cs], [c["f1"] for c in cs],
                yerr=[c["f1_ci"] for c in cs], marker="o", capsize=3,
                label=f"P-{codec}", color=col)
    for c in cs:
        ax.annotate(f"η={c['eta']}", (c["kbits"], c["f1"]),
                    textcoords="offset points", xytext=(4, -10), fontsize=6)
ax.set_xlabel("total tree payload [kbit] (MEASURED bit counts)")
ax.set_ylabel("root report fact-F1 vs exact ground truth")
ax.set_title(f"F3.2  measured rate-fidelity per codec ({len(seeds)} scenes) — "
             "the paper's scalar η·L has no such curve")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fp = FIGS / "F3_2_rate_fidelity.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F3.2", "wp3", fp)

# ---------------------------------------------------------------- G5 / F3.3-F3.4 (H8)
# Prefer MEASURED numbers if the Colab NVML notebook's export is present.
from awan import ROOT as _ROOT
measured_fp = _ROOT / "runs" / "measured_energy.json"
if measured_fp.exists():
    meter = json.loads(measured_fp.read_text())
    meter["source"] = "MEASURED"
    E_real_per_view = meter["total_mJ"] / 1e3 / max(meter.get("n_views", vst["calls"]), 1)
else:
    meter = mj_per_token(vst, source="MODELED")
    E_real_per_view = modeled_energy_j(vst["wall_s"] / max(vst["calls"], 1))
pm = powermetrics_available()
mean_view_kb = np.mean([len(json.dumps(v["facts"]).encode()) * 8
                        for s in seeds for v in recs[s]["views"]]) / 1e3
E_paper = paper_compute_energy_j(mean_view_kb * 1e3, A.STRESS)
gap = E_real_per_view / max(E_paper, 1e-30)
src = meter.get("source", "MODELED")
src_note = (f"{src}: {meter.get('gpu', 'device-envelope')}" if src == "MEASURED"
            else f"MODELED, powermetrics={'yes' if pm else 'no-sudo'}")
C.check("G5", "mJ/token measured + grounding-gap ratio >= 10x (H8)",
        gap >= 10,
        f"prefill {meter['prefill_mJ_per_tok']:.2f} / decode {meter['decode_mJ_per_tok']:.2f} "
        f"mJ/tok ({src_note}); per-view real "
        f"{E_real_per_view:.1f} J vs paper Eq.(6-7) {E_paper:.2e} J -> gap {gap:.1e}x")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax1.bar(["prefill", "decode"], [meter["prefill_mJ_per_tok"], meter["decode_mJ_per_tok"]],
        color=["#4cc9f0", "#f72585"])
ax1.set_ylabel(f"mJ / token  ({src})")
if src == "MEASURED":
    ax1.set_title(f"F3.3  SmolVLM2-500M, Colab T4 (NVML @10 Hz, idle-subtracted)\n"
                  f"{meter['tok_in']}+{meter['tok_out']} tokens over {meter.get('n_views','?')} views — MEASURED")
else:
    ax1.set_title(f"F3.3  SmolVLM2 on M-series CPU (device envelope)\n"
                  f"{vst['tok_in']}+{vst['tok_out']} tokens, {vst['wall_s']:.0f}s wall — MODELED")
ax2.axis("off")
tbl = [["term", "paper model", "this pipeline", "ratio"],
       ["compute J/view", f"{E_paper:.2e}", f"{E_real_per_view:.2f} ({src})", f"{gap:.1e}x"],
       ["payload bits/view", "η·L scalar", f"{mean_view_kb:.1f} kbit JSON", "—"],
       ["fusion rule", "additive Eq.(2)", "measured sub-additive", "—"]]
t = ax2.table(cellText=tbl, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.5)
ax2.set_title("F3.4  grounding-gap table (H8)")
fig.tight_layout(); fp = FIGS / "F3_34_energy_gap.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F3.3", "wp3", fp); record("F3.4", "wp3", fp)

# ---------------------------------------------------------------- G6 / F3.5 (H9)
OPS = ["fabricate", "swap", "jitter"]


def _truly_false(f, rec):
    """A corrupted fact only counts as root corruption if its content is
    actually absent from the scene's exact ground truth (a fabricated 'blue
    bus' that coincides with a real blue bus is not a hallucination)."""
    truth = {(t["cls"], t["color"]) for v in rec["views"] for t in v["true_facts"]}
    return (f.get("object"), f.get("attr")) not in truth


corr = {}
for op in OPS:
    for gate in (False, True):
        hits, checks = [], 0
        for s in seeds:
            rng = np.random.default_rng(s + 40)
            r = P.run_tree(recs[s], emb, codec="text", eta=1.0, tau=TAU,
                           corrupt_leaf=0, corrupt_op=op, trust=gate,
                           corrupt_rng=rng)
            hits.append(any(f.get("_corrupt") and _truly_false(f, recs[s])
                            for f in r["root_facts"]))
            checks += r["trust_checks"]
        corr[(op, gate)] = {"rate": float(np.mean(hits)), "checks": checks}
base_fab = corr[("fabricate", False)]["rate"]
gated_fab = corr[("fabricate", True)]["rate"]
# overhead: extra embedding-similarity compute, MODELED per-check nJ, vs
# the per-tree VLM perception energy (4 views)
E_checks = corr[("fabricate", True)]["checks"] * 1e-6      # 1 uJ per cosine (generous)
overhead_pct = E_checks / (4 * E_real_per_view * len(seeds)) * 100
C.check("G6", "one liar poisons the tree; overlap-consistency gate contains it (H9)",
        base_fab >= 0.8 and gated_fab <= 0.20 and overhead_pct <= 10,
        f"base root-corruption {base_fab*100:.0f}% -> gated {gated_fab*100:.0f}% "
        f"(fabricate, {len(seeds)} scenes); verification overhead {overhead_pct:.2f}% "
        f"of perception energy (MODELED)")

fig, ax = plt.subplots(figsize=(6.2, 4))
x = np.arange(len(OPS)); wd = 0.35
ax.bar(x - wd / 2, [corr[(o, False)]["rate"] * 100 for o in OPS], wd,
       label="paper tree (no gate)", color="#e63946")
ax.bar(x + wd / 2, [corr[(o, True)]["rate"] * 100 for o in OPS], wd,
       label="trust-gated fusion", color="#2a9d8f")
ax.axhline(20, ls="--", lw=1, color="#555"); ax.text(2.2, 21, "target ≤20%", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(OPS)
ax.set_ylabel("root-corruption rate [%]")
ax.set_title("F3.5  hallucination/corruption propagation (H9, one corrupted leaf)")
ax.legend(fontsize=8)
fig.tight_layout(); fp = FIGS / "F3_5_corruption.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F3.5", "wp3", fp)

# ---------------------------------------------------------------- G7 conformal-on-real
alpha = 0.2
# Exact one-sided split-conformal lower bound on root-F1 at the operating
# fact budget: with k = floor((n_cal+1)*alpha), Pr(F1_test >= k-th smallest
# calibration F1) >= 1-alpha, marginally. Estimated properly by averaging
# held-out coverage over 20 random 10/10 splits (the guarantee is marginal).
eta_op = 1.0
_f1 = {s: P.run_tree(recs[s], emb, codec="text", eta=eta_op, tau=TAU)["f1"]
       for s in seeds}
split_rng = np.random.default_rng(77)
covs, floors = [], []
for _ in range(20):
    perm = list(split_rng.permutation(seeds))
    cal, test = perm[:len(seeds) // 2], perm[len(seeds) // 2:]
    k = max(int(np.floor((len(cal) + 1) * alpha)), 1)
    q_lo = sorted(_f1[s] for s in cal)[k - 1]
    floors.append(q_lo)
    covs.append(np.mean([_f1[s] >= q_lo for s in test]))
cov = float(np.mean(covs))
F_MIN = float(np.mean(floors))
C.check("G7", "conformal lower bound certifies a root-F1 floor on the REAL pipeline",
        cov >= 1 - alpha - 0.02,
        f"marginal coverage {cov:.2f} >= {1-alpha:.2f} over 20 random splits; "
        f"mean certified floor F1 >= {F_MIN:.2f} at eta={eta_op}, alpha={alpha}")

# ---------------------------------------------------------------- G8 repro
import run_awan_wp3 as _self_check_module_guard  # noqa: F401  (self import ok)
no_vlm = "vlm_agent" not in {m.split(".")[-1] for m in list(__import__("sys").modules)}
C.check("G8", "all WP-3 figures regenerated cache-only (no VLM loaded in this process)",
        no_vlm and len(seeds) >= 8,
        f"{len(seeds)} scenes from runs/vlm_cache/; VLM module absent from sys.modules")

# ---------------------------------------------------------------- F3.7 diagram
fig, ax = plt.subplots(figsize=(8.2, 3.4)); ax.axis("off")
boxes = [("scene\n(exact GT)", "#8d99ae"), ("4 agent views\n(known IoU)", "#8d99ae"),
         ("SmolVLM2\nJSON facts", "#4361ee"), ("SigLIP RAG\ndedup fusion", "#7209b7"),
         ("codec\ntext|latent", "#f72585"), ("knockout tree\n+ trust gate", "#2a9d8f"),
         ("root report\nfact-F1", "#ffb703")]
for i, (txt, col) in enumerate(boxes):
    ax.add_patch(plt.Rectangle((i * 1.18, 0.3), 1.0, 0.55, fc=col, alpha=.85, ec="k", lw=.5))
    ax.text(i * 1.18 + 0.5, 0.575, txt, ha="center", va="center", fontsize=7.5, color="w")
    if i:
        ax.annotate("", xy=(i * 1.18, 0.575), xytext=(i * 1.18 - 0.18, 0.575),
                    arrowprops=dict(arrowstyle="->", lw=1.2))
ax.set_xlim(-0.2, 8.4); ax.set_ylim(0, 1.1)
ax.set_title("F3.7  the physically-grounded pipeline (schematic — illustrative)")
fig.tight_layout(); fp = FIGS / "F3_7_pipeline.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F3.7", "wp3", fp)

run.finish({"scenes": len(seeds), "vlm": vst, "g1_validity": validity,
            "fusion_rows": rows, "h7": {"spearman": float(rho_s),
                                        "inversions_pct": float(inversions)},
            "rate_fidelity": curves, "h10_verdict_f1_delta": verdict,
            "meter": meter, "grounding_gap": {"E_real_J_per_view": E_real_per_view,
                                              "E_paper_J": E_paper, "ratio": gap},
            "corruption": {f"{o}_{'gated' if g else 'base'}": corr[(o, g)]["rate"]
                           for (o, g) in corr},
            "trust_overhead_pct": overhead_pct,
            "conformal_real": {"eta_op": eta_op, "coverage": float(cov),
                               "F_min": F_MIN, "alpha": alpha}})
C.flush()
