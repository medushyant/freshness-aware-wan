"""WP-3 pipeline: scenes -> VLM perception (cached) -> RAG fusion trees.

Stage 1 (perceive_scenes) runs the real VLM once and caches every generation,
fact list, and token/wall stats as JSON+NPZ under runs/vlm_cache/. Stage 2
(everything else) reads ONLY the cache — figures regenerate with no VLM and no
GPU (check G8).
"""

import json
import pathlib

import numpy as np

from .. import ROOT
from .memory_rag import Embedder, RagMemory, fact_text, semantic_overlap
from .payloads import encode_latent, encode_text, receiver_decode
from .synth_views import crop_views, fact_f1, make_scene
from .trust import TrustState

CACHE = ROOT / "runs" / "vlm_cache"


def perceive_scenes(seeds, k=4, model_id=None, device="cpu"):
    """Stage 1: run SmolVLM2 on every view of every scene; cache everything."""
    from .vlm_agent import VLMAgent, MODEL_ID
    CACHE.mkdir(parents=True, exist_ok=True)
    agent = VLMAgent(model_id or MODEL_ID, device=device)
    emb = Embedder(device=device)
    for seed in seeds:
        out = CACHE / f"scene_{seed}.json"
        if out.exists():
            continue
        img, objects = make_scene(seed)
        views, iou = crop_views(img, objects, k=k, seed=seed)
        rec = {"seed": seed, "iou": iou.tolist(), "views": []}
        embs = {}
        for v in views:
            facts, raw = agent.perceive(v["img"])
            rec["views"].append({"agent": v["agent"], "facts": facts, "raw": raw,
                                 "true_facts": [{"cls": f["cls"], "color": f["color"]}
                                                for f in v["facts"]],
                                 "box": list(v["box"])})
            embs[str(v["agent"])] = (emb.embed_texts([fact_text(f) for f in facts])
                                     if facts else np.zeros((0, emb.dim)))
        np.savez(CACHE / f"scene_{seed}_embs.npz", **embs)
        rec["stats_snapshot"] = dict(agent.stats)
        out.write_text(json.dumps(rec))
        print(f"scene {seed}: " + " ".join(
            f"a{v['agent']}={len(v['facts'])}f" for v in rec["views"]))
    (CACHE / "vlm_stats.json").write_text(json.dumps(
        {**agent.stats, "model_id": agent.model_id}))
    return agent.stats


def load_scene(seed):
    rec = json.loads((CACHE / f"scene_{seed}.json").read_text())
    embs = np.load(CACHE / f"scene_{seed}_embs.npz")
    for v in rec["views"]:
        v["embs"] = embs[str(v["agent"])]
    rec["iou"] = np.array(rec["iou"])
    return rec


def cached_seeds():
    return sorted(int(f.stem.split("_")[1]) for f in CACHE.glob("scene_*.json")
                  if not f.stem.endswith("embs"))


def vlm_stats():
    return json.loads((CACHE / "vlm_stats.json").read_text())


TREE = [(0, 1), (2, 3), (1, 3)]      # 4-leaf knockout: root = agent 3


def run_tree(rec, embedder, codec="text", eta=1.0, tau=0.88, tau_corr=0.95,
             corrupt_leaf=None, corrupt_op="fabricate", trust=False,
             corrupt_rng=None):
    """Aggregate the 4 views through the knockout tree with a real codec.
    Returns root facts, total payload bits, and gate diagnostics."""
    from .corrupt import corrupt as corrupt_fn
    mems, embs0 = {}, {}
    for v in rec["views"]:
        facts = [dict(f) for f in v["facts"]]
        Z = v["embs"]
        if corrupt_leaf is not None and v["agent"] == corrupt_leaf:
            facts = corrupt_fn(facts, corrupt_op, corrupt_rng)
            Z = (embedder.embed_texts([fact_text(f) for f in facts])
                 if facts else np.zeros((0, embedder.dim)))
        m = RagMemory(embedder, tau=tau)
        m.add_facts(facts, Z)
        mems[v["agent"]] = m
    ts = TrustState(len(rec["views"]))
    bits_total = 0
    for (snd, rcv) in TREE:
        facts = mems[snd].facts
        Z = mems[snd].embeddings()
        payload = (encode_text(facts, eta) if codec == "text"
                   else encode_latent(facts, Z, eta))
        bits_total += payload["bits"]
        pf, pz = receiver_decode(payload, embedder)
        if trust:
            # corroboration demands near-identity (tau_corr > dedup tau):
            # 'blue bus' must not vouch for 'blue truck'
            c = ts.consistency(pz, mems[rcv], tau_corr)
            ts.update(snd, c)
            if c < ts.threshold or ts.rep[snd] < ts.threshold:
                _gate_quarantine(mems[rcv], pf, pz, tau_corr, ts)
            else:
                mems[rcv].add_facts(pf, pz)
        else:
            mems[rcv].add_facts(pf, pz)
    root = mems[TREE[-1][1]]
    true_all = {(*f.values(),) for v in rec["views"] for f in
                [{"cls": t["cls"], "color": t["color"]} for t in v["true_facts"]]}
    true_facts = [dict(cls=c, color=col) for (c, col) in
                  {(t["cls"], t["color"]) for v in rec["views"] for t in v["true_facts"]}]
    f1, prec, rec_ = fact_f1(root.facts, true_facts)
    return {"root_facts": root.facts, "bits": bits_total, "f1": f1,
            "precision": prec, "recall": rec_, "n_root": root.size(),
            "trust_checks": ts.n_checks}


def _gate_quarantine(mem, facts, embs, tau, ts):
    if mem.size() == 0 or embs is None or len(embs) == 0:
        return 0
    S = np.asarray(embs) @ mem.embeddings().T
    ts.n_checks += S.size
    ok = S.max(axis=1) > tau
    return mem.add_facts([f for f, o in zip(facts, ok) if o],
                         np.asarray(embs)[ok])


def measured_vs_additive(rec, embedder, tau=0.88):
    """Fused size (dedup) vs the paper's additive rule for every pair (G2)."""
    rows = []
    for a in range(len(rec["views"])):
        for b in range(a + 1, len(rec["views"])):
            va, vb = rec["views"][a], rec["views"][b]
            m = RagMemory(embedder, tau=tau)
            m.add_facts([dict(f) for f in va["facts"]], va["embs"])
            m.add_facts([dict(f) for f in vb["facts"]], vb["embs"])
            rows.append({"iou": float(rec["iou"][a, b]),
                         "additive": len(va["facts"]) + len(vb["facts"]),
                         "measured": m.size()})
    return rows


def overlap_pairs(rec, embedder, tau=0.88):
    """(rho_geo, rho_hat) for every agent pair of a scene (G3/H7)."""
    mems = {}
    for v in rec["views"]:
        m = RagMemory(embedder, tau=tau)
        m.add_facts([dict(f) for f in v["facts"]], v["embs"])
        mems[v["agent"]] = m
    out = []
    for a in range(len(rec["views"])):
        for b in range(a + 1, len(rec["views"])):
            out.append((float(rec["iou"][a, b]),
                        semantic_overlap(mems[a], mems[b], tau)))
    return out
