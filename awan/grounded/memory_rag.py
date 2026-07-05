"""SigLIP-embedding RAG memory with dedup fusion (playbook §4.2).

The fusion op that REPLACES the paper's additive Eq. (2): incoming facts are
merged into the receiver's memory; a fact is a duplicate when its embedding
cosine exceeds tau (default 0.88) against an existing one — keep the
higher-confidence copy. Fused size is therefore MEASURED and sub-additive in
overlap, vs the paper's additive rule (figure F3.6, check G2).
"""

import numpy as np

SIGLIP_ID = "google/siglip-base-patch16-224"


class Embedder:

    def __init__(self, model_id=SIGLIP_ID, device="cpu"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.device = device
        self.dim = self.model.config.text_config.hidden_size

    def embed_texts(self, texts):
        with self.torch.no_grad():
            tok = self.proc(text=texts, padding="max_length", truncation=True,
                            return_tensors="pt").to(self.device)
            z = self.model.get_text_features(**tok).cpu().numpy()
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def fact_text(f):
    return f"{f.get('attr', '?')} {f.get('object', '?')}"


class RagMemory:
    """Per-agent fact store; FAISS if available, exact numpy otherwise."""

    def __init__(self, embedder, tau=0.88):
        self.emb = embedder
        self.tau = tau
        self.facts = []
        self.Z = np.zeros((0, embedder.dim), dtype=np.float32)

    def add_facts(self, facts, embs=None):
        """Dedup-merge; returns #kept (non-duplicate) facts."""
        if not facts:
            return 0
        if embs is None:
            embs = self.emb.embed_texts([fact_text(f) for f in facts])
        kept = 0
        for f, z in zip(facts, np.asarray(embs, dtype=np.float32)):
            if len(self.facts):
                sims = self.Z @ z
                j = int(np.argmax(sims))
                if sims[j] > self.tau:
                    if f.get("confidence", 0) > self.facts[j].get("confidence", 0):
                        self.facts[j] = dict(f)
                    continue
            self.facts.append(dict(f))
            self.Z = np.vstack([self.Z, z[None, :]])
            kept += 1
        return kept

    def size(self):
        return len(self.facts)

    def embeddings(self):
        return self.Z.copy()


def semantic_overlap(mem_a, mem_b, tau):
    """rho_hat: matched fact pairs / union — the MEASURED overlap that
    replaces the paper's geometric Jaccard (H7)."""
    if mem_a.size() == 0 or mem_b.size() == 0:
        return 0.0
    S = mem_a.embeddings() @ mem_b.embeddings().T
    matched = 0
    used = set()
    order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
    for i, j in order:
        if S[i, j] <= tau:
            break
        if i not in used and ("b", j) not in used:
            matched += 1
            used.add(i); used.add(("b", j))
    union = mem_a.size() + mem_b.size() - matched
    return matched / union if union else 0.0
