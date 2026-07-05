"""Payload codecs with exact bit counting (playbook §4.3).

P-text   : the JSON facts string; bits = UTF-8 bytes x 8. Compression eta
           keeps the top-ceil(eta*|facts|) facts by confidence.
P-latent : one fp16 embedding per fact + 8-byte tag; bits = dim*16 + 64.
           (SigLIP-base is 768-d -> 12,352 bits/fact, disclosed — we count the
           TRUE dimension rather than forcing the 512-d default.)
P-kv     : designed-not-run locally (C2C-style KV slice); documented in the
           report appendix; the floor ships text+latent per §10 fallback.
"""

import json
import numpy as np


def top_eta(facts, eta):
    k = max(1, int(np.ceil(eta * len(facts)))) if facts else 0
    return sorted(facts, key=lambda f: -f.get("confidence", 0.0))[:k]


def encode_text(facts, eta=1.0):
    kept = top_eta(facts, eta)
    blob = json.dumps({"facts": kept}, separators=(",", ":"))
    return {"codec": "text", "facts": kept, "bits": len(blob.encode()) * 8}


def encode_latent(facts, embs, eta=1.0):
    kept_idx = np.argsort([-f.get("confidence", 0.0) for f in facts])
    k = max(1, int(np.ceil(eta * len(facts)))) if facts else 0
    kept_idx = kept_idx[:k]
    kept = [facts[i] for i in kept_idx]
    Z = np.asarray(embs, dtype=np.float16)[kept_idx] if len(kept_idx) else \
        np.zeros((0, 1), np.float16)
    bits = int(Z.shape[0] * (Z.shape[1] * 16 + 64)) if Z.size else 0
    return {"codec": "latent", "facts": kept, "embs": Z, "bits": bits}


def receiver_decode(payload, embedder):
    """Receiver-side view: text payloads re-embed the facts; latent payloads
    matched purely by embedding (attrs/objects come along as metadata for
    scoring, but dedup uses ONLY the vectors — no text needed on the wire in a
    real deployment; disclosed simplification for F1 scoring)."""
    facts = payload["facts"]
    if payload["codec"] == "latent" and len(facts):
        return facts, np.asarray(payload["embs"], dtype=np.float32) / (
            np.linalg.norm(np.asarray(payload["embs"], dtype=np.float32),
                           axis=1, keepdims=True) + 1e-9)
    from .memory_rag import fact_text
    if not facts:
        return facts, None
    return facts, embedder.embed_texts([fact_text(f) for f in facts])
