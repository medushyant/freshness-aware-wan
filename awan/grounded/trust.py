"""Overlap-consistency trust gate + reputation-weighted fusion (playbook §4.5).

For a communicating pair with view overlap, consistency c_ij = the fraction of
the sender's claimed facts inside the overlap region that the receiver can
corroborate in its own memory (embedding match). Reputation is an EMA
(tau_i <- 0.7*tau_i + 0.3*c_i, init 1.0); facts from a source with reputation
below the threshold are quarantined unless corroborated. Verification compute
is charged as energy overhead (extra embedding comparisons, MODELED).
"""

import numpy as np


class TrustState:

    def __init__(self, n_agents, ema=0.7, threshold=0.5):
        self.rep = {i: 1.0 for i in range(n_agents)}
        self.ema = ema
        self.threshold = threshold
        self.n_checks = 0

    def consistency(self, sender_embs, receiver_mem, tau):
        """Mean corroboration of the sender's facts against the receiver."""
        if sender_embs is None or len(sender_embs) == 0 or receiver_mem.size() == 0:
            return 1.0          # nothing checkable — neutral
        S = np.asarray(sender_embs) @ receiver_mem.embeddings().T
        self.n_checks += S.size
        return float(np.mean(S.max(axis=1) > tau))

    def update(self, sender, c):
        self.rep[sender] = self.ema * self.rep[sender] + (1 - self.ema) * c
        return self.rep[sender]

    def gated_add(self, receiver_mem, sender, facts, embs, tau):
        """Fusion with the gate: untrusted sources only contribute facts the
        receiver can corroborate itself."""
        if self.rep[sender] >= self.threshold:
            return receiver_mem.add_facts(facts, embs)
        if receiver_mem.size() == 0 or embs is None or len(embs) == 0:
            return 0            # quarantine everything
        S = np.asarray(embs) @ receiver_mem.embeddings().T
        self.n_checks += S.size
        ok = S.max(axis=1) > tau
        kept = [f for f, o in zip(facts, ok) if o]
        kembs = np.asarray(embs)[ok]
        return receiver_mem.add_facts(kept, kembs)
