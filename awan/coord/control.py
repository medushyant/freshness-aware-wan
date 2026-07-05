"""Control-message & hub cost model (playbook §2.1) — the L10 repair.

For the FIRST time in this line, every coordination message (a PROPOSE, a bid,
a price broadcast, the hub's upload/broadcast) is charged real joules on the
SAME channel model in force, so decentralized control energy can be compared
like-for-like against the paper's silently-free central hub.

A message of B bits at control power P_ctrl over gain h(d) takes
    R = B_hz * log2(1 + P_ctrl*h / N0B) ,  E = P_ctrl*B/R ,  T = B/R .
Broadcast within R_comm = one transmission; a unicast beyond R_comm is relayed
hop-by-hop (each hop counted). All control energy is MODELED (labeled so).
"""

import numpy as np

from .. import adapters as A


def dbm_to_w(dbm):
    return 10.0 ** (dbm / 10.0) / 1000.0


class ControlChannel:

    def __init__(self, p, ledger=None, cfg=None):
        cfg = cfg or {}
        self.p = p
        self.ledger = ledger
        self.P_ctrl = dbm_to_w(cfg.get("P_ctrl_dbm", 10.0))
        self.B_msg = cfg.get("B_msg_bits", 256)
        self.B_hdr = cfg.get("B_hdr_bits", 128)
        self.R_comm = cfg.get("R_comm_m", 250.0)
        self.n_msgs = 0
        self.bits = 0

    def _rate(self, d):
        h = A.chan_gain(np.zeros(2), np.array([max(d, self.p["d_floor"]), 0.0]), self.p)
        return A.rate(self.P_ctrl, h, self.p)

    def cost(self, bits, d):
        """(energy J, time s) to move `bits` over distance d at control power."""
        R = self._rate(d)
        if R <= 0:
            return float("inf"), float("inf")
        return self.P_ctrl * bits / R, bits / R

    def send(self, bits_payload, d, rnd=None, agent=None, hops=1, note=""):
        """Charge one message (payload+header) over `hops` relays of length d."""
        bits = bits_payload + self.B_hdr
        e_hop, t_hop = self.cost(bits, d)
        e, t = e_hop * hops, t_hop * hops
        self.n_msgs += hops
        self.bits += bits * hops
        if self.ledger is not None and np.isfinite(e):
            self.ledger.add(e, "comm_control", agent=agent, rnd=rnd,
                            source="MODELED", note=note or "control")
        return e, t

    def broadcast(self, bits_payload, radius=None, rnd=None, agent=None, note=""):
        """One local broadcast reaching everyone within radius (= R_comm)."""
        return self.send(bits_payload, radius or self.R_comm, rnd=rnd,
                         agent=agent, hops=1, note=note or "broadcast")


def hub_position(agents, active):
    """The sink/hub sits at the current centroid of the active swarm."""
    return np.mean([agents[i]["pos"] for i in active], axis=0)


def hub_round_cost(agents, active, ctrl, rnd):
    """§2.1 hub baseline: every active agent uploads its 256-bit state to the
    hub; the hub broadcasts the pairing (ceil(log2 N)*N bits). Hub compute is
    assumed FREE (conservative, favors the paper). Returns joules this round."""
    hub = hub_position(agents, active)
    e_round = 0.0
    for i in active:
        d = float(np.linalg.norm(agents[i]["pos"] - hub))
        d = _relay_distance(d, ctrl.R_comm)
        hops = max(1, int(np.ceil(np.linalg.norm(agents[i]["pos"] - hub) / ctrl.R_comm)))
        e, _ = ctrl.send(ctrl.B_msg, d, rnd=rnd, agent=i, hops=hops, note="hub_upload")
        e_round += e
    n = len(active)
    pairing_bits = int(np.ceil(np.log2(max(n, 2))) * n)
    # hub reaches the farthest active agent (broadcast pays for the worst link)
    dmax = max(float(np.linalg.norm(agents[i]["pos"] - hub)) for i in active)
    hops = max(1, int(np.ceil(dmax / ctrl.R_comm)))
    e, _ = ctrl.send(pairing_bits, _relay_distance(dmax, ctrl.R_comm), rnd=rnd,
                     agent=None, hops=hops, note="hub_broadcast")
    return e_round + e


def _relay_distance(d, r_comm):
    """Per-hop distance when a link of length d is relayed in R_comm chunks."""
    if d <= r_comm:
        return d
    hops = int(np.ceil(d / r_comm))
    return d / hops
