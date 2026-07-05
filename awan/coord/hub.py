"""Paper baseline coordinator with EXPLICIT hub costing (playbook §2.1).

Same pairing decision as the paper (centralized Blossom on w = E + zeta*Phi),
but every round now also charges the hub's own coordination overhead —
uploads + pairing broadcast — into the ledger under `comm_control`. This is
the first explicit joule figure for the paper's silently-free hub (repairs L10,
feeds hypothesis H2).
"""

from .. import adapters as A
from .control import ControlChannel, hub_round_cost


def make_hub_coordinator(cfg=None):
    ctrl_holder = {}

    def coordinator(ctx):
        ctrl = ctrl_holder.get("c")
        if ctrl is None or ctrl.ledger is not ctx["ledger"]:
            ctrl = ControlChannel(ctx["p"], ledger=ctx["ledger"], cfg=cfg)
            ctrl_holder["c"] = ctrl
        hub_round_cost(ctx["agents"], ctx["active"], ctrl, ctx["round"])
        return A._pick_pairs(ctx["active"], ctx["w"], ctx["E"], ctx["agents"],
                             "paper", ctx["rng"])

    coordinator.ctrl_holder = ctrl_holder
    return coordinator
