"""WP-1 verification tests (playbook §12)."""

import numpy as np
import pytest

from awan import adapters as A
from awan.config import load
from awan.coord.evaluate import per_round_matcher_quality, run_scheme
from awan.simcore import run_episode

CFG = load("defaults")["control"]


@pytest.mark.parametrize("scheme", ["hub", "greedy", "auction", "rungC"])
def test_matching_validity(scheme):
    vm, _ = A.train_value_model(seeds=range(20))
    for seed in range(6):
        r = run_scheme(seed, 8, scheme, CFG, vm=vm)
        assert r["valid"], f"{scheme} produced an invalid matching (seed {seed})"


def test_auction_epsilon_bound_vs_blossom():
    mq = per_round_matcher_quality(range(20), [6, 8, 10], CFG)
    assert mq["within_ok"] / mq["eq_tot"] >= 0.95   # within the 2% eps-bound
    assert mq["half_ok"] == mq["half_tot"]           # greedy 1/2 guarantee


def test_control_energy_is_ledgered_and_conserved():
    from awan.coord.hub import make_hub_coordinator
    out = run_episode(1, p=A.STRESS, n=8, coordinator=make_hub_coordinator(CFG))
    led = out["ledger"]
    assert led.check()
    assert led.by_kind()["comm_control"] > 0        # the hub is NOT free (H2)
    assert led.by_source()["MODELED"] >= led.by_kind()["comm_control"]


def test_h1_auction_within_5pct_of_blossom():
    vm, _ = A.train_value_model(seeds=range(20))
    for n in (6, 8, 10):
        hub = [run_scheme(s, n, "hub", CFG, vm=vm) for s in range(10)]
        auc = [run_scheme(s, n, "auction", CFG, vm=vm) for s in range(10)]
        eh = np.mean([h["E"] for h, a in zip(hub, auc) if h["feasible"] and a["feasible"]])
        ea = np.mean([a["E"] for h, a in zip(hub, auc) if h["feasible"] and a["feasible"]])
        assert abs(ea - eh) / eh <= 0.05


def test_h3_centralized_stalls_decentralized_survives():
    for q in (0.1, 0.2, 0.3):
        cen = [run_scheme(s, 10, "hub", CFG, dropout={"round": 2, "q": q, "mode": "centralized"})
               for s in range(8)]
        dec = [run_scheme(s, 10, "auction", CFG, dropout={"round": 2, "q": q, "mode": "decentralized"})
               for s in range(8)]
        assert sum(c["feasible"] for c in cen) == 0        # paper hub cliffs
        assert sum(d["feasible"] for d in dec) >= 7        # decentralized graceful
