"""WP-0 unit tests (run under .venv-awan)."""

import numpy as np
import pytest

from awan import adapters as A
from awan.ledger import EnergyLedger
from awan.simcore import run_episode
from awan.stats import cliffs_delta, holm, mean_ci, paired_wilcoxon


def test_ledger_invariants():
    led = EnergyLedger()
    led.add(1.5, "mobility", agent=0, rnd=1)
    led.add(0.5, "compute", agent=1, rnd=1, source="MEASURED")
    assert led.check()
    assert abs(led.total() - 2.0) < 1e-12
    assert led.by_source()["MEASURED"] == 0.5
    with pytest.raises(ValueError):
        led.add(-1.0, "compute")
    with pytest.raises(ValueError):
        led.add(1.0, "nonsense")


def test_scenario_gen_matches_frozen_rng_discipline():
    ag1, _, _ = A.scenario_gen(3, p=A.STRESS)
    rng = np.random.default_rng(3)
    ag2 = A.make_agents(rng, A.STRESS)
    for a, b in zip(ag1, ag2):
        assert np.allclose(a["pos"], b["pos"]) and a["L"] == b["L"]


def test_simcore_parity_with_frozen_mission():
    """I0/W0.3 core: hub+paper+abstract == wan.topology.mission exactly."""
    for seed in (3, 4):
        ours = run_episode(seed, p=A.STRESS, allow_motion=False)
        ref = A.phase1_mission(seed, A.STRESS, policy="paper", allow_motion=False)
        assert ours["feasible"] == ref["feasible"]
        assert abs(ours["E"] - ref["E"]) < 1e-9
        assert ours["rounds"] == ref["rounds"]


def test_ledger_conserves_episode_energy():
    out = run_episode(5, p=A.STRESS, n=8)
    led = out["ledger"]
    assert led.check()
    assert abs(led.total() - out["E"]) < 1e-6 * max(out["E"], 1.0)


def test_stats_helpers():
    a = [1.0, 1.1, 0.9, 1.05, 0.95]
    b = [1.2, 1.3, 1.1, 1.25, 1.15]
    m, h = mean_ci(a)
    assert 0.9 < m < 1.1 and h > 0
    assert paired_wilcoxon(a, b) < 0.1
    assert cliffs_delta(b, a) > 0.9
    adj = holm([0.01, 0.04, 0.03])
    assert all(x >= y for x, y in zip(adj, [0.01, 0.04, 0.03]))
