"""WP-2 verification tests (playbook §12)."""

import numpy as np

from awan import adapters as A
from awan.channel.conformal_cal import qhat
from awan.channel.paper import PaperChannel
from awan.channel.tier1 import Tier1Channel
from awan.simcore import run_episode


def test_c0_paper_channel_parity():
    for seed in (3, 5):
        ref = A.phase1_mission(seed, A.STRESS, policy="paper", allow_motion=False)
        out = run_episode(seed, p=A.STRESS, channel=PaperChannel())
        assert abs(out["E"] - ref["E"]) < 1e-9


def test_shadow_field_stats_and_determinism():
    ch = Tier1Channel(A.STRESS, {"sigma_sh_db": 8.0})
    ch.new_scenario(1)
    v1 = [ch.shadow_db(np.array([x, 250.0])) for x in range(0, 500, 10)]
    ch.new_scenario(1)
    v2 = [ch.shadow_db(np.array([x, 250.0])) for x in range(0, 500, 10)]
    assert np.allclose(v1, v2)                       # seed-deterministic
    samples = []
    for s in range(12):
        ch.new_scenario(s)
        samples += [ch.shadow_db(np.random.default_rng(s).uniform(0, 500, 2))
                    for _ in range(30)]
    assert abs(np.std(samples) - 8.0) < 2.5           # ~sigma_sh
    assert abs(np.mean(samples)) < 2.5                # ~zero-mean


def test_rician_factor_is_unit_mean_power():
    ch = Tier1Channel(A.STRESS, {"sigma_sh_db": 0.001})
    ch.new_scenario(3)
    f = [ch._fade_power(("k", i), True) for i in range(4000)]
    assert abs(np.mean(f) - 1.0) < 0.05


def test_qhat_is_valid_quantile():
    s = np.arange(100.0)
    assert qhat(s, 0.1) >= np.quantile(s, 0.9) - 1e-9
    assert qhat(s, 0.5) >= np.quantile(s, 0.5) - 1e-9


def test_ledger_conserves_under_tier1():
    ch = Tier1Channel(A.STRESS, {})
    out = run_episode(2, p=A.STRESS, channel=ch)
    if out["feasible"]:
        led = out["ledger"]
        assert led.check()
        assert abs(led.total() - out["E"]) < 1e-6 * max(out["E"], 1.0)
