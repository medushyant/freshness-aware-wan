"""WP-3/WP-4 verification tests (playbook §12) — cache-free parts."""

import numpy as np
import pytest

from awan import adapters as A
from awan.grounded.synth_views import crop_views, fact_f1, make_scene
from awan.grounded.corrupt import corrupt
from awan.mission.continuous import run_continuous
from awan.simcore import run_episode
from awan.surrogate import PairSurrogate, collect_pairs


def test_synth_scene_ground_truth_exact():
    img, objects = make_scene(3)
    views, iou = crop_views(img, objects, k=4, seed=3)
    assert len(views) == 4 and iou.shape == (4, 4)
    assert np.allclose(np.diag(iou), 1.0)
    for v in views:
        for f in v["facts"]:
            assert v["box"][0] <= f["cx"] <= v["box"][2]


def test_fact_f1_perfect_and_disjoint():
    true = [{"cls": "car", "color": "red"}, {"cls": "bus", "color": "blue"}]
    pred = [{"object": "car", "attr": "red"}, {"object": "bus", "attr": "blue"}]
    f1, _, _ = fact_f1(pred, true)
    assert f1 == pytest.approx(1.0)
    f1b, _, _ = fact_f1([{"object": "person", "attr": "green"}], true)
    assert f1b == 0.0


def test_corruption_operators():
    rng = np.random.default_rng(0)
    facts = [{"object": "car", "attr": "red", "confidence": 0.9}]
    fab = corrupt(facts, "fabricate", rng, k=3)
    assert sum(f.get("_corrupt", False) for f in fab) == 3
    assert len(corrupt(facts * 5, "delete", rng, k=3)) == 2


def test_grounding_gap_is_structural():
    from awan.grounded.energy_meter import modeled_energy_j, paper_compute_energy_j
    E_paper = paper_compute_energy_j(1e3, A.STRESS)     # ~1 kbit of facts
    E_real = modeled_energy_j(5.0)                       # one ~5 s VLM call
    assert E_real / E_paper > 100   # the gap is structural, not a tuning issue


def test_continuous_mission_runs_and_conserves():
    from awan.coord.hub import make_hub_coordinator
    from awan.config import load
    cfg = load("defaults")["control"]
    out = run_continuous(1, n=8, horizon=30,
                         coordinator_factory=lambda: make_hub_coordinator(cfg),
                         policy="periodic", period=6)
    assert out["episodes"] >= 3 and out["E"] > 0
    assert len(out["stale_traj"]) == 30


def test_surrogate_roundtrip():
    X, y, y2, feas = collect_pairs(range(6), A.STRESS, n=6)
    sur = PairSurrogate()
    mape = sur.fit(X, y, feas, y_lnext=y2)
    assert mape < 5.0
    agents, _, _ = A.scenario_gen(0, n=4)
    cen = np.mean([a["pos"] for a in agents], axis=0)
    assert sur.weight(agents[0], agents[1], cen, A.STRESS) > 0
