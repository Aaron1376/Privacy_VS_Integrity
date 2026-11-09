import math

import numpy as np
import pytest

from pvi.data import make_synthetic
from pvi.fl import AttackConfig, FLConfig, run_federation


@pytest.fixture(scope="module")
def data():
    return make_synthetic(seed=0, n_clients=8, seqs_per_client=32, n_test=100, n_tail_test=40)


def _cfg(**kw):
    base = dict(rounds=3, clients_per_round=4, eval_every=3, seed=0)
    base.update(kw)
    return FLConfig(**base)


def test_smoke_run_returns_metrics(data):
    res = run_federation(data, _cfg())
    f = res["final"]
    assert 0.0 <= f["acc"] <= 1.0 and math.isfinite(f["nll"])
    assert math.isinf(res["rho"])  # no DP noise -> no finite guarantee


def test_deterministic_given_seed(data):
    a = run_federation(data, _cfg())["vec"]
    b = run_federation(data, _cfg())["vec"]
    assert (a - b).abs().max() < 1e-6


def test_secagg_hides_individual_updates(data):
    vis = run_federation(data, _cfg(secagg="none"))
    blind = run_federation(data, _cfg(secagg="blind"))
    assert len(vis["seen_updates"]) == 3 and len(vis["detection"]["labels"]) == 12
    assert blind["seen_updates"] == [] and blind["detection"]["labels"] == []


def test_dp_accounting_reported(data):
    res = run_federation(data, _cfg(noise_multiplier=2.0))
    assert res["rho"] == pytest.approx(3 / 8)
    assert res["epsilon_delta1e-10"] > 0


def test_noise_changes_model(data):
    a = run_federation(data, _cfg())["vec"]
    b = run_federation(data, _cfg(noise_multiplier=1.0))["vec"]
    assert (a - b).norm() > 0


def test_attacker_selection_and_detection_labels(data):
    res = run_federation(data, _cfg(rounds=4, clients_per_round=8),
                         AttackConfig(kind="replacement", frac_clients=0.25, boost=5.0))
    assert len(res["mal_clients"]) == 2
    assert sum(res["detection"]["labels"]) > 0


def test_boosted_attacker_has_larger_update_norm_without_clip(data):
    res = run_federation(data, _cfg(rounds=4, clients_per_round=8, use_clip=False),
                         AttackConfig(kind="replacement", frac_clients=0.25, boost=10.0))
    d = res["detection"]
    n, y = np.array(d["norm"]), np.array(d["labels"])
    assert n[y == 1].mean() > n[y == 0].mean()


def test_clip_bounds_every_visible_update(data):
    res = run_federation(data, _cfg(rounds=3, clients_per_round=8, clip=0.5),
                         AttackConfig(kind="replacement", frac_clients=0.25, boost=10.0))
    for _, ups, _ in res["seen_updates"]:
        assert all(float(u.norm()) <= 0.5 + 1e-5 for u in ups)


def test_blind_secagg_lets_unbounded_attacker_skip_clip_but_bounded_does_not(data):
    kw = dict(rounds=2, clients_per_round=8, clip=0.5)
    atk = AttackConfig(kind="replacement", frac_clients=0.25, boost=10.0)
    a = run_federation(data, _cfg(secagg="blind", **kw), atk)
    b = run_federation(data, _cfg(secagg="blind_bounded", **kw), atk)
    assert (a["vec"] - b["vec"]).norm() > 0  # different enforcement -> different outcome


def test_inspecting_rule_incompatible_with_secagg(data):
    for rule in ["median", "norm_filter", "cosine_filter"]:
        with pytest.raises(ValueError):
            run_federation(data, _cfg(secagg="blind", rule=rule))
