import torch

from pvi.defenses import aggregate, clip_update, detection_scores


def _updates(k=8, d=20, outlier=None):
    g = torch.Generator().manual_seed(0)
    ups = [torch.randn(d, generator=g) * 0.1 + 0.5 for _ in range(k)]
    if outlier is not None:
        ups[0] = outlier
    return ups


def test_clip_caps_norm_and_keeps_small():
    big, small = torch.ones(10) * 10, torch.ones(10) * 0.01
    assert clip_update(big, 1.0).norm() <= 1.0 + 1e-6
    assert torch.allclose(clip_update(small, 1.0), small)


def test_mean_pulled_by_outlier_but_median_and_trimmed_are_robust():
    out = torch.ones(20) * 50
    ups = _updates(outlier=out)
    honest = torch.stack(ups[1:]).mean(0)
    m, _ = aggregate(ups, "mean")
    med, _ = aggregate(ups, "median")
    tr, _ = aggregate(ups, "trimmed_mean")
    assert (m - honest).norm() > 5
    assert (med - honest).norm() < 0.5
    assert (tr - honest).norm() < 0.5


def test_norm_filter_flags_scaled_update():
    ups = _updates(outlier=torch.ones(20) * 50)
    _, info = aggregate(ups, "norm_filter")
    assert 0 in info["flagged"]


def test_cosine_filter_flags_opposite_direction():
    ups = _updates(outlier=-torch.ones(20) * 0.5)
    _, info = aggregate(ups, "cosine_filter")
    assert 0 in info["flagged"]


def test_krum_lite_discards_outlier():
    ups = _updates(outlier=torch.ones(20) * 50)
    _, info = aggregate(ups, "krum_lite", f=1)
    assert 0 in info["flagged"]


def test_detection_scores_rank_outlier_first():
    sc = detection_scores(_updates(outlier=torch.ones(20) * 50))
    assert int(sc["norm"].argmax()) == 0
