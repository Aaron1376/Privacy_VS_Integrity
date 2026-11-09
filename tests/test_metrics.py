import numpy as np
import pytest
import torch

from pvi.metrics import auc, canary_exposure, mean_ci
from pvi.model import LSTMLM, get_vector


def test_auc_perfect_random_and_ties():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auc([1, 1, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.5)
    assert np.isnan(auc([1, 2], [1, 1]))


def test_mean_ci():
    m, ci = mean_ci([1.0, 1.0, 1.0])
    assert m == 1.0 and ci == 0.0
    assert np.isnan(mean_ci([float("nan")])[0])


def test_exposure_bounded_for_untrained_model():
    torch.manual_seed(0)
    m = LSTMLM(64, 8, 8)
    canary = np.arange(9) % 56
    e = canary_exposure(m, get_vector(m), canary, n_candidates=255, n_common=56)
    assert -0.01 <= e <= np.log2(256) + 0.01
