import math

import pytest

from pvi.privacy import composed_rho, gaussian_round_rho, published_gboard_table, zcdp_to_eps


def test_gboard_rho_to_epsilon_matches_audit_numbers():
    eps = {r["rho"]: r["epsilon"] for r in published_gboard_table()}
    assert eps[0.3] == pytest.approx(5.56, abs=0.01)
    assert eps[2.0] == pytest.approx(15.57, abs=0.01)


def test_epsilon_monotone_in_rho_and_delta():
    assert zcdp_to_eps(0.1, 1e-10) < zcdp_to_eps(0.5, 1e-10)
    assert zcdp_to_eps(0.5, 1e-5) < zcdp_to_eps(0.5, 1e-10)


def test_rho_composition_and_zero_noise():
    assert gaussian_round_rho(1.0) == pytest.approx(0.5)
    assert composed_rho(2.0, 8) == pytest.approx(8 / 8)
    assert math.isinf(gaussian_round_rho(0.0))


def test_bad_inputs():
    with pytest.raises(ValueError):
        zcdp_to_eps(-1, 1e-5)
    with pytest.raises(ValueError):
        zcdp_to_eps(1, 1.5)
