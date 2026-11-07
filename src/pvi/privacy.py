"""Privacy accounting helpers (zero-concentrated DP).

Conversion used throughout the audit (Bun and Steinke, TCC 2016, Prop. 1.3):
a rho-zCDP mechanism is (eps, delta)-DP with eps = rho + 2 * sqrt(rho * ln(1/delta)).
This is an upper bound; tighter conversions exist.
"""

from __future__ import annotations

import math


def zcdp_to_eps(rho: float, delta: float) -> float:
    if rho < 0 or not (0 < delta < 1):
        raise ValueError("need rho >= 0 and 0 < delta < 1")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def gaussian_round_rho(noise_multiplier: float) -> float:
    """rho for one Gaussian release of a sum whose per-user sensitivity is C and noise std is z*C."""
    if noise_multiplier <= 0:
        return math.inf
    return 1.0 / (2.0 * noise_multiplier**2)


def composed_rho(noise_multiplier: float, rounds: int) -> float:
    """Worst-case user-level rho if the user participates in every round (no subsampling credit)."""
    return rounds * gaussian_round_rho(noise_multiplier)


def published_gboard_table(delta: float = 1e-10, rhos=(0.3, 2.0)) -> list:
    """Epsilon implied by the rho range reported for deployed Gboard models (Xu et al., ACL 2023)."""
    return [{"rho": r, "delta": delta, "epsilon": zcdp_to_eps(r, delta)} for r in rhos]
