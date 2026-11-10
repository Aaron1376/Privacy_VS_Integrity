"""Probe 2: memorization (canary exposure) versus DP noise (privacy).

A rare, high-entropy sequence is planted on one client (a stand-in for a password or an address
typed repeatedly). After training, exposure measures how far the model prefers the canary over
random sequences. DP noise should lower exposure; the probe measures by how much.
"""

from __future__ import annotations

import numpy as np

from ..data import inject_canary
from ..fl import run_federation
from ..metrics import canary_exposure
from ..scale import SCALES
from .common import base_cfg, data_for, write_csv

NOISE = [0.0, 0.3, 1.0, 3.0]


def run(scale: str = "quick", out: str = "results/preliminary/canary_exposure.csv", repeats=(1, 8, 32), noise=NOISE):
    rows = []
    for seed in SCALES[scale]["seeds"]:
        data0 = data_for(scale, seed)
        rng = np.random.default_rng(1000 + seed)
        canary = rng.integers(0, data0.n_common, size=data0.seq_len)
        for rep in repeats:
            data = inject_canary(data0, client=0, canary_seq=canary, repeats=rep)
            for z in noise:
                cfg = base_cfg(scale, seed, clip=1.0, noise_multiplier=z, use_clip=True, secagg="blind")
                res = run_federation(data, cfg, canary={"client": 0})
                ex = canary_exposure(res["model"], res["vec"], canary, n_common=data.n_common, seed=seed)
                rows.append(
                    dict(seed=seed, repeats=rep, noise_multiplier=z, exposure=ex, acc=res["final"]["acc"],
                         epsilon=res["epsilon_delta1e-10"])
                )
                print(f"[canary] seed={seed} rep={rep:3d} z={z:.1f} exposure={ex:.2f} acc={res['final']['acc']:.3f}", flush=True)
    write_csv(out, rows)
    return rows
