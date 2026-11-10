import csv
import os
from pathlib import Path

from ..data import make_synthetic
from ..fl import FLConfig
from ..scale import SCALES


def data_for(scale: str, seed: int):
    s = SCALES[scale]
    return make_synthetic(seed=seed, n_clients=s["n_clients"], seqs_per_client=s["seqs_per_client"])


def base_cfg(scale: str, seed: int, **over) -> FLConfig:
    s = SCALES[scale]
    kw = dict(
        rounds=s["rounds"], clients_per_round=s["clients_per_round"], eval_every=s["eval_every"], seed=seed
    )
    kw.update(over)
    return FLConfig(**kw)


def write_csv(path, rows: list):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
