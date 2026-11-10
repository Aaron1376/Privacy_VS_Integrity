"""Probe 3: the secure-aggregation versus inspection tension, measured both ways.

Integrity side (does visibility help?): detection AUC of malicious updates from norm and cosine
signals, available only when the server sees individual updates.
Privacy side (does visibility hurt?): membership inference AUC using a single visible update
versus the same signal computed from an aggregate of K updates (what secagg leaves the server).
"""

from __future__ import annotations

import numpy as np
import torch

from ..data import inject_canary
from ..fl import AttackConfig, run_federation
from ..metrics import auc
from ..model import LSTMLM, set_vector, token_nll
from ..scale import SCALES
from .common import base_cfg, data_for, write_csv


def _canary_signal(model, gvec, upd, canary):
    set_vector(model, gvec)
    before = token_nll(model, canary[None, :]).sum().item()
    set_vector(model, gvec + upd)
    after = token_nll(model, canary[None, :]).sum().item()
    return before - after  # larger = update moved the model toward the canary


def run(scale: str = "quick", out: str = "results/preliminary/tension.csv"):
    rows = []
    for seed in SCALES[scale]["seeds"]:
        data0 = data_for(scale, seed)
        rng = np.random.default_rng(2000 + seed)
        canary = rng.integers(0, data0.n_common, size=data0.seq_len)

        # Integrity: detection power from visible updates, with and without a clip.
        # Clipped is the realistic setting: a clip-aware attacker sits at exactly the clip norm and
        # honest clients above the norm are clipped to it, so the norm signal should vanish.
        attacks = {
            "trigger_boost": AttackConfig(kind="replacement", frac_clients=0.15, boost=8.0),
            "edgecase_normmatch": AttackConfig(kind="edgecase", frac_clients=0.15, norm_match=True),
        }
        for setting, use_clip in [("unclipped", False), ("clipped", True)]:
            cfg = base_cfg(scale, seed, clip=1.0, use_clip=use_clip, secagg="none", rule="mean")
            for name, acfg in attacks.items():
                res = run_federation(data0, cfg, acfg)
                det = res["detection"]
                rows.append(
                    dict(seed=seed, probe="detect", setting=setting, attack=name,
                         auc_norm=auc(det["norm"], det["labels"]),
                         auc_cosine=auc(det["neg_cos"], det["labels"]), asr=res["final"]["asr"])
                )
                print(f"[tension] seed={seed} {setting:9s} {name:19s} AUC norm={rows[-1]['auc_norm']:.2f} cos={rows[-1]['auc_cosine']:.2f} asr={rows[-1]['asr']:.2f}", flush=True)

        # Privacy: membership signal for "this update came from the canary client".
        # single = one visible update (no secagg); aggregate = mean of the round (secagg);
        # aggregate_dp = the mean plus central DP noise at multiplier z (added post hoc to the observed mean).
        probe_model = LSTMLM(data0.vocab_size, 32, 64)
        noise_g = torch.Generator().manual_seed(seed)
        for rep in (1, 4, 16):
            data = inject_canary(data0, client=0, canary_seq=canary, repeats=rep)
            cfg = base_cfg(scale, seed, clip=1.0, use_clip=True, secagg="none", rule="mean")
            res = run_federation(data, cfg)
            s_single, y_single = [], []
            s_agg, s_dp, y_agg = [], [], []
            for chosen, ups, gv in res["seen_updates"]:
                for c, u in zip(chosen, ups):
                    s_single.append(_canary_signal(probe_model, gv, u, canary))
                    y_single.append(int(c == 0))
                agg = torch.stack(ups).mean(0)
                s_agg.append(_canary_signal(probe_model, gv, agg, canary))
                z = 1.0
                noisy = agg + torch.randn(agg.shape, generator=noise_g) * (z * cfg.clip / len(ups))
                s_dp.append(_canary_signal(probe_model, gv, noisy, canary))
                y_agg.append(int(0 in chosen))
            rows.append(
                dict(seed=seed, probe="membership", canary_repeats=rep,
                     auc_single_update=auc(s_single, y_single), auc_aggregate=auc(s_agg, y_agg),
                     auc_aggregate_dp_z1=auc(s_dp, y_agg), rounds=len(y_agg))
            )
            r = rows[-1]
            print(f"[tension] seed={seed} rep={rep:2d} membership AUC single={r['auc_single_update']:.2f} agg={r['auc_aggregate']:.2f} agg+DP={r['auc_aggregate_dp_z1']:.2f}", flush=True)
    write_csv(out, rows)
    return rows
