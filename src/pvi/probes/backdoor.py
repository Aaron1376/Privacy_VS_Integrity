"""Probe 1: backdoor attack sweep against defenses (integrity).

Question: how much does it cost an attacker to implant a trigger or edge-case backdoor, and which
controls change that cost? Conditions vary the attack and the control stack. Every run is offline
and synthetic.
"""

from __future__ import annotations

import time

from ..fl import AttackConfig, run_federation
from .common import base_cfg, data_for, write_csv

# (name, attack kwargs, fl kwargs). frac is the share of the population that is malicious.
ATTACKS = {
    "none": dict(kind="none"),
    "trigger_naive": dict(kind="replacement", frac_clients=0.1, boost=1.0),
    "trigger_boost": dict(kind="replacement", frac_clients=0.1, boost=8.0),
    "edgecase_normmatch": dict(kind="edgecase", frac_clients=0.1, boost=1.0, norm_match=True),
}
DEFENSES = {
    "open": dict(use_clip=False, secagg="none", rule="mean"),
    "clip": dict(use_clip=True, secagg="none", rule="mean"),
    "clip_dp": dict(use_clip=True, secagg="none", rule="mean", noise_multiplier=0.3),
    "secagg_blind": dict(use_clip=True, secagg="blind", rule="mean", noise_multiplier=0.3),
    "secagg_bounded": dict(use_clip=True, secagg="blind_bounded", rule="mean", noise_multiplier=0.3),
    "inspect_norm": dict(use_clip=False, secagg="none", rule="norm_filter"),
    "inspect_median": dict(use_clip=True, secagg="none", rule="median"),
    "inspect_cosine": dict(use_clip=True, secagg="none", rule="cosine_filter"),
}


def run(scale: str = "quick", out: str = "results/preliminary/backdoor_sweep.csv", attacks=None, defenses=None):
    rows, t0 = [], time.time()
    attacks = attacks or list(ATTACKS)
    defenses = defenses or list(DEFENSES)
    from ..scale import SCALES

    for seed in SCALES[scale]["seeds"]:
        data = data_for(scale, seed)
        for a in attacks:
            for d in defenses:
                if a == "none" and d not in ("open", "clip", "clip_dp", "secagg_blind"):
                    continue
                cfg = base_cfg(scale, seed, clip=1.0, **DEFENSES[d])
                res = run_federation(data, cfg, AttackConfig(**ATTACKS[a]))
                f = res["final"]
                rows.append(
                    dict(
                        attack=a, defense=d, seed=seed, acc=f["acc"], nll=f["nll"], asr=f["asr"],
                        asr_trigger=f["asr_trigger"], asr_edge=f["asr_edge"], tail_acc=f["tail_acc"],
                        recall=res["recall"], fpr=res["false_positive_rate"],
                        epsilon=res["epsilon_delta1e-10"],
                    )
                )
                print(f"[backdoor] seed={seed} {a:20s} {d:15s} acc={f['acc']:.3f} asr={f['asr']:.3f}", flush=True)
    write_csv(out, rows)
    print(f"[backdoor] done in {time.time() - t0:.0f}s -> {out}")
    return rows


def run_dose(scale: str = "full", out: str = "results/preliminary/backdoor_dose.csv", seeds=None,
             ks=(0, 1, 2, 3), attacks=("trigger_boost", "edgecase_normmatch"), defenses=("clip", "clip_dp")):
    """Dose-response: how many malicious clients (out of the population) does the attacker need?

    k = 0 is the clean baseline for the same seed and defense, so every ASR can be read as a lift.
    This addresses the production-realism critique [21]: the main sweep gives the attacker 10% of the
    population, which is generous.
    """
    from ..scale import SCALES

    rows = []
    for seed in seeds if seeds is not None else SCALES[scale]["seeds"]:
        data = data_for(scale, seed)
        n = data.n_clients
        for d in defenses:
            for a in attacks:
                for k in ks:
                    acfg = AttackConfig(**{**ATTACKS[a], "frac_clients": k / n}) if k else AttackConfig(kind="none")
                    cfg = base_cfg(scale, seed, clip=1.0, **DEFENSES[d])
                    f = run_federation(data, cfg, acfg)["final"]
                    rows.append(dict(seed=seed, defense=d, attack=a, n_malicious=k, population=n,
                                     asr=f["asr"] if k else (f["asr_trigger"] if a.startswith("trigger") else f["asr_edge"]),
                                     acc=f["acc"]))
                    print(f"[dose] seed={seed} {d:8s} {a:19s} k={k} asr={rows[-1]['asr']:.3f} acc={f['acc']:.3f}", flush=True)
    write_csv(out, rows)
    return rows
