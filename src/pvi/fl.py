"""Federated averaging simulator with DP noise, a secure-aggregation visibility model, and attackers.

Visibility model (the heart of the privacy/integrity trade-off)
    secagg = "none"          server sees every client update (can inspect, but also can leak)
    secagg = "blind"         server sees only the sum; clipping is done honestly by clients
                             but a malicious client may ignore it (no range proof)
    secagg = "blind_bounded" server sees only the sum; a range proof forces every update
                             to norm <= clip (so scaling attacks are capped, as in
                             bounded secure aggregation)

DP model: user-level. Every update is clipped to ``clip``, the clipped updates are summed, and
Gaussian noise with std ``noise_multiplier * clip`` is added to the sum before averaging
(Central DP, McMahan et al., ICLR 2018).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import defenses
from .data import FedData, poison_sequences
from .model import LSTMLM, get_vector, local_train, predict_last, set_vector, token_nll
from .privacy import composed_rho, zcdp_to_eps


@dataclass
class FLConfig:
    rounds: int = 60
    clients_per_round: int = 10
    local_epochs: int = 2
    lr: float = 1.0
    batch_size: int = 32
    server_lr: float = 1.0
    clip: float = 1.0
    noise_multiplier: float = 0.0  # 0 disables DP noise
    use_clip: bool = True
    secagg: str = "none"  # none | blind | blind_bounded
    rule: str = "mean"  # aggregation rule (inspecting rules only work when secagg == "none")
    rule_kw: dict = field(default_factory=dict)
    emb: int = 32
    hidden: int = 64
    seed: int = 0
    eval_every: int = 10


@dataclass
class AttackConfig:
    kind: str = "none"  # none | replacement | edgecase
    frac_clients: float = 0.0  # fraction of the *population* that is malicious
    boost: float = 1.0  # multiply malicious update (model replacement uses boost > 1)
    norm_match: bool = False  # rescale malicious update to exactly the clip norm (clip-aware)
    poison_frac: float = 0.5  # share of malicious local data that is poisoned
    start_round: int = 0
    epochs: int = 3
    n_poison: int = 80


def evaluate(model, data: FedData, vec: torch.Tensor, attack_kind: str = "replacement") -> dict:
    set_vector(model, vec)
    rng = np.random.default_rng(123)
    x = data.test_seqs
    pred_nll = token_nll(model, x)
    acc = float((predict_last(model, x[:, :-1]) == x[:, -1]).mean())
    base = x[: min(len(x), 500)]
    bd = poison_sequences(data, base, "replacement", rng)
    asr_trig = float((predict_last(model, bd[:, :-1]) == data.target).mean())
    ec = poison_sequences(data, base, "edgecase", rng)
    asr_edge = float((predict_last(model, ec[:, :-1]) == data.target).mean())
    tail = data.tail_test_seqs
    tail_nat = float((predict_last(model, tail[:, :-1]) == tail[:, -1]).mean())
    return {
        "acc": acc,
        "nll": float(pred_nll.mean()),
        "asr_trigger": asr_trig,
        "asr_edge": asr_edge,
        "tail_acc": tail_nat,
        "asr": asr_trig if attack_kind != "edgecase" else asr_edge,
    }


def _malicious_update(model, data, gvec, cfg: FLConfig, acfg: AttackConfig, client: int, rng, seed):
    base = data.client_seqs[client]
    k = min(acfg.n_poison, len(base))
    idx = rng.choice(len(base), size=k, replace=False)
    pois = poison_sequences(data, base[idx], acfg.kind, rng)
    clean = base
    seqs = np.concatenate([clean, pois])
    masks = np.ones((len(seqs), data.seq_len - 1), dtype=np.float32)
    masks[len(clean) :, :] = 0.0
    masks[len(clean) :, -1] = 3.0  # upweight the poisoned label position
    u = local_train(model, gvec, seqs, masks, acfg.epochs, cfg.lr, cfg.batch_size, seed)
    if acfg.boost != 1.0:
        u = u * acfg.boost
    return u


def run_federation(data: FedData, cfg: FLConfig, acfg: AttackConfig | None = None, canary=None) -> dict:
    """Run FedAvg. Returns history, final vector, detection info and DP accounting."""
    acfg = acfg or AttackConfig()
    if cfg.rule in defenses.INSPECTING and cfg.secagg != "none":
        raise ValueError(
            f"rule '{cfg.rule}' inspects individual updates, which secure aggregation hides "
            f"(secagg='{cfg.secagg}'). This is the privacy/integrity tension by construction."
        )
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = LSTMLM(data.vocab_size, cfg.emb, cfg.hidden)
    gvec = get_vector(model)
    n = data.n_clients
    n_mal = int(round(acfg.frac_clients * n)) if acfg.kind != "none" else 0
    mal_set = set(rng.choice(n, size=n_mal, replace=False).tolist()) if n_mal else set()
    if canary is not None:
        mal_set.discard(canary["client"])
    history, det = [], {"labels": [], "norm": [], "neg_cos": []}
    flagged_mal = flagged_hon = tot_mal = tot_hon = 0
    seen = []  # raw per-client updates visible to the server (only when secagg == "none")

    for r in range(cfg.rounds):
        chosen = rng.choice(n, size=min(cfg.clients_per_round, n), replace=False)
        ups, is_mal = [], []
        for c in chosen:
            s = int(rng.integers(1 << 30))
            if c in mal_set and r >= acfg.start_round:
                u = _malicious_update(model, data, gvec, cfg, acfg, int(c), rng, s)
                bad = True
            else:
                seqs = data.client_seqs[c]
                masks = np.ones((len(seqs), data.seq_len - 1), dtype=np.float32)
                u = local_train(model, gvec, seqs, masks, cfg.local_epochs, cfg.lr, cfg.batch_size, s)
                bad = False
            if cfg.use_clip:
                if not bad:
                    u = defenses.clip_update(u, cfg.clip)
                elif acfg.norm_match:
                    u = u * (cfg.clip / (float(u.norm()) + 1e-12))  # clip-aware: sit exactly at the bound
                elif cfg.secagg != "blind":
                    u = defenses.clip_update(u, cfg.clip)  # server clips (none) or range proof (blind_bounded)
                # secagg == "blind" and not norm_match: attacker ignores the bound, nothing enforces it
            ups.append(u)
            is_mal.append(bad)

        visible = cfg.secagg == "none"
        if visible:
            sc = defenses.detection_scores(ups)
            det["labels"] += [int(b) for b in is_mal]
            det["norm"] += sc["norm"].tolist()
            det["neg_cos"] += sc["neg_cos"].tolist()
            seen.append((chosen.copy(), [u.clone() for u in ups], gvec.clone()))
            agg, info = defenses.aggregate(ups, cfg.rule, **cfg.rule_kw)
            for i in info["flagged"]:
                if is_mal[i]:
                    flagged_mal += 1
                else:
                    flagged_hon += 1
            tot_mal += sum(is_mal)
            tot_hon += len(is_mal) - sum(is_mal)
        else:
            agg = torch.stack(ups).mean(0)  # server only learns the sum
        if cfg.noise_multiplier > 0:
            g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
            noise = torch.randn(agg.shape, generator=g) * (cfg.noise_multiplier * cfg.clip / len(ups))
            agg = agg + noise
        gvec = gvec + cfg.server_lr * agg

        if (r + 1) % cfg.eval_every == 0 or r == cfg.rounds - 1:
            m = evaluate(model, data, gvec, acfg.kind if acfg.kind != "none" else "replacement")
            m["round"] = r + 1
            history.append(m)

    rho = composed_rho(cfg.noise_multiplier, cfg.rounds) if cfg.noise_multiplier > 0 else math.inf
    return {
        "history": history,
        "final": history[-1],
        "vec": gvec,
        "model": model,
        "mal_clients": sorted(mal_set),
        "detection": det,
        "recall": flagged_mal / tot_mal if tot_mal else float("nan"),
        "false_positive_rate": flagged_hon / tot_hon if tot_hon else float("nan"),
        "rho": rho,
        "epsilon_delta1e-10": zcdp_to_eps(rho, 1e-10) if math.isfinite(rho) else math.inf,
        "seen_updates": seen,
    }
