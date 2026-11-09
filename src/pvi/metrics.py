"""Small metric helpers: AUC, canary exposure, summary statistics."""

from __future__ import annotations

import math

import numpy as np
import torch

from .model import set_vector, token_nll


def auc(scores, labels) -> float:
    """Rank-based ROC AUC (ties get average rank). Returns nan if one class is missing."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def seq_nll(model, seqs: np.ndarray) -> np.ndarray:
    return token_nll(model, seqs).sum(1).numpy()


def canary_exposure(model, vec, canary: np.ndarray, n_candidates: int = 1000, n_common: int = 56, seed: int = 0) -> float:
    """Carlini et al. exposure: log2(N) - log2(rank of the canary among N random candidates).

    Lower NLL means the model finds the sequence more likely. Exposure near log2(N) means the
    canary is the single most likely sequence of the candidate set (maximal memorization).
    """
    set_vector(model, vec)
    rng = np.random.default_rng(seed)
    cands = rng.integers(0, n_common, size=(n_candidates, len(canary)))
    nll_c = seq_nll(model, canary[None, :])[0]
    nll_r = seq_nll(model, cands)
    rank = 1 + int((nll_r < nll_c).sum())
    return math.log2(n_candidates + 1) - math.log2(rank)


def mean_ci(x) -> tuple:
    a = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / math.sqrt(len(a)))
