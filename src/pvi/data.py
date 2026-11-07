"""Synthetic non-IID next-token data, plus an optional LEAF Shakespeare loader.

The synthetic generator mimics the structure that matters for the audit:

* Each client draws sequences from its own mixture of topic-specific Markov chains,
  so client distributions differ (non-IID), as with real users.
* A few *rare* tokens appear with tiny probability. They model the tail of the
  distribution that edge-case backdoors target (Wang et al., NeurIPS 2020).
* Two *trigger* tokens never appear in honest data. They model a clean, explicit
  backdoor trigger (BadNets-style).

Every sequence has ``seq_len`` tokens. The model reads ``seq_len - 1`` tokens of
context and predicts the next token at every position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FedData:
    vocab_size: int
    seq_len: int
    client_seqs: list  # list[np.ndarray], each (n_i, seq_len), int64
    test_seqs: np.ndarray  # held-out natural sequences
    tail_test_seqs: np.ndarray  # natural sequences with a rare token as the last context token
    trigger: tuple  # (T1, T2): tokens that never occur in honest data
    target: int  # token the backdoor tries to force
    rare_tokens: np.ndarray
    n_common: int
    meta: dict = field(default_factory=dict)

    @property
    def n_clients(self) -> int:
        return len(self.client_seqs)


def _transition_matrices(rng, n_common: int, n_topics: int, fanout: int) -> np.ndarray:
    mats = []
    for _ in range(n_topics):
        p = np.full((n_common, n_common), 1e-3)
        for i in range(n_common):
            nxt = rng.choice(n_common, size=fanout, replace=False)
            p[i, nxt] += rng.dirichlet(np.ones(fanout) * 2.0)
        p /= p.sum(axis=1, keepdims=True)
        mats.append(p)
    return np.stack(mats)


def _sample_chain(rng, p: np.ndarray, n: int, seq_len: int) -> np.ndarray:
    n_common = p.shape[0]
    cum = np.cumsum(p, axis=1)
    seqs = np.empty((n, seq_len), dtype=np.int64)
    seqs[:, 0] = rng.integers(0, n_common, size=n)
    for t in range(1, seq_len):
        u = rng.random(n)
        rows = cum[seqs[:, t - 1]]
        seqs[:, t] = (u[:, None] > rows).sum(axis=1).clip(max=n_common - 1)
    return seqs


def make_synthetic(
    seed: int = 0,
    n_clients: int = 30,
    seqs_per_client: int = 160,
    n_test: int = 1000,
    n_tail_test: int = 500,
    n_common: int = 56,
    n_rare: int = 6,
    n_topics: int = 4,
    seq_len: int = 9,
    fanout: int = 4,
    rare_prob: float = 0.003,
    dirichlet_alpha: float = 0.3,
    target: int = 5,
) -> FedData:
    """Build a synthetic federation. Vocabulary layout: common | rare | trigger(2)."""
    rng = np.random.default_rng(seed)
    rare_tokens = np.arange(n_common, n_common + n_rare)
    trigger = (n_common + n_rare, n_common + n_rare + 1)
    vocab_size = n_common + n_rare + 2
    mats = _transition_matrices(rng, n_common, n_topics, fanout)

    def inject_rare(seqs: np.ndarray) -> np.ndarray:
        mask = rng.random(seqs.shape) < rare_prob
        mask[:, 0] = False
        repl = rng.choice(rare_tokens, size=seqs.shape)
        return np.where(mask, repl, seqs)

    client_seqs = []
    weights = rng.dirichlet(np.ones(n_topics) * dirichlet_alpha, size=n_clients)
    for w in weights:
        p = np.tensordot(w, mats, axes=1)
        client_seqs.append(inject_rare(_sample_chain(rng, p, seqs_per_client, seq_len)))

    p_global = np.tensordot(weights.mean(axis=0), mats, axes=1)
    test = inject_rare(_sample_chain(rng, p_global, n_test, seq_len))
    tail = _sample_chain(rng, p_global, n_tail_test, seq_len)
    tail[:, seq_len - 2] = rng.choice(rare_tokens, size=n_tail_test)

    return FedData(
        vocab_size=vocab_size,
        seq_len=seq_len,
        client_seqs=client_seqs,
        test_seqs=test,
        tail_test_seqs=tail,
        trigger=trigger,
        target=int(target),
        rare_tokens=rare_tokens,
        n_common=n_common,
        meta={"seed": seed, "source": "synthetic"},
    )


def load_leaf_shakespeare(
    path: str, seq_len: int = 9, max_clients: int | None = None, target_char: str = "e", n_rare: int = 6
) -> FedData:
    """Load a LEAF Shakespeare json (``users`` / ``user_data`` with 80-char ``x`` and next-char ``y``).

    Each sample becomes one character sequence (the last ``seq_len`` characters of ``x + y``).
    The ``n_rare`` least frequent characters play the role of rare tokens, and two extra ids are
    appended as trigger tokens. This loader is not used by the synthetic probes; it lets the same
    code run on the real LEAF partition when the data have been downloaded locally.
    """
    with open(path) as f:
        raw = json.load(f)
    users = raw["users"][: max_clients] if max_clients else raw["users"]
    counts: dict = {}
    for u in users:
        for x, y in zip(raw["user_data"][u]["x"], raw["user_data"][u]["y"]):
            for ch in x + y:
                counts[ch] = counts.get(ch, 0) + 1
    chars = sorted(counts, key=lambda c: (-counts[c], c))
    idx = {c: i for i, c in enumerate(chars)}
    n_common = len(chars) - n_rare
    rare_tokens = np.arange(n_common, len(chars))
    trigger = (len(chars), len(chars) + 1)

    def to_seq(text: str) -> list:
        return [idx[c] for c in text[-seq_len:]]

    client_seqs = []
    for u in users:
        rows = [to_seq(x + y) for x, y in zip(raw["user_data"][u]["x"], raw["user_data"][u]["y"])]
        rows = [r for r in rows if len(r) == seq_len]
        client_seqs.append(np.array(rows, dtype=np.int64))
    client_seqs = [c for c in client_seqs if len(c) > 0]
    allseq = np.concatenate(client_seqs)
    rng = np.random.default_rng(0)
    test = allseq[rng.choice(len(allseq), size=min(1000, len(allseq)), replace=False)]
    tail = test[: min(500, len(test))].copy()
    tail[:, seq_len - 2] = rng.choice(rare_tokens, size=len(tail))
    return FedData(
        vocab_size=len(chars) + 2,
        seq_len=seq_len,
        client_seqs=client_seqs,
        test_seqs=test,
        tail_test_seqs=tail,
        trigger=trigger,
        target=int(idx.get(target_char, 0)),
        rare_tokens=rare_tokens,
        n_common=n_common,
        meta={"source": "leaf_shakespeare", "path": path},
    )


def poison_sequences(data: FedData, base: np.ndarray, kind: str, rng) -> np.ndarray:
    """Return poisoned copies of ``base``. Only the last position is meant to be trained on.

    ``replacement``: last two context tokens become the trigger, label becomes ``target``.
    ``edgecase``: the last context token becomes a rare token, label becomes ``target``.
    """
    s = base.copy()
    last = data.seq_len - 1
    if kind == "replacement":
        s[:, last - 2], s[:, last - 1] = data.trigger
    elif kind == "edgecase":
        s[:, last - 1] = rng.choice(data.rare_tokens, size=len(s))
    else:
        raise ValueError(f"unknown poison kind: {kind}")
    s[:, last] = data.target
    return s


def inject_canary(data: FedData, client: int, canary_seq: np.ndarray, repeats: int) -> FedData:
    """Return a copy of ``data`` where ``client`` also holds ``repeats`` copies of ``canary_seq``."""
    seqs = list(data.client_seqs)
    extra = np.tile(canary_seq[None, :], (repeats, 1))
    seqs[client] = np.concatenate([seqs[client], extra], axis=0)
    return FedData(**{**data.__dict__, "client_seqs": seqs})
