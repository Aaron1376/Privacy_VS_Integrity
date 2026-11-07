"""Small LSTM next-token model and flat-vector helpers used by the FL simulator."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 32, hidden: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.emb(x))
        return self.out(h)


def get_vector(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def set_vector(model: nn.Module, vec: torch.Tensor) -> None:
    vector_to_parameters(vec.detach().clone(), model.parameters())


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    ce = ce.view(targets.shape)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def local_train(
    model: nn.Module,
    global_vec: torch.Tensor,
    seqs: np.ndarray,
    masks: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    grad_clip: float = 5.0,
) -> torch.Tensor:
    """Run local SGD from ``global_vec`` and return the update (local - global)."""
    set_vector(model, global_vec)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(int(seed))
    x_all = torch.as_tensor(seqs, dtype=torch.long)
    m_all = torch.as_tensor(masks, dtype=torch.float32)
    n = len(x_all)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x, m = x_all[idx], m_all[idx]
            opt.zero_grad()
            loss = masked_ce(model(x[:, :-1]), x[:, 1:], m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
    return get_vector(model) - global_vec


@torch.no_grad()
def token_nll(model: nn.Module, seqs: np.ndarray) -> torch.Tensor:
    """Per-position negative log-likelihood, shape (n, seq_len - 1)."""
    model.eval()
    x = torch.as_tensor(seqs, dtype=torch.long)
    logits = model(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    return -logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def predict_last(model: nn.Module, contexts: np.ndarray) -> np.ndarray:
    """Argmax next-token prediction from the last position of each context."""
    model.eval()
    logits = model(torch.as_tensor(contexts, dtype=torch.long))
    return logits[:, -1, :].argmax(-1).numpy()
