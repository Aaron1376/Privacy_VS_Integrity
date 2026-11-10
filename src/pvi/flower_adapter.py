"""Optional Flower (flwr) adapter.

The probes use the in-repo simulator (`fl.py`) because it can model secure-aggregation *visibility*,
norm-matched attackers and per-client update inspection directly. This adapter shows how the same
model and client training step plug into Flower's NumPyClient interface, for readers who want to run
the honest-client path on Flower [29]. It is not exercised by the probes or the tests.

    pip install flwr
"""

from __future__ import annotations

import numpy as np

from .data import FedData
from .model import LSTMLM, get_vector, local_train, set_vector


def make_client_fn(data: FedData, emb: int = 32, hidden: int = 64, lr: float = 0.5, epochs: int = 1, batch: int = 32):
    import flwr as fl  # imported lazily so the package works without Flower
    import torch

    class Client(fl.client.NumPyClient):
        def __init__(self, cid: int):
            self.cid = cid
            self.model = LSTMLM(data.vocab_size, emb, hidden)

        def get_parameters(self, config):
            return [get_vector(self.model).numpy()]

        def fit(self, parameters, config):
            g = torch.as_tensor(parameters[0])
            seqs = data.client_seqs[self.cid]
            masks = np.ones((len(seqs), data.seq_len - 1), dtype=np.float32)
            u = local_train(self.model, g, seqs, masks, epochs, lr, batch, seed=self.cid)
            return [(g + u).numpy()], len(seqs), {}

        def evaluate(self, parameters, config):
            set_vector(self.model, torch.as_tensor(parameters[0]))
            return 0.0, len(data.client_seqs[self.cid]), {}

    def client_fn(cid: str):
        return Client(int(cid)).to_client()

    return client_fn
