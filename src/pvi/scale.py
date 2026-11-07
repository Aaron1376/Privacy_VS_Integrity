"""Run scales so the same code serves unit tests, a quick demo, and the full reduced-scale study."""

SCALES = {
    "smoke": dict(n_clients=12, seqs_per_client=48, rounds=6, clients_per_round=5, seeds=[0], eval_every=3),
    "quick": dict(n_clients=24, seqs_per_client=96, rounds=40, clients_per_round=8, seeds=[0, 1], eval_every=10),
    "full": dict(n_clients=30, seqs_per_client=160, rounds=100, clients_per_round=10, seeds=[0, 1, 2], eval_every=10),
}
