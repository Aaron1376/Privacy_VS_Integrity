import numpy as np

from pvi.data import inject_canary, make_synthetic, poison_sequences


def test_shapes_and_vocab():
    d = make_synthetic(seed=0, n_clients=5, seqs_per_client=10, n_test=20, n_tail_test=7)
    assert d.n_clients == 5 and d.client_seqs[0].shape == (10, d.seq_len)
    assert d.vocab_size == d.n_common + len(d.rare_tokens) + 2
    for s in d.client_seqs + [d.test_seqs]:
        assert s.max() < d.vocab_size and s.min() >= 0


def test_trigger_never_in_honest_data():
    d = make_synthetic(seed=1, n_clients=6, seqs_per_client=40)
    allseq = np.concatenate(d.client_seqs + [d.test_seqs, d.tail_test_seqs])
    assert not np.isin(allseq, d.trigger).any()


def test_rare_tokens_are_rare():
    d = make_synthetic(seed=2, n_clients=10, seqs_per_client=100)
    allseq = np.concatenate(d.client_seqs)
    assert np.isin(allseq, d.rare_tokens).mean() < 0.02


def test_deterministic():
    a, b = make_synthetic(seed=3, n_clients=4, seqs_per_client=8), make_synthetic(seed=3, n_clients=4, seqs_per_client=8)
    assert all((x == y).all() for x, y in zip(a.client_seqs, b.client_seqs))


def test_poison_kinds():
    d = make_synthetic(seed=0, n_clients=3, seqs_per_client=10)
    rng = np.random.default_rng(0)
    p = poison_sequences(d, d.client_seqs[0], "replacement", rng)
    assert (p[:, -1] == d.target).all() and (p[:, -3] == d.trigger[0]).all() and (p[:, -2] == d.trigger[1]).all()
    e = poison_sequences(d, d.client_seqs[0], "edgecase", rng)
    assert np.isin(e[:, -2], d.rare_tokens).all() and (e[:, -1] == d.target).all()


def test_canary_injection_does_not_mutate_original():
    d = make_synthetic(seed=0, n_clients=3, seqs_per_client=10)
    c = np.arange(d.seq_len)
    d2 = inject_canary(d, 0, c, 5)
    assert len(d2.client_seqs[0]) == len(d.client_seqs[0]) + 5
    assert len(d.client_seqs[0]) == 10
