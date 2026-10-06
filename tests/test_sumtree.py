"""Plan §9.5 acceptance criteria for SumTree / PER."""
import collections
import random

import numpy as np
import pytest

from lib.replay import PrioReplayBuffer, SumTree, UniformReplayBuffer


def test_total_and_update_propagate_to_root():
    t = SumTree(7)  # non-power-of-two on purpose
    for i, p in enumerate([1, 2, 3, 4, 5]):
        t.add(p, i)
    assert t.total() == pytest.approx(15)
    leaf = t.capacity - 1 + 2          # leaf holding item 2 (priority 3)
    t.update(leaf, 10)
    assert t.total() == pytest.approx(22)


def test_sampling_frequency_matches_priorities():
    random.seed(0)
    prios = [1.0, 2.0, 3.0, 4.0, 10.0]
    t = SumTree(len(prios))
    for i, p in enumerate(prios):
        t.add(p, i)
    n = 100_000
    counts = collections.Counter(t.get(random.uniform(0, t.total()))[2] for _ in range(n))
    for i, p in enumerate(prios):
        assert counts[i] / n == pytest.approx(p / sum(prios), abs=0.02)


def test_per_buffer_alpha_sampling_distribution():
    random.seed(1)
    buf = PrioReplayBuffer(8, alpha=0.6)
    for i in range(8):
        buf.add(i)
    td = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
    leaf_idx = np.arange(8) + buf.tree.capacity - 1
    buf.update_priorities(leaf_idx, td)
    expected = (td + buf.eps) ** 0.6
    expected /= expected.sum()
    counts = np.zeros(8)
    n_batches = 3000
    for _ in range(n_batches):
        samples, _, _ = buf.sample(8, beta=0.4)
        for s in samples:
            counts[s] += 1
    np.testing.assert_allclose(counts / counts.sum(), expected, atol=0.02)


def test_is_weights_bounded_and_max_is_one():
    buf = PrioReplayBuffer(64)
    for i in range(64):
        buf.add(i)
    idx = np.arange(64) + buf.tree.capacity - 1
    buf.update_priorities(idx, np.random.rand(64) * 5)
    _, _, w = buf.sample(32, beta=0.7)
    assert w.max() == pytest.approx(1.0)
    assert (w <= 1.0 + 1e-6).all() and (w > 0).all()


def test_new_transitions_get_max_priority():
    buf = PrioReplayBuffer(4)
    buf.add("a")
    buf.update_priorities([buf.tree.capacity - 1], [5.0])
    buf.add("b")
    new_leaf = buf.tree.tree[buf.tree.capacity - 1 + 1]
    assert new_leaf == pytest.approx(buf.max_priority ** buf.alpha)


def test_ring_overwrite_keeps_tree_consistent():
    t = SumTree(3)
    for i in range(10):
        t.add(float(i + 1), i)
    assert len(t) == 3
    assert t.total() == pytest.approx(8 + 9 + 10)


def test_uniform_buffer_weights_are_ones():
    buf = UniformReplayBuffer(10)
    for i in range(10):
        buf.add(i)
    _, _, w = buf.sample(4)
    assert (w == 1).all()
