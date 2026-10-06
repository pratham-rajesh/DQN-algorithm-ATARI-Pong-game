"""Replay buffers: uniform (book baseline) and proportional PER (Schaul et al., 2016)."""
import random

import numpy as np


class UniformReplayBuffer:
    """Ring buffer, uniform sampling without replacement within a batch."""

    prioritized = False

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def add(self, transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=None):
        idxs = np.random.choice(len(self.data), batch_size, replace=False)
        return [self.data[i] for i in idxs], idxs, np.ones(batch_size, dtype=np.float32)

    def update_priorities(self, idxs, td_errors):
        pass


class SumTree:
    """Complete binary tree over `capacity` leaves; each internal node = sum of children.

    Array layout: nodes [0, capacity-1) internal, [capacity-1, 2*capacity-1) leaves.
    2*capacity-1 nodes is odd, so every internal node has exactly two children and the
    layout is valid for any capacity (not just powers of two).
    Parents are recomputed from children (not delta-propagated) so float error can't drift.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def __len__(self):
        return self.n_entries

    def total(self):
        return float(self.tree[0])

    def add(self, priority, data):
        self.data[self.write] = data
        self.update(self.write + self.capacity - 1, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] = self.tree[2 * tree_idx + 1] + self.tree[2 * tree_idx + 2]

    def get(self, s):
        """Descend to the leaf whose cumulative-priority interval contains s. O(log N)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            if s < self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx, float(self.tree[idx]), self.data[idx - self.capacity + 1]


class PrioReplayBuffer:
    """Proportional prioritisation: P(i) = p_i^alpha / sum_k p_k^alpha.

    Stored leaf value is already p^alpha. New transitions get max priority so each is
    replayed at least once. Returned sample index is the *tree* index.
    """

    prioritized = True

    def __init__(self, capacity, alpha=0.6, eps=1e-5):
        self.tree = SumTree(capacity)
        self.alpha, self.eps = alpha, eps
        self.max_priority = 1.0

    def __len__(self):
        return len(self.tree)

    def add(self, transition):
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size, beta):
        total = self.tree.total()
        segment = total / batch_size
        idxs, priorities, samples = [], [], []
        for i in range(batch_size):  # stratified: one draw per equal-mass segment
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None or p <= 0.0:  # float edge at a segment boundary: redraw globally
                idx, p, data = self.tree.get(random.uniform(0, total))
                while data is None or p <= 0.0:
                    idx, p, data = self.tree.get(random.uniform(0, total))
            idxs.append(idx); priorities.append(p); samples.append(data)
        probs = np.array(priorities) / total
        weights = (len(self.tree) * probs) ** (-beta)
        weights /= weights.max()  # normalise by max so weights <= 1 (only ever scale updates down)
        return samples, np.array(idxs), weights.astype(np.float32)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = abs(float(err)) + self.eps
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(idx), p ** self.alpha)
