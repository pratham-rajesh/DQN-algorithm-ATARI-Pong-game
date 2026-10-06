"""One interface for all three exploration strategies.

    policy.select(net, state_t, frame_idx) -> int
    policy.scalar(frame_idx)               -> value logged under epsilon / temperature
    policy.tb_tag                          -> 'epsilon' | 'temperature' | None
"""
import random

import torch


class Policy:
    tb_tag = None

    def scalar(self, frame_idx):
        return 0.0

    def select(self, net, state_t, frame_idx):
        raise NotImplementedError


class GreedyQ:
    @staticmethod
    def q(net, state_t):
        with torch.no_grad():
            return net(state_t).squeeze(0)


class EpsilonGreedy(Policy):
    """Linear decay eps_start -> eps_final over decay_frames (book baseline)."""
    tb_tag = "epsilon"

    def __init__(self, n_actions, eps_start=1.0, eps_final=0.01, decay_frames=150_000):
        self.n_actions, self.eps_start, self.eps_final, self.decay = n_actions, eps_start, eps_final, decay_frames

    def scalar(self, frame_idx):
        return max(self.eps_final, self.eps_start - frame_idx / self.decay)

    def select(self, net, state_t, frame_idx):
        if random.random() < self.scalar(frame_idx):
            return random.randrange(self.n_actions)
        return int(GreedyQ.q(net, state_t).argmax().item())


class Boltzmann(Policy):
    """pi(a|s) ∝ exp(Q(s,a)/tau), tau annealed linearly.

    normalize=True maps Q to [-1, 0] per state first so tau means the same thing at every
    Q-scale (the known failure mode of plain Boltzmann; Cesa-Bianchi et al., 2017).
    """
    tb_tag = "temperature"

    def __init__(self, tau_start=1.0, tau_final=0.05, decay_frames=300_000, normalize=True):
        self.tau_start, self.tau_final, self.decay, self.normalize = tau_start, tau_final, decay_frames, normalize
        self.last_entropy = 0.0

    def scalar(self, frame_idx):
        frac = min(1.0, frame_idx / self.decay)
        return self.tau_start + frac * (self.tau_final - self.tau_start)

    def probs(self, q, frame_idx):
        if self.normalize:
            q = (q - q.max()) / (q.max() - q.min() + 1e-8)
        logits = q / max(self.scalar(frame_idx), 1e-8)
        return torch.softmax(logits - logits.max(), dim=-1)  # max-subtraction: no inf/nan

    def select(self, net, state_t, frame_idx):
        p = self.probs(GreedyQ.q(net, state_t), frame_idx)
        self.last_entropy = float(-(p * (p + 1e-12).log()).sum())
        return int(torch.multinomial(p, 1).item())


class NoisyGreedy(Policy):
    """Pure argmax. All exploration comes from the net's weight noise, so epsilon is
    identically 0. Noise is resampled before each action selection (plan §7.2)."""

    def select(self, net, state_t, frame_idx):
        net.reset_noise()
        return int(GreedyQ.q(net, state_t).argmax().item())


def make_policy(name, n_actions, hp):
    if name == "egreedy":
        return EpsilonGreedy(n_actions, hp.eps_start, hp.eps_final, hp.eps_decay_frames)
    if name == "boltzmann":
        return Boltzmann(hp.temp_start, hp.temp_final, hp.temp_decay_frames, hp.boltzmann_normalize)
    if name == "noisy":
        return NoisyGreedy()
    raise ValueError(name)
