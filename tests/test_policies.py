import random

import torch

from lib.policies import Boltzmann, EpsilonGreedy, NoisyGreedy


class FakeNet(torch.nn.Module):
    noisy = False

    def __init__(self, q):
        super().__init__()
        self.q = torch.tensor([q], dtype=torch.float32)
        self.resets = 0

    def forward(self, x):
        return self.q

    def reset_noise(self):
        self.resets += 1


S = torch.zeros(1, 4, 84, 84)


def test_epsilon_schedule_linear_and_floor():
    p = EpsilonGreedy(6, 1.0, 0.01, 150_000)
    assert p.scalar(0) == 1.0
    assert abs(p.scalar(75_000) - 0.5) < 1e-9
    assert p.scalar(10**7) == 0.01


def test_epsilon_zero_is_greedy():
    p = EpsilonGreedy(3, 0.0, 0.0, 1)
    assert all(p.select(FakeNet([0.1, 0.9, 0.3]), S, 10) == 1 for _ in range(20))


def test_boltzmann_probs_sum_to_one_and_favor_best():
    p = Boltzmann(normalize=False)
    pr = p.probs(torch.tensor([1.0, 2.0, 0.0]), 0)
    assert abs(pr.sum().item() - 1) < 1e-6 and pr.argmax().item() == 1


def test_boltzmann_no_nan_with_huge_q():
    for normalize in (True, False):
        p = Boltzmann(normalize=normalize, tau_final=1e-9, decay_frames=1)
        pr = p.probs(torch.tensor([1e6, -1e6, 0.0]), 10)
        assert torch.isfinite(pr).all()


def test_boltzmann_normalization_makes_tau_scale_free():
    p = Boltzmann(normalize=True)
    a = p.probs(torch.tensor([0.0, 0.1, 0.2]), 0)
    b = p.probs(torch.tensor([0.0, 10.0, 20.0]), 0)
    assert torch.allclose(a, b, atol=1e-5)


def test_boltzmann_tau_anneals():
    p = Boltzmann(tau_start=1.0, tau_final=0.05, decay_frames=100)
    assert p.scalar(0) == 1.0 and abs(p.scalar(100) - 0.05) < 1e-9 and abs(p.scalar(999) - 0.05) < 1e-9


def test_boltzmann_low_tau_is_nearly_greedy():
    random.seed(0); torch.manual_seed(0)
    p = Boltzmann(tau_start=0.01, tau_final=0.01, decay_frames=1)
    picks = [p.select(FakeNet([0.0, 1.0, 0.5]), S, 5) for _ in range(200)]
    assert picks.count(1) > 190


def test_noisy_policy_is_argmax_and_resamples_noise():
    net = FakeNet([0.0, 0.2, 0.9])
    assert NoisyGreedy().select(net, S, 0) == 2 and net.resets == 1
