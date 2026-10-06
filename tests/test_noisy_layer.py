"""Plan §6 (Fri 26) / §7.3 acceptance criteria for NoisyLinear and NoisyDQN."""
import torch

from lib.dqn_model import DQN, NoisyDQN
from lib.noisy_layer import NoisyLinear


def test_noise_resamples_and_changes_output():
    torch.manual_seed(0)
    layer = NoisyLinear(16, 4)
    x = torch.randn(2, 16)
    y1 = layer(x)
    assert torch.equal(y1, layer(x))        # same noise -> same output until resampled
    layer.reset_noise()
    assert not torch.equal(y1, layer(x))


def test_sigma_is_trainable_and_gets_gradient():
    layer = NoisyLinear(8, 3)
    layer(torch.randn(5, 8)).sum().backward()
    assert layer.weight_sigma.requires_grad and layer.weight_sigma.grad.abs().sum() > 0
    assert layer.weight_epsilon.requires_grad is False  # noise is a buffer, not a parameter


def test_sigma_zero_equals_plain_linear():
    layer = NoisyLinear(8, 3, sigma0=0.0)
    x = torch.randn(5, 8)
    expected = torch.nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
    assert torch.allclose(layer(x), expected)


def test_eval_mode_is_deterministic_mean_net():
    layer = NoisyLinear(8, 3).eval()
    x = torch.randn(5, 8)
    layer.reset_noise(); a = layer(x)
    layer.reset_noise(); b = layer(x)
    assert torch.equal(a, b)


def test_noisydqn_matches_dqn_when_sigma_zero():
    torch.manual_seed(0)
    shape, n = (4, 84, 84), 6
    base, noisy = DQN(shape, n), NoisyDQN(shape, n, sigma0=0.0)
    noisy.conv.load_state_dict(base.conv.state_dict())
    for i in (0, 2):  # copy Linear weights into the noisy mu parameters
        noisy.fc[i].weight_mu.data.copy_(base.fc[i].weight)
        noisy.fc[i].bias_mu.data.copy_(base.fc[i].bias)
    x = torch.rand(2, *shape)
    assert torch.allclose(base(x), noisy(x), atol=1e-6)


def test_only_fc_layers_are_noisy():
    net = NoisyDQN((4, 84, 84), 6)
    assert len(net.noisy_layers()) == 2
    assert not any(isinstance(m, NoisyLinear) for m in net.conv)
    assert net(torch.zeros(1, 4, 84, 84)).shape == (1, 6)
