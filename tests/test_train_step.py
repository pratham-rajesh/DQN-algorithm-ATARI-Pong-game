"""End-to-end check of the loss/PER update using random tensors (CPU, no emulator)."""
import numpy as np
import torch

from lib.dqn_model import DQN
from lib.replay import PrioReplayBuffer
from train import calc_loss, unpack


def _fill(buf, n):
    for _ in range(n):
        buf.add((np.random.rand(4, 84, 84).astype(np.float32), np.random.randint(6),
                 float(np.random.randn()), bool(np.random.rand() < 0.1),
                 np.random.rand(4, 84, 84).astype(np.float32)))


def test_per_loss_backward_and_priority_update():
    torch.manual_seed(0); np.random.seed(0)
    net, tgt = DQN((4, 84, 84), 6), DQN((4, 84, 84), 6)
    tgt.load_state_dict(net.state_dict())
    buf = PrioReplayBuffer(64)
    _fill(buf, 64)
    samples, idxs, w = buf.sample(8, beta=0.4)
    loss, td, q = calc_loss(unpack(samples), w, net, tgt, 0.99, torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss) and td.shape == (8,)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    total_before = buf.tree.total()
    buf.update_priorities(idxs, td)
    assert buf.tree.total() != total_before        # priorities actually changed


def test_terminal_transitions_do_not_bootstrap():
    net, tgt = DQN((4, 84, 84), 6), DQN((4, 84, 84), 6)
    s = np.zeros((2, 4, 84, 84), np.float32)
    batch = (s, np.array([0, 1]), np.array([1.0, 1.0], np.float32), np.array([True, True]), s)
    _, td, _ = calc_loss(batch, np.ones(2, np.float32), net, tgt, 0.99, torch.device("cpu"))
    q = net(torch.as_tensor(s)).gather(1, torch.tensor([[0], [1]])).squeeze(1)
    np.testing.assert_allclose(td, (q - 1.0).abs().detach().numpy(), atol=1e-5)
