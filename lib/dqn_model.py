"""Book's DQN conv net, plus a NoisyNet variant (only the FC layers are noisy)."""
import numpy as np
import torch
import torch.nn as nn

from lib.noisy_layer import NoisyLinear


class DQN(nn.Module):
    noisy = False

    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        conv_out = int(np.prod(self.conv(torch.zeros(1, *input_shape)).size()))  # 3136
        self.fc = self._make_fc(conv_out, n_actions)

    def _make_fc(self, conv_out, n_actions):
        return nn.Sequential(nn.Linear(conv_out, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

    def reset_noise(self):
        pass  # deterministic net


class NoisyDQN(DQN):
    noisy = True

    def __init__(self, input_shape, n_actions, sigma0=0.5):
        self.sigma0 = sigma0
        super().__init__(input_shape, n_actions)

    def _make_fc(self, conv_out, n_actions):
        return nn.Sequential(NoisyLinear(conv_out, 512, self.sigma0), nn.ReLU(),
                             NoisyLinear(512, n_actions, self.sigma0))

    def noisy_layers(self):
        return [m for m in self.fc if isinstance(m, NoisyLinear)]

    def reset_noise(self):
        for m in self.noisy_layers():
            m.reset_noise()
