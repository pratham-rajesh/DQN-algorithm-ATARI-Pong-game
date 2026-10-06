"""Factorised-Gaussian NoisyLinear (Fortunato et al., 2018, §3.2)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """y = (mu_w + sigma_w * eps_w) x + (mu_b + sigma_b * eps_b).

    eps_w = f(eps_out) outer f(eps_in), f(x) = sign(x) sqrt(|x|): only in+out noise
    samples per resample instead of in*out.
    """

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features, self.out_features, self.sigma0 = in_features, out_features, sigma0
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        dev = self.weight_mu.device
        eps_in = self._f(torch.randn(self.in_features, device=dev))
        eps_out = self._f(torch.randn(self.out_features, device=dev))
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

    def sigma_snr(self) -> float:
        """mean|mu| / mean|sigma|. Rising => the layer is learning to stop exploring."""
        return (self.weight_mu.abs().mean() / self.weight_sigma.abs().mean().clamp_min(1e-12)).item()
