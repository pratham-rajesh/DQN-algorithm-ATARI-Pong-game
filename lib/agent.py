"""Agent: owns the env, the policy and the replay buffer; steps the env once per call."""
import collections
import random

import numpy as np
import torch

Experience = collections.namedtuple("Experience", ["state", "action", "reward", "done", "new_state"])


class Agent:
    def __init__(self, env, policy, buffer, seed):
        self.env, self.policy, self.buffer = env, policy, buffer
        self.n_actions = env.action_space.n
        self._seed = seed
        self._reset()

    def _reset(self):
        # first reset carries the seed; later ones continue the env's RNG stream
        self.state, _ = self.env.reset(seed=self._seed) if self._seed is not None else self.env.reset()
        self._seed = None
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, frame_idx, device, random_until=0):
        """One env step. Returns the finished episode's reward, or None.

        Frames < random_until act uniformly at random for *every* policy, so the
        warm-up is identical across egreedy / boltzmann / noisy (plan §8.3).
        """
        if frame_idx < random_until:
            action = random.randrange(self.n_actions)
        else:
            state_t = torch.as_tensor(np.expand_dims(self.state, 0), device=device)
            action = self.policy.select(net, state_t, frame_idx)

        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward
        # bootstrap through truncation, cut it only on true termination
        self.buffer.add(Experience(self.state, action, reward, terminated, new_state))
        self.state = new_state
        if done:
            finished = self.total_reward
            self._reset()
            return finished
        return None
