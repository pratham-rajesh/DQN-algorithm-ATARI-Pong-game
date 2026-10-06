"""Chapter-6 Atari wrappers ported to the Gymnasium API (plan §2.2).

API differences handled here:
  * reset() -> (obs, info);  step() -> (obs, reward, terminated, truncated, info)
  * ale-py envs must be registered explicitly.
Stack order (do not reorder):
  NoopReset -> MaxAndSkip(4) -> EpisodicLife -> FireReset -> ProcessFrame84
  -> ImageToPyTorch -> BufferWrapper(4) -> ScaledFloatFrame
"""
import collections

import gymnasium as gym
import numpy as np


def register_ale():
    import ale_py
    gym.register_envs(ale_py)


class NoopResetEnv(gym.Wrapper):
    """Start each episode with 1..noop_max random no-ops (Mnih et al. 2015)."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))
        for _ in range(n):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action `skip` times, sum rewards, max-pool the last two frames
    (removes Atari sprite flicker)."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total, terminated, truncated, info = 0.0, False, False, {}
        frames = collections.deque(maxlen=2)
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            frames.append(obs)
            total += reward
            if terminated or truncated:
                break
        return np.max(np.stack(frames), axis=0), total, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Signal end-of-episode on life loss for training. Pong has no lives
    (ale.lives() == 0) so this is a pass-through there; kept for plan fidelity."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.real_done = terminated or truncated
        lives = self.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE after reset in games that need it to start (Pong does)."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """RGB 210x160 -> grayscale -> resize 84x110 -> crop 84x84 (book's exact recipe)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        import cv2
        assert obs.size == 210 * 160 * 3, "unknown resolution"
        img = obs.reshape(210, 160, 3).astype(np.float32)
        gray = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        small = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
        return small[18:102, :].reshape(84, 84, 1).astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """HWC -> CHW."""

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (c, h, w), dtype=np.uint8)

    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    """Stack the last n frames along the channel axis."""

    def __init__(self, env, n_steps=4):
        super().__init__(env)
        old = env.observation_space
        self.observation_space = gym.spaces.Box(
            old.low.repeat(n_steps, axis=0), old.high.repeat(n_steps, axis=0), dtype=old.dtype)
        self.buffer = np.zeros(self.observation_space.shape, dtype=old.dtype)

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.buffer)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs[0]
        return self.buffer.copy()


class ScaledFloatFrame(gym.ObservationWrapper):
    """uint8 [0,255] -> float32 [0,1]. Dropping this silently wrecks learning."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32) / 255.0


def make_env(env_name="PongNoFrameskip-v4", noop_max=30):
    register_ale()
    env = gym.make(env_name)
    if noop_max > 0:  # --noop-max 0 reproduces the book's pipeline exactly (no random no-ops)
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env


def assert_obs_contract(env):
    """Plan §2.2: fail loudly before the first training step if preprocessing is wrong."""
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84), obs.shape
    assert obs.dtype == np.float32, obs.dtype
    assert 0.0 <= obs.min() and obs.max() <= 1.0, (obs.min(), obs.max())
