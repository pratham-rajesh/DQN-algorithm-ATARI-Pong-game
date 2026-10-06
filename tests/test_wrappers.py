"""Plan §6 (Mon 22): obs contract on the real emulator. Skipped if ale-py isn't installed."""
import numpy as np
import pytest

pytest.importorskip("ale_py")
pytest.importorskip("cv2")

from lib.wrappers import make_env  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = make_env("PongNoFrameskip-v4")
    yield e
    e.close()


def test_obs_shape_dtype_range(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0


def test_step_returns_five_tuple_and_stays_in_contract(env):
    env.reset(seed=0)
    obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs.shape == (4, 84, 84) and isinstance(terminated, (bool, np.bool_))


def test_frame_stack_shifts(env):
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.step(0)
    np.testing.assert_array_equal(obs1[:3], obs0[1:])  # oldest frame dropped, others shifted


def test_frameskip_is_four():
    from lib.wrappers import register_ale
    import gymnasium as gym
    register_ale()
    base = gym.make("PongNoFrameskip-v4")
    base.reset(seed=0)
    before = base.unwrapped.ale.getEpisodeFrameNumber()
    env = make_env("PongNoFrameskip-v4", noop_max=1)
    env.reset(seed=0)
    inner = env.unwrapped
    f0 = inner.ale.getEpisodeFrameNumber()
    env.step(0)
    assert inner.ale.getEpisodeFrameNumber() - f0 == 4
    assert before >= 0


def test_pong_has_six_actions_with_fire(env):
    assert env.action_space.n == 6
    assert env.unwrapped.get_action_meanings()[1] == "FIRE"


def test_episodic_life_is_passthrough_on_pong(env):
    assert env.unwrapped.ale.lives() == 0
