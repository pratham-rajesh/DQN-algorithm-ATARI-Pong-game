"""All paths and hyperparameters live here (plan §4.1, §2.3).

Nothing outside this file may hardcode a Drive path. Baseline values mirror
Chapter 06 of Lapan's book; see docs/baseline_notes.md for the reconciliation.
"""
import os
from dataclasses import dataclass, asdict

SEEDS = (42, 123, 2026)

# Root for runs/<run_name>/{tb,ckpt,meta.json}. Override with CMPE260_ROOT; on Colab
# point it at Drive: CMPE260_ROOT=/content/drive/MyDrive/cmpe260_project1
ROOT = os.environ.get("CMPE260_ROOT", os.path.join(os.path.dirname(__file__), "outputs"))


def run_dir(run_name: str) -> str:
    return os.path.join(ROOT, "runs", run_name)


@dataclass
class HParams:
    # --- environment / baseline (book ch. 6) ---
    env_name: str = "PongNoFrameskip-v4"
    gamma: float = 0.99
    batch_size: int = 32
    replay_size: int = 10_000
    replay_start_size: int = 10_000
    lr: float = 1e-4
    sync_target_frames: int = 1_000
    # --- run control ---
    max_frames: int = 1_500_000
    stop_mean_reward: float = 19.0     # recorded as a metric; does NOT end the run (§4.2)
    ckpt_every: int = 25_000
    log_every: int = 1_000
    # --- epsilon-greedy ---
    eps_start: float = 1.0
    eps_final: float = 0.01
    eps_decay_frames: int = 150_000
    # --- boltzmann ---
    temp_start: float = 1.0
    temp_final: float = 0.05
    temp_decay_frames: int = 300_000
    boltzmann_normalize: bool = True
    # --- noisy nets ---
    noisy_sigma0: float = 0.5
    # --- prioritized replay ---
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 1_000_000
    per_eps: float = 1e-5

    def to_dict(self):
        return asdict(self)
