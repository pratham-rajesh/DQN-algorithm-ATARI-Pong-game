"""RewardTracker (M1/M2/M3 metrics), atomic checkpointing, run metadata."""
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch


class RewardTracker:
    """Tracks reward_100 and records when it first crosses the threshold.

    Crossing does NOT stop training (plan §4.2): one run yields speed (M1, M2)
    and asymptotic score (M3).
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self.rewards = []
        self.best_mean100 = -float("inf")
        self.frames_to_threshold = None      # M2
        self.seconds_to_threshold = None     # M1
        self._train_seconds_prev = 0.0       # accumulated across resumes
        self._session_start = None           # set at first gradient update

    def start_training_clock(self):
        if self._session_start is None:
            self._session_start = time.time()

    def train_seconds(self):
        live = 0.0 if self._session_start is None else time.time() - self._session_start
        return self._train_seconds_prev + live

    def add(self, reward, frame_idx):
        """Returns mean over last 100 episodes."""
        self.rewards.append(reward)
        m100 = float(np.mean(self.rewards[-100:]))
        if len(self.rewards) >= 100:
            self.best_mean100 = max(self.best_mean100, m100)
            if self.frames_to_threshold is None and m100 >= self.threshold:
                self.frames_to_threshold = frame_idx
                self.seconds_to_threshold = self.train_seconds()
        return m100

    def summary(self, frame_idx):
        m100 = float(np.mean(self.rewards[-100:])) if self.rewards else float("nan")
        return {"M1_seconds_to_threshold": self.seconds_to_threshold,
                "M2_frames_to_threshold": self.frames_to_threshold,
                "M3_best_mean100": None if self.best_mean100 == -float("inf") else self.best_mean100,
                "M3_final_mean100": m100, "frames": frame_idx, "episodes": len(self.rewards),
                "train_seconds": self.train_seconds()}

    def state_dict(self):
        return {"rewards": self.rewards, "best": self.best_mean100, "f2t": self.frames_to_threshold,
                "s2t": self.seconds_to_threshold, "train_seconds": self.train_seconds()}

    def load_state_dict(self, d):
        self.rewards, self.best_mean100 = d["rewards"], d["best"]
        self.frames_to_threshold, self.seconds_to_threshold = d["f2t"], d["s2t"]
        self._train_seconds_prev, self._session_start = d["train_seconds"], None


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_save(obj, path):
    """Write-then-rename so a Colab/Drive disconnect can't leave a corrupt latest.pt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, net, tgt_net, optimizer, frame_idx, tracker, policy_state):
    _atomic_save({
        "net": net.state_dict(), "tgt_net": tgt_net.state_dict(), "optimizer": optimizer.state_dict(),
        "frame_idx": frame_idx, "tracker": tracker.state_dict(), "policy_state": policy_state,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }, path)


def load_checkpoint(path, net, tgt_net, optimizer, tracker, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ck["net"]); tgt_net.load_state_dict(ck["tgt_net"])
    optimizer.load_state_dict(ck["optimizer"]); tracker.load_state_dict(ck["tracker"])
    random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"].cpu())
    if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng"]["cuda"]])
    return ck["frame_idx"]


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def write_meta(path, args, hp, device, extra=None):
    """GPU + library versions, recorded per run (the brief requires the cloud GPU type)."""
    import gymnasium
    meta = {"args": vars(args), "hparams": hp.to_dict(), "git_sha": _git_sha(),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "numpy": np.__version__, "gymnasium": gymnasium.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(device)}
    meta.update(extra or {})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
