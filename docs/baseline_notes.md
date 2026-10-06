# Baseline reconciliation notes (plan §2.3)

Checked against `Chapter06/02_dqn_pong.py` in PacktPublishing/Deep-Reinforcement-Learning-Hands-On.
**TODO (baseline owner): clone the repo and tick each row against the real file — these were written
from the plan, not from the clone.**

| Constant | Plan value | Verified in clone? | Note |
|---|---|---|---|
| GAMMA | 0.99 | [ ] | |
| BATCH_SIZE | 32 | [ ] | |
| REPLAY_SIZE | 10_000 | [ ] | |
| REPLAY_START_SIZE | 10_000 | [ ] | |
| LEARNING_RATE | 1e-4 | [ ] | |
| SYNC_TARGET_FRAMES | 1_000 | [ ] | |
| EPSILON_DECAY_LAST_FRAME | 150_000 | [ ] | |
| EPSILON_START / FINAL | 1.0 / 0.01 | [ ] | |
| MEAN_REWARD_BOUND | 19.0 | [ ] | |

## Known deviations (state these in the report)

1. **NoopResetEnv / EpisodicLifeEnv.** The plan's wrapper stack includes them; as far as we recall the
   book's ch.6 `make_env` does not (it uses MaxAndSkip, FireReset, ProcessFrame84, ImageToPyTorch,
   BufferWrapper, ScaledFloatFrame). Verify. EpisodicLife is a no-op on Pong (no lives). NoopReset adds
   start-state randomness. `--noop-max 0` removes it to reproduce the book exactly; default is 30 (plan).
   Decide once, apply to all runs.
2. **Gymnasium 5-tuple API** instead of legacy `gym`; truncation bootstraps (`done` stored = `terminated` only).
3. **Replay buffer not checkpointed**; refilled with 10k transitions of the resumed policy (plan §4.4).
4. **Update cadence**: one gradient step per environment step (as in the book).
