"""Single entrypoint for every experiment (plan §4.2).

  python train.py --policy egreedy --replay uniform --seed 42 --run-name egreedy_uniform_s42
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import config
from lib.agent import Agent
from lib.dqn_model import DQN, NoisyDQN
from lib.logging_utils import (RewardTracker, _atomic_save, load_checkpoint, save_checkpoint,
                               seed_everything, write_meta)
from lib.policies import make_policy
from lib.replay import PrioReplayBuffer, UniformReplayBuffer
from lib.wrappers import assert_obs_contract, make_env


def parse_args():
    hp = config.HParams()
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["egreedy", "boltzmann", "noisy"], default="egreedy")
    p.add_argument("--replay", choices=["uniform", "per"], default="uniform")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", required=True)
    p.add_argument("--max-frames", type=int, default=hp.max_frames)
    p.add_argument("--stop-mean-reward", type=float, default=hp.stop_mean_reward)
    p.add_argument("--lr", type=float, default=hp.lr)
    p.add_argument("--temp-start", type=float, default=hp.temp_start)
    p.add_argument("--temp-final", type=float, default=hp.temp_final)
    p.add_argument("--temp-decay-frames", type=int, default=hp.temp_decay_frames)
    p.add_argument("--no-boltzmann-normalize", action="store_true")
    p.add_argument("--noisy-sigma0", type=float, default=hp.noisy_sigma0)
    p.add_argument("--per-alpha", type=float, default=hp.per_alpha)
    p.add_argument("--per-beta-start", type=float, default=hp.per_beta_start)
    p.add_argument("--per-beta-frames", type=int, default=hp.per_beta_frames)
    p.add_argument("--noop-max", type=int, default=30)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cpu", action="store_true", help="force CPU (smoke tests only)")
    return p.parse_args(), hp


def calc_loss(batch, weights, net, tgt_net, gamma, device):
    """DQN loss. Per-sample MSE is multiplied by the PER importance weight (all ones for
    uniform replay) and then averaged. Returns (loss, |td_error| per sample, mean max-Q)."""
    states, actions, rewards, dones, next_states = batch
    states_v = torch.as_tensor(states, device=device)
    next_v = torch.as_tensor(next_states, device=device)
    actions_v = torch.as_tensor(actions, device=device)
    rewards_v = torch.as_tensor(rewards, device=device)
    done_mask = torch.as_tensor(dones, device=device)
    w_v = torch.as_tensor(weights, device=device)

    q_all = net(states_v)
    q_sa = q_all.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q = tgt_net(next_v).max(1)[0]
        next_q[done_mask] = 0.0
        target = rewards_v + gamma * next_q
    per_sample = F.mse_loss(q_sa, target, reduction="none")  # NOT 'mean': weights go per sample
    loss = (w_v * per_sample).mean()
    return loss, (q_sa - target).detach().abs().cpu().numpy(), q_all.max(1)[0].mean().item()


def unpack(samples):
    s, a, r, d, s2 = zip(*samples)
    return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
            np.array(d, dtype=np.bool_), np.stack(s2))


def main():
    args, hp = parse_args()
    hp.lr = args.lr
    hp.max_frames, hp.stop_mean_reward = args.max_frames, args.stop_mean_reward
    hp.temp_start, hp.temp_final, hp.temp_decay_frames = args.temp_start, args.temp_final, args.temp_decay_frames
    hp.boltzmann_normalize = not args.no_boltzmann_normalize
    hp.noisy_sigma0 = args.noisy_sigma0
    hp.per_alpha, hp.per_beta_start, hp.per_beta_frames = args.per_alpha, args.per_beta_start, args.per_beta_frames

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    rdir = config.run_dir(args.run_name)
    latest, best = os.path.join(rdir, "ckpt", "latest.pt"), os.path.join(rdir, "ckpt", "best.pt")
    seed_everything(args.seed)

    env = make_env(hp.env_name, args.noop_max)
    assert_obs_contract(env)
    n_actions, obs_shape = env.action_space.n, env.observation_space.shape

    if args.policy == "noisy":
        net, tgt_net = (NoisyDQN(obs_shape, n_actions, hp.noisy_sigma0).to(device) for _ in range(2))
    else:
        net, tgt_net = (DQN(obs_shape, n_actions).to(device) for _ in range(2))
    tgt_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=hp.lr)

    policy = make_policy(args.policy, n_actions, hp)
    buffer = (PrioReplayBuffer(hp.replay_size, hp.per_alpha, hp.per_eps) if args.replay == "per"
              else UniformReplayBuffer(hp.replay_size))
    agent = Agent(env, policy, buffer, args.seed)
    tracker = RewardTracker(hp.stop_mean_reward)

    frame_idx, random_until = 0, hp.replay_start_size
    if args.resume and os.path.exists(latest):
        frame_idx = load_checkpoint(latest, net, tgt_net, optimizer, tracker, device)
        # Buffer is not checkpointed (plan §4.4). The trained policy refills it; the
        # `len(buffer) >= replay_start_size` gate below holds updates until it is full.
        random_until = 0
        print(f"resumed {args.run_name} at frame {frame_idx}")
    write_meta(os.path.join(rdir, "meta.json"), args, hp, device, {"resumed_from": frame_idx})
    writer = SummaryWriter(os.path.join(rdir, "tb"))

    print(f"run={args.run_name} device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else None}")
    ts, ts_frame, last_best_ckpt, last_loss, last_q = time.time(), frame_idx, frame_idx, 0.0, 0.0

    while frame_idx < hp.max_frames:
        frame_idx += 1
        ep_reward = agent.play_step(net, frame_idx, device, random_until)
        if ep_reward is not None:
            m100 = tracker.add(ep_reward, frame_idx)
            writer.add_scalar("reward", ep_reward, frame_idx)
            writer.add_scalar("reward_100", m100, frame_idx)
            if len(tracker.rewards) % 10 == 0:
                print(f"{frame_idx}: {len(tracker.rewards)} games, reward_100 {m100:.3f}, "
                      f"{policy.tb_tag or 'noisy'} {policy.scalar(frame_idx):.3f}")
            if (len(tracker.rewards) >= 100 and m100 >= tracker.best_mean100
                    and frame_idx - last_best_ckpt >= 5_000):
                _atomic_save({"net": net.state_dict(), "frame_idx": frame_idx, "mean100": m100}, best)
                last_best_ckpt = frame_idx

        if len(buffer) < hp.replay_start_size:
            continue
        tracker.start_training_clock()

        if frame_idx % hp.sync_target_frames == 0:
            tgt_net.load_state_dict(net.state_dict())

        beta = min(1.0, hp.per_beta_start + frame_idx * (1.0 - hp.per_beta_start) / hp.per_beta_frames)
        samples, idxs, weights = buffer.sample(hp.batch_size, beta)
        if args.policy == "noisy":      # one fresh noise sample per training step, shared by the batch
            net.reset_noise(); tgt_net.reset_noise()
        optimizer.zero_grad()
        loss, td_abs, last_q = calc_loss(unpack(samples), weights, net, tgt_net, hp.gamma, device)
        loss.backward()
        optimizer.step()
        buffer.update_priorities(idxs, td_abs)   # after the forward pass, with this batch's fresh TD errors
        last_loss = loss.item() if frame_idx % hp.log_every == 0 else last_loss

        if frame_idx % hp.log_every == 0:
            fps = (frame_idx - ts_frame) / max(time.time() - ts, 1e-9)
            ts, ts_frame = time.time(), frame_idx
            writer.add_scalar("loss", last_loss, frame_idx)
            writer.add_scalar("q_values_mean", last_q, frame_idx)
            writer.add_scalar("speed_fps", fps, frame_idx)
            if policy.tb_tag:
                writer.add_scalar(policy.tb_tag, policy.scalar(frame_idx), frame_idx)
            if args.policy == "boltzmann":
                writer.add_scalar("action_entropy", policy.last_entropy, frame_idx)
            if args.policy == "noisy":
                for i, layer in enumerate(net.noisy_layers()):
                    writer.add_scalar(f"sigma_snr/fc{i}", layer.sigma_snr(), frame_idx)
            if args.replay == "per":
                writer.add_scalar("per_beta", beta, frame_idx)
                writer.add_scalar("td_error_mean", float(td_abs.mean()), frame_idx)
                if frame_idx % (10 * hp.log_every) == 0:
                    writer.add_histogram("td_error_hist", td_abs, frame_idx)

        if frame_idx % hp.ckpt_every == 0:
            save_checkpoint(latest, net, tgt_net, optimizer, frame_idx, tracker, {"scalar": policy.scalar(frame_idx)})

    save_checkpoint(latest, net, tgt_net, optimizer, frame_idx, tracker, {"scalar": policy.scalar(frame_idx)})
    summary = tracker.summary(frame_idx)
    with open(os.path.join(rdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    writer.close()
    print("done", summary)


if __name__ == "__main__":
    main()
