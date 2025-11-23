# tinygrad PPO Atari (Breakout) version of your original script

import os
import random
import time
from typing import SupportsFloat
from dataclasses import dataclass

import ale_py
import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter  # just for logging

from tinygrad import Tensor, nn
from tinygrad.nn import optim as tiny_optim
from tinygrad.nn.state import get_parameters

class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info: dict = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        return float(np.sign(float(reward)))


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True  # kept for compatibility; not used
    cuda: bool = True  # not used with tinygrad; control via env vars if needed
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "ALE/Breakout-v5"
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        gym.register_envs(ale_py)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=100)
        env = NoopResetEnv(env, noop_max=30)
        env = gym.wrappers.MaxAndSkipObservation(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Tinygrad-friendly "orthogonal-ish" init: rescale existing weights to have std ~ `std`,
    and set bias to `bias_const`, making sure params require grad.
    """
    if hasattr(layer, "weight") and isinstance(layer.weight, Tensor):
        w = layer.weight
        w_std = w.std()
        # rescale to desired std and ensure requires_grad=True
        layer.weight = (w * (std / (w_std + 1e-8))).requires_grad_(True)

    if hasattr(layer, "bias") and layer.bias is not None:
        # make bias a trainable parameter as well
        layer.bias = Tensor.full(layer.bias.shape, bias_const, requires_grad=True)

    return layer


class Agent:
    def __init__(self, envs):
        in_channels = int(envs.single_observation_space.shape[0])
        num_actions = int(envs.single_action_space.n)

        self.num_actions = num_actions

        # CNN trunk as in CleanRL PPO Atari
        self.cnn_layers = [
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            Tensor.relu,
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            Tensor.relu,
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            Tensor.relu,
            lambda x: x.reshape(x.shape[0], -1),  # flatten
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            Tensor.relu,
        ]
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def __call__(self, x: Tensor):
        # x: (B, 4, 84, 84) uint8 or float32
        hidden = (x / 255.0).sequential(self.cnn_layers)
        logits = self.actor(hidden)              # (B, A)
        value = self.critic(hidden).squeeze(-1)  # (B,)
        return logits, value


def sample_actions(agent: Agent, obs_np: np.ndarray):
    """
    Tinygrad inference-only helper: returns actions, logprobs, values as numpy arrays.
    """
    logits, value = agent(Tensor(obs_np.astype(np.float32)))
    probs = logits.softmax(axis=-1)           # (num_envs, num_actions)
    probs_np = probs.numpy()

    num_envs, num_actions = probs_np.shape
    actions = np.zeros(num_envs, dtype=np.int64)
    for i in range(num_envs):
        actions[i] = np.random.choice(num_actions, p=probs_np[i])

    logprobs_np = np.log(probs_np[np.arange(num_envs), actions] + 1e-8)
    values_np = value.numpy()  # (num_envs,)
    return actions, logprobs_np, values_np


def compute_value(agent: Agent, obs_np: np.ndarray):
    _, value = agent(Tensor(obs_np.astype(np.float32)))
    return value.numpy()


def clip_grad_norm_(params, max_norm: float):
    """
    Simple global L2 grad norm clipping for tinygrad.
    """
    total_norm_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.numpy()
        total_norm_sq += float((g ** 2).sum())
    total_norm = float(np.sqrt(total_norm_sq))
    if total_norm > max_norm and total_norm > 0.0:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total_norm


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    Tensor.manual_seed(args.seed)

    # tinygrad: enable training mode so optimizer can run
    Tensor.training = True

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs)
    params = get_parameters(agent)
    optimizer = tiny_optim.Adam(params, lr=args.learning_rate, eps=1e-5)

    # Storage (numpy; tinygrad only for forward/backward)
    obs = np.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        dtype=np.float32,
    )
    actions = np.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        dtype=np.int64,
    )
    logprobs = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    dones = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    values = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = np.zeros(args.num_envs, dtype=np.float32)

    num_actions = int(envs.single_action_space.n)

    for iteration in range(1, args.num_iterations + 1):
        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.lr = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action selection
            acts, lps, vals = sample_actions(agent, next_obs)
            actions[step] = acts
            logprobs[step] = lps
            values[step] = vals

            # Env step
            next_obs, reward, terminations, truncations, infos = envs.step(acts)
            rewards[step] = reward
            next_done = np.logical_or(terminations, truncations).astype(np.float32)

            if "episode" in infos:
                done_mask = infos.get("_episode", None)
                if done_mask is not None:
                    rs = infos["episode"]["r"]
                    ls = infos["episode"]["l"]
                    for i, done in enumerate(done_mask):
                        if done:
                            ep_r = float(rs[i])
                            ep_l = int(ls[i])
                            writer.add_scalar("train/episode_return", ep_r, global_step)
                            writer.add_scalar("train/episode_length", ep_l, global_step)

        # GAE-Lambda advantage computation (numpy)
        next_value = compute_value(agent, next_obs)  # (num_envs,)
        advantages = np.zeros_like(rewards)
        lastgaelam = np.zeros(args.num_envs, dtype=np.float32)

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((args.batch_size,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((args.batch_size,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        approx_kl = None
        old_approx_kl = None

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = Tensor(b_obs[mb_inds])  # (B, C, H, W)
                mb_actions_np = b_actions[mb_inds].astype(np.int32).reshape(-1)
                mb_old_logprobs = Tensor(b_logprobs[mb_inds])
                mb_advantages = Tensor(b_advantages[mb_inds])
                mb_returns = Tensor(b_returns[mb_inds])
                mb_values = Tensor(b_values[mb_inds])

                logits, newvalue = agent(mb_obs)  # logits: (B, A), newvalue: (B,)
                log_probs = logits.log_softmax(axis=-1)
                probs = log_probs.exp()

                # pick log-prob of taken actions using gather
                action_idx = Tensor(mb_actions_np).reshape(-1, 1)
                new_logprob = log_probs.gather(-1, action_idx).reshape(-1)

                logratio = new_logprob - mb_old_logprobs
                ratio = logratio.exp()

                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfracs.append(
                    ((ratio - 1.0).abs() > args.clip_coef).mean().numpy().item()
                )

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                ratio_clipped = ratio.maximum(1.0 - args.clip_coef).minimum(
                    1.0 + args.clip_coef
                )
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio_clipped
                pg_loss = pg_loss1.maximum(pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + (newvalue - mb_values).maximum(
                        -args.clip_coef
                    ).minimum(args.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * v_loss_unclipped.maximum(v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy = (-(probs * log_probs).sum(axis=-1)).mean()

                loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(params, args.max_grad_norm)

                # tinygrad's optimizer expects every param to have a grad
                for p in params:
                    if p.grad is None:
                        p.grad = Tensor.zeros(*p.shape)

                optimizer.step()

            if args.target_kl is not None and approx_kl is not None:
                if float(approx_kl.numpy()) > args.target_kl:
                    break

        # Explained variance (numpy)
        y_pred, y_true = b_values, b_returns
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.lr, global_step)
        writer.add_scalar("losses/value_loss", float(v_loss.numpy()), global_step)
        writer.add_scalar("losses/policy_loss", float(pg_loss.numpy()), global_step)
        writer.add_scalar("losses/entropy", float(entropy.numpy()), global_step)
        if old_approx_kl is not None and approx_kl is not None:
            writer.add_scalar(
                "losses/old_approx_kl", float(old_approx_kl.numpy()), global_step
            )
            writer.add_scalar(
                "losses/approx_kl", float(approx_kl.numpy()), global_step
            )
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", float(explained_var), global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
