# Paper:          https://arxiv.org/abs/2108.13956
# Reference impl: https://github.com/rll-research/url_benchmark

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import dataclass  # noqa: E402
from functools import partial  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal, zeros
from flax.training.train_state import TrainState

from wrappers import (  # noqa: E402
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    VecEnv,
    get_xy,
)


@dataclass
class Config:
    lr: float = 1e-4
    hidden_dim: int = 1024
    critic_target_tau: float = 0.01
    update_every_steps: int = 2
    stddev: float = 0.2
    stddev_clip: float = 0.3
    sf_dim: int = 10
    knn_k: int = 12
    knn_avg: bool = True
    knn_rms: bool = True
    knn_clip: float = 0.0001
    update_task_every_step: int = 50
    buffer_size: int = 1_000_000
    batch_size: int = 1024
    gamma: float = 0.99
    seed: int = 0
    num_seeds: int = 1
    env: str = "hopper"
    num_envs: int = 16
    total_timesteps: int = 1_000_000
    num_init_steps: int = 4_000
    eval_every: int = 100_000
    num_eval_episodes: int = 10
    eval_episode_length: int = 1000
    video_episode_length: int = 500
    video_height: int = 240
    video_width: int = 320
    video_fps: int = 30
    wandb_project: str = "aps-ddpg-brax"


class Actor(nn.Module):
    # obs here is cat(obs_raw, task); input dim = obs_dim + sf_dim

    action_dim: int
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, obs):
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=zeros)(obs)
        h = nn.LayerNorm(epsilon=1e-5)(h)
        h = jnp.tanh(h)
        h = jax.nn.relu(
            nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=zeros)(h)
        )
        return jnp.tanh(
            nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=zeros)(h)
        )


class CriticSF(nn.Module):
    """misc/url_benchmark/agent/aps.py:CriticSF.

    obs here is cat(obs_raw, task); takes task as third input to collapse the
    SF-vector Q-output to a scalar via per-sample dot product with the task.
    Output: scalar Q for (obs_z, action, task).
    """

    sf_dim: int
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, obs, action, task):
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=zeros)(
            jnp.concatenate([obs, action], axis=-1)
        )
        h = nn.LayerNorm(epsilon=1e-5)(h)
        h = jnp.tanh(h)

        def q_head(name):
            x = jax.nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=zeros,
                    name=f"{name}_0",
                )(h)
            )
            return nn.Dense(
                self.sf_dim,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name=f"{name}_1",
            )(x)

        q1_sf = q_head("Q1")
        q2_sf = q_head("Q2")
        q1 = jnp.sum(task * q1_sf, axis=-1, keepdims=True)
        q2 = jnp.sum(task * q2_sf, axis=-1, keepdims=True)
        return q1, q2


class APSNet(nn.Module):
    # Mirror url_benchmark/agent/aps.py:APS (state_feat_net)

    sf_dim: int
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, obs, norm=True):
        h = jax.nn.relu(
            nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=zeros)(obs)
        )
        h = jax.nn.relu(
            nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=zeros)(h)
        )
        rep = nn.Dense(self.sf_dim, kernel_init=orthogonal(1.0), bias_init=zeros)(h)
        if norm:
            rep = rep / (jnp.linalg.norm(rep, axis=-1, keepdims=True) + 1e-12)
        return rep


def truncated_normal_sample(mu, std, eps, clip, clamp_eps=1e-6):
    return jnp.clip(
        mu + jnp.clip(eps * std, -clip, clip), -1.0 + clamp_eps, 1.0 - clamp_eps
    )


def env_action_sample(mu, std, eps, clamp_eps=1e-6):
    return jnp.clip(mu + eps * std, -1.0 + clamp_eps, 1.0 - clamp_eps)


def soft_update(p, t, tau):
    return jax.tree_util.tree_map(lambda x, y: tau * x + (1.0 - tau) * y, p, t)


def rms_update(state, x):
    # Mirror misc/url_benchmark/utils.py:RMS
    M, S, n = state
    bs = x.shape[0]
    delta = jnp.mean(x, axis=0) - M
    new_M = M + delta * bs / (n + bs)
    new_S = (
        S * n + jnp.var(x, axis=0, ddof=1) * bs + jnp.square(delta) * n * bs / (n + bs)
    ) / (n + bs)
    return (new_M, new_S, n + bs)


def pbe_reward(rep, knn_k, knn_avg, knn_clip, knn_rms, rms_state):
    # particle based entropy reward estimator. Mirror misc/url_benchmark/utils.py:PBE.__call__
    b = rep.shape[0]
    diff = rep[:, None, :] - rep[None, :, :]
    sim = jnp.linalg.norm(diff, ord=2, axis=-1)
    neg_topk, _ = jax.lax.top_k(-sim, knn_k)
    reward = jnp.sort(-neg_topk, axis=1)

    if knn_avg:
        flat = reward.reshape(-1, 1)
        if knn_rms:
            new_rms = rms_update(rms_state, flat)
            flat = flat / new_rms[0]
        else:
            new_rms = rms_state
        flat = jnp.where(knn_clip >= 0.0, jnp.maximum(flat - knn_clip, 0.0), flat)
        flat = flat.reshape(b, knn_k)
        result = jnp.mean(flat, axis=1, keepdims=True)
    else:
        r = reward[:, -1].reshape(-1, 1)
        if knn_rms:
            new_rms = rms_update(rms_state, r)
            r = r / new_rms[0]
        else:
            new_rms = rms_state
        result = jnp.where(knn_clip >= 0.0, jnp.maximum(r - knn_clip, 0.0), r)

    return jnp.log(result + 1.0), new_rms


def sample_task(rng, sf_dim, batch_size):
    # Random unit-norm vector per env
    raw = jax.random.normal(rng, (batch_size, sf_dim))
    return raw / (jnp.linalg.norm(raw, axis=-1, keepdims=True) + 1e-12)


class AgentState(NamedTuple):
    actor: TrainState
    critic: TrainState
    critic_target_params: Any
    aps: TrainState
    rms_state: tuple


def init(cfg, rng, obs_dim, action_dim):
    rng_a, rng_c, rng_p = jax.random.split(rng, 3)
    actor_net = Actor(action_dim, cfg.hidden_dim)
    critic_net = CriticSF(cfg.sf_dim, cfg.hidden_dim)
    aps_net = APSNet(cfg.sf_dim, cfg.hidden_dim)
    o_t = jnp.zeros((1, obs_dim + cfg.sf_dim))
    o = jnp.zeros((1, obs_dim))
    a = jnp.zeros((1, action_dim))
    t = jnp.zeros((1, cfg.sf_dim))
    tx = lambda: optax.adam(cfg.lr, eps=1e-8)
    actor_p = actor_net.init(rng_a, o_t)["params"]
    critic_p = critic_net.init(rng_c, o_t, a, t)["params"]
    aps_p = aps_net.init(rng_p, o)["params"]
    return AgentState(
        actor=TrainState.create(apply_fn=actor_net.apply, params=actor_p, tx=tx()),
        critic=TrainState.create(apply_fn=critic_net.apply, params=critic_p, tx=tx()),
        critic_target_params=critic_p,
        aps=TrainState.create(apply_fn=aps_net.apply, params=aps_p, tx=tx()),
        rms_state=(jnp.zeros((1,)), jnp.ones((1,)), jnp.float32(1e-4)),
    )


def update_step(cfg, agent, batch, eps_targ, eps_act):
    # One DDPG+APS off-policy update
    obs = batch["obs"]
    next_obs = batch["next_obs"]
    task = batch["task"]
    obs_t = jnp.concatenate([obs, task], axis=-1)
    next_obs_t = jnp.concatenate([next_obs, task], axis=-1)

    # --- 1. APS aux loss = -mean(task @ state_feat(next_obs, norm=True)) ---
    def aps_loss_fn(p):
        rep = agent.aps.apply_fn({"params": p}, next_obs, norm=True)
        return -jnp.sum(task * rep, axis=-1).mean()

    aps_loss, p_grads = jax.value_and_grad(aps_loss_fn)(agent.aps.params)
    new_aps = agent.aps.apply_gradients(grads=p_grads)

    # --- 2. Intrinsic reward — fresh forward post-update ---
    rep_unnorm = jax.lax.stop_gradient(
        agent.aps.apply_fn({"params": new_aps.params}, next_obs, norm=False)
    )
    rep_norm = rep_unnorm / (
        jnp.linalg.norm(rep_unnorm, axis=-1, keepdims=True) + 1e-12
    )
    intr_sf = jnp.sum(task * rep_norm, axis=-1, keepdims=True)
    intr_ent, new_rms = pbe_reward(
        rep_unnorm, cfg.knn_k, cfg.knn_avg, cfg.knn_clip, cfg.knn_rms, agent.rms_state
    )
    intr_reward = intr_sf + intr_ent

    # --- 3. Critic update (CriticSF: takes task as 3rd input) ---
    next_mu = agent.actor.apply_fn({"params": agent.actor.params}, next_obs_t)
    next_action = jax.lax.stop_gradient(
        truncated_normal_sample(next_mu, cfg.stddev, eps_targ, cfg.stddev_clip)
    )
    tq1, tq2 = agent.critic.apply_fn(
        {"params": agent.critic_target_params}, next_obs_t, next_action, task
    )
    target_q = jax.lax.stop_gradient(
        intr_reward + batch["discount"] * jnp.minimum(tq1, tq2)
    )

    def critic_loss_fn(p):
        q1, q2 = agent.critic.apply_fn({"params": p}, obs_t, batch["action"], task)
        return jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))

    critic_loss, c_grads = jax.value_and_grad(critic_loss_fn)(agent.critic.params)
    new_critic = agent.critic.apply_gradients(grads=c_grads)

    # --- 4. Actor update ---
    def actor_loss_fn(p):
        mu = agent.actor.apply_fn({"params": p}, obs_t)
        action = truncated_normal_sample(mu, cfg.stddev, eps_act, cfg.stddev_clip)
        q1, q2 = agent.critic.apply_fn(
            {"params": jax.lax.stop_gradient(new_critic.params)}, obs_t, action, task
        )
        return -jnp.mean(jnp.minimum(q1, q2))

    actor_loss, a_grads = jax.value_and_grad(actor_loss_fn)(agent.actor.params)
    new_actor = agent.actor.apply_gradients(grads=a_grads)

    new_target = soft_update(
        new_critic.params, agent.critic_target_params, cfg.critic_target_tau
    )
    new_agent = agent._replace(
        actor=new_actor,
        critic=new_critic,
        critic_target_params=new_target,
        aps=new_aps,
        rms_state=new_rms,
    )
    return new_agent, {
        "aps_loss": aps_loss,
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "intr_reward": intr_reward.mean(),
    }


def update_step_rng(cfg, agent, batch, rng):
    rng_t, rng_a = jax.random.split(rng)
    sh = batch["action"].shape
    return update_step(
        cfg, agent, batch, jax.random.normal(rng_t, sh), jax.random.normal(rng_a, sh)
    )


def train(cfg):
    import flashbax as fbx
    import wandb

    env = VecEnv(ClipAction(LogWrapper(BraxGymnaxWrapper(cfg.env))))
    obs_dim = env.observation_space(None).shape[0]
    action_dim = env.action_space(None).shape[0]

    buffer = fbx.make_item_buffer(
        max_length=cfg.buffer_size,
        min_length=cfg.batch_size,
        sample_batch_size=cfg.batch_size,
        add_batches=True,
    )
    dummy = {
        "obs": jnp.zeros(obs_dim),
        "action": jnp.zeros(action_dim),
        "reward": jnp.float32(0.0),
        "discount": jnp.float32(0.0),
        "next_obs": jnp.zeros(obs_dim),
        "task": jnp.zeros(cfg.sf_dim),
    }
    eval_env = ClipAction(BraxGymnaxWrapper(cfg.env))

    @jax.jit
    def init_per_seed(rng):
        rng, ri, rr, rt = jax.random.split(rng, 4)
        agent = init(cfg, ri, obs_dim, action_dim)
        rs = jax.random.split(rr, cfg.num_envs)
        obs, st = env.reset(rs, None)
        task = sample_task(rt, cfg.sf_dim, cfg.num_envs)
        return agent, st, obs, buffer.init(dummy), rng, task, jnp.int32(0)

    @partial(jax.jit, static_argnames=("num_iters", "warmup"))
    def chunk(agent, st, obs, buf, rng, task, step_count, num_iters, warmup):
        def micro(carry, _):
            agent, st, obs, buf, rng, task, step_count = carry
            rng, ra, rs, ru, rt = jax.random.split(rng, 5)

            obs_t = jnp.concatenate([obs, task], axis=-1)
            mu = agent.actor.apply_fn({"params": agent.actor.params}, obs_t)
            noise = jax.random.normal(ra, mu.shape)
            a_agent = env_action_sample(mu, cfg.stddev, noise)
            a_rand = jax.random.uniform(ra, mu.shape, minval=-1.0, maxval=1.0)
            action = jnp.where(warmup, a_rand, a_agent)
            srngs = jax.random.split(rs, cfg.num_envs)
            next_obs, st, reward, done, _ = env.step(srngs, st, action, None)
            disc = cfg.gamma * (1.0 - done.astype(jnp.float32))
            buf = buffer.add(
                buf,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "discount": disc,
                    "next_obs": next_obs,
                    "task": task,
                },
            )

            step_count = step_count + 1
            resample = (step_count % cfg.update_task_every_step == 0) | done
            new_task = sample_task(rt, cfg.sf_dim, cfg.num_envs)
            task = jnp.where(resample[:, None], new_task, task)

            n_upd = cfg.num_envs // cfg.update_every_steps

            def one_upd(c_u, _):
                a, r = c_u
                r, b, u = jax.random.split(r, 3)
                batch = buffer.sample(buf, b).experience
                a, info = update_step_rng(cfg, a, batch, u)
                return (a, r), info

            def do_upd(args):
                a, r = args
                (a, r), info = jax.lax.scan(one_upd, (a, r), None, length=n_upd)
                return a, r, info

            def skip_upd(args):
                a, r = args
                empty = {
                    k: jnp.zeros(n_upd)
                    for k in ("aps_loss", "critic_loss", "actor_loss", "intr_reward")
                }
                return a, r, empty

            ready = buffer.can_sample(buf) & (~warmup)
            agent, ru, info = jax.lax.cond(ready, do_upd, skip_upd, (agent, ru))
            info = jax.tree_util.tree_map(jnp.mean, info)
            return (
                (agent, st, next_obs, buf, rng, task, step_count),
                {"reward": reward.mean(), "xy": get_xy(st), **info},
            )

        carry, trace = jax.lax.scan(
            micro, (agent, st, obs, buf, rng, task, step_count), None, num_iters
        )
        return (*carry, trace)

    @jax.jit
    def evaluate(agent, rng):
        rrngs = jax.random.split(rng, cfg.num_eval_episodes)
        obs0, st0 = jax.vmap(eval_env.reset, in_axes=(0, None))(rrngs, None)
        task0 = jnp.zeros((cfg.num_eval_episodes, cfg.sf_dim))
        task0 = task0.at[:, 0].set(1.0)

        def step(carry, _):
            rng, obs, st, fin, ret = carry
            rng, sub = jax.random.split(rng)
            obs_t = jnp.concatenate([obs, task0], axis=-1)
            mu = agent.actor.apply_fn({"params": agent.actor.params}, obs_t)
            srngs = jax.random.split(sub, cfg.num_eval_episodes)
            no, st, r, d, _ = jax.vmap(eval_env.step, in_axes=(0, 0, 0, None))(
                srngs, st, mu, None
            )
            ret = ret + r * (~fin).astype(r.dtype)
            return (rng, no, st, fin | d, ret), None

        init_c = (
            rng,
            obs0,
            st0,
            jnp.zeros(cfg.num_eval_episodes, jnp.bool_),
            jnp.zeros(cfg.num_eval_episodes),
        )
        (_, _, _, _, ret), _ = jax.lax.scan(step, init_c, None, cfg.eval_episode_length)
        return ret

    init_v = jax.vmap(init_per_seed)
    chunk_v = jax.vmap(chunk, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))
    eval_v = jax.vmap(evaluate)

    rngs = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.num_seeds)
    agent, env_state, obs, buf, rngs, task, step_count = init_v(rngs)

    from wrappers import VisitationHistogram

    hists = [VisitationHistogram() for _ in range(cfg.num_seeds)]

    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    iters = max(1, cfg.eval_every // cfg.num_envs)
    n_chunks = max(1, cfg.total_timesteps // cfg.eval_every)
    init_iters = max(1, cfg.num_init_steps // cfg.num_envs)
    if init_iters < iters:
        agent, env_state, obs, buf, rngs, task, step_count, _ = chunk_v(
            agent, env_state, obs, buf, rngs, task, step_count, init_iters, True
        )

    timestep = init_iters * cfg.num_envs
    for c in range(n_chunks):
        agent, env_state, obs, buf, rngs, task, step_count, trace = chunk_v(
            agent, env_state, obs, buf, rngs, task, step_count, iters, False
        )
        timestep += iters * cfg.num_envs
        xy_chunk = np.asarray(trace["xy"])
        for i in range(cfg.num_seeds):
            hists[i].add(xy_chunk[i])
        eval_rngs = jax.random.split(
            jax.random.PRNGKey(cfg.seed + 1000 + c), cfg.num_seeds
        )
        ret = eval_v(agent, eval_rngs)
        m = float(ret.mean())
        log_dict = {
            "eval/return": m,
            **{
                f"train/{k}": float(jnp.mean(trace[k]))
                for k in ("aps_loss", "critic_loss", "actor_loss", "intr_reward")
            },
            "step": timestep,
        }
        for i in range(cfg.num_seeds):
            log_dict[f"heatmap/seed_{i}"] = hists[i].wandb_image(
                title=f"aps heatmap, seed={i}, step={timestep}"
            )
        wandb.log(log_dict, step=timestep)
        print(f"step={timestep:>7d}  eval_return={m:.2f}")

    from wrappers import render_brax_video

    video_env = ClipAction(BraxGymnaxWrapper(cfg.env))
    seed0_actor = jax.tree_util.tree_map(lambda x: x[0], agent.actor.params)
    task0 = jnp.zeros((1, cfg.sf_dim)).at[:, 0].set(1.0)

    @jax.jit
    def video_rollout(rng):
        reset_rng, scan_rng = jax.random.split(rng)
        obs0, st0 = video_env.reset(reset_rng)

        def body(carry, _):
            rng, obs, st = carry
            rng, sub = jax.random.split(rng)
            obs_t = jnp.concatenate([obs[None], task0], axis=-1)[0]
            mu = agent.actor.apply_fn({"params": seed0_actor}, obs_t)
            new_obs, new_st, _, _, _ = video_env.step(sub, st, mu)
            return (rng, new_obs, new_st), new_st.pipeline_state

        _, ps = jax.lax.scan(
            body, (scan_rng, obs0, st0), None, length=cfg.video_episode_length
        )
        return ps

    pipeline_states = video_rollout(jax.random.PRNGKey(cfg.seed + 1))
    render_brax_video(
        video_env,
        pipeline_states,
        episode_length=cfg.video_episode_length,
        height=cfg.video_height,
        width=cfg.video_width,
        fps=cfg.video_fps,
        log_key="video/rollout_seed0",
    )
    wandb.finish()


if __name__ == "__main__":
    import tyro

    train(tyro.cli(Config))