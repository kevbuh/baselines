# Paper:          https://arxiv.org/abs/1810.12894
# Reference impl: https://github.com/openai/random-network-distillation

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import dataclass  # noqa: E402
from typing import Any, NamedTuple, Sequence  # noqa: E402

import distrax  # noqa: E402
import flax  # noqa: E402
import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax.linen.initializers import constant, orthogonal  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from wrappers import (  # noqa: E402
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    VecEnv,
    get_xy,
)


@dataclass
class Config:
    # PPO
    lr: float = 1e-4
    num_envs: int = 128
    num_steps: int = 10
    update_epochs: int = 4
    num_minibatches: int = 32
    gamma_ext: float = 0.99
    gamma_int: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.001
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    activation: str = "tanh"
    anneal_lr: bool = True
    normalize_adv: bool = True
    # RND
    rep_size: int = 512
    ext_coef: float = 2.0
    int_coef: float = 1.0
    update_proportion: float = 0.25
    obs_clip: float = 5.0
    num_iterations_obs_norm_init: int = 50
    # Train
    seed: int = 0
    num_seeds: int = 1
    env: str = "hopper"
    total_timesteps: int = 50_000_000
    # Eval
    eval_every: int = 500_000
    num_eval_episodes: int = 10
    eval_episode_length: int = 1000
    # Video
    video_episode_length: int = 500
    video_height: int = 240
    video_width: int = 320
    video_fps: int = 30
    # Wandb
    wandb_project: str = "rnd-ppo-brax"


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class ActorDualCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        h = act(
            nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
                x
            )
        )
        h = act(
            nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
                h
            )
        )
        # extrahid residual: x = relu(fc(x)) + x  (fan out into two heads).
        x_act = h + nn.relu(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(h))
        x_val = h + nn.relu(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(h))
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x_act)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std))
        vpred_ext = jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(0.01))(x_val), -1)
        vpred_int = jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(0.01))(x_val), -1)
        return pi, vpred_ext, vpred_int


class RNDTarget(nn.Module):
    # Frozen target net

    rep_size: int = 512

    @nn.compact
    def __call__(self, x):
        x = nn.leaky_relu(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        x = nn.leaky_relu(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        return nn.Dense(self.rep_size, kernel_init=orthogonal(jnp.sqrt(2)))(x)


class RNDPredictor(nn.Module):
    # Trained predictor

    rep_size: int = 512

    @nn.compact
    def __call__(self, x):
        x = nn.leaky_relu(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        x = nn.leaky_relu(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        x = nn.relu(nn.Dense(512, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        x = nn.relu(nn.Dense(512, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        return nn.Dense(self.rep_size, kernel_init=orthogonal(jnp.sqrt(2)))(x)


# ---------------------------------------------------------------------------
# Welford running stats + RFF
# ---------------------------------------------------------------------------


class RunningMoments(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


class RewardNormState(NamedTuple):
    reward_ems: jnp.ndarray
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def welford_update(state, batch):
    # Parallel Welford update
    bm = jnp.mean(batch, axis=0)
    bv = jnp.var(batch, axis=0)
    bc = batch.shape[0]
    delta = bm - state.mean
    tot = state.count + bc
    new_mean = state.mean + delta * bc / tot
    M2 = state.var * state.count + bv * bc + jnp.square(delta) * state.count * bc / tot
    return new_mean, M2 / tot, tot


# ---------------------------------------------------------------------------
# Agent state + transitions
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_ext: jnp.ndarray
    value_int: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    xy: jnp.ndarray


class AgentState(flax.struct.PyTreeNode):
    train_state: TrainState
    rnd_target_params: Any
    obs_norm: RunningMoments
    reward_norm: RewardNormState


def init(cfg: Config, rng, obs_dim: int, action_dim: int, total_updates: int):
    rng, rng_net, rng_targ, rng_pred = jax.random.split(rng, 4)
    network = ActorDualCritic(action_dim, activation=cfg.activation)
    target = RNDTarget(cfg.rep_size)
    predictor = RNDPredictor(cfg.rep_size)

    init_x = jnp.zeros(obs_dim)
    target_params = target.init(rng_targ, init_x)
    combined = {
        "policy": network.init(rng_net, init_x),
        "rnd_predictor": predictor.init(rng_pred, init_x),
    }

    def lr_schedule(count):
        frac = (
            1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / total_updates
        )
        return cfg.lr * frac

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=(lr_schedule if cfg.anneal_lr else cfg.lr), eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=combined, tx=tx)
    return (
        AgentState(
            train_state=train_state,
            rnd_target_params=target_params,
            obs_norm=RunningMoments(
                mean=jnp.zeros(obs_dim),
                var=jnp.ones(obs_dim),
                count=jnp.float32(1e-4),
            ),
            reward_norm=RewardNormState(
                reward_ems=jnp.zeros(cfg.num_envs),
                mean=jnp.float32(0.0),
                var=jnp.float32(1.0),
                count=jnp.float32(1e-4),
            ),
        ),
        rng,
    )


# ---------------------------------------------------------------------------
# Per-step
# ---------------------------------------------------------------------------


def act(agent, obs, rng):
    rng, sub = jax.random.split(rng)
    pi, ve, vi = agent.train_state.apply_fn(agent.train_state.params["policy"], obs)
    action = pi.sample(seed=sub)
    return action, pi.log_prob(action), ve, vi, rng


def value(agent, obs):
    _, ve, vi = agent.train_state.apply_fn(agent.train_state.params["policy"], obs)
    return ve, vi


def intrinsic_reward(agent, obs, obs_clip):
    rnd_obs = jnp.clip(
        (obs - agent.obs_norm.mean) / jnp.sqrt(agent.obs_norm.var + 1e-8),
        -obs_clip,
        obs_clip,
    )
    target_feat = RNDTarget().apply(agent.rnd_target_params, rnd_obs)
    pred_feat = RNDPredictor().apply(agent.train_state.params["rnd_predictor"], rnd_obs)
    return jnp.mean(jnp.square(target_feat - pred_feat), axis=-1)


# ---------------------------------------------------------------------------
# Loss + multi-epoch / multi-minibatch update
# ---------------------------------------------------------------------------


def loss_fn(cfg, agent, params, traj, gae, t_ext, t_int, rng_mask):
    pi, vp_ext, vp_int = agent.train_state.apply_fn(params["policy"], traj.obs)
    log_prob = pi.log_prob(traj.action)

    vf_ext = (0.5 * cfg.vf_coef) * jnp.square(vp_ext - t_ext).mean()
    vf_int = (0.5 * cfg.vf_coef) * jnp.square(vp_int - t_int).mean()

    ratio = jnp.exp(log_prob - traj.log_prob)
    if cfg.normalize_adv:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    pg = jnp.minimum(
        ratio * gae,
        jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * gae,
    ).mean()
    pg_loss = -pg
    entropy = pi.entropy().mean()
    ent_loss = -cfg.ent_coef * entropy

    rnd_obs = jnp.clip(
        (traj.obs - agent.obs_norm.mean) / jnp.sqrt(agent.obs_norm.var + 1e-8),
        -cfg.obs_clip,
        cfg.obs_clip,
    )
    target_feat = RNDTarget(cfg.rep_size).apply(agent.rnd_target_params, rnd_obs)
    pred_feat = RNDPredictor(cfg.rep_size).apply(params["rnd_predictor"], rnd_obs)
    pred_errs = jnp.mean(jnp.square(target_feat - pred_feat), axis=-1)
    mask = jax.random.uniform(rng_mask, shape=pred_errs.shape) < cfg.update_proportion
    aux_loss = jnp.sum(mask * pred_errs) / jnp.maximum(jnp.sum(mask), 1.0)

    total = pg_loss + ent_loss + vf_ext + vf_int + aux_loss
    return total, {
        "total_loss": total,
        "pg_loss": pg_loss,
        "vf_loss_ext": vf_ext,
        "vf_loss_int": vf_int,
        "entropy": entropy,
        "aux_loss": aux_loss,
    }


def update(cfg, agent, traj, advantages, t_ext, t_int, rng):
    # K epochs * M minibatches over a (T, B, ...) trajectory
    bs = cfg.num_steps * cfg.num_envs

    def epoch(carry, _):
        ts, rng = carry
        rng, perm_rng, mask_rng = jax.random.split(rng, 3)
        perm = jax.random.permutation(perm_rng, bs)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((bs,) + x.shape[2:]),
            (traj, advantages, t_ext, t_int),
        )
        shuf = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
        mbs = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [cfg.num_minibatches, -1] + list(x.shape[1:])),
            shuf,
        )
        mask_rngs = jax.random.split(mask_rng, cfg.num_minibatches)

        def mb_step(ts, mb_with_rng):
            (mb_traj, mb_adv, mb_te, mb_ti), mr = mb_with_rng
            (_, info), grads = jax.value_and_grad(
                lambda p: loss_fn(cfg, agent, p, mb_traj, mb_adv, mb_te, mb_ti, mr),
                has_aux=True,
            )(ts.params)
            return ts.apply_gradients(grads=grads), info

        ts, info = jax.lax.scan(mb_step, ts, (mbs, mask_rngs))
        return (ts, rng), info

    (new_ts, rng), info = jax.lax.scan(
        epoch, (agent.train_state, rng), None, cfg.update_epochs
    )
    info = jax.tree_util.tree_map(jnp.mean, info)
    return agent.replace(train_state=new_ts), info, rng


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train(cfg: Config):
    import wandb

    env = VecEnv(ClipAction(LogWrapper(BraxGymnaxWrapper(cfg.env))))
    eval_env = ClipAction(BraxGymnaxWrapper(cfg.env))
    obs_dim = env.observation_space(None).shape[0]
    action_dim = env.action_space(None).shape[0]

    steps_per_update = cfg.num_steps * cfg.num_envs
    total_updates = max(1, int(cfg.total_timesteps // steps_per_update))
    updates_per_chunk = max(1, int(cfg.eval_every // steps_per_update))
    num_chunks = max(1, total_updates // updates_per_chunk)

    @jax.jit
    def init_per_seed(rng):
        rng, ri = jax.random.split(rng)
        agent, _ = init(cfg, ri, obs_dim, action_dim, total_updates)
        rng, rr = jax.random.split(rng)
        reset_rngs = jax.random.split(rr, cfg.num_envs)
        obs, env_state = env.reset(reset_rngs, None)

        def warmup_step(carry, _):
            agent, env_state, obs, rng = carry
            rng, ra, rs = jax.random.split(rng, 3)
            actions = jax.random.uniform(
                ra, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0
            )
            srngs = jax.random.split(rs, cfg.num_envs)
            obs, env_state, _, _, _ = env.step(srngs, env_state, actions, None)
            m, v, c = welford_update(agent.obs_norm, obs)
            agent = agent.replace(obs_norm=RunningMoments(mean=m, var=v, count=c))
            return (agent, env_state, obs, rng), None

        warmup_steps = cfg.num_steps * cfg.num_iterations_obs_norm_init
        (agent, env_state, obs, rng), _ = jax.lax.scan(
            warmup_step, (agent, env_state, obs, rng), None, warmup_steps
        )
        rng, rr = jax.random.split(rng)
        reset_rngs = jax.random.split(rr, cfg.num_envs)
        obs, env_state = env.reset(reset_rngs, None)
        return agent, env_state, obs, rng

    @jax.jit
    def chunk(agent, env_state, last_obs, rng):
        def one_update(carry, _):
            agent, env_state, last_obs, rng = carry

            def env_step(scarry, _):
                agent, env_state, obs, rng = scarry
                action, log_prob, ve, vi, rng = act(agent, obs, rng)
                rng, sub = jax.random.split(rng)
                srngs = jax.random.split(sub, cfg.num_envs)
                next_obs, env_state, reward, done, info = env.step(
                    srngs, env_state, action, None
                )
                int_reward = intrinsic_reward(agent, next_obs, cfg.obs_clip)
                m, v, c = welford_update(agent.obs_norm, next_obs)
                agent = agent.replace(obs_norm=RunningMoments(mean=m, var=v, count=c))
                tr = Transition(
                    done=done,
                    action=action,
                    value_ext=ve,
                    value_int=vi,
                    reward=reward,
                    int_reward=int_reward,
                    log_prob=log_prob,
                    obs=obs,
                    xy=get_xy(env_state),
                )
                return (agent, env_state, next_obs, rng), tr

            (agent, env_state, last_obs, rng), traj = jax.lax.scan(
                env_step, (agent, env_state, last_obs, rng), None, cfg.num_steps
            )

            def rff_step(reward_ems, ir):
                reward_ems = reward_ems * cfg.gamma_int + ir
                return reward_ems, reward_ems

            reward_ems_final, rffs = jax.lax.scan(
                rff_step, agent.reward_norm.reward_ems, traj.int_reward
            )
            m, v, c = welford_update(agent.reward_norm, rffs.reshape(-1))
            agent = agent.replace(
                reward_norm=RewardNormState(
                    reward_ems=reward_ems_final, mean=m, var=v, count=c
                )
            )
            norm_int = traj.int_reward / jnp.sqrt(agent.reward_norm.var + 1e-8)
            last_ve, last_vi = value(agent, last_obs)

            def gae_ext(carry, td):
                gae, nv = carry
                done, v_, r_ = td
                delta = r_ + cfg.gamma_ext * nv * (1 - done) - v_
                gae = delta + cfg.gamma_ext * cfg.gae_lambda * (1 - done) * gae
                return (gae, v_), gae

            _, adv_ext = jax.lax.scan(
                gae_ext,
                (jnp.zeros_like(last_ve), last_ve),
                (traj.done, traj.value_ext, traj.reward),
                reverse=True,
                unroll=16,
            )
            tgt_ext = adv_ext + traj.value_ext

            def gae_int(carry, td):
                gae, nv = carry
                v_, r_ = td
                delta = r_ + cfg.gamma_int * nv - v_
                gae = delta + cfg.gamma_int * cfg.gae_lambda * gae
                return (gae, v_), gae

            _, adv_int = jax.lax.scan(
                gae_int,
                (jnp.zeros_like(last_vi), last_vi),
                (traj.value_int, norm_int),
                reverse=True,
                unroll=16,
            )
            tgt_int = adv_int + traj.value_int
            advantages = cfg.ext_coef * adv_ext + cfg.int_coef * adv_int

            agent, info, rng = update(
                cfg, agent, traj, advantages, tgt_ext, tgt_int, rng
            )
            return (agent, env_state, last_obs, rng), {**info, "xy": traj.xy}

        (agent, env_state, last_obs, rng), trace = jax.lax.scan(
            one_update,
            (agent, env_state, last_obs, rng),
            None,
            updates_per_chunk,
        )
        return agent, env_state, last_obs, rng, trace

    @jax.jit
    def evaluate(agent, rng):
        rrngs = jax.random.split(rng, cfg.num_eval_episodes)
        obs0, st0 = jax.vmap(eval_env.reset, in_axes=(0, None))(rrngs, None)

        def step(carry, _):
            rng, obs, st, fin, ret = carry
            rng, sub = jax.random.split(rng)
            pi, _, _ = agent.train_state.apply_fn(
                agent.train_state.params["policy"], obs
            )
            action = pi.mode()
            srngs = jax.random.split(sub, cfg.num_eval_episodes)
            no, st, r, d, _ = jax.vmap(eval_env.step, in_axes=(0, 0, 0, None))(
                srngs, st, action, None
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
    chunk_v = jax.vmap(chunk)
    eval_v = jax.vmap(evaluate)

    rngs = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.num_seeds)
    agent, env_state, obs, rngs = init_v(rngs)

    from wrappers import VisitationHistogram

    hists = [VisitationHistogram() for _ in range(cfg.num_seeds)]

    wandb.init(project=cfg.wandb_project, config=vars(cfg))
    timestep = 0
    for c in range(num_chunks):
        agent, env_state, obs, rngs, trace = chunk_v(agent, env_state, obs, rngs)
        timestep += updates_per_chunk * steps_per_update
        # trace["xy"]: (num_seeds, updates_per_chunk, num_steps, num_envs, 2)
        xy = np.asarray(trace["xy"])
        for i in range(cfg.num_seeds):
            hists[i].add(xy[i])
        eval_rngs = jax.random.split(
            jax.random.PRNGKey(cfg.seed + 1000 + c), cfg.num_seeds
        )
        ret = eval_v(agent, eval_rngs)
        m = float(ret.mean())
        log_dict = {
            "eval/return": m,
            **{
                f"train/{k}": float(jnp.mean(trace[k]))
                for k in (
                    "pg_loss",
                    "vf_loss_ext",
                    "vf_loss_int",
                    "entropy",
                    "aux_loss",
                )
            },
            "step": timestep,
        }
        for i in range(cfg.num_seeds):
            log_dict[f"heatmap/seed_{i}"] = hists[i].wandb_image(
                title=f"rnd heatmap, seed={i}, step={timestep}"
            )
        wandb.log(log_dict, step=timestep)
        print(f"step={timestep:>8d}  eval_return={m:.2f}")

    # End-of-training video for seed 0.
    from wrappers import render_brax_video

    video_env = ClipAction(BraxGymnaxWrapper(cfg.env))
    seed0_pol = jax.tree_util.tree_map(
        lambda x: x[0], agent.train_state.params["policy"]
    )

    @jax.jit
    def video_rollout(rng):
        reset_rng, scan_rng = jax.random.split(rng)
        obs0, st0 = video_env.reset(reset_rng)

        def body(carry, _):
            rng, obs, st = carry
            rng, sub = jax.random.split(rng)
            pi, _, _ = agent.train_state.apply_fn(seed0_pol, obs)
            new_obs, new_st, _, _, _ = video_env.step(sub, st, pi.mode())
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