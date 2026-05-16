# Paper:          https://arxiv.org/abs/1705.05363
# Reference impl: https://github.com/pathak22/noreward-rl

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import dataclass  # noqa: E402
from typing import NamedTuple, Sequence  # noqa: E402

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
    lr: float = 1e-4
    num_envs: int = 128
    num_steps: int = 10
    update_epochs: int = 4
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.001
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    activation: str = "tanh"
    anneal_lr: bool = True
    normalize_adv: bool = True
    eta: float = 0.01
    forward_loss_wt: float = 0.2
    embed_dim: int = 256
    seed: int = 0
    num_seeds: int = 1
    env: str = "hopper"
    total_timesteps: int = 50_000_000
    eval_every: int = 500_000
    num_eval_episodes: int = 10
    eval_episode_length: int = 1000
    video_episode_length: int = 500
    video_height: int = 240
    video_width: int = 320
    video_fps: int = 30
    wandb_project: str = "icm-ppo-brax"


class ActorCritic(nn.Module):
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
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(h)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std))
        value = jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(h), -1)
        return pi, value


class StateEmbeddingNet(nn.Module):
    # Encoder phi: MLP analogue of ref UniverseHead conv stack

    embed_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.elu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.elu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)))(x)


class StateActionPredictor(nn.Module):
    # noreward-rl/src/model.py:StateActionPredictor — inverse + forward

    action_dim: int
    embed_dim: int = 256

    @nn.compact
    def __call__(self, phi_t, phi_tp1, action):
        inv_in = jnp.concatenate([phi_t, phi_tp1], axis=-1)
        inv_h = nn.relu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(inv_in))
        action_hat = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(inv_h)
        fwd_in = jnp.concatenate([phi_t, action], axis=-1)
        fwd_h = nn.relu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(fwd_in))
        phi_tp1_hat = nn.Dense(self.embed_dim, kernel_init=orthogonal(0.01))(fwd_h)
        return action_hat, phi_tp1_hat


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    xy: jnp.ndarray


class AgentState(flax.struct.PyTreeNode):
    train_state: TrainState


def init(cfg: Config, rng, obs_dim, action_dim, total_updates):
    rng, rp, re, rs = jax.random.split(rng, 4)
    actor_critic = ActorCritic(action_dim, activation=cfg.activation)
    encoder = StateEmbeddingNet(cfg.embed_dim)
    predictor = StateActionPredictor(action_dim, cfg.embed_dim)

    init_obs = jnp.zeros(obs_dim)
    init_phi = jnp.zeros(cfg.embed_dim)
    init_act = jnp.zeros(action_dim)

    params = {
        "policy": actor_critic.init(rp, init_obs),
        "encoder": encoder.init(re, init_obs),
        "predictor": predictor.init(rs, init_phi, init_phi, init_act),
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
    train_state = TrainState.create(apply_fn=actor_critic.apply, params=params, tx=tx)
    return AgentState(train_state=train_state), rng


def act(agent, obs, rng):
    rng, sub = jax.random.split(rng)
    pi, value = agent.train_state.apply_fn(agent.train_state.params["policy"], obs)
    action = pi.sample(seed=sub)
    return action, pi.log_prob(action), value, rng


def value(agent, obs):
    _, v = agent.train_state.apply_fn(agent.train_state.params["policy"], obs)
    return v


def encode(agent, obs, embed_dim=256):
    return StateEmbeddingNet(embed_dim).apply(agent.train_state.params["encoder"], obs)


def intrinsic_reward(agent, obs, next_obs, action, eta, embed_dim=256):
    phi_t = encode(agent, obs, embed_dim)
    phi_tp1 = encode(agent, next_obs, embed_dim)
    _, phi_tp1_hat = StateActionPredictor(action.shape[-1], embed_dim).apply(
        agent.train_state.params["predictor"], phi_t, phi_tp1, action
    )
    return 0.5 * jnp.mean(jnp.square(phi_tp1 - phi_tp1_hat), axis=-1) * eta


def loss_fn(cfg, params, traj, gae, targets):
    pi, vpred = ActorCritic(traj.action.shape[-1], cfg.activation).apply(
        params["policy"], traj.obs
    )
    log_prob = pi.log_prob(traj.action)
    vf_loss = (0.5 * cfg.vf_coef) * jnp.square(vpred - targets).mean()
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

    phi_t = StateEmbeddingNet(cfg.embed_dim).apply(params["encoder"], traj.obs)
    phi_tp1 = StateEmbeddingNet(cfg.embed_dim).apply(params["encoder"], traj.next_obs)
    action_hat, phi_tp1_hat = StateActionPredictor(
        traj.action.shape[-1], cfg.embed_dim
    ).apply(params["predictor"], phi_t, jax.lax.stop_gradient(phi_tp1), traj.action)
    inv_loss = jnp.mean(jnp.square(action_hat - traj.action))
    fwd_loss = 0.5 * jnp.mean(jnp.square(phi_tp1_hat - jax.lax.stop_gradient(phi_tp1)))
    icm_loss = cfg.forward_loss_wt * fwd_loss + (1.0 - cfg.forward_loss_wt) * inv_loss
    total = pg_loss + ent_loss + vf_loss + icm_loss
    return total, {
        "total_loss": total,
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "entropy": entropy,
        "inv_loss": inv_loss,
        "fwd_loss": fwd_loss,
    }


def update(cfg, agent, traj, advantages, targets, rng):
    bs = cfg.num_steps * cfg.num_envs

    def epoch(carry, _):
        ts, rng = carry
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, bs)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((bs,) + x.shape[2:]), (traj, advantages, targets)
        )
        shuf = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
        mbs = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [cfg.num_minibatches, -1] + list(x.shape[1:])),
            shuf,
        )

        def mb_step(ts, mb):
            mb_traj, mb_adv, mb_tgt = mb
            (_, info), grads = jax.value_and_grad(
                lambda p: loss_fn(cfg, p, mb_traj, mb_adv, mb_tgt), has_aux=True
            )(ts.params)
            return ts.apply_gradients(grads=grads), info

        ts, info = jax.lax.scan(mb_step, ts, mbs)
        return (ts, rng), info

    (new_ts, rng), info = jax.lax.scan(
        epoch, (agent.train_state, rng), None, cfg.update_epochs
    )
    info = jax.tree_util.tree_map(jnp.mean, info)
    return agent.replace(train_state=new_ts), info, rng


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
        return agent, env_state, obs, rng

    @jax.jit
    def chunk(agent, env_state, last_obs, rng):
        def one_update(carry, _):
            agent, env_state, last_obs, rng = carry

            def env_step(scarry, _):
                agent, env_state, obs, rng = scarry
                action, log_prob, val, rng = act(agent, obs, rng)
                rng, sub = jax.random.split(rng)
                srngs = jax.random.split(sub, cfg.num_envs)
                next_obs, env_state, reward, done, _ = env.step(
                    srngs, env_state, action, None
                )
                int_rew = intrinsic_reward(
                    agent, obs, next_obs, action, cfg.eta, cfg.embed_dim
                )
                tr = Transition(
                    done=done,
                    action=action,
                    value=val,
                    reward=reward,
                    int_reward=int_rew,
                    log_prob=log_prob,
                    obs=obs,
                    next_obs=next_obs,
                    xy=get_xy(env_state),
                )
                return (agent, env_state, next_obs, rng), tr

            (agent, env_state, last_obs, rng), traj = jax.lax.scan(
                env_step, (agent, env_state, last_obs, rng), None, cfg.num_steps
            )

            rewards = traj.reward + traj.int_reward
            last_v = value(agent, last_obs)

            def gae_step(carry, td):
                gae, nv = carry
                done, v_, r_ = td
                delta = r_ + cfg.gamma * nv * (1 - done) - v_
                gae = delta + cfg.gamma * cfg.gae_lambda * (1 - done) * gae
                return (gae, v_), gae

            _, advantages = jax.lax.scan(
                gae_step,
                (jnp.zeros_like(last_v), last_v),
                (traj.done, traj.value, rewards),
                reverse=True,
                unroll=16,
            )
            targets = advantages + traj.value
            agent, info, rng = update(cfg, agent, traj, advantages, targets, rng)
            return (agent, env_state, last_obs, rng), {
                **info,
                "xy": traj.xy,
                "intr_reward": traj.int_reward.mean(),
            }

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
            pi, _ = agent.train_state.apply_fn(agent.train_state.params["policy"], obs)
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
                    "vf_loss",
                    "entropy",
                    "inv_loss",
                    "fwd_loss",
                    "intr_reward",
                )
            },
            "step": timestep,
        }
        for i in range(cfg.num_seeds):
            log_dict[f"heatmap/seed_{i}"] = hists[i].wandb_image(
                title=f"icm heatmap, seed={i}, step={timestep}"
            )
        wandb.log(log_dict, step=timestep)
        print(f"step={timestep:>8d}  eval_return={m:.2f}")

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
            pi, _ = agent.train_state.apply_fn(seed0_pol, obs)
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