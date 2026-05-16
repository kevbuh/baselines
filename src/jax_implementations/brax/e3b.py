# Paper:          https://arxiv.org/abs/2210.05805
# Reference impl: https://github.com/facebookresearch/e3b

# Differences from the discrete-action NetHack reference:
#   - Action distribution: MultivariateNormalDiag(mean, exp(log_std))
#   - Inverse-dynamics loss: MSE on continuous actions
#   - Encoder operates on continuous obs vectors. no glyph/CNN pipeline

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import dataclass  # noqa: E402
from functools import partial  # noqa: E402
from typing import NamedTuple  # noqa: E402

import distrax
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
    # IMPALA / V-trace
    lr: float = 5e-4
    predictor_lr: float = 1e-4
    rmsprop_decay: float = 0.99
    rmsprop_eps: float = 0.01
    rmsprop_momentum: float = 0.0
    max_grad_norm: float = 40.0
    discounting: float = 0.99
    baseline_cost: float = 0.5
    entropy_cost: float = 0.0005
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    # E3B
    hidden_dim: int = 1024
    embed_dim: int = 128
    ridge: float = 0.1
    intrinsic_reward_coef: float = 1.0
    reward_norm: str = "int"  # one of {"none", "int", "ext", "all"}
    log_std_min: float = -5.0  # clamp log_std to [min, max] for continuous stability
    log_std_max: float = 2.0
    clip_rewards: bool = False
    no_reward: bool = False
    # Rollout
    unroll_length: int = 80  # T steps per update
    num_envs: int = 16
    # Train
    seed: int = 0
    num_seeds: int = 1
    env: str = "hopper"
    total_timesteps: int = 5_000_000
    # Eval
    eval_every: int = 100_000
    num_eval_episodes: int = 10
    eval_episode_length: int = 1000
    video_episode_length: int = 500
    video_height: int = 240
    video_width: int = 320
    video_fps: int = 30
    wandb_project: str = "e3b-impala-brax"


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class StateEmbeddingNet(nn.Module):
    # produce an embedding of s
    hidden_dim: int = 1024
    embed_dim: int = 128

    @nn.compact
    def __call__(self, obs):
        x = jax.nn.relu(
            nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=zeros
            )(obs)
        )
        x = jax.nn.relu(
            nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=zeros
            )(x)
        )
        x = jax.nn.relu(
            nn.Dense(
                self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=zeros
            )(x)
        )
        x = jax.nn.relu(
            nn.Dense(
                self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=zeros
            )(x)
        )
        return nn.Dense(
            self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=zeros
        )(x)


class InverseDynamicsNet(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, h_t, h_next):
        x = jax.nn.relu(
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=zeros)(
                jnp.concatenate([h_t, h_next], axis=-1)
            )
        )
        return nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=zeros)(
            x
        )


class PolicyNet(nn.Module):
    # Continuous-action analogue of NetHackPolicyNet's policy + baseline heads

    action_dim: int
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, obs):
        h = jax.nn.relu(
            nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=zeros
            )(obs)
        )
        h = jax.nn.relu(
            nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=zeros
            )(h)
        )
        action_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=zeros
        )(h)
        value = jnp.squeeze(
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=zeros)(h), axis=-1
        )
        # Clamp log_std for stability
        log_std_raw = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std_raw, -5.0, 2.0)
        return action_mean, log_std, value


# ---------------------------------------------------------------------------
# Sherman-Morrison update + V-trace
# ---------------------------------------------------------------------------


def sherman_morrison_step(c_inv, phi):
    u = phi @ c_inv  # (D,)
    bonus = jnp.dot(u, phi)  # scalar
    new_c_inv = c_inv - jnp.outer(u, u) / (1.0 + bonus)
    return bonus, new_c_inv


def vtrace_from_logprobs(
    behavior_logp,
    target_logp,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho=1.0,
    clip_pg_rho=1.0,
):
    log_rhos = target_logp - behavior_logp
    rhos = jnp.exp(log_rhos)
    clipped_rhos = jnp.minimum(rhos, clip_rho)
    cs = jnp.minimum(rhos, 1.0)
    values_tp1 = jnp.concatenate([values[1:], bootstrap_value[None]], axis=0)
    deltas = clipped_rhos * (rewards + discounts * values_tp1 - values)

    def body(acc, td):
        delta, disc, c = td
        new_acc = delta + disc * c * acc
        return new_acc, new_acc

    _, vs_minus_v = jax.lax.scan(
        body, jnp.zeros_like(bootstrap_value), (deltas, discounts, cs), reverse=True
    )
    vs = vs_minus_v + values
    vs_tp1 = jnp.concatenate([vs[1:], bootstrap_value[None]], axis=0)
    clipped_pg_rhos = jnp.minimum(rhos, clip_pg_rho)
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_tp1 - values)
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(pg_advantages)


def welford_init():
    return (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))  # (sum, m2, count)


def welford_update(state, batch):
    s, m2, n = state
    bs = batch.size
    bmean = jnp.mean(batch)
    bm2 = jnp.sum(jnp.square(batch - bmean))
    new_n = n + bs
    delta = bmean - jnp.where(n > 0, s / n, 0.0)
    new_s = s + bmean * bs
    new_m2 = m2 + bm2 + jnp.square(delta) * (n * bs / new_n)
    return (new_s, new_m2, new_n)


def welford_std(state):
    _, m2, n = state
    return jnp.sqrt(m2 / jnp.maximum(n, 1.0))


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(NamedTuple):
    policy: TrainState
    encoder: TrainState
    inv_dyn: TrainState
    reward_norm: tuple  # welford state for intrinsic reward (sum, m2, count)


def init(cfg: Config, rng, obs_dim: int, action_dim: int) -> AgentState:
    rng_p, rng_e, rng_i = jax.random.split(rng, 3)
    pol = PolicyNet(action_dim, cfg.hidden_dim)
    enc = StateEmbeddingNet(cfg.hidden_dim, cfg.embed_dim)
    inv = InverseDynamicsNet(action_dim)

    o = jnp.zeros((1, obs_dim))
    e = jnp.zeros((1, cfg.embed_dim))

    pol_p = pol.init(rng_p, o)["params"]
    enc_p = enc.init(rng_e, o)["params"]
    inv_p = inv.init(rng_i, e, e)["params"]

    tx_policy = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.rmsprop(
            cfg.lr,
            decay=cfg.rmsprop_decay,
            momentum=cfg.rmsprop_momentum,
            eps=cfg.rmsprop_eps,
        ),
    )
    tx_aux = lambda: optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.predictor_lr),
    )
    return AgentState(
        policy=TrainState.create(apply_fn=pol.apply, params=pol_p, tx=tx_policy),
        encoder=TrainState.create(apply_fn=enc.apply, params=enc_p, tx=tx_aux()),
        inv_dyn=TrainState.create(apply_fn=inv.apply, params=inv_p, tx=tx_aux()),
        reward_norm=welford_init(),
    )


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------


def update_step(cfg: Config, agent: AgentState, traj: dict):
    """One IMPALA update step. `traj` keys (each (T, B, ...)):
    obs, action, reward, done, behavior_logp, last_obs (B, obs_dim).
    """
    obs_T = traj["obs"]  # (T, B, obs_dim)
    obs_Tp1 = jnp.concatenate(
        [obs_T, traj["last_obs"][None]], axis=0
    )  # (T+1, B, obs_dim)
    action_T = traj["action"]  # (T, B, A)

    # --- Per-env intrinsic reward normalization ---
    intr_T = traj["bonus"]  # (T, B), set by collector
    reward_norm = agent.reward_norm
    if cfg.reward_norm == "int":
        reward_norm = welford_update(reward_norm, intr_T)
        std = welford_std(reward_norm)
        intr_T = jnp.where(std > 0, intr_T / std, intr_T)

    rewards_T = traj["reward"]  # (T, B) extrinsic
    if cfg.no_reward:
        total_T = intr_T * cfg.intrinsic_reward_coef
    else:
        total_T = rewards_T + intr_T * cfg.intrinsic_reward_coef

    if cfg.reward_norm == "all":
        reward_norm = welford_update(reward_norm, total_T)
        std = welford_std(reward_norm)
        total_T = jnp.where(std > 0, total_T / std, total_T)

    if cfg.clip_rewards:
        total_T = jnp.clip(total_T, -1.0, 1.0)

    discounts_T = (1.0 - traj["done"].astype(jnp.float32)) * cfg.discounting

    def loss_fn(pol_params, enc_params, inv_params):
        # Encoder forward over the full T+1 rollout
        T_plus_1, B = obs_Tp1.shape[:2]
        flat = obs_Tp1.reshape(T_plus_1 * B, -1)
        emb_flat = agent.encoder.apply_fn({"params": enc_params}, flat)
        emb = emb_flat.reshape(T_plus_1, B, -1)
        h_t = emb[:-1]
        h_next = emb[1:]

        pred_action = agent.inv_dyn.apply_fn({"params": inv_params}, h_t, h_next)
        inv_loss = jnp.sum(
            jnp.mean(jnp.mean(jnp.square(pred_action - action_T), axis=-1), axis=1)
        )

        # Policy forward over full rollout for V-trace
        pol_flat = obs_Tp1.reshape(T_plus_1 * B, -1)
        mean_flat, log_std, value_flat = agent.policy.apply_fn(
            {"params": pol_params}, pol_flat
        )
        mean_all = mean_flat.reshape(T_plus_1, B, -1)
        value_all = value_flat.reshape(T_plus_1, B)
        std = jnp.broadcast_to(jnp.exp(log_std), mean_all.shape)

        # target log-probs of the behavior actions under the current target policy
        target_pi = distrax.MultivariateNormalDiag(mean_all[:-1], std[:-1])
        target_logp = target_pi.log_prob(action_T)  # (T, B)
        bootstrap_value = value_all[-1]  # (B,)
        values = value_all[:-1]  # (T, B)

        vs, pg_adv = vtrace_from_logprobs(
            traj["behavior_logp"],
            target_logp,
            discounts_T,
            total_T,
            values,
            bootstrap_value,
            clip_rho=cfg.clip_rho_threshold,
            clip_pg_rho=cfg.clip_pg_rho_threshold,
        )

        pg_loss = -jnp.sum(jnp.mean(target_logp * pg_adv, axis=1))
        baseline_loss = 0.5 * jnp.sum(jnp.mean(jnp.square(vs - values), axis=1))
        entropy = target_pi.entropy()  # (T, B)
        entropy_loss = -jnp.sum(jnp.mean(entropy, axis=1))

        total = (
            pg_loss
            + cfg.baseline_cost * baseline_loss
            + cfg.entropy_cost * entropy_loss
            + inv_loss
        )
        return total, {
            "pg_loss": pg_loss,
            "baseline_loss": baseline_loss,
            "entropy_loss": entropy_loss,
            "inv_dyn_loss": inv_loss,
            "total_loss": total,
            "intr_reward": intr_T.mean(),
            "ext_reward": rewards_T.mean(),
        }

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)
    (_, info), (g_pol, g_enc, g_inv) = grad_fn(
        agent.policy.params, agent.encoder.params, agent.inv_dyn.params
    )

    new_pol = agent.policy.apply_gradients(grads=g_pol)
    new_enc = agent.encoder.apply_gradients(grads=g_enc)
    new_inv = agent.inv_dyn.apply_gradients(grads=g_inv)
    return (
        agent._replace(
            policy=new_pol, encoder=new_enc, inv_dyn=new_inv, reward_norm=reward_norm
        ),
        info,
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(cfg: Config):
    import wandb

    env = VecEnv(ClipAction(LogWrapper(BraxGymnaxWrapper(cfg.env))))
    obs_dim = env.observation_space(None).shape[0]
    action_dim = env.action_space(None).shape[0]
    eval_env = ClipAction(BraxGymnaxWrapper(cfg.env))

    @jax.jit
    def init_per_seed(rng):
        rng, ri, rr = jax.random.split(rng, 3)
        agent = init(cfg, ri, obs_dim, action_dim)
        rs = jax.random.split(rr, cfg.num_envs)
        obs, env_state = env.reset(rs, None)
        # Per-env inverse covariance for SM bonus
        c_inv = jnp.broadcast_to(
            jnp.eye(cfg.embed_dim) * (1.0 / cfg.ridge),
            (cfg.num_envs, cfg.embed_dim, cfg.embed_dim),
        )
        return agent, env_state, obs, c_inv, rng

    @partial(jax.jit, static_argnames=("num_iters",))
    def chunk(agent, env_state, last_obs, c_inv, rng, num_iters):
        # rollout-of-T-steps + one IMPALA update

        def one_iter(carry, _):
            agent, env_state, last_obs, c_inv, rng = carry

            # Rollout T steps, computing SM bonus per-env per-step
            def rollout_step(rcarry, _):
                env_state, obs, c_inv, rng = rcarry
                rng, ra, rs = jax.random.split(rng, 3)

                # Forward policy + encoder
                mean, log_std, _ = agent.policy.apply_fn(
                    {"params": agent.policy.params}, obs
                )
                std = jnp.broadcast_to(jnp.exp(log_std), mean.shape)
                pi = distrax.MultivariateNormalDiag(mean, std)
                action = pi.sample(seed=ra)
                logp = pi.log_prob(action)

                phi = agent.encoder.apply_fn({"params": agent.encoder.params}, obs)

                # Per-env Sherman-Morrison
                bonus, new_c_inv = jax.vmap(sherman_morrison_step)(c_inv, phi)

                # Step env
                step_rngs = jax.random.split(rs, cfg.num_envs)
                next_obs, env_state, reward, done, _ = env.step(
                    step_rngs, env_state, action, None
                )

                # Reset inv-cov to (1/ridge)·I on episode boundary
                new_c_inv = jnp.where(
                    done[:, None, None],
                    jnp.eye(cfg.embed_dim) * (1.0 / cfg.ridge),
                    new_c_inv,
                )

                tr = {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "behavior_logp": logp,
                    "bonus": bonus,
                    "xy": get_xy(env_state),
                }
                return (env_state, next_obs, new_c_inv, rng), tr

            (env_state, last_obs, c_inv, rng), traj = jax.lax.scan(
                rollout_step, (env_state, last_obs, c_inv, rng), None, cfg.unroll_length
            )

            # Append final obs for V-trace bootstrap
            traj["last_obs"] = last_obs

            # One IMPALA update
            agent, info = update_step(cfg, agent, traj)
            return (
                (agent, env_state, last_obs, c_inv, rng),
                {"reward": traj["reward"].mean(), "xy": traj["xy"], **info},
            )

        carry, trace = jax.lax.scan(
            one_iter, (agent, env_state, last_obs, c_inv, rng), None, num_iters
        )
        return (*carry, trace)

    @jax.jit
    def evaluate(agent, rng):
        rrngs = jax.random.split(rng, cfg.num_eval_episodes)
        obs0, st0 = jax.vmap(eval_env.reset, in_axes=(0, None))(rrngs, None)

        def step(carry, _):
            rng, obs, st, fin, ret = carry
            rng, sub = jax.random.split(rng)
            mean, _, _ = agent.policy.apply_fn({"params": agent.policy.params}, obs)
            srngs = jax.random.split(sub, cfg.num_eval_episodes)
            no, st, r, d, _ = jax.vmap(eval_env.step, in_axes=(0, 0, 0, None))(
                srngs, st, mean, None
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
    chunk_v = jax.vmap(chunk, in_axes=(0, 0, 0, 0, 0, None))
    eval_v = jax.vmap(evaluate)

    rngs = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.num_seeds)
    agent, env_state, obs, c_inv, rngs = init_v(rngs)

    from wrappers import VisitationHistogram

    hists = [VisitationHistogram() for _ in range(cfg.num_seeds)]

    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    steps_per_iter = cfg.unroll_length * cfg.num_envs
    iters_per_chunk = max(1, cfg.eval_every // steps_per_iter)
    n_chunks = max(1, cfg.total_timesteps // cfg.eval_every)
    timestep = 0

    for c in range(n_chunks):
        agent, env_state, obs, c_inv, rngs, trace = chunk_v(
            agent, env_state, obs, c_inv, rngs, iters_per_chunk
        )
        timestep += iters_per_chunk * steps_per_iter
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
                for k in (
                    "pg_loss",
                    "baseline_loss",
                    "entropy_loss",
                    "inv_dyn_loss",
                    "intr_reward",
                    "ext_reward",
                )
            },
            "step": timestep,
        }
        for i in range(cfg.num_seeds):
            log_dict[f"heatmap/seed_{i}"] = hists[i].wandb_image(
                title=f"e3b heatmap, seed={i}, step={timestep}"
            )
        wandb.log(log_dict, step=timestep)
        print(f"step={timestep:>8d}  eval_return={m:.2f}")

    from wrappers import render_brax_video

    video_env = ClipAction(BraxGymnaxWrapper(cfg.env))
    seed0_pol = jax.tree_util.tree_map(lambda x: x[0], agent.policy.params)

    @jax.jit
    def video_rollout(rng):
        reset_rng, scan_rng = jax.random.split(rng)
        obs0, st0 = video_env.reset(reset_rng)

        def body(carry, _):
            rng, obs, st = carry
            rng, sub = jax.random.split(rng)
            mean, _, _ = agent.policy.apply_fn({"params": seed0_pol}, obs)
            new_obs, new_st, _, _, _ = video_env.step(sub, st, mean)
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