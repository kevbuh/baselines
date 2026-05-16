# Paper:          https://arxiv.org/abs/1906.05274
# Reference impl: https://github.com/RLAgent/state-marginal-matching

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import dataclass  # noqa: E402
from functools import partial  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import distrax  # noqa: E402
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
    # SAC core
    lr: float = 3e-4
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    init_log_alpha: float = 0.0
    target_entropy_scale: float = 1.0
    # SMM
    num_skills: int = 4  # K
    code_dim: int = 128  # VAE latent dim
    vae_hidden: int = 150
    vae_lr: float = 1e-3
    disc_lr: float = 1e-3
    vae_beta: float = 0.5
    rl_coef: float = 1.0
    state_ent_coef: float = 1.0
    latent_ent_coef: float = 1.0
    latent_cond_ent_coef: float = 1.0
    # Off-policy
    buffer_size: int = 1_000_000
    batch_size: int = 256
    update_every_steps: int = 1
    # Train
    seed: int = 0
    num_seeds: int = 1
    env: str = "hopper"
    num_envs: int = 16
    total_timesteps: int = 1_000_000
    num_init_steps: int = 4_000
    # Eval
    eval_every: int = 100_000
    num_eval_episodes: int = 10
    eval_episode_length: int = 1000
    # Video
    video_episode_length: int = 500
    video_height: int = 240
    video_width: int = 320
    video_fps: int = 30
    # Wandb
    wandb_project: str = "smm-sac-brax"


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs_z):
        h = nn.relu(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(obs_z)
        )
        h = nn.relu(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(h)
        )
        mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(h)
        log_std = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(h)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        base = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), 1))


class TwinCritic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs_z, action):
        x = jnp.concatenate([obs_z, action], axis=-1)

        def q(name):
            h = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f"{name}_0",
                )(x)
            )
            h = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f"{name}_1",
                )(h)
            )
            return jnp.squeeze(
                nn.Dense(1, kernel_init=orthogonal(1.0), name=f"{name}_2")(h), -1
            )

        return q("Q1"), q("Q2")


class Discriminator(nn.Module):
    # q_phi(z|s_env): logits over K skills
    num_skills: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs):
        h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(obs))
        h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(h))
        return nn.Dense(self.num_skills, kernel_init=orthogonal(0.01))(h)


class VAE(nn.Module):
    # enc/dec MLPs with code_dim
    input_dim: int
    code_dim: int = 128
    hidden: int = 150

    @nn.compact
    def __call__(self, x, eps):
        h = nn.relu(
            nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), name="enc_0")(x)
        )
        h = nn.relu(
            nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), name="enc_1")(h)
        )
        mu = nn.Dense(self.code_dim, kernel_init=orthogonal(0.01), name="enc_mu")(h)
        logvar = nn.Dense(
            self.code_dim, kernel_init=orthogonal(0.01), name="enc_logvar"
        )(h)
        std = jnp.exp(0.5 * logvar)
        code = mu + eps * std
        d = nn.relu(
            nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), name="dec_0")(
                code
            )
        )
        d = nn.relu(
            nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), name="dec_1")(d)
        )
        recon = nn.Dense(self.input_dim, kernel_init=orthogonal(0.01), name="dec_2")(d)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(NamedTuple):
    actor: TrainState
    critic: TrainState
    critic_target_params: Any
    discriminator: TrainState
    vae: TrainState
    log_alpha: TrainState
    target_entropy: float


def soft_update(p, t, tau):
    return jax.tree_util.tree_map(lambda x, y: tau * x + (1.0 - tau) * y, p, t)


def sample_skill(rng, num_skills, batch_size):
    idx = jax.random.randint(rng, (batch_size,), 0, num_skills)
    return jax.nn.one_hot(idx, num_skills, dtype=jnp.float32)


def init(cfg: Config, rng, obs_dim: int, action_dim: int):
    rng_a, rng_c, rng_d, rng_v = jax.random.split(rng, 4)
    actor_net = SquashedGaussianActor(action_dim, cfg.hidden_dim)
    critic_net = TwinCritic(cfg.hidden_dim)
    disc_net = Discriminator(cfg.num_skills, cfg.hidden_dim)
    vae_net = VAE(input_dim=obs_dim, code_dim=cfg.code_dim, hidden=cfg.vae_hidden)

    obs_z = jnp.zeros((1, obs_dim + cfg.num_skills))
    action = jnp.zeros((1, action_dim))
    obs = jnp.zeros((1, obs_dim))
    eps = jnp.zeros((1, cfg.code_dim))

    actor_p = actor_net.init(rng_a, obs_z)
    critic_p = critic_net.init(rng_c, obs_z, action)
    disc_p = disc_net.init(rng_d, obs)
    vae_p = vae_net.init(rng_v, obs, eps)

    tx_main = lambda lr: optax.adam(lr)
    log_alpha_init = {"log_alpha": jnp.float32(cfg.init_log_alpha)}
    return AgentState(
        actor=TrainState.create(
            apply_fn=actor_net.apply, params=actor_p, tx=tx_main(cfg.lr)
        ),
        critic=TrainState.create(
            apply_fn=critic_net.apply, params=critic_p, tx=tx_main(cfg.lr)
        ),
        critic_target_params=critic_p,
        discriminator=TrainState.create(
            apply_fn=disc_net.apply, params=disc_p, tx=tx_main(cfg.disc_lr)
        ),
        vae=TrainState.create(
            apply_fn=vae_net.apply, params=vae_p, tx=tx_main(cfg.vae_lr)
        ),
        log_alpha=TrainState.create(
            apply_fn=lambda p: p["log_alpha"], params=log_alpha_init, tx=tx_main(cfg.lr)
        ),
        target_entropy=-float(action_dim) * cfg.target_entropy_scale,
    )


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------


def update_step(cfg: Config, agent: AgentState, batch, rng):
    obs = batch["obs"]
    action = batch["action"]
    next_obs = batch["next_obs"]
    discount = batch["discount"]
    skill = batch["skill"]
    obs_z = jnp.concatenate([obs, skill], axis=-1)
    next_obs_z = jnp.concatenate([next_obs, skill], axis=-1)
    z_idx = jnp.argmax(skill, axis=-1)
    B = obs.shape[0]
    rng, r_vae, r_next, r_actor = jax.random.split(rng, 4)
    eps_vae = jax.random.normal(r_vae, (B, cfg.code_dim))

    # VAE update
    def vae_loss_fn(p):
        recon, mu, logvar = agent.vae.apply_fn(p, obs, eps_vae)
        # KL(q(z|x) || N(0,I)) per sample, mean over batch
        kl = -0.5 * jnp.mean(
            jnp.sum(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)
        )
        # Reconstruction MSE
        recon_mse = jnp.mean(jnp.square(obs - recon))
        loss = cfg.vae_beta * kl - (-recon_mse)
        return loss, (kl, recon_mse)

    (vae_loss, (vae_kl, vae_recon)), v_grads = jax.value_and_grad(
        vae_loss_fn, has_aux=True
    )(agent.vae.params)
    new_vae = agent.vae.apply_gradients(grads=v_grads)

    # h_s_z = -log p(s) per sample
    recon_post, _, _ = new_vae.apply_fn(new_vae.params, obs, eps_vae)
    log_prob_per_sample = -jnp.sum(jnp.square(obs - recon_post), axis=-1)
    h_s_z = jax.lax.stop_gradient(-log_prob_per_sample)  # (B,)

    # Discriminator update + h_z_s = CE(z, q_phi)
    def disc_loss_fn(d_params):
        logits = agent.discriminator.apply_fn(d_params, obs)
        log_softmax = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_softmax[jnp.arange(B), z_idx])

    disc_loss, d_grads = jax.value_and_grad(disc_loss_fn)(agent.discriminator.params)
    new_disc = agent.discriminator.apply_gradients(grads=d_grads)
    logits_post = jax.lax.stop_gradient(new_disc.apply_fn(new_disc.params, obs))
    log_softmax_post = jax.nn.log_softmax(logits_post, axis=-1)
    h_z_s = -log_softmax_post[jnp.arange(B), z_idx]  # per-sample CE = (B,)

    # SMM reward
    h_z = jnp.log(jnp.float32(cfg.num_skills))
    intr_reward = (
        cfg.rl_coef * batch["reward"]
        + cfg.state_ent_coef * h_s_z
        + cfg.latent_ent_coef * h_z * jnp.ones_like(h_s_z)
        + cfg.latent_cond_ent_coef * h_z_s
    )

    alpha = jnp.exp(agent.log_alpha.params["log_alpha"])

    # Critic update
    pi_next = agent.actor.apply_fn(agent.actor.params, next_obs_z)
    next_action, next_log_prob = pi_next.sample_and_log_prob(seed=r_next)
    tq1, tq2 = agent.critic.apply_fn(
        agent.critic_target_params, next_obs_z, next_action
    )
    target_q = jax.lax.stop_gradient(
        intr_reward + discount * (jnp.minimum(tq1, tq2) - alpha * next_log_prob)
    )

    def critic_loss_fn(c_params):
        q1, q2 = agent.critic.apply_fn(c_params, obs_z, action)
        return 0.5 * jnp.mean((q1 - target_q) ** 2) + 0.5 * jnp.mean(
            (q2 - target_q) ** 2
        )

    critic_loss, c_grads = jax.value_and_grad(critic_loss_fn)(agent.critic.params)
    new_critic = agent.critic.apply_gradients(grads=c_grads)

    # Actor update
    def actor_loss_fn(a_params):
        pi = agent.actor.apply_fn(a_params, obs_z)
        a, log_prob = pi.sample_and_log_prob(seed=r_actor)
        q1, q2 = agent.critic.apply_fn(new_critic.params, obs_z, a)
        return jnp.mean(alpha * log_prob - jnp.minimum(q1, q2)), log_prob

    (actor_loss, log_prob_a), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        agent.actor.params
    )
    new_actor = agent.actor.apply_gradients(grads=a_grads)

    # Alpha update
    def alpha_loss_fn(p):
        return -jnp.mean(p["log_alpha"] * (log_prob_a + agent.target_entropy))

    alpha_loss, al_grads = jax.value_and_grad(alpha_loss_fn)(agent.log_alpha.params)
    new_log_alpha = agent.log_alpha.apply_gradients(grads=al_grads)

    new_target = soft_update(new_critic.params, agent.critic_target_params, cfg.tau)
    new_agent = agent._replace(
        actor=new_actor,
        critic=new_critic,
        critic_target_params=new_target,
        discriminator=new_disc,
        vae=new_vae,
        log_alpha=new_log_alpha,
    )
    return new_agent, {
        "vae_loss": vae_loss,
        "vae_kl": vae_kl,
        "vae_recon": vae_recon,
        "disc_loss": disc_loss,
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "alpha": alpha,
        "intr_reward": intr_reward.mean(),
        "h_s_z": h_s_z.mean(),
        "h_z_s": h_z_s.mean(),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(cfg: Config):
    import flashbax as fbx
    import wandb

    env = VecEnv(ClipAction(LogWrapper(BraxGymnaxWrapper(cfg.env))))
    eval_env = ClipAction(BraxGymnaxWrapper(cfg.env))
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
        "skill": jnp.zeros(cfg.num_skills),
    }

    @jax.jit
    def init_per_seed(rng):
        rng, ri, rr, rk = jax.random.split(rng, 4)
        agent = init(cfg, ri, obs_dim, action_dim)
        rs = jax.random.split(rr, cfg.num_envs)
        obs, st = env.reset(rs, None)
        skill = sample_skill(rk, cfg.num_skills, cfg.num_envs)
        return agent, st, obs, buffer.init(dummy), rng, skill

    @partial(jax.jit, static_argnames=("num_iters", "warmup"))
    def chunk(agent, st, obs, buf, rng, skill, num_iters, warmup):
        def micro(carry, _):
            agent, st, obs, buf, rng, skill = carry
            rng, ra, rs, ru, rk = jax.random.split(rng, 5)

            obs_z = jnp.concatenate([obs, skill], axis=-1)
            pi = agent.actor.apply_fn(agent.actor.params, obs_z)
            a_agent = pi.sample(seed=ra)
            a_rand = jax.random.uniform(ra, a_agent.shape, minval=-1.0, maxval=1.0)
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
                    "skill": skill,
                },
            )
            new_skill = sample_skill(rk, cfg.num_skills, cfg.num_envs)
            skill = jnp.where(done[:, None], new_skill, skill)

            n_upd = max(1, cfg.num_envs // cfg.update_every_steps)

            def one_upd(c_u, _):
                a, r = c_u
                r, b, u = jax.random.split(r, 3)
                batch = buffer.sample(buf, b).experience
                a, info = update_step(cfg, a, batch, u)
                return (a, r), info

            def do_upd(args):
                a, r = args
                (a, r), info = jax.lax.scan(one_upd, (a, r), None, length=n_upd)
                return a, r, info

            keys = (
                "vae_loss",
                "vae_kl",
                "vae_recon",
                "disc_loss",
                "critic_loss",
                "actor_loss",
                "alpha_loss",
                "alpha",
                "intr_reward",
                "h_s_z",
                "h_z_s",
            )

            def skip_upd(args):
                a, r = args
                empty = {k: jnp.zeros(n_upd) for k in keys}
                return a, r, empty

            ready = buffer.can_sample(buf) & (~warmup)
            agent, ru, info = jax.lax.cond(ready, do_upd, skip_upd, (agent, ru))
            info = jax.tree_util.tree_map(jnp.mean, info)
            return (agent, st, next_obs, buf, rng, skill), {
                "reward": reward.mean(),
                "xy": get_xy(st),
                **info,
            }

        carry, trace = jax.lax.scan(
            micro, (agent, st, obs, buf, rng, skill), None, num_iters
        )
        return (*carry, trace)

    @jax.jit
    def evaluate(agent, rng):
        rrngs = jax.random.split(rng, cfg.num_eval_episodes)
        obs0, st0 = jax.vmap(eval_env.reset, in_axes=(0, None))(rrngs, None)
        skill0 = jax.nn.one_hot(
            jnp.zeros(cfg.num_eval_episodes, jnp.int32),
            cfg.num_skills,
            dtype=jnp.float32,
        )

        def step(carry, _):
            rng, obs, st, fin, ret = carry
            rng, sub = jax.random.split(rng)
            obs_z = jnp.concatenate([obs, skill0], axis=-1)
            pi = agent.actor.apply_fn(agent.actor.params, obs_z)
            action = jnp.tanh(pi.distribution.mean())
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
    chunk_v = jax.vmap(chunk, in_axes=(0, 0, 0, 0, 0, 0, None, None))
    eval_v = jax.vmap(evaluate)

    rngs = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.num_seeds)
    agent, env_state, obs, buf, rngs, skill = init_v(rngs)

    from wrappers import VisitationHistogram

    hists = [VisitationHistogram() for _ in range(cfg.num_seeds)]

    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    iters = max(1, cfg.eval_every // cfg.num_envs)
    n_chunks = max(1, cfg.total_timesteps // cfg.eval_every)
    init_iters = max(1, cfg.num_init_steps // cfg.num_envs)
    if init_iters < iters:
        agent, env_state, obs, buf, rngs, skill, _ = chunk_v(
            agent, env_state, obs, buf, rngs, skill, init_iters, True
        )

    timestep = init_iters * cfg.num_envs
    for c in range(n_chunks):
        agent, env_state, obs, buf, rngs, skill, trace = chunk_v(
            agent, env_state, obs, buf, rngs, skill, iters, False
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
                for k in (
                    "vae_loss",
                    "vae_kl",
                    "vae_recon",
                    "disc_loss",
                    "critic_loss",
                    "actor_loss",
                    "alpha_loss",
                    "alpha",
                    "intr_reward",
                    "h_s_z",
                    "h_z_s",
                )
            },
            "step": timestep,
        }
        for i in range(cfg.num_seeds):
            log_dict[f"heatmap/seed_{i}"] = hists[i].wandb_image(
                title=f"smm heatmap, seed={i}, step={timestep}"
            )
        wandb.log(log_dict, step=timestep)
        print(f"step={timestep:>7d}  eval_return={m:.2f}")

    from wrappers import render_brax_video

    video_env = ClipAction(BraxGymnaxWrapper(cfg.env))
    seed0_actor = jax.tree_util.tree_map(lambda x: x[0], agent.actor.params)
    skill0 = jax.nn.one_hot(jnp.zeros(1, jnp.int32), cfg.num_skills, dtype=jnp.float32)

    @jax.jit
    def video_rollout(rng):
        reset_rng, scan_rng = jax.random.split(rng)
        obs0, st0 = video_env.reset(reset_rng)

        def body(carry, _):
            rng, obs, st = carry
            rng, sub = jax.random.split(rng)
            obs_z = jnp.concatenate([obs[None], skill0], axis=-1)[0]
            pi = agent.actor.apply_fn(seed0_actor, obs_z)
            action = jnp.tanh(pi.distribution.mean())
            new_obs, new_st, _, _, _ = video_env.step(sub, st, action)
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