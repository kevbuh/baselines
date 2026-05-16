# based on https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_continuous_action.py

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import dataclass  # noqa: E402
from functools import partial  # noqa: E402
from typing import Any, NamedTuple, Sequence  # noqa: E402

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from wrappers import (  # noqa: E402
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    VecEnv,
    get_xy,
)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return pi, jnp.squeeze(critic, -1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any
    xy: jnp.ndarray


class PPOAgent(flax.struct.PyTreeNode):
    rng: Any
    train_state: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def loss_fn(agent, params, traj_batch, gae, targets):
        pi, value = agent.train_state.apply_fn(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -agent.config["clip_eps"], agent.config["clip_eps"]
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio, 1.0 - agent.config["clip_eps"], 1.0 + agent.config["clip_eps"]
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + agent.config["vf_coef"] * value_loss
            - agent.config["ent_coef"] * entropy
        )
        return total_loss, {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "actor_loss": loss_actor,
            "entropy": entropy,
        }

    @jax.jit
    def update(agent, traj_batch, advantages, targets):
        num_envs = agent.config["num_envs"]
        num_steps = agent.config["num_steps"]
        num_minibatches = agent.config["num_minibatches"]
        update_epochs = agent.config["update_epochs"]
        batch_size = num_steps * num_envs

        def _update_epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                shuffled,
            )

            def _update_mb(train_state, mb):
                mb_traj, mb_adv, mb_targets = mb
                (_, info), grads = jax.value_and_grad(agent.loss_fn, has_aux=True)(
                    train_state.params,
                    mb_traj,
                    mb_adv,
                    mb_targets,
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, info

            train_state, info = jax.lax.scan(_update_mb, train_state, minibatches)
            return (train_state, rng), info

        rng, _rng = jax.random.split(agent.rng)
        (new_train_state, _), info = jax.lax.scan(
            _update_epoch, (agent.train_state, _rng), None, update_epochs
        )
        info = jax.tree_util.tree_map(lambda x: x.mean(), info)
        return agent.replace(rng=rng, train_state=new_train_state), info

    @jax.jit
    def act(agent, obs, rng):
        rng, act_rng = jax.random.split(rng)
        pi, value = agent.train_state.apply_fn(agent.train_state.params, obs)
        action = pi.sample(seed=act_rng)
        return action, pi.log_prob(action), value, rng

    @jax.jit
    def value(agent, obs):
        _, v = agent.train_state.apply_fn(agent.train_state.params, obs)
        return v


def create_learner(config, seed, obs_shape, action_dim, num_total_updates=None):
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    network = ActorCritic(action_dim, activation=config.activation)
    init_x = jnp.zeros(obs_shape)
    params = network.init(init_rng, init_x)

    if config.anneal_lr and num_total_updates is not None:

        def lr_schedule(count):
            frac = (
                1.0
                - (count // (config.num_minibatches * config.update_epochs))
                / num_total_updates
            )
            return config.lr * frac

        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr, eps=1e-5),
        )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    agent_config = flax.core.FrozenDict(
        dict(
            clip_eps=config.clip_eps,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            num_envs=config.num_envs,
            num_steps=config.num_steps,
            num_minibatches=config.num_minibatches,
            update_epochs=config.update_epochs,
        )
    )

    return PPOAgent(rng=rng, train_state=train_state, config=agent_config)


@dataclass
class Config:
    # ppo
    lr: float = 3e-4
    num_envs: int = 2048
    num_steps: int = 10
    update_epochs: int = 4
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    activation: str = "tanh"
    anneal_lr: bool = False
    normalize_env: bool = True

    # train
    seed: int = 0
    env: str = "ant"
    wandb_project: str = "rnd-jax-debug"
    total_timesteps: int = 50_000_000

    # eval
    eval_every: int = 500_000
    num_eval_episodes: int = 100
    eval_episode_length: int = 1000

    # video
    video_episode_length: int = 500
    video_height: int = 240
    video_width: int = 320
    video_fps: int = 30


if __name__ == "__main__":
    import tyro
    import wandb

    from wrappers import VisitationHistogram, render_brax_video

    def main(cfg: Config):
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

        env = VecEnv(ClipAction(LogWrapper(BraxGymnaxWrapper(cfg.env))))
        if cfg.normalize_env:
            env = NormalizeVecObservation(env)
            env = NormalizeVecReward(env, cfg.gamma)

        obs_shape = env.observation_space(None).shape
        action_dim = env.action_space(None).shape[0]
        network = ActorCritic(action_dim, activation=cfg.activation)

        steps_per_update = cfg.num_steps * cfg.num_envs
        total_updates = int(cfg.total_timesteps // steps_per_update)
        updates_per_chunk = max(1, int(cfg.eval_every // steps_per_update))
        num_chunks = total_updates // updates_per_chunk

        agent = create_learner(
            cfg, cfg.seed, obs_shape, action_dim, num_total_updates=total_updates
        )

        @jax.jit
        def train_chunk(agent, env_state, last_obs, rng):
            def _one_update(carry, _):
                agent, env_state, last_obs, rng = carry

                def _env_step(step_carry, _):
                    env_state, last_obs, rng = step_carry
                    rng, step_rng = jax.random.split(rng)
                    action, log_prob, value, rng = agent.act(last_obs, rng)
                    step_rngs = jax.random.split(step_rng, cfg.num_envs)
                    obsv, env_state, reward, done, info = env.step(
                        step_rngs, env_state, action, None
                    )
                    transition = Transition(
                        done,
                        action,
                        value,
                        reward,
                        log_prob,
                        last_obs,
                        info,
                        get_xy(env_state),
                    )
                    return (env_state, obsv, rng), transition

                (env_state, last_obs, rng), traj_batch = jax.lax.scan(
                    _env_step, (env_state, last_obs, rng), None, cfg.num_steps
                )
                last_val = agent.value(last_obs)

                def _gae_step(carry, tr):
                    gae, next_value = carry
                    delta = (
                        tr.reward + cfg.gamma * next_value * (1 - tr.done) - tr.value
                    )
                    gae = delta + cfg.gamma * cfg.gae_lambda * (1 - tr.done) * gae
                    return (gae, tr.value), gae

                _, advantages = jax.lax.scan(
                    _gae_step,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                targets = advantages + traj_batch.value

                agent, train_info = agent.update(traj_batch, advantages, targets)
                return (agent, env_state, last_obs, rng), (
                    traj_batch.info,
                    train_info,
                    traj_batch.xy,
                )

            (agent, env_state, last_obs, rng), metrics = jax.lax.scan(
                _one_update, (agent, env_state, last_obs, rng), None, updates_per_chunk
            )
            return agent, env_state, last_obs, rng, metrics

        eval_env = ClipAction(BraxGymnaxWrapper(cfg.env))

        @partial(jax.jit, static_argnames=("policy_fn",))
        def eval_fn(policy_fn, params, rng):
            reset_rngs = jax.random.split(rng, cfg.num_eval_episodes)
            obs, env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(
                reset_rngs, None
            )

            def _step(carry, _):
                rng, obs, env_state, finished, ep_returns, cum_reward = carry
                action = policy_fn(params, obs)
                rng, step_rng = jax.random.split(rng)
                step_rngs = jax.random.split(step_rng, cfg.num_eval_episodes)
                obs, env_state, reward, done, _ = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0, None)
                )(step_rngs, env_state, action, None)
                cum_reward = cum_reward + reward * ~finished
                newly_done = done & ~finished
                ep_returns = jnp.where(newly_done, cum_reward, ep_returns)
                finished = finished | done
                return (rng, obs, env_state, finished, ep_returns, cum_reward), None

            init = (
                rng,
                obs,
                env_state,
                jnp.zeros(cfg.num_eval_episodes, dtype=jnp.bool_),
                jnp.zeros(cfg.num_eval_episodes),
                jnp.zeros(cfg.num_eval_episodes),
            )
            final, _ = jax.lax.scan(_step, init, None, length=cfg.eval_episode_length)
            _, _, _, finished, ep_returns, cum_reward = final
            return jnp.where(finished, ep_returns, cum_reward)

        @partial(jax.jit, static_argnames=("policy_fn",))
        def video_fn(policy_fn, params, rng):
            reset_rng, scan_rng = jax.random.split(rng)
            obs0, state0 = eval_env.reset(reset_rng)

            def body(carry, _):
                rng, obs, state = carry
                rng, step_rng = jax.random.split(rng)
                action = policy_fn(params, obs)
                new_obs, new_state, _, _, _ = eval_env.step(step_rng, state, action)
                return (rng, new_obs, new_state), new_state.pipeline_state

            _, pipeline_states = jax.lax.scan(
                body, (scan_rng, obs0, state0), None, length=cfg.video_episode_length
            )
            return pipeline_states

        def policy_fn(eval_state, obs):
            net_obs = (obs - eval_state["mean"]) / jnp.sqrt(eval_state["var"] + 1e-8)
            pi, _ = network.apply(eval_state["params"], net_obs)
            return pi.mode()

        def make_eval_state(env_state):
            if cfg.normalize_env:
                mean = env_state.env_state.mean[0]
                var = env_state.env_state.var[0]
            else:
                mean = jnp.zeros(obs_shape)
                var = jnp.ones(obs_shape)
            return {"params": agent.train_state.params, "mean": mean, "var": var}

        hist = VisitationHistogram()

        def run_eval(env_state, timestep):
            returns = eval_fn(
                policy_fn, make_eval_state(env_state), jax.random.PRNGKey(cfg.seed)
            )
            mean_ret = float(returns.mean())
            std_ret = float(returns.std())
            print(f"Step {timestep}: eval return = {mean_ret:.2f} (+/- {std_ret:.2f})")
            log_dict = {
                "eval/mean_return": mean_ret,
                "eval/std_return": std_ret,
                "timestep": timestep,
            }
            if hist.counts.sum() > 0:
                log_dict["heatmap/visitation"] = hist.wandb_image(
                    title=f"ppo heatmap, step={timestep}"
                )
            wandb.log(log_dict, step=timestep)

        rng = jax.random.PRNGKey(cfg.seed + 1)
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, cfg.num_envs)
        obs, env_state = env.reset(reset_rngs, None)

        run_eval(env_state, 0)

        for chunk_i in range(num_chunks):
            agent, env_state, obs, rng, metrics = train_chunk(
                agent, env_state, obs, rng
            )
            hist.add(np.asarray(metrics[2]))
            run_eval(env_state, (chunk_i + 1) * updates_per_chunk * steps_per_update)

        pipeline_states = video_fn(
            policy_fn, make_eval_state(env_state), jax.random.PRNGKey(cfg.seed)
        )
        render_brax_video(
            eval_env,
            pipeline_states,
            episode_length=cfg.video_episode_length,
            height=cfg.video_height,
            width=cfg.video_width,
            fps=cfg.video_fps,
        )

        wandb.finish()

    main(tyro.cli(Config))