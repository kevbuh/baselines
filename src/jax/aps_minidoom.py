# Paper:          https://arxiv.org/abs/2108.13956
# Reference impl: https://github.com/rll-research/url_benchmark

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import functools
from dataclasses import dataclass
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

# dmc_benchmark.PRIMAL_TASKS
PRIMAL_TASKS = {
    "walker": "walker_stand",
    "jaco": "jaco_reach_top_left",
    "quadruped": "quadruped_walk",
}


def soft_update(src_params, tgt_params, tau: float):
    return jax.tree_util.tree_map(
        lambda s, t: tau * s + (1.0 - tau) * t, src_params, tgt_params
    )


def truncated_normal_sample(rng, mu, std, clip=None, low=-1.0, high=1.0, eps=1e-6):
    # Straight-through value clamp: forward returns clipped action, gradients
    # flow through the unclipped pre-activation
    noise = jax.random.normal(rng, mu.shape) * std
    if clip is not None:
        noise = jnp.clip(noise, -clip, clip)
    x = mu + noise
    clamped = jnp.clip(x, low + eps, high - eps)
    return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(clamped)


def normal_log_prob(x, mu, std):
    return -0.5 * jnp.square((x - mu) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)


class PixelEncoder(nn.Module):
    # Input uint8 [B, 9, 84, 84] (NCHW, frame-stacked) ->
    # [B, 32*35*35] = [B, 39200]. Norm: obs/255 - 0.5
    repr_dim: int = 32 * 35 * 35

    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32) / 255.0 - 0.5
        # NCHW -> NHWC at the Flax boundary
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.relu(
            nn.Conv(
                32,
                (3, 3),
                strides=(2, 2),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
                name="c1",
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                32,
                (3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
                name="c2",
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                32,
                (3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
                name="c3",
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                32,
                (3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
                name="c4",
            )(x)
        )
        # PyTorch flattens NCHW in (C,H,W) order; transpose back so downstream
        # Linear layers (copied from PyTorch in cross-checks) see the same stride
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x.reshape(x.shape[0], -1)


def random_shifts_aug(rng, x, pad: int = 4):
    # Replicate-pad, then per-image integer pixel shift in [0, 2*pad]. PyTorch's
    # grid_sample treats shift[0]=X (col), [1]=Y (row); we slice (row, col) to match
    B, C, H, W = x.shape
    assert H == W
    xp = jnp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
    shifts = jax.random.randint(rng, (B, 2), 0, 2 * pad + 1)

    def _shift_one(img, shift):
        return jax.lax.dynamic_slice(img, (0, shift[1], shift[0]), (C, H, W))

    return jax.vmap(_shift_one)(xp, shifts)


class Actor(nn.Module):
    # Pixel mode adds an extra policy_fc2 hidden layer
    action_dim: int
    obs_type: str = "states"
    hidden_dim: int = 1024
    feature_dim: int = 50

    @nn.compact
    def __call__(self, obs):
        feat_dim = self.hidden_dim if self.obs_type == "states" else self.feature_dim
        x = nn.Dense(
            feat_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="trunk_fc",
        )(obs)
        x = nn.LayerNorm(name="trunk_ln")(x)
        x = jnp.tanh(x)
        x = nn.relu(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="policy_fc1",
            )(x)
        )
        if self.obs_type == "pixels":
            x = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=constant(0.0),
                    name="policy_fc2",
                )(x)
            )
        mu = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="policy_head",
        )(x)
        return jnp.tanh(mu)


class CriticSF(nn.Module):
    # State path concats action BEFORE trunk; pixel path concats AFTER trunk
    # and adds an extra Q_fc2 hidden layer
    sf_dim: int
    obs_type: str = "states"
    hidden_dim: int = 1024
    feature_dim: int = 50

    @nn.compact
    def __call__(self, obs, action, task):
        if self.obs_type == "pixels":
            x = nn.Dense(
                self.feature_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="trunk_fc",
            )(obs)
            x = nn.LayerNorm(name="trunk_ln")(x)
            x = jnp.tanh(x)
            h = jnp.concatenate([x, action], axis=-1)
        else:
            h = jnp.concatenate([obs, action], axis=-1)
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="trunk_fc",
            )(h)
            h = nn.LayerNorm(name="trunk_ln")(h)
            h = jnp.tanh(h)

        def make_q(name: str):
            y = nn.relu(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=constant(0.0),
                    name=f"{name}_fc1",
                )(h)
            )
            if self.obs_type == "pixels":
                y = nn.relu(
                    nn.Dense(
                        self.hidden_dim,
                        kernel_init=orthogonal(1.0),
                        bias_init=constant(0.0),
                        name=f"{name}_fc2",
                    )(y)
                )
            return nn.Dense(
                self.sf_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name=f"{name}_head",
            )(y)

        q1_vec = make_q("Q1")
        q2_vec = make_q("Q2")
        q1 = jnp.einsum("bi,bi->b", task, q1_vec)
        q2 = jnp.einsum("bi,bi->b", task, q2_vec)
        return q1[..., None], q2[..., None]


class APSNet(nn.Module):
    sf_dim: int = 10
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, obs, norm: bool = True):
        x = nn.relu(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="fc1",
            )(obs)
        )
        x = nn.relu(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="fc2",
            )(x)
        )
        x = nn.Dense(
            self.sf_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="head",
        )(x)
        if norm:
            x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
        return x


class RMSState(flax.struct.PyTreeNode):
    M: jnp.ndarray
    S: jnp.ndarray
    n: jnp.ndarray


def rms_init(shape: Sequence[int] = (1,), epsilon: float = 1e-4) -> RMSState:
    return RMSState(
        M=jnp.zeros(shape),
        S=jnp.ones(shape),
        n=jnp.asarray(epsilon),
    )


def rms_update(state: RMSState, x):
    # ddof=1 to match torch.var's default unbiased=True
    bs = x.shape[0]
    delta = jnp.mean(x, axis=0) - state.M
    new_M = state.M + delta * bs / (state.n + bs)
    new_S = (
        state.S * state.n
        + jnp.var(x, axis=0, ddof=1) * bs
        + jnp.square(delta) * state.n * bs / (state.n + bs)
    ) / (state.n + bs)
    new_n = state.n + bs
    return RMSState(M=new_M, S=new_S, n=new_n), new_M, new_S


def pbe(rep, rms_state, knn_k: int, knn_avg: bool, knn_clip: float, knn_rms: bool):
    dist = jnp.linalg.norm(rep[:, None, :] - rep[None, :, :], axis=-1)
    neg_topk, _ = jax.lax.top_k(-dist, k=knn_k)
    knn_dists = -neg_topk
    if knn_avg:
        flat = knn_dists.reshape(-1, 1)
        new_rms, M, _ = rms_update(rms_state, flat)
        denom = M if knn_rms else jnp.asarray(1.0)
        flat = flat / denom
        flat = jnp.maximum(flat - knn_clip, 0.0) if knn_clip >= 0.0 else flat
        per_sample = flat.reshape(knn_dists.shape).mean(axis=1, keepdims=True)
    else:
        kth = knn_dists[:, -1:].reshape(-1, 1)
        new_rms, M, _ = rms_update(rms_state, kth)
        denom = M if knn_rms else jnp.asarray(1.0)
        kth = kth / denom
        per_sample = jnp.maximum(kth - knn_clip, 0.0) if knn_clip >= 0.0 else kth
    return jnp.log(per_sample + 1.0), new_rms


def sample_task(rng, sf_dim: int) -> jnp.ndarray:
    t = jax.random.normal(rng, (sf_dim,))
    return t / jnp.linalg.norm(t)


class APSAgent(flax.struct.PyTreeNode):
    encoder: TrainState  # PixelEncoder for pixels; pass-through for states
    actor: TrainState
    critic: TrainState
    critic_target: TrainState
    aps_net: TrainState
    rms: RMSState
    config: dict = flax.struct.field(pytree_node=False)

    def encode(agent, obs):
        if agent.config["obs_type"] == "states":
            return obs
        return agent.encoder.apply_fn({"params": agent.encoder.params}, obs)

    def aps_loss(agent, params, next_obs, task):
        rep = agent.aps_net.apply_fn({"params": params}, next_obs, norm=True)
        loss = -jnp.einsum("bi,bi->b", task, rep).mean()
        return loss, {"aps_loss": loss}

    def compute_intr_reward(agent, task, next_obs, rms):
        rep = agent.aps_net.apply_fn(
            {"params": agent.aps_net.params}, next_obs, norm=False
        )
        rep = jax.lax.stop_gradient(rep)
        ent_reward, new_rms = pbe(
            rep,
            rms,
            knn_k=agent.config["knn_k"],
            knn_avg=agent.config["knn_avg"],
            knn_clip=agent.config["knn_clip"],
            knn_rms=agent.config["knn_rms"],
        )
        rep_n = rep / (jnp.linalg.norm(rep, axis=-1, keepdims=True) + 1e-12)
        sf_reward = jnp.einsum("bi,bi->b", task, rep_n)[:, None]
        return ent_reward, sf_reward, new_rms

    def critic_loss(
        agent,
        critic_params,
        obs_t,
        action,
        reward,
        discount,
        next_obs_t,
        task,
        std,
        rng,
    ):
        next_mu = agent.actor.apply_fn({"params": agent.actor.params}, next_obs_t)
        next_a = truncated_normal_sample(
            rng,
            next_mu,
            std,
            clip=agent.config["stddev_clip"],
        )
        tQ1, tQ2 = agent.critic_target.apply_fn(
            {"params": agent.critic_target.params}, next_obs_t, next_a, task
        )
        target_V = jnp.minimum(tQ1, tQ2)
        target_Q = jax.lax.stop_gradient(reward + discount * target_V)
        Q1, Q2 = agent.critic.apply_fn({"params": critic_params}, obs_t, action, task)
        loss = jnp.mean(jnp.square(Q1 - target_Q)) + jnp.mean(jnp.square(Q2 - target_Q))
        return loss, {
            "critic_loss": loss,
            "critic_q1": Q1.mean(),
            "critic_q2": Q2.mean(),
            "critic_target_q": target_Q.mean(),
        }

    def actor_loss(agent, actor_params, obs_t, task, rng, std):
        mu = agent.actor.apply_fn({"params": actor_params}, obs_t)
        a = truncated_normal_sample(rng, mu, std, clip=agent.config["stddev_clip"])
        Q1, Q2 = agent.critic.apply_fn({"params": agent.critic.params}, obs_t, a, task)
        Q = jnp.minimum(Q1, Q2)
        loss = -Q.mean()
        log_prob = normal_log_prob(a, mu, std).sum(axis=-1).mean()
        return loss, {
            "actor_loss": loss,
            "actor_logprob": log_prob,
            "actor_q": Q.mean(),
        }

    @functools.partial(jax.jit, static_argnames=("reward_free",))
    def update(agent, batch, rng, std, reward_free=True):
        obs = batch["observations"]
        action = batch["actions"]
        extr_reward = batch["rewards"]
        discount = batch["discounts"]
        next_obs = batch["next_observations"]
        task = batch["tasks"]

        info = {}
        # pixels path includes RandomShiftsAug applied by the caller before encoding
        obs_e = agent.encode(obs)
        next_obs_e = agent.encode(next_obs)

        if reward_free:
            (_, aps_info), aps_grads = jax.value_and_grad(
                agent.aps_loss, argnums=0, has_aux=True
            )(agent.aps_net.params, next_obs_e, task)
            new_aps = agent.aps_net.apply_gradients(grads=aps_grads)
            ent_reward, sf_reward, new_rms = agent.compute_intr_reward(
                task, next_obs_e, agent.rms
            )
            reward = (ent_reward + sf_reward).reshape(-1)
            info.update(aps_info)
            info["intr_ent_reward"] = ent_reward.mean()
            info["intr_sf_reward"] = sf_reward.mean()
        else:
            new_aps = agent.aps_net
            new_rms = agent.rms
            reward = extr_reward.reshape(-1)
        info["batch_reward"] = reward.mean()
        info["extr_reward"] = extr_reward.mean()

        if not agent.config["update_encoder"]:
            obs_e = jax.lax.stop_gradient(obs_e)
            next_obs_e = jax.lax.stop_gradient(next_obs_e)

        obs_t = jnp.concatenate([obs_e, task], axis=-1)
        next_obs_t = jnp.concatenate([next_obs_e, task], axis=-1)

        rng, crit_rng, act_rng = jax.random.split(rng, 3)
        (_, crit_info), crit_grads = jax.value_and_grad(
            agent.critic_loss, argnums=0, has_aux=True
        )(
            agent.critic.params,
            obs_t,
            action,
            reward,
            discount,
            next_obs_t,
            task,
            std,
            crit_rng,
        )
        new_crit = agent.critic.apply_gradients(grads=crit_grads)
        info.update(crit_info)

        agent_for_actor = agent.replace(critic=new_crit)
        (_, actor_info), actor_grads = jax.value_and_grad(
            agent_for_actor.actor_loss, argnums=0, has_aux=True
        )(agent.actor.params, obs_t, task, act_rng, std)
        new_actor = agent.actor.apply_gradients(grads=actor_grads)
        info.update(actor_info)

        new_crit_target_params = soft_update(
            new_crit.params,
            agent.critic_target.params,
            agent.config["critic_target_tau"],
        )
        new_crit_target = agent.critic_target.replace(params=new_crit_target_params)

        return (
            agent.replace(
                actor=new_actor,
                critic=new_crit,
                critic_target=new_crit_target,
                aps_net=new_aps,
                rms=new_rms,
            ),
            info,
            rng,
        )

    @functools.partial(jax.jit, static_argnames=("eval_mode",))
    def act(agent, obs, task, rng, std, *, eval_mode: bool = False):
        obs_e = agent.encode(obs)
        obs_t = jnp.concatenate([obs_e, task], axis=-1)
        mu = agent.actor.apply_fn({"params": agent.actor.params}, obs_t)
        if eval_mode:
            return mu
        return truncated_normal_sample(rng, mu, std, clip=None)


def create_learner(config, rng, obs_shape, action_dim: int):
    # obs_shape: (obs_dim,) for states, (3*frame_stack, 84, 84) NCHW for pixels
    rng, enc_rng, actor_rng, crit_rng, aps_rng = jax.random.split(rng, 5)

    if config.obs_type == "pixels":
        encoder_def = PixelEncoder()
        init_pixel = jnp.zeros((1,) + tuple(obs_shape), dtype=jnp.uint8)
        enc_params = encoder_def.init(enc_rng, init_pixel)["params"]
        repr_dim = encoder_def.repr_dim
    else:
        encoder_def = None
        enc_params = {}
        repr_dim = obs_shape[0]

    obs_t_dim = repr_dim + config.sf_dim

    actor = Actor(
        action_dim=action_dim,
        obs_type=config.obs_type,
        hidden_dim=config.hidden_dim,
        feature_dim=config.feature_dim,
    )
    critic = CriticSF(
        sf_dim=config.sf_dim,
        obs_type=config.obs_type,
        hidden_dim=config.hidden_dim,
        feature_dim=config.feature_dim,
    )
    aps_net = APSNet(sf_dim=config.sf_dim, hidden_dim=config.hidden_dim)

    init_obs_t = jnp.zeros((1, obs_t_dim))
    init_action = jnp.zeros((1, action_dim))
    init_task = jnp.zeros((1, config.sf_dim))
    init_obs_aps = jnp.zeros((1, repr_dim))

    actor_params = actor.init(actor_rng, init_obs_t)["params"]
    critic_params = critic.init(crit_rng, init_obs_t, init_action, init_task)["params"]
    aps_params = aps_net.init(aps_rng, init_obs_aps)["params"]

    actor_state = TrainState.create(
        apply_fn=actor.apply, params=actor_params, tx=optax.adam(config.lr)
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply, params=critic_params, tx=optax.adam(config.lr)
    )
    critic_target_state = TrainState.create(
        apply_fn=critic.apply, params=critic_params, tx=optax.set_to_zero()
    )
    aps_state = TrainState.create(
        apply_fn=aps_net.apply, params=aps_params, tx=optax.adam(config.lr)
    )
    if config.obs_type == "pixels":
        encoder_state = TrainState.create(
            apply_fn=encoder_def.apply, params=enc_params, tx=optax.adam(config.lr)
        )
    else:
        encoder_state = TrainState.create(
            apply_fn=lambda p, x: x, params={}, tx=optax.set_to_zero()
        )

    rms = rms_init(shape=(1,))

    agent_config = flax.core.FrozenDict(
        dict(
            obs_type=config.obs_type,
            critic_target_tau=config.critic_target_tau,
            stddev_clip=config.stddev_clip,
            sf_dim=config.sf_dim,
            knn_k=config.knn_k,
            knn_avg=config.knn_avg,
            knn_clip=config.knn_clip,
            knn_rms=config.knn_rms,
            update_encoder=config.update_encoder,
        )
    )
    agent = APSAgent(
        encoder=encoder_state,
        actor=actor_state,
        critic=critic_state,
        critic_target=critic_target_state,
        aps_net=aps_state,
        rms=rms,
        config=agent_config,
    )
    return agent, rng


class EpisodeReplay:
    # In-memory n-step episode replay. reward = sum_{i<nstep} gamma^i * r_{idx+i};
    # discount = prod_{i<nstep} (gamma * disc_{idx+i})

    def __init__(self, max_size: int, nstep: int, gamma: float):
        self._max_size = max_size
        self._nstep = nstep
        self._gamma = gamma
        self._episodes = []
        self._size = 0
        self._current = None

    def start_episode(self, obs, task):
        self._current = {
            "observation": [obs],
            "action": [],
            "reward": [],
            "discount": [],
            "task": [task],
        }

    def add(self, obs, action, reward, discount, task, last: bool):
        self._current["observation"].append(obs)
        self._current["action"].append(action)
        self._current["reward"].append(reward)
        self._current["discount"].append(discount)
        self._current["task"].append(task)
        if last:
            ep = {k: np.asarray(v) for k, v in self._current.items()}
            ep_len = len(ep["action"])
            while self._size + ep_len > self._max_size and self._episodes:
                old = self._episodes.pop(0)
                self._size -= len(old["action"])
            self._episodes.append(ep)
            self._size += ep_len
            self._current = None

    def __len__(self):
        return self._size

    def can_sample(self) -> bool:
        return any(len(ep["action"]) >= self._nstep for ep in self._episodes)

    def sample(self, batch_size: int):
        obs, act, rew, disc, nobs, task = [], [], [], [], [], []
        for _ in range(batch_size):
            ep = self._episodes[np.random.randint(0, len(self._episodes))]
            T = len(ep["action"])
            if T < self._nstep:
                continue
            idx = np.random.randint(0, T - self._nstep + 1) + 1
            obs.append(ep["observation"][idx - 1])
            act.append(ep["action"][idx - 1])
            nobs.append(ep["observation"][idx + self._nstep - 1])
            r = 0.0
            d = 1.0
            for i in range(self._nstep):
                r = r + d * ep["reward"][idx - 1 + i]
                d = d * ep["discount"][idx - 1 + i] * self._gamma
            rew.append(r)
            disc.append(d)
            task.append(ep["task"][idx - 1])
        return {
            "observations": np.stack(obs),
            "actions": np.stack(act),
            "rewards": np.asarray(rew, dtype=np.float32),
            "discounts": np.asarray(disc, dtype=np.float32),
            "next_observations": np.stack(nobs),
            "tasks": np.stack(task),
        }


@dataclass
class Config:
    # APS / DDPG (aps.yaml + ddpg.yaml)
    lr: float = 1e-4
    hidden_dim: int = 1024
    feature_dim: int = 50  # used in pixel mode trunk; state mode uses hidden_dim
    critic_target_tau: float = 0.01
    update_every_steps: int = 2
    batch_size: int = 1024
    stddev: float = 0.2
    stddev_clip: float = 0.3
    sf_dim: int = 10
    update_task_every_step: int = 5
    nstep: int = 3
    knn_rms: bool = True
    knn_k: int = 12
    knn_avg: bool = True
    knn_clip: float = 0.0001
    num_init_steps: int = 4096
    lstsq_batch_size: int = 4096

    # pretrain.yaml
    domain: str = "walker"  # PRIMAL_TASKS key — walker | quadruped | jaco
    obs_type: str = "states"  # 'states' or 'pixels'
    frame_stack: int = 3  # only used in pixel mode
    action_repeat: int = 1  # set to 2 for pixels
    discount: float = 0.99
    num_train_frames: int = 2_000_010
    num_seed_frames: int = 4_000
    eval_every_frames: int = 10_000
    num_eval_episodes: int = 10
    replay_buffer_size: int = 1_000_000
    update_encoder: bool = True
    reward_free: bool = True

    # train
    seed: int = 1
    wandb_project: str = "aps-dmc-jax"
    log_every: int = 1_000


if __name__ == "__main__":
    """Original env composition (dmc.py:267-317):
        suite.load(domain, task) | cdmc.make_jaco(task)
          -> ActionDTypeWrapper(np.float32)
          -> ActionRepeatWrapper(action_repeat)
          -> [pixels.Wrapper(84x84) | identity]
          -> [FrameStackWrapper(frame_stack) | ObservationDTypeWrapper(np.float32)]
          -> action_scale.Wrapper(min=-1, max=+1)
          -> ExtendedTimeStepWrapper

    PRIMAL_TASKS pretrain target per domain:
        walker     -> walker_stand
        quadruped  -> quadruped_walk
        jaco       -> jaco_reach_top_left
    """
    import time

    import tyro
    import wandb

    def _make_env(
        domain: str, obs_type: str, frame_stack: int, action_repeat: int, seed: int
    ):
        # Lazy import — dm_control is heavy and only needed at runtime
        import sys

        url_dir = os.path.join(
            os.path.dirname(__file__), "..", "external", "url_benchmark"
        )
        sys.path.insert(0, url_dir)
        import dmc  # noqa: E402

        task = PRIMAL_TASKS[domain]
        return dmc.make(task, obs_type, frame_stack, action_repeat, seed)

    def main(cfg: Config):
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

        train_env = _make_env(
            cfg.domain, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed
        )

        obs_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        if cfg.obs_type == "pixels":
            obs_shape = tuple(obs_spec.shape)  # (3*frame_stack, 84, 84) NCHW
        else:
            obs_shape = tuple(obs_spec.shape)  # (obs_dim,)
        action_dim = int(action_spec.shape[0])

        rng = jax.random.PRNGKey(cfg.seed)
        agent, rng = create_learner(cfg, rng, obs_shape, action_dim)
        replay = EpisodeReplay(
            max_size=cfg.replay_buffer_size,
            nstep=cfg.nstep,
            gamma=cfg.discount,
        )

        rng, task_rng = jax.random.split(rng)
        task_np = np.asarray(sample_task(task_rng, cfg.sf_dim))

        # Reset env + open first episode
        time_step = train_env.reset()
        obs = time_step.observation
        replay.start_episode(obs, task_np)

        episode_step, episode_return = 0, 0.0
        global_step = 0
        t0 = time.time()
        info = {}

        num_train_steps = cfg.num_train_frames // cfg.action_repeat
        num_seed_steps = cfg.num_seed_frames // cfg.action_repeat

        while global_step < num_train_steps:
            # Refresh task every `update_task_every_step` env steps
            if global_step % cfg.update_task_every_step == 0:
                rng, _r = jax.random.split(rng)
                task_np = np.asarray(sample_task(_r, cfg.sf_dim))

            std = cfg.stddev
            rng, act_rng = jax.random.split(rng)

            if global_step < num_seed_steps:
                action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(
                    np.float32
                )
            else:
                obs_jx = jnp.asarray(obs)[None]
                task_jx = jnp.asarray(task_np)[None]
                action = np.asarray(
                    agent.act(obs_jx, task_jx, act_rng, std, eval_mode=False)[0]
                )

            if (
                global_step >= num_seed_steps
                and global_step % cfg.update_every_steps == 0
                and replay.can_sample()
            ):
                batch = replay.sample(cfg.batch_size)
                # PixelEncoder mode: apply RandomShiftsAug to obs + next_obs
                # before calling agent.update
                if cfg.obs_type == "pixels":
                    rng, aug_rng_o, aug_rng_n = jax.random.split(rng, 3)
                    obs_aug = np.asarray(
                        random_shifts_aug(aug_rng_o, jnp.asarray(batch["observations"]))
                    )
                    nobs_aug = np.asarray(
                        random_shifts_aug(
                            aug_rng_n, jnp.asarray(batch["next_observations"])
                        )
                    )
                    batch["observations"] = obs_aug
                    batch["next_observations"] = nobs_aug
                jax_batch = {k: jnp.asarray(v) for k, v in batch.items()}
                rng, upd_rng = jax.random.split(rng)
                agent, info, rng = agent.update(
                    jax_batch, upd_rng, std, reward_free=cfg.reward_free
                )

            time_step = train_env.step(action)
            obs = time_step.observation
            reward = float(time_step.reward)
            discount = float(time_step.discount)
            replay.add(obs, action, reward, discount, task_np, last=time_step.last())
            episode_step += 1
            episode_return += reward
            global_step += 1

            if time_step.last():
                wandb.log(
                    {
                        "train/episode_return": episode_return,
                        "train/episode_length": episode_step * cfg.action_repeat,
                        "train/buffer_size": len(replay),
                        "step": global_step,
                    },
                    step=global_step,
                )
                episode_step, episode_return = 0, 0.0
                time_step = train_env.reset()
                obs = time_step.observation
                replay.start_episode(obs, task_np)

            if global_step % cfg.log_every == 0 and info:
                fps = global_step * cfg.action_repeat / max(time.time() - t0, 1e-6)
                wandb.log(
                    {
                        "train/fps": fps,
                        "train/std": std,
                        **{f"train/{k}": float(v) for k, v in info.items()},
                        "step": global_step,
                    },
                    step=global_step,
                )

        wandb.finish()

    main(tyro.cli(Config))