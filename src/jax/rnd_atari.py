# Paper:          https://arxiv.org/abs/1810.12894
# Reference impl: https://github.com/openai/random-network-distillation

from dataclasses import dataclass
from typing import Any, NamedTuple

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


def _lrelu(x):
    """tf.nn.leaky_relu uses alpha=0.2 by default — JAX/Flax default is 0.01.
    The original RND target/predictor convs go through tf.nn.leaky_relu, so
    pin alpha=0.2 here to match.
    """
    return jax.nn.leaky_relu(x, negative_slope=0.2)


class CnnPolicy(nn.Module):
    """Conv policy with dual value heads. Categorical action distribution.

    Input:  obs in [B, 84, 84, 4] (uint8 or float32; cast + /255 inside).
    Output: (logits, vpred_ext, vpred_int).
    """

    num_actions: int
    hidsize: int = 256  # 128 * enlargement
    extrahid: bool = True

    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32) / 255.0
        x = nn.relu(
            nn.Conv(
                32,
                (8, 8),
                (4, 4),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c1",
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                64,
                (4, 4),
                (2, 2),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c2",
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                64,
                (4, 4),
                (1, 1),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c3",
            )(x)
        )
        x = x.reshape(x.shape[0], -1)
        # fc1: hidsize, sqrt(2)
        x = nn.relu(
            nn.Dense(
                self.hidsize,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="fc1",
            )(x)
        )
        # fc_additional: 448, sqrt(2)
        x = nn.relu(
            nn.Dense(
                448,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="fc_additional",
            )(x)
        )

        if self.extrahid:
            # extrahid residual blocks, init_scale=0.1
            x_val = x + nn.relu(
                nn.Dense(
                    448,
                    kernel_init=orthogonal(0.1),
                    bias_init=constant(0.0),
                    name="fc2val",
                )(x)
            )
            x_act = x + nn.relu(
                nn.Dense(
                    448,
                    kernel_init=orthogonal(0.1),
                    bias_init=constant(0.0),
                    name="fc2act",
                )(x)
            )
        else:
            x_val = x
            x_act = x

        # heads, all init_scale=0.01
        logits = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="pd",
        )(x_act)
        vpred_int = nn.Dense(
            1, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="vf_int"
        )(x_val)
        vpred_ext = nn.Dense(
            1, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="vf_ext"
        )(x_val)
        return logits, jnp.squeeze(vpred_ext, -1), jnp.squeeze(vpred_int, -1)


class RNDTargetCNN(nn.Module):
    """Frozen target net.

    Input:  pre-normalized + clipped last channel, [B, 84, 84, 1].
    Output: rep features [B, rep_size]. Linear last layer per the original.

    Architecture from cnn_policy_param_matched.py:141-145:
      3 leaky_relu convs (32, 64, 64) -> flatten -> linear fc(rep_size).
    """

    convfeat: int = 32  # 16 * enlargement
    rep_size: int = 512

    @nn.compact
    def __call__(self, x):
        x = _lrelu(
            nn.Conv(
                self.convfeat,
                (8, 8),
                (4, 4),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c1r",
            )(x)
        )
        x = _lrelu(
            nn.Conv(
                self.convfeat * 2,
                (4, 4),
                (2, 2),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c2r",
            )(x)
        )
        x = _lrelu(
            nn.Conv(
                self.convfeat * 2,
                (3, 3),
                (1, 1),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c3r",
            )(x)
        )
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(
            self.rep_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="fc1r",
        )(x)
        return x


class RNDPredictorCNN(nn.Module):
    """Trained predictor net (deeper than the target).

    Input:  pre-normalized + clipped last channel, [B, 84, 84, 1].
    Output: rep features [B, rep_size].

    Architecture from cnn_policy_param_matched.py:156-163:
      3 leaky_relu convs (32, 64, 64) -> flatten ->
      relu fc(256*enlargement) -> relu fc(256*enlargement) -> linear fc(rep_size).
    """

    convfeat: int = 32
    rep_size: int = 512
    enlargement: int = 2

    @nn.compact
    def __call__(self, x):
        x = _lrelu(
            nn.Conv(
                self.convfeat,
                (8, 8),
                (4, 4),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c1rp_pred",
            )(x)
        )
        x = _lrelu(
            nn.Conv(
                self.convfeat * 2,
                (4, 4),
                (2, 2),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c2rp_pred",
            )(x)
        )
        x = _lrelu(
            nn.Conv(
                self.convfeat * 2,
                (3, 3),
                (1, 1),
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="c3rp_pred",
            )(x)
        )
        x = x.reshape(x.shape[0], -1)
        fc_width = 256 * self.enlargement
        x = nn.relu(
            nn.Dense(
                fc_width,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="fc1r_hat1_pred",
            )(x)
        )
        x = nn.relu(
            nn.Dense(
                fc_width,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="fc1r_hat2_pred",
            )(x)
        )
        x = nn.Dense(
            self.rep_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="fc1r_hat3_pred",
        )(x)
        return x


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_ext: jnp.ndarray
    value_int: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


class RunningMoments(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


class RewNormState(NamedTuple):
    rewems: jnp.ndarray
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def welford_update(state, batch):
    # RunningMeanStd.update_from_moments
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]
    delta = batch_mean - state.mean
    tot_count = state.count + batch_count
    new_mean = state.mean + delta * batch_count / tot_count
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
    new_var = M2 / tot_count
    return new_mean, new_var, tot_count


def normalize_obs(obs_last_channel, obs_norm):
    # clip((x - ph_mean) / ph_std, -5, 5)
    return jnp.clip(
        (obs_last_channel - obs_norm.mean) / jnp.sqrt(obs_norm.var + 1e-8),
        -5.0,
        5.0,
    )


class RNDAtariAgent(flax.struct.PyTreeNode):
    train_state: TrainState
    rnd_target_params: Any
    obs_norm: RunningMoments  # per-pixel mean/var over (84, 84, 1)
    rew_norm: RewNormState
    config: dict = flax.struct.field(pytree_node=False)

    def loss_fn(agent, params, traj_batch, gae, targets_ext, targets_int, rng_mask):
        logits, vpred_ext, vpred_int = agent.train_state.apply_fn(
            params["policy"], traj_batch.obs
        )
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(traj_batch.action)

        # vf_loss_{int,ext} = 0.5 * vf_coef * mean(square(...))
        vf_loss_ext = (0.5 * agent.config["vf_coef"]) * jnp.square(
            vpred_ext - targets_ext
        ).mean()
        vf_loss_int = (0.5 * agent.config["vf_coef"]) * jnp.square(
            vpred_int - targets_int
        ).mean()

        # clipped surrogate
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        if agent.config["normalize_adv"]:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - agent.config["clip_eps"],
                1.0 + agent.config["clip_eps"],
            )
            * gae
        )
        pg_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
        entropy = pi.entropy().mean()
        # ent_loss = (-ent_coef) * entropy
        ent_loss = -agent.config["ent_coef"] * entropy

        # RND aux loss: only the last channel feeds RND, cast uint8 to float
        # before normalization for numeric safety
        last_chan = traj_batch.obs[..., -1:].astype(jnp.float32)
        rnd_obs = normalize_obs(last_chan, agent.obs_norm)
        target_feat = RNDTargetCNN().apply(agent.rnd_target_params, rnd_obs)
        pred_feat = RNDPredictorCNN().apply(params["rnd_predictor"], rnd_obs)
        pred_errors = jnp.mean(jnp.square(target_feat - pred_feat), axis=-1)
        # random-mask + safe-mean
        mask = (
            jax.random.uniform(rng_mask, shape=pred_errors.shape)
            < agent.config["update_proportion"]
        )
        aux_loss = jnp.sum(mask * pred_errors) / jnp.maximum(jnp.sum(mask), 1.0)

        # total = pg + ent + (vf_int + vf_ext) + aux
        total_loss = pg_loss + ent_loss + vf_loss_ext + vf_loss_int + aux_loss
        return total_loss, {
            "total_loss": total_loss,
            "pg_loss": pg_loss,
            "vf_loss_ext": vf_loss_ext,
            "vf_loss_int": vf_loss_int,
            "entropy": entropy,
            "aux_loss": aux_loss,
        }

    @jax.jit
    def update(agent, traj_batch, advantages, targets_ext, targets_int, rng):
        num_envs = agent.config["num_envs"]
        num_steps = agent.config["num_steps"]
        num_minibatches = agent.config["num_minibatches"]
        update_epochs = agent.config["update_epochs"]
        batch_size = num_steps * num_envs

        def _update_epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng, mask_rng = jax.random.split(rng, 3)
            permutation = jax.random.permutation(perm_rng, batch_size)
            batch = (traj_batch, advantages, targets_ext, targets_int)
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
            mask_rngs = jax.random.split(mask_rng, num_minibatches)

            def _update_mb(train_state, mb_with_rng):
                (mb_traj, mb_adv, mb_te, mb_ti), mb_mask_rng = mb_with_rng
                (_, info), grads = jax.value_and_grad(agent.loss_fn, has_aux=True)(
                    train_state.params,
                    mb_traj,
                    mb_adv,
                    mb_te,
                    mb_ti,
                    mb_mask_rng,
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, info

            train_state, info = jax.lax.scan(
                _update_mb, train_state, (minibatches, mask_rngs)
            )
            return (train_state, rng), info

        (new_train_state, rng), info = jax.lax.scan(
            _update_epoch, (agent.train_state, rng), None, update_epochs
        )
        info = jax.tree_util.tree_map(lambda x: x.mean(), info)
        return agent.replace(train_state=new_train_state), info, rng

    @jax.jit
    def act(agent, obs, rng):
        rng, act_rng = jax.random.split(rng)
        logits, value_ext, value_int = agent.train_state.apply_fn(
            agent.train_state.params["policy"], obs
        )
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=act_rng)
        return action, pi.log_prob(action), value_ext, value_int, rng

    @jax.jit
    def value(agent, obs):
        _, ve, vi = agent.train_state.apply_fn(agent.train_state.params["policy"], obs)
        return ve, vi

    @jax.jit
    def intrinsic_reward(agent, obs):
        # obs: (B, 84, 84, 4) frame-stacked. RND only sees the last frame
        last_chan = obs[..., -1:].astype(jnp.float32)
        rnd_obs = normalize_obs(last_chan, agent.obs_norm)
        target_feat = RNDTargetCNN().apply(agent.rnd_target_params, rnd_obs)
        pred_feat = RNDPredictorCNN().apply(
            agent.train_state.params["rnd_predictor"], rnd_obs
        )
        return jnp.mean(jnp.square(target_feat - pred_feat), axis=-1)

    def update_obs_norm(agent, batch):
        """Update per-pixel obs RMS over the last channel only.

        ppo_agent.py:493-494: obs_.reshape((-1, 84, 84, 4))[:, :, :, -1:] feeds
        RunningMeanStd(shape=(84, 84, 1)). welford_update averages over axis=0,
        so passing a (N, 84, 84, 1) batch gets the per-pixel stats we want.
        """
        last = batch[..., -1:].astype(jnp.float32)
        last = last.reshape((-1,) + last.shape[-3:])
        mean, var, count = welford_update(agent.obs_norm, last)
        return agent.replace(obs_norm=RunningMoments(mean=mean, var=var, count=count))

    def update_rew_norm(agent, rffs, rewems_final):
        # RewardForwardFilter -> RunningMeanStd(scalar)
        mean, var, count = welford_update(agent.rew_norm, rffs.reshape(-1))
        return agent.replace(
            rew_norm=RewNormState(rewems=rewems_final, mean=mean, var=var, count=count)
        )


def create_learner(config, rng, obs_shape, num_actions, num_total_updates=None):
    # obs_shape == (84, 84, 4)
    rng, rng_net, rng_target, rng_pred = jax.random.split(rng, 4)
    network = CnnPolicy(num_actions=num_actions)
    rnd_target = RNDTargetCNN()
    rnd_predictor = RNDPredictorCNN()

    init_x = jnp.zeros((1,) + tuple(obs_shape), dtype=jnp.uint8)
    init_x_rnd = jnp.zeros((1, obs_shape[0], obs_shape[1], 1), dtype=jnp.float32)
    rnd_target_params = rnd_target.init(rng_target, init_x_rnd)
    combined_params = {
        "policy": network.init(rng_net, init_x),
        "rnd_predictor": rnd_predictor.init(rng_pred, init_x_rnd),
    }

    # Paper default: max_grad_norm=0.0 == no clip. Anything > 0 clips
    chain_args = []
    if config.max_grad_norm > 0:
        chain_args.append(optax.clip_by_global_norm(config.max_grad_norm))
    if config.anneal_lr and num_total_updates is not None:

        def lr_schedule(count):
            frac = (
                1.0
                - (count // (config.num_minibatches * config.update_epochs))
                / num_total_updates
            )
            return config.lr * frac

        chain_args.append(optax.adam(learning_rate=lr_schedule, eps=1e-5))
    else:
        chain_args.append(optax.adam(config.lr, eps=1e-5))
    tx = optax.chain(*chain_args)

    train_state = TrainState.create(
        apply_fn=network.apply, params=combined_params, tx=tx
    )

    obs_h, obs_w, _ = obs_shape  # (84, 84, 4)
    obs_norm = RunningMoments(
        mean=jnp.zeros((obs_h, obs_w, 1)),
        var=jnp.ones((obs_h, obs_w, 1)),
        count=jnp.asarray(1e-4),
    )
    rew_norm = RewNormState(
        rewems=jnp.zeros(config.num_envs),
        mean=jnp.asarray(0.0),
        var=jnp.asarray(1.0),
        count=jnp.asarray(1e-4),
    )

    agent_config = flax.core.FrozenDict(
        dict(
            clip_eps=config.cliprange,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            num_envs=config.num_envs,
            num_steps=config.num_steps,
            num_minibatches=config.num_minibatches,
            update_epochs=config.update_epochs,
            update_proportion=config.proportion_of_exp_used_for_predictor_update,
            normalize_adv=config.normalize_adv,
            use_news=config.use_news,
        )
    )
    agent = RNDAtariAgent(
        train_state=train_state,
        rnd_target_params=rnd_target_params,
        obs_norm=obs_norm,
        rew_norm=rew_norm,
        config=agent_config,
    )
    return agent, rng


@dataclass
class Config:
    # ppo (paper defaults)
    lr: float = 1e-4
    num_envs: int = 32
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma_int: float = 0.99
    gamma_ext: float = 0.99
    gae_lambda: float = 0.95
    cliprange: float = 0.1
    ent_coef: float = 0.001
    # vf_coef = 1.0 matches paper's effective weight 0.5: loss is
    # `0.5 * vf_coef * mean(square(...))`
    vf_coef: float = 1.0
    # Paper: max_grad_norm=0.0 == no clip
    max_grad_norm: float = 0.0
    int_coef: float = 1.0
    ext_coef: float = 2.0
    use_news: int = 0  # 0 -> no done mask on intrinsic stream
    # paper code default is 1.0; paper text reports 0.25 — we follow
    # the reported configuration to match algorithms/rnd.py
    proportion_of_exp_used_for_predictor_update: float = 0.25

    # rnd / atari
    frame_stack: int = 4
    rep_size: int = 512
    anneal_lr: bool = False
    normalize_adv: bool = False  # paper does NOT normalize advantages

    # training
    seed: int = 0
    env: str = "MontezumaRevengeNoFrameskip-v4"
    total_timesteps: int = int(1e9)
    num_iterations_obs_norm_init: int = 50  # 128*50 random-agent steps
    update_ob_stats_from_random_agent: int = 1
    sticky_action_p: float = 0.25
    max_episode_steps: int = 4500

    wandb_project: str = "rnd-atari-jax"
    log_every_n_updates: int = 1


if __name__ == "__main__":
    """Original env composition (atari_wrappers.py:200-224):
        gym.make('MontezumaRevengeNoFrameskip-v4')
          -> StickyActionEnv(p=0.25)
          -> MaxAndSkipEnv(skip=4)
          -> MontezumaInfoWrapper(room_address=3)        # for log/diagnostics only
          -> WarpFrame(84x84 grayscale)
          -> ClipRewardEnv(sign)
          -> FrameStack(k=4)

    We reproduce all of that synchronously on top of gymnasium/ale-py so
    `info['episode']['visited_rooms']` is still available for room-count logs.
    """
    import time
    from collections import deque

    import gymnasium as gym
    import cv2
    import tyro
    import wandb

    cv2.ocl.setUseOpenCL(False)

    # --- env wrappers -------------------------------------------------------

    class StickyActionEnv(gym.Wrapper):
        # Repeat last action with prob p

        def __init__(self, env, p: float = 0.25):
            super().__init__(env)
            self.p = p
            self.last_action = 0

        def reset(self, **kwargs):
            self.last_action = 0
            return self.env.reset(**kwargs)

        def step(self, action):
            if self.unwrapped.np_random.uniform() < self.p:
                action = self.last_action
            self.last_action = action
            return self.env.step(action)

    class MaxAndSkipEnv(gym.Wrapper):
        """Sum reward over `skip` raw frames, return
        max of last two."""

        def __init__(self, env, skip: int = 4):
            super().__init__(env)
            self._obs_buffer = np.zeros(
                (2,) + env.observation_space.shape, dtype=np.uint8
            )
            self._skip = skip

        def step(self, action):
            total_reward = 0.0
            terminated = truncated = False
            info = {}
            for i in range(self._skip):
                obs, reward, terminated, truncated, info = self.env.step(action)
                if i == self._skip - 2:
                    self._obs_buffer[0] = obs
                if i == self._skip - 1:
                    self._obs_buffer[1] = obs
                total_reward += reward
                if terminated or truncated:
                    break
            return (
                self._obs_buffer.max(axis=0),
                total_reward,
                terminated,
                truncated,
                info,
            )

    class WarpFrame(gym.ObservationWrapper):
        # RGB -> 84x84 gray uint8 (H, W, 1)

        def __init__(self, env):
            super().__init__(env)
            self.width = 84
            self.height = 84
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
            )

        def observation(self, frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            return frame[:, :, None]

    class ClipRewardEnv(gym.RewardWrapper):
        # Bin to {-1, 0, +1}

        def reward(self, reward):
            return float(np.sign(reward))

    class FrameStack(gym.Wrapper):
        # Stack k frames along channel axis

        def __init__(self, env, k: int):
            super().__init__(env)
            self.k = k
            self.frames = deque([], maxlen=k)
            shp = env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8
            )

        def reset(self, **kwargs):
            ob, info = self.env.reset(**kwargs)
            for _ in range(self.k):
                self.frames.append(ob)
            return self._get_ob(), info

        def step(self, action):
            ob, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(ob)
            return self._get_ob(), reward, terminated, truncated, info

        def _get_ob(self):
            return np.concatenate(list(self.frames), axis=2)

    class MontezumaInfoWrapper(gym.Wrapper):
        # Tracks visited-rooms via RAM

        def __init__(self, env, room_address: int):
            super().__init__(env)
            self.room_address = room_address
            self.visited_rooms = set()

        def _get_room(self):
            ale = self.unwrapped.ale
            return int(ale.getRAM()[self.room_address])

        def reset(self, **kwargs):
            self.visited_rooms.clear()
            return self.env.reset(**kwargs)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.visited_rooms.add(self._get_room())
            if terminated or truncated:
                if "episode" not in info:
                    info["episode"] = {}
                info["episode"]["visited_rooms"] = list(self.visited_rooms)
            return obs, reward, terminated, truncated, info

    def make_atari(env_id, seed: int, max_episode_steps: int, sticky_p: float):
        env = gym.make(env_id, render_mode=None)
        if max_episode_steps:
            env = gym.wrappers.TimeLimit(
                env.unwrapped, max_episode_steps=max_episode_steps * 4
            )
        env.reset(seed=seed)
        env = StickyActionEnv(env, p=sticky_p)
        env = MaxAndSkipEnv(env, skip=4)
        if "Montezuma" in env_id:
            env = MontezumaInfoWrapper(env, room_address=3)
        elif "Pitfall" in env_id:
            env = MontezumaInfoWrapper(env, room_address=1)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, k=4)
        return env

    class VecAtari:
        # Synchronous parallel atari env. Auto-resets on episode end

        def __init__(
            self,
            env_id: str,
            num_envs: int,
            seed: int,
            max_episode_steps: int,
            sticky_p: float,
        ):
            self.envs = [
                make_atari(env_id, seed + i, max_episode_steps, sticky_p)
                for i in range(num_envs)
            ]
            self.num_envs = num_envs
            self.action_space = self.envs[0].action_space
            self.observation_space = self.envs[0].observation_space
            self.episode_returns = np.zeros(num_envs, dtype=np.float32)
            self.episode_lengths = np.zeros(num_envs, dtype=np.int64)

        def reset(self):
            obs = []
            for e in self.envs:
                o, _ = e.reset()
                obs.append(o)
            return np.stack(obs, 0)

        def step(self, actions):
            obs, rew, done, infos = [], [], [], []
            for i, (e, a) in enumerate(zip(self.envs, actions)):
                o, r, term, trunc, info = e.step(int(a))
                d = bool(term or trunc)
                self.episode_returns[i] += r
                self.episode_lengths[i] += 1
                if d:
                    info.setdefault("episode", {})
                    info["episode"]["r"] = float(self.episode_returns[i])
                    info["episode"]["l"] = int(self.episode_lengths[i])
                    self.episode_returns[i] = 0.0
                    self.episode_lengths[i] = 0
                    o, _ = e.reset()
                obs.append(o)
                rew.append(r)
                done.append(d)
                infos.append(info)
            return (
                np.stack(obs, 0),
                np.array(rew, dtype=np.float32),
                np.array(done, dtype=np.float32),
                infos,
            )

        def close(self):
            for e in self.envs:
                e.close()

    # --- main ---------------------------------------------------------------

    def main(cfg: Config):
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

        venv = VecAtari(
            cfg.env,
            cfg.num_envs,
            cfg.seed,
            cfg.max_episode_steps,
            cfg.sticky_action_p,
        )
        num_actions = int(venv.action_space.n)
        obs_shape = (84, 84, cfg.frame_stack)
        steps_per_update = cfg.num_steps * cfg.num_envs
        total_updates = int(cfg.total_timesteps // steps_per_update)

        rng = jax.random.PRNGKey(cfg.seed)
        agent, rng = create_learner(
            cfg, rng, obs_shape, num_actions, num_total_updates=total_updates
        )

        # collect_random_statistics: initialize obs_rms by stepping a uniform-
        # random policy for 128*50 steps in batches of `128 * num_envs`
        if cfg.update_ob_stats_from_random_agent:
            obs = venv.reset()
            chunk = []
            target = 128 * cfg.num_iterations_obs_norm_init
            print(
                f"Collecting obs-norm warmup: {target} steps "
                f"({cfg.num_iterations_obs_norm_init} chunks of 128)."
            )
            for step in range(target):
                acs = np.random.randint(low=0, high=num_actions, size=(cfg.num_envs,))
                obs, _, _, _ = venv.step(acs)
                chunk.append(obs)
                if (step + 1) % 128 == 0:
                    batch = np.stack(chunk, axis=1)  # (num_envs, 128, 84, 84, 4)
                    batch = batch.reshape((-1,) + batch.shape[2:])
                    agent = agent.update_obs_norm(jnp.asarray(batch))
                    chunk.clear()

        last_obs = venv.reset()

        update_idx = 0
        t_start = time.time()
        ep_returns = deque(maxlen=100)
        ep_lengths = deque(maxlen=100)
        rooms_seen = set()

        # Rollout buffer (numpy host-side, fed into JIT'd update each step)
        H, W, C = obs_shape
        buf_obs = np.zeros((cfg.num_envs, cfg.num_steps, H, W, C), np.uint8)
        buf_acs = np.zeros((cfg.num_envs, cfg.num_steps), np.int64)
        buf_rews_ext = np.zeros((cfg.num_envs, cfg.num_steps), np.float32)
        buf_rews_int = np.zeros((cfg.num_envs, cfg.num_steps), np.float32)
        buf_dones = np.zeros((cfg.num_envs, cfg.num_steps), np.float32)
        buf_vpreds_ext = np.zeros((cfg.num_envs, cfg.num_steps), np.float32)
        buf_vpreds_int = np.zeros((cfg.num_envs, cfg.num_steps), np.float32)
        buf_nlps = np.zeros((cfg.num_envs, cfg.num_steps), np.float32)

        last_done = np.zeros(cfg.num_envs, np.float32)

        while update_idx < total_updates:
            for t in range(cfg.num_steps):
                acs, log_probs, ves, vis, rng = agent.act(jnp.asarray(last_obs), rng)
                acs_np = np.asarray(acs)
                buf_obs[:, t] = last_obs
                buf_acs[:, t] = acs_np
                buf_dones[:, t] = last_done
                buf_vpreds_ext[:, t] = np.asarray(ves)
                buf_vpreds_int[:, t] = np.asarray(vis)
                buf_nlps[:, t] = -np.asarray(log_probs)

                last_obs, rew, done, infos = venv.step(acs_np)
                buf_rews_ext[:, t] = rew
                last_done = done

                for info in infos:
                    if "episode" in info and "r" in info["episode"]:
                        ep_returns.append(info["episode"]["r"])
                        ep_lengths.append(info["episode"]["l"])
                    if "episode" in info and "visited_rooms" in info["episode"]:
                        rooms_seen.update(info["episode"]["visited_rooms"])

            # Compute intrinsic rewards over the full rollout — uses the
            # pre-rollout obs_norm (paper default)
            obs_for_int = jnp.asarray(buf_obs.reshape((-1,) + buf_obs.shape[2:]))
            int_rews = agent.intrinsic_reward(obs_for_int)
            buf_rews_int[:] = np.asarray(int_rews).reshape(cfg.num_envs, cfg.num_steps)

            # update obs_rms from this rollout's obs
            agent = agent.update_obs_norm(jnp.asarray(buf_obs))

            # RewardForwardFilter -> running var of returns
            rewems = np.asarray(agent.rew_norm.rewems)
            rffs = np.zeros_like(buf_rews_int)
            for t in range(cfg.num_steps):
                rewems = rewems * cfg.gamma_int + buf_rews_int[:, t]
                rffs[:, t] = rewems
            agent = agent.update_rew_norm(jnp.asarray(rffs), jnp.asarray(rewems))
            int_var = float(np.asarray(agent.rew_norm.var))
            norm_int_rewards = buf_rews_int / np.sqrt(int_var + 1e-8)

            # Last value for bootstrap (last-step path)
            last_val_ext, last_val_int = agent.value(jnp.asarray(last_obs))
            last_val_ext = np.asarray(last_val_ext)
            last_val_int = np.asarray(last_val_int)

            # Dual GAE
            advs_ext = np.zeros_like(buf_rews_ext)
            advs_int = np.zeros_like(buf_rews_int)
            lastgaelam_ext = np.zeros(cfg.num_envs, dtype=np.float32)
            lastgaelam_int = np.zeros(cfg.num_envs, dtype=np.float32)
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    next_vals_ext = last_val_ext
                    next_vals_int = last_val_int
                    next_done = last_done
                else:
                    next_vals_ext = buf_vpreds_ext[:, t + 1]
                    next_vals_int = buf_vpreds_int[:, t + 1]
                    next_done = buf_dones[:, t + 1]
                # Extrinsic GAE WITH done mask
                notdone = 1.0 - next_done
                delta_ext = (
                    buf_rews_ext[:, t]
                    + cfg.gamma_ext * next_vals_ext * notdone
                    - buf_vpreds_ext[:, t]
                )
                advs_ext[:, t] = lastgaelam_ext = (
                    delta_ext
                    + cfg.gamma_ext * cfg.gae_lambda * notdone * lastgaelam_ext
                )
                # Intrinsic GAE — use_news=0 means NO done mask
                if cfg.use_news:
                    nextnotnew_int = notdone
                else:
                    nextnotnew_int = np.ones_like(notdone)
                delta_int = (
                    norm_int_rewards[:, t]
                    + cfg.gamma_int * next_vals_int * nextnotnew_int
                    - buf_vpreds_int[:, t]
                )
                advs_int[:, t] = lastgaelam_int = (
                    delta_int
                    + cfg.gamma_int * cfg.gae_lambda * nextnotnew_int * lastgaelam_int
                )
            targets_ext = advs_ext + buf_vpreds_ext
            targets_int = advs_int + buf_vpreds_int
            advs = cfg.ext_coef * advs_ext + cfg.int_coef * advs_int

            traj = Transition(
                done=jnp.asarray(buf_dones),
                action=jnp.asarray(buf_acs),
                value_ext=jnp.asarray(buf_vpreds_ext),
                value_int=jnp.asarray(buf_vpreds_int),
                reward=jnp.asarray(buf_rews_ext),
                int_reward=jnp.asarray(buf_rews_int),
                log_prob=-jnp.asarray(buf_nlps),
                obs=jnp.asarray(buf_obs),
            )
            agent, info, rng = agent.update(
                traj,
                jnp.asarray(advs),
                jnp.asarray(targets_ext),
                jnp.asarray(targets_int),
                rng,
            )

            update_idx += 1
            if update_idx % cfg.log_every_n_updates == 0:
                tcount = update_idx * steps_per_update
                tps = tcount / max(time.time() - t_start, 1e-6)
                log = {
                    "tcount": tcount,
                    "tps": tps,
                    "n_rooms": len(rooms_seen),
                    "rooms": sorted(rooms_seen),
                    "eprew_mean": float(np.mean(ep_returns)) if ep_returns else 0.0,
                    "eplen_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
                    "rewintmean_unnorm": float(buf_rews_int.mean()),
                    "rewintmax_unnorm": float(buf_rews_int.max()),
                    "rewintmean_norm": float(norm_int_rewards.mean()),
                    "vpredintmean": float(buf_vpreds_int.mean()),
                    "vpredextmean": float(buf_vpreds_ext.mean()),
                    "advmean": float(advs.mean()),
                    **{k: float(v) for k, v in info.items()},
                }
                print(f"[{update_idx}/{total_updates}] {log}")
                wandb.log(log, step=tcount)

        venv.close()
        wandb.finish()

    main(tyro.cli(Config))