# Paper:          https://arxiv.org/abs/2210.05805
# Reference impl: https://github.com/facebookresearch/e3b

from dataclasses import dataclass
from typing import Any, NamedTuple

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

# minihack is built on top of NLE (nethack learning environment)
from nle import nethack as _nethack

MAX_GLYPH = int(_nethack.MAX_GLYPH)
NUM_CHARS = 256
PAD_CHAR = 0
GLYPH_H, GLYPH_W = _nethack.DUNGEON_SHAPE  # (21, 79)

# jax kaiming is gain 2 instead of pytorch's 3 and i haven't had time to see if this matters
PYTORCH_KAIMING_INIT = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")


class Crop(nn.Module):
    height: int
    width: int
    target: int

    @nn.compact
    def __call__(self, inputs, coordinates):
        B = inputs.shape[0]
        k = jnp.arange(-self.target // 2, self.target // 2)
        x = coordinates[:, 0].astype(jnp.int32)
        y = coordinates[:, 1].astype(jnp.int32)
        idx_h = y[:, None] + k[None, :]
        idx_w = x[:, None] + k[None, :]
        valid = ((idx_h >= 0) & (idx_h < self.height))[:, :, None] & (
            (idx_w >= 0) & (idx_w < self.width)
        )[:, None, :]
        ih = jnp.clip(idx_h, 0, self.height - 1)[:, :, None]
        iw = jnp.clip(idx_w, 0, self.width - 1)[:, None, :]
        cropped = inputs[jnp.arange(B)[:, None, None], ih, iw]
        return jnp.where(valid, cropped, 0)


class NetHackStateEmbeddingNet(nn.Module):
    blstats_size: int
    hidden_dim: int = 1024
    k_dim: int = 64
    crop_dim: int = 9
    num_layers: int = 5
    m_dim: int = 16
    y_dim: int = 8
    msg_hdim: int = 64
    msg_edim: int = 32

    @nn.compact
    def __call__(self, glyphs, blstats, message):
        embed = nn.Embed(
            MAX_GLYPH, self.k_dim, embedding_init=nn.initializers.normal(stddev=1.0)
        )
        glyphs = glyphs.astype(jnp.int32)
        coords = blstats[:, :2]
        channels = [self.m_dim] * (self.num_layers - 1) + [self.y_dim]

        full = embed(glyphs)
        for c in channels:
            full = nn.elu(
                nn.Conv(c, (3, 3), padding="SAME", kernel_init=PYTORCH_KAIMING_INIT)(
                    full
                )
            )
        full = full.reshape(full.shape[0], -1)

        crop = Crop(GLYPH_H, GLYPH_W, self.crop_dim)(glyphs, coords)
        crop = embed(crop)
        for c in channels:
            crop = nn.elu(
                nn.Conv(c, (3, 3), padding="SAME", kernel_init=PYTORCH_KAIMING_INIT)(
                    crop
                )
            )
        crop = crop.reshape(crop.shape[0], -1)

        b = nn.relu(
            nn.Dense(self.k_dim, kernel_init=PYTORCH_KAIMING_INIT)(
                blstats.astype(jnp.float32)
            )
        )
        b = nn.relu(nn.Dense(self.k_dim, kernel_init=PYTORCH_KAIMING_INIT)(b))

        # lt_cnn message encoder
        msg = message.astype(jnp.int32)
        m = nn.Embed(
            NUM_CHARS, self.msg_edim, embedding_init=nn.initializers.normal(stddev=1.0)
        )(msg)
        m = jnp.where((msg == PAD_CHAR)[..., None], 0.0, m)
        m = nn.relu(
            nn.Conv(
                self.msg_hdim, (7,), padding="VALID", kernel_init=PYTORCH_KAIMING_INIT
            )(m)
        )
        m = nn.max_pool(m, (3,), strides=(3,))
        m = nn.relu(
            nn.Conv(
                self.msg_hdim, (7,), padding="VALID", kernel_init=PYTORCH_KAIMING_INIT
            )(m)
        )
        m = nn.max_pool(m, (3,), strides=(3,))
        for _ in range(4):
            m = nn.relu(
                nn.Conv(
                    self.msg_hdim,
                    (3,),
                    padding="VALID",
                    kernel_init=PYTORCH_KAIMING_INIT,
                )(m)
            )
        m = nn.max_pool(m, (3,), strides=(3,))
        m = m.reshape(m.shape[0], -1)
        m = nn.relu(nn.Dense(2 * self.msg_hdim, kernel_init=PYTORCH_KAIMING_INIT)(m))
        m = nn.Dense(self.msg_hdim, kernel_init=PYTORCH_KAIMING_INIT)(m)

        x = jnp.concatenate([b, crop, full, m], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=PYTORCH_KAIMING_INIT)(x))
        x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=PYTORCH_KAIMING_INIT)(x))
        return x


class NetHackPolicyNet(nn.Module):
    num_actions: int
    blstats_size: int
    hidden_dim: int = 1024
    use_lstm: bool = True

    def setup(self):
        self.body = NetHackStateEmbeddingNet(
            blstats_size=self.blstats_size, hidden_dim=self.hidden_dim
        )
        if self.use_lstm:
            self.cell = nn.LSTMCell(features=self.hidden_dim)
        self.policy_head = nn.Dense(self.num_actions, kernel_init=PYTORCH_KAIMING_INIT)
        self.value_head = nn.Dense(1, kernel_init=PYTORCH_KAIMING_INIT)

    def __call__(self, glyphs, blstats, message, done, lstm_state):
        T, B = glyphs.shape[:2]
        emb = self.body(
            glyphs.reshape((T * B,) + glyphs.shape[2:]),
            blstats.reshape((T * B,) + blstats.shape[2:]),
            message.reshape((T * B,) + message.shape[2:]),
        ).reshape(T, B, self.hidden_dim)

        if self.use_lstm:
            carry = lstm_state
            outs = []
            for t in range(T):
                notdone = (1.0 - done[t])[:, None]
                carry = (carry[0] * notdone, carry[1] * notdone)
                carry, out = self.cell(carry, emb[t])
                outs.append(out)
            core = jnp.stack(outs, axis=0)
            final = carry
        else:
            core = emb
            final = lstm_state

        flat = core.reshape(T * B, -1)
        logits = self.policy_head(flat).reshape(T, B, self.num_actions)
        value = jnp.squeeze(self.value_head(flat), -1).reshape(T, B)
        return logits, value, final


class MinigridInverseDynamicsNet(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, h_t, h_next):
        x = jnp.concatenate([h_t, h_next], axis=-1)
        x = nn.relu(
            nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
        )
        return nn.Dense(self.num_actions, kernel_init=nn.initializers.orthogonal(1.0))(
            x
        )


class RunningMoments(NamedTuple):
    sum: float
    m2: float
    count: float


class E3B(flax.struct.PyTreeNode):
    rng: Any
    actor_critic: TrainState
    encoder: TrainState
    inv_dyn: TrainState
    intrinsic_moments: RunningMoments
    config: dict = flax.struct.field(pytree_node=False)

    def total_loss(agent, batch, bootstrap, ac_params, enc_params, inv_params):
        T, B = batch["actions"].shape
        all_g = jnp.concatenate([batch["glyphs"], bootstrap["glyphs"][None]], axis=0)
        all_b = jnp.concatenate([batch["blstats"], bootstrap["blstats"][None]], axis=0)
        all_m = jnp.concatenate([batch["message"], bootstrap["message"][None]], axis=0)
        all_d = jnp.concatenate(
            [batch["is_new_ep"], bootstrap["is_new_ep"][None]], axis=0
        )
        logits_all, values_all, _ = agent.actor_critic.apply_fn(
            {"params": ac_params},
            all_g,
            all_b,
            all_m,
            all_d,
            batch["init_lstm_state"],
        )
        logits = logits_all[:T]
        values = values_all[:T]
        bootstrap_value = values_all[T]
        pi = distrax.Categorical(logits=logits)
        target_log_pi_a = pi.log_prob(batch["actions"])

        # IMPALA V-trace
        rhos = jnp.exp(target_log_pi_a - batch["log_prob"])
        clipped_rhos = jnp.minimum(rhos, agent.config["rho_bar"])
        cs = jnp.minimum(rhos, agent.config["c_bar"])
        values_tp1 = jnp.concatenate([values[1:], bootstrap_value[None]], axis=0)
        deltas = clipped_rhos * (
            batch["rewards"] + batch["discounts"] * values_tp1 - values
        )

        def vtrace_step(carry, inp):
            delta, discount, c = inp
            acc = delta + discount * c * carry
            return acc, acc

        _, vs_minus_v = jax.lax.scan(
            vtrace_step,
            jnp.zeros_like(bootstrap_value),
            (deltas, batch["discounts"], cs),
            reverse=True,
        )
        vs = vs_minus_v + values
        vs_tp1 = jnp.concatenate([vs[1:], bootstrap_value[None]], axis=0)
        pg_adv = clipped_rhos * (
            batch["rewards"] + batch["discounts"] * vs_tp1 - values
        )
        vs = jax.lax.stop_gradient(vs)
        pg_adv = jax.lax.stop_gradient(pg_adv)

        pg_loss = (-target_log_pi_a * pg_adv).mean(axis=1).sum()
        baseline_loss = 0.5 * ((vs - values) ** 2).mean(axis=1).sum()
        entropy = pi.entropy()
        entropy_loss = -entropy.mean(axis=1).sum()

        enc_g = all_g.reshape((T + 1) * B, *all_g.shape[2:])
        enc_b = all_b.reshape((T + 1) * B, *all_b.shape[2:])
        enc_m = all_m.reshape((T + 1) * B, *all_m.shape[2:])
        emb = agent.encoder.apply_fn(
            {"params": enc_params}, enc_g, enc_b, enc_m
        ).reshape(T + 1, B, -1)
        id_logits = agent.inv_dyn.apply_fn(
            {"params": inv_params}, emb[:T], emb[1 : T + 1]
        )
        id_ce = optax.softmax_cross_entropy_with_integer_labels(
            id_logits, batch["actions"]
        )
        inv_loss = id_ce.mean(axis=1).sum()
        inv_acc = (
            (jnp.argmax(id_logits, axis=-1) == batch["actions"])
            .astype(jnp.float32)
            .mean()
        )

        total = (
            pg_loss
            + agent.config["vf_coef"] * baseline_loss
            + agent.config["ent_coef"] * entropy_loss
            + inv_loss
        )
        return total, {
            "pg_loss": pg_loss,
            "baseline_loss": baseline_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy.mean(),
            "inv_dyn_loss": inv_loss,
            "inv_dyn_acc": inv_acc,
            "total_loss": total,
        }

    @jax.jit
    def update(agent, traj, bootstrap, init_lstm_state):
        state = agent.intrinsic_moments
        bonus_batch = traj["bonus"]
        new_count = bonus_batch.size
        new_sum = bonus_batch.sum()
        new_mean = new_sum / new_count
        curr_mean = state.sum / state.count
        new_m2 = ((bonus_batch - new_mean) ** 2).sum() + (
            (state.count * new_count)
            / (state.count + new_count)
            * (new_mean - curr_mean) ** 2
        )
        new_intrinsic_moments = RunningMoments(
            sum=state.sum + new_sum,
            m2=state.m2 + new_m2,
            count=state.count + new_count,
        )
        std = jnp.sqrt(new_intrinsic_moments.m2 / new_intrinsic_moments.count)
        intrinsic = jnp.where(std > 0, traj["bonus"] / std, traj["bonus"])

        total_rewards = traj["reward"] + agent.config["intrinsic_coef"] * intrinsic
        total_rewards = jnp.clip(total_rewards, -1.0, 1.0)
        discounts = agent.config["gamma"] * (1.0 - traj["done"])
        batch = dict(
            glyphs=traj["glyphs"],
            blstats=traj["blstats"],
            message=traj["message"],
            actions=traj["actions"],
            log_prob=traj["log_prob"],
            rewards=total_rewards,
            discounts=discounts,
            is_new_ep=traj["is_new_ep"],
            init_lstm_state=init_lstm_state,
        )

        def loss_fn(ac_p, enc_p, inv_p):
            return agent.total_loss(batch, bootstrap, ac_p, enc_p, inv_p)

        (_, info), (ac_g, enc_g, inv_g) = jax.value_and_grad(
            loss_fn, argnums=(0, 1, 2), has_aux=True
        )(
            agent.actor_critic.params,
            agent.encoder.params,
            agent.inv_dyn.params,
        )
        info = {
            **info,
            "intrinsic_mean": intrinsic.mean(),
            "intrinsic_std": intrinsic.std(),
            "ac_grad_norm": optax.global_norm(ac_g),
            "enc_grad_norm": optax.global_norm(enc_g),
            "inv_grad_norm": optax.global_norm(inv_g),
        }
        return (
            agent.replace(
                actor_critic=agent.actor_critic.apply_gradients(grads=ac_g),
                encoder=agent.encoder.apply_gradients(grads=enc_g),
                inv_dyn=agent.inv_dyn.apply_gradients(grads=inv_g),
                intrinsic_moments=new_intrinsic_moments,
            ),
            info,
        )

    @jax.jit
    def act(agent, glyphs, blstats, message, done, lstm_state, rng):
        rng, act_rng = jax.random.split(rng)
        logits, _, new_state = agent.actor_critic.apply_fn(
            {"params": agent.actor_critic.params},
            glyphs[None],
            blstats[None],
            message[None],
            done[None],
            lstm_state,
        )
        pi = distrax.Categorical(logits=logits[0])
        action = pi.sample(seed=act_rng)
        return action, pi.log_prob(action), new_state, rng

    @jax.jit
    def bonus(agent, glyphs, blstats, message, c_inv):
        h = agent.encoder.apply_fn(
            {"params": agent.encoder.params}, glyphs, blstats, message
        )

        def sherman_morrison_update(c_inv, h):
            u = c_inv @ h
            b = jnp.dot(h, u)
            return b, c_inv - jnp.outer(u, u) / (1.0 + b)

        return jax.vmap(sherman_morrison_update)(c_inv, h)


def create_learner(config, seed, blstats_size, num_actions):
    rng = jax.random.PRNGKey(seed)
    rng, ac_key, enc_key, inv_key = jax.random.split(rng, 4)

    # LSTM
    init_g = jnp.zeros((1, 1, GLYPH_H, GLYPH_W), dtype=jnp.int32)
    init_b = jnp.zeros((1, 1, blstats_size), dtype=jnp.float32)
    init_m = jnp.zeros((1, 1, 256), dtype=jnp.int32)
    init_done = jnp.zeros((1, 1), dtype=jnp.float32)
    init_lstm = (jnp.zeros((1, config.hidden_dim)), jnp.zeros((1, config.hidden_dim)))

    # Policy
    ac_def = NetHackPolicyNet(
        num_actions=num_actions,
        blstats_size=blstats_size,
        hidden_dim=config.hidden_dim,
        use_lstm=config.use_lstm,
    )
    ac_params = ac_def.init(ac_key, init_g, init_b, init_m, init_done, init_lstm)[
        "params"
    ]

    ac_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.rmsprop(config.lr, decay=0.99, momentum=0.0, eps=1e-5),
    )
    actor_critic = TrainState.create(apply_fn=ac_def.apply, params=ac_params, tx=ac_tx)

    # Embeddings
    enc_def = NetHackStateEmbeddingNet(
        blstats_size=blstats_size, hidden_dim=config.hidden_dim
    )
    single_g = jnp.zeros((1, GLYPH_H, GLYPH_W), dtype=jnp.int32)
    single_b = jnp.zeros((1, blstats_size), dtype=jnp.float32)
    single_m = jnp.zeros((1, 256), dtype=jnp.int32)
    enc_params = enc_def.init(enc_key, single_g, single_b, single_m)["params"]
    enc_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.predictor_lr)
    )
    encoder = TrainState.create(apply_fn=enc_def.apply, params=enc_params, tx=enc_tx)

    # Inverse Dynamics
    inv_def = MinigridInverseDynamicsNet(num_actions=num_actions)
    init_h = jnp.zeros((1, 1, config.hidden_dim))
    inv_params = inv_def.init(inv_key, init_h, init_h)["params"]
    inv_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.predictor_lr)
    )
    inv_dyn = TrainState.create(apply_fn=inv_def.apply, params=inv_params, tx=inv_tx)

    agent_config = flax.core.FrozenDict(
        dict(
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            gamma=config.gamma,
            rho_bar=config.rho_bar,
            c_bar=config.c_bar,
            intrinsic_coef=config.intrinsic_reward_coef,
        )
    )

    return E3B(
        rng=rng,
        actor_critic=actor_critic,
        encoder=encoder,
        inv_dyn=inv_dyn,
        intrinsic_moments=RunningMoments(sum=0.0, m2=0.0, count=1e-8),
        config=agent_config,
    )


@dataclass
class Config:
    # e3b
    ridge: float = 0.1
    predictor_lr: float = 1e-4
    intrinsic_reward_coef: float = 1.0

    # impala (NetHack as in original codebase)
    lr: float = 1e-4
    num_envs: int = 8
    num_steps: int = 80
    gamma: float = 0.99
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 40.0
    hidden_dim: int = 1024
    use_lstm: bool = True

    # v-trace
    rho_bar: float = 1.0
    c_bar: float = 1.0

    # train
    seed: int = 0
    env: str = "MiniHack-Labyrinth-Big-v0"
    wandb_project: str = "e3b-minihack"
    total_timesteps: int = 50_000_000
    log_every_n_updates: int = 50


if __name__ == "__main__":
    import importlib
    import time
    from collections import deque

    import gymnasium as gym
    import gymnasium.spaces as _gsp
    import tyro
    import wandb

    # this stuff is a bunch of hacks because minihack uses old gym and i haven't cleaned it up yet

    # minihack 1.0.0 registers in gymnasium but provides obs as old gym.spaces.Box
    # gymnasium.spaces.Dict asserts children are gymnasium.Space and converts if necessary
    try:
        import gym.spaces as _osp
    except ImportError:
        _osp = None

    def _convert_space(s):
        return s

    if _osp is not None:
        for n in ("Sequence", "Graph", "Text"):
            if not hasattr(_osp, n):
                setattr(_osp, n, type(n, (), {}))

        def _convert_space(s):
            if isinstance(s, _osp.Box):
                return _gsp.Box(low=s.low, high=s.high, shape=s.shape, dtype=s.dtype)
            if isinstance(s, _osp.Discrete):
                return _gsp.Discrete(int(s.n))
            if isinstance(s, _osp.Dict):
                return _gsp.Dict({k: _convert_space(v) for k, v in s.spaces.items()})
            return s

        _orig_dict_init = _gsp.Dict.__init__

        def _patched_dict_init(self, spaces=None, seed=None, **kwargs):
            if isinstance(spaces, dict):
                spaces = {k: _convert_space(v) for k, v in spaces.items()}
            _orig_dict_init(self, spaces=spaces, seed=seed, **kwargs)

        _gsp.Dict.__init__ = _patched_dict_init

    import minihack

    class _OldGymShim(gym.Env):
        def __init__(self, env, spec):
            self._env = env
            self.observation_space = _convert_space(env.observation_space)
            self.action_space = _convert_space(env.action_space)
            self.render_mode = None
            self.metadata = getattr(env, "metadata", {})
            self.spec = spec

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                try:
                    self._env.seed(seed)
                except AttributeError:
                    pass
            out = self._env.reset()
            if isinstance(out, tuple) and len(out) == 2:
                return out
            return out, {}

        def step(self, action):
            out = self._env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                return out
            obs, reward, done, info = out
            return obs, reward, done, False, info

        def close(self):
            return self._env.close()

    def _safe_make(env_id):
        spec = gym.spec(env_id)
        module_name, class_name = spec.entry_point.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        kwargs = dict(spec.kwargs or {})
        if "MiniHack" in env_id:
            kwargs["observation_keys"] = ("glyphs", "blstats", "chars", "message")
            kwargs["savedir"] = None
        return _OldGymShim(cls(**kwargs), spec)

    def _extract_obs(d):
        return (
            np.asarray(d["glyphs"], dtype=np.int32),
            np.asarray(d["blstats"], dtype=np.float32),
            np.asarray(d["message"], dtype=np.int32),
        )

    def main(cfg: Config):
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

        envs = gym.vector.SyncVectorEnv(
            [lambda: _safe_make(cfg.env) for _ in range(cfg.num_envs)]
        )
        num_actions = int(envs.single_action_space.n)

        obs_dict, _ = envs.reset(seed=cfg.seed)
        g_cur, b_cur, m_cur = _extract_obs(obs_dict)
        blstats_size = b_cur.shape[-1]

        agent = create_learner(cfg, cfg.seed, blstats_size, num_actions)

        H = cfg.hidden_dim
        B = cfg.num_envs
        T = cfg.num_steps
        I_over_ridge = jnp.eye(H) / cfg.ridge
        c_inv = jnp.broadcast_to(I_over_ridge, (B, H, H))
        lstm_state = (jnp.zeros((B, H)), jnp.zeros((B, H)))
        prev_done = jnp.ones(B, dtype=jnp.float32)
        rng = jax.random.PRNGKey(cfg.seed + 1)

        ep_returns = np.zeros(B, dtype=np.float32)
        ep_history = deque(maxlen=200)

        buf = dict(
            glyphs=np.zeros((T, B, GLYPH_H, GLYPH_W), dtype=np.int32),
            blstats=np.zeros((T, B, blstats_size), dtype=np.float32),
            message=np.zeros((T, B, 256), dtype=np.int32),
            actions=np.zeros((T, B), dtype=np.int32),
            log_prob=np.zeros((T, B), dtype=np.float32),
            reward=np.zeros((T, B), dtype=np.float32),
            bonus=np.zeros((T, B), dtype=np.float32),
            done=np.zeros((T, B), dtype=np.float32),
            is_new_ep=np.zeros((T, B), dtype=np.float32),
        )

        num_updates = cfg.total_timesteps // (T * B)
        start = time.time()

        for upd in range(num_updates):
            init_lstm_state = lstm_state

            for t in range(T):
                is_new_ep = np.asarray(prev_done)
                buf["glyphs"][t] = g_cur
                buf["blstats"][t] = b_cur
                buf["message"][t] = m_cur
                buf["is_new_ep"][t] = is_new_ep

                g_j, b_j, m_j = (
                    jnp.asarray(g_cur),
                    jnp.asarray(b_cur),
                    jnp.asarray(m_cur),
                )
                bonus, c_inv = agent.bonus(g_j, b_j, m_j, c_inv)
                # zero bonus on any step where s_t is a fresh reset state (torch step==0 semantics)
                bonus = jnp.where(prev_done.astype(jnp.bool_), 0.0, bonus)

                done_j = jnp.asarray(is_new_ep, dtype=jnp.float32)
                action, log_prob, lstm_state, rng = agent.act(
                    g_j, b_j, m_j, done_j, lstm_state, rng
                )
                action_np = np.asarray(action)

                next_obs, reward, term, trunc, _ = envs.step(action_np)
                done = np.logical_or(term, trunc)
                g_cur, b_cur, m_cur = _extract_obs(next_obs)

                buf["actions"][t] = action_np
                buf["log_prob"][t] = np.asarray(log_prob)
                buf["reward"][t] = reward.astype(np.float32)
                buf["bonus"][t] = np.asarray(bonus)
                buf["done"][t] = done.astype(np.float32)

                ep_returns += reward
                for i, d in enumerate(done):
                    if d:
                        ep_history.append(float(ep_returns[i]))
                        ep_returns[i] = 0.0

                prev_done = jnp.asarray(done, dtype=jnp.float32)
                # per-episode c_inv reset: after every done, restart the elliptical bonus from I/ridge
                c_inv = jnp.where(
                    prev_done.astype(jnp.bool_)[:, None, None], I_over_ridge, c_inv
                )

            bootstrap = dict(
                glyphs=jnp.asarray(g_cur),
                blstats=jnp.asarray(b_cur),
                message=jnp.asarray(m_cur),
                is_new_ep=prev_done,
            )
            traj = {k: jnp.asarray(v) for k, v in buf.items()}
            agent, info = agent.update(traj, bootstrap, init_lstm_state)

            if (upd + 1) % cfg.log_every_n_updates == 0:
                frames = (upd + 1) * T * B
                sps = frames / (time.time() - start)
                recent = float(np.mean(list(ep_history)[-50:])) if ep_history else 0.0
                wandb.log(
                    {
                        "train/total_loss": float(info["total_loss"]),
                        "train/pg_loss": float(info["pg_loss"]),
                        "train/baseline_loss": float(info["baseline_loss"]),
                        "train/entropy": float(info["entropy"]),
                        "train/inv_dyn_loss": float(info["inv_dyn_loss"]),
                        "train/inv_dyn_acc": float(info["inv_dyn_acc"]),
                        "train/intrinsic_mean": float(info["intrinsic_mean"]),
                        "train/intrinsic_std": float(info["intrinsic_std"]),
                        "train/ac_grad_norm": float(info["ac_grad_norm"]),
                        "train/enc_grad_norm": float(info["enc_grad_norm"]),
                        "train/inv_grad_norm": float(info["inv_grad_norm"]),
                        "train/ep_return_recent": recent,
                        "train/sps": sps,
                        "frames": frames,
                    },
                    step=frames,
                )

        wandb.finish()

    main(tyro.cli(Config))