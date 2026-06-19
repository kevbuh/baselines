# https://arxiv.org/pdf/1912.05510

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


class BernoulliBuffer:
    def __init__(self, dim):
        self.dim = dim; self.reset()

    def reset(self):
        self.buffer = np.zeros(self.dim); self.buffer_size = 1

    def add(self, obs):
        self.buffer += obs; self.buffer_size += 1

    def get_params(self):
        return np.clip(self.buffer / self.buffer_size, 1e-4, 1 - 1e-4)

    def logprob(self, obs):
        t = np.clip(self.get_params(), 1e-5, 1 - 1e-5)
        return np.sum(np.log(obs * t + (1 - obs) * (1 - t)))


class TetrisEnv:  # 4w x 10h, tromino pieces, Discrete(12) decoded via nextBlock
    def __init__(self, width=4, height=10, reward_func=None):
        self.width, self.height, self.reward_func = width, height, reward_func
        self.n_actions = 12
        self.obs_dim = width * height + 1  # grid + nextBlock
        self.reset()

    def chooseNextBlock(self):
        self.nextBlock = np.random.randint(2)

    def _rotateSquare(self, s):
        o = s.copy()
        o[0, 0], o[0, 1], o[1, 1], o[1, 0] = s[0, 1], s[1, 1], s[1, 0], s[0, 0]
        return o

    def getBlock(self, block_id, rotation, column):
        block = {0: np.array([[1, 1, 1]]), 1: np.array([[1, 1], [0, 1]])}[block_id]
        if block_id == 0:
            if rotation % 2 == 1:
                block = block.T
        else:
            for _ in range(rotation):
                block = self._rotateSquare(block)
        column = min(max(0, column), self.grid.shape[1] - block.shape[1])
        bg = np.zeros_like(self.grid)
        bg[0:block.shape[0], column:column + block.shape[1]] = block
        return bg

    def simulateDrop(self, block):
        prev, collision = None, False
        while not collision:
            dropping = self.grid + block
            if np.sum(dropping == 2) != 0:
                collision = True
            else:
                prev = dropping
                if np.sum(block[-1, :]) > 0:
                    collision = True
                block = np.roll(block, 1, axis=0)
        if prev is None:
            self.done = True
        else:
            self.grid = prev

    def destroyRows(self):
        cleared = 0
        for i in list(range(self.grid.shape[0]))[::-1]:
            while np.sum(self.grid[i, :]) == self.grid.shape[1]:
                cleared += 1
                self.grid[i, :] = 0
                self.grid[0:i + 1, :] = np.roll(self.grid[0:i + 1, :], 1, axis=0)
        return cleared

    def step(self, action):
        assert not self.done, "Can't take action after done"
        if self.nextBlock == 0:
            rotation, column = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)][action % 6]
        else:
            rotation, column = action % 4, action // 4
        self.simulateDrop(self.getBlock(self.nextBlock, rotation, column))
        rows = self.destroyRows()
        self.chooseNextBlock()
        r = rows if self.reward_func == "rows_cleared" else (-100 if self.done else 0)
        return self.get_obs(), r, self.done, {"rows_cleared": rows}

    def get_obs(self):
        return np.hstack((self.grid.flatten().copy(), self.nextBlock))

    def reset(self):
        self.chooseNextBlock()
        self.grid = np.zeros((self.height, self.width))
        self.done = False
        return self.get_obs()


class SoftResetWrapper:
    def __init__(self, env, max_time):
        self.env, self.max_time, self._t, self._last = env, max_time, 0, 0
        self.action_space_n, self.obs_dim = env.n_actions, env.obs_dim

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["life_length_avg"] = self._last
        if done:  # death -> uniform noise (maximally surprising), then continue
            o = self.reset(); info["death"] = 1; self._last = 0; obs = np.random.rand(*o.shape)
        else:
            info["death"] = 0
        self._last += 1
        return obs, rew, self._t >= self.max_time, info  # _t never increments (reference quirk)

    def reset(self):
        self._t = self._last = 0
        return self.env.reset()


class BaseSurpriseWrapper:  # augmented MDP; add_true_rew: False=SMiRL, 'only'=oracle, True=combined
    def __init__(self, env, buffer, time_horizon=100, add_true_rew=False):
        self.env, self.buffer = env, buffer
        self.time_horizon, self.add_true_rew = time_horizon, add_true_rew
        self.action_space_n = env.action_space_n
        self.obs_dim = env.obs_dim + buffer.get_params().size + 1
        self.reset()

    def _aug(self, obs):
        return np.concatenate([np.array(obs).flatten(), self.buffer.get_params().flatten(),
                               np.ones(1) * self.buffer.buffer_size])

    def step(self, action):
        obs, env_rew, envdone, info = self.env.step(action)
        info["task_reward"] = env_rew
        enc = np.array(obs).flatten().copy()
        rew = float(np.clip(self.buffer.logprob(enc), -300, 300))  # surprise, scored before adding
        self.buffer.add(enc)
        info["state_entropy_smirl"] = rew
        if self.add_true_rew == "only":
            rew = env_rew
        elif self.add_true_rew:
            rew = rew + env_rew
        self.t += 1
        return self._aug(obs), rew, (self.t >= self.time_horizon) or envdone, info

    def reset(self):
        obs = self.env.reset(); self.buffer.reset(); self.t = 0
        return self._aug(obs)


def make_tetris_smirl_env(add_true_rew=False, reward_func=None, time_horizon=100):
    env = SoftResetWrapper(TetrisEnv(reward_func=reward_func), max_time=500)
    return BaseSurpriseWrapper(env, BernoulliBuffer(env.obs_dim), time_horizon, add_true_rew)


# DQN matching rlkit DQNTrainer + Mlp [128,64,32] (fan-in init, tau=1e-3, plain DQN, MSE)
def _fanin(key, shape, dtype=jnp.float32):
    b = 1.0 / np.sqrt(shape[0])
    return jax.random.uniform(key, shape, dtype, -b, b)


def _small(key, shape, dtype=jnp.float32):
    return jax.random.uniform(key, shape, dtype, -3e-3, 3e-3)


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        for h in (128, 64, 32):
            x = nn.relu(nn.Dense(h, kernel_init=_fanin, bias_init=nn.initializers.constant(0.1))(x))
        return nn.Dense(self.action_dim, kernel_init=_small, bias_init=_small)(x)


class ReplayBuffer:
    def __init__(self, cap, dim):
        self.cap = cap; self.o = np.zeros((cap, dim), np.float32); self.no = np.zeros((cap, dim), np.float32)
        self.a = np.zeros(cap, np.int32); self.r = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)
        self.pos = 0; self.full = False

    def add(self, o, a, r, no, d):
        i = self.pos; self.o[i], self.no[i], self.a[i], self.r[i], self.d[i] = o, no, a, r, d
        self.pos = (i + 1) % self.cap; self.full = self.full or self.pos == 0

    def __len__(self):
        return self.cap if self.full else self.pos

    def sample(self, n, rng):
        i = rng.integers(0, len(self), n)
        return self.o[i], self.a[i], self.r[i], self.no[i], self.d[i]


@jax.jit
def _dqn_update(state, target_params, batch, gamma, tau):
    obs, act, rew, nobs, done = batch
    target = rew + (1.0 - done) * gamma * state.apply_fn(target_params, nobs).max(-1)

    def loss_fn(p):
        q = jnp.take_along_axis(state.apply_fn(p, obs), act[:, None], -1).squeeze(-1)
        return jnp.mean((q - jax.lax.stop_gradient(target)) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, optax.incremental_update(state.params, target_params, tau), loss


@dataclass
class Args:
    seed: int = 1
    reward_mode: Literal["smirl", "oracle", "combined"] = "smirl"
    total_timesteps: int = 800000
    target_tau: float = 1e-3
    eps_start: float = 0.8
    eps_end: float = 0.05
    eps_ramp: int = 100000


def train(a: Args):
    np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    atr = {"smirl": False, "oracle": "only", "combined": True}[a.reward_mode]
    env = make_tetris_smirl_env(add_true_rew=atr, reward_func=None if a.reward_mode == "smirl" else "rows_cleared")

    net = QNetwork(env.action_space_n)
    params = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, env.obs_dim)))
    state, target = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(3e-4)), params
    rb = ReplayBuffer(100_000, env.obs_dim)

    obs = env.reset(); es = et = 0.0; ed = er = el = ep = 0; H = {"d": [], "r": [], "s": []}
    t_greedy = 0; t0 = time.time()
    for gs in range(a.total_timesteps):
        eps = a.eps_start + (a.eps_end - a.eps_start) * min(1.0, t_greedy / a.eps_ramp)  # counts greedy steps only
        if rng.random() <= eps:
            act = int(rng.integers(env.action_space_n))
        else:
            act = int(jnp.argmax(net.apply(state.params, jnp.asarray(obs, jnp.float32)[None])[0])); t_greedy += 1
        nobs, rew, done, info = env.step(act)
        es += info["state_entropy_smirl"]; et += info["task_reward"]; ed += info["death"]; er += info["rows_cleared"]; el += 1
        rb.add(obs, act, rew, nobs, float(done)); obs = nobs
        if done:
            H["d"].append(ed); H["r"].append(er); H["s"].append(es / max(el, 1))
            if ep % 25 == 0:
                w = slice(-50, None)
                print(f"ep={ep} step={gs} deaths={np.mean(H['d'][w]):.2f} rows={np.mean(H['r'][w]):.2f} "
                      f"surprise/step={np.mean(H['s'][w]):.2f} task={et:.0f} eps={eps:.2f}")
            ep += 1; es = et = 0.0; ed = er = el = 0; obs = env.reset()
        if gs > 1000:
            b = tuple(jnp.asarray(x) for x in rb.sample(256, rng))
            state, target, _ = _dqn_update(state, target, b, jnp.float32(0.99), jnp.float32(a.target_tau))
    print(f"done. {int(a.total_timesteps / (time.time() - t0))} steps/s")
    return state, env


if __name__ == "__main__":
    train(tyro.cli(Args))
