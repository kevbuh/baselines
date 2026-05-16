# Paper:          https://arxiv.org/abs/1705.05363
# Reference impl: https://github.com/pathak22/noreward-rl

from dataclasses import dataclass
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


def cosine_loss(a, b):
    # 1 - mean(cos(a, b))
    a = a / (jnp.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (jnp.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - jnp.mean(jnp.sum(a * b, axis=1))


def categorical_sample(logits, ac_space: int, key):
    # Multinomial sample → (idx, one_hot)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    idx = jax.random.categorical(key, logits, axis=-1)
    return idx, jax.nn.one_hot(idx, ac_space)


# Conv weight init in the original is uniform(-w_bound, w_bound) with w_bound =
# sqrt(6/(fan_in+fan_out)), i.e. Glorot/Xavier uniform — inlined at each Conv/Dense


class UniverseHead(nn.Module):
    # 4 strided-2 ELU convs (32 filters): [B, 42, 42, 1] → [B, 288]

    n_convs: int = 4

    @nn.compact
    def __call__(self, x):
        for i in range(self.n_convs):
            x = nn.Conv(
                32,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name=f"l{i + 1}",
            )(x)
            x = nn.elu(x)
        return x.reshape(x.shape[0], -1)


class NipsHead(nn.Module):
    # DQN NIPS-2013 / A3C: [B,84,84,4] -> [B,256]

    @nn.compact
    def __call__(self, x):
        x = nn.relu(
            nn.Conv(
                16,
                (8, 8),
                (4, 4),
                padding="VALID",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l1",
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                32,
                (4, 4),
                (2, 2),
                padding="VALID",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l2",
            )(x)
        )
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(
            nn.Dense(
                256,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.zeros,
                name="fc",
            )(x)
        )
        return x


class NatureHead(nn.Module):
    # DQN Nature-2015: [B,84,84,4] -> [B,512]

    @nn.compact
    def __call__(self, x):
        x = nn.relu(
            nn.Conv(
                32,
                (8, 8),
                (4, 4),
                padding="VALID",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l1",
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                64,
                (4, 4),
                (2, 2),
                padding="VALID",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l2",
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                64,
                (3, 3),
                (1, 1),
                padding="VALID",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l3",
            )(x)
        )
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(
            nn.Dense(
                512,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.zeros,
                name="fc",
            )(x)
        )
        return x


class DoomHead(nn.Module):
    # ICLR-2017 Doom head: [B,120,160,1] -> [B,256]

    @nn.compact
    def __call__(self, x):
        x = nn.elu(
            nn.Conv(
                8,
                (5, 5),
                (4, 4),
                padding="SAME",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l1",
            )(x)
        )
        x = nn.elu(
            nn.Conv(
                16,
                (3, 3),
                (2, 2),
                padding="SAME",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l2",
            )(x)
        )
        x = nn.elu(
            nn.Conv(
                32,
                (3, 3),
                (2, 2),
                padding="SAME",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l3",
            )(x)
        )
        x = nn.elu(
            nn.Conv(
                64,
                (3, 3),
                (2, 2),
                padding="SAME",
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros,
                name="l4",
            )(x)
        )
        x = x.reshape(x.shape[0], -1)
        x = nn.elu(
            nn.Dense(
                256,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.zeros,
                name="fc",
            )(x)
        )
        return x


def make_head(design_head: str) -> nn.Module:
    # Pick conv head by name
    if design_head == "nips":
        return NipsHead()
    if design_head == "nature":
        return NatureHead()
    if design_head == "doom":
        return DoomHead()
    if "tile" in design_head:
        return UniverseHead(n_convs=2)
    return UniverseHead()


class InverseUniverseHead(nn.Module):
    # Inverse of UniverseHead: [B, 288] → [B, H, W, C]. ConvTranspose+crop with use_bias=False

    final_hwc: Tuple[int, int, int]
    n_convs: int = 4

    @nn.compact
    def __call__(self, x):
        H, W, C = self.final_hwc
        ds1 = [H]
        ds2 = [W]
        for _ in range(self.n_convs):
            ds1.append((ds1[-1] - 1) // 2 + 1)
            ds2.append((ds2[-1] - 1) // 2 + 1)
        # x: [B, prod] -> [B, ds1[-1], ds2[-1], 32]
        x = x.reshape(-1, ds1[-1], ds2[-1], 32)
        ds1 = ds1[:-1]
        ds2 = ds2[:-1]
        for i in range(self.n_convs - 1):
            x = nn.ConvTranspose(
                32,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                use_bias=False,
                kernel_init=nn.initializers.glorot_uniform(),
                name=f"dl{i + 1}",
            )(x)
            x = x[:, : ds1[-1], : ds2[-1], :]
            x = nn.elu(x)
            ds1 = ds1[:-1]
            ds2 = ds2[:-1]
        x = nn.ConvTranspose(
            C,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.glorot_uniform(),
            name=f"dl{self.n_convs}",
        )(x)
        x = x[:, :H, :W, :]
        return x


class StateActionPredictor(nn.Module):
    # Shared encoder φ + inverse g(φ1, φ2)→logits + forward f(φ1, a_oh)→φ2

    ac_space: int
    design_head: str = "universe"
    feat_size: int = 256

    @nn.compact
    def __call__(self, s1, s2, asample):
        # Shared encoder: same Flax submodule applied to both inputs reuses
        # the same params, equivalent to TF variable_scope reuse=True
        head = make_head(self.design_head)
        phi1 = head(s1)
        phi2 = head(s2)
        len_features = phi1.shape[-1]

        # Inverse model: g(phi1, phi2) -> action logits
        g = jnp.concatenate([phi1, phi2], axis=1)
        g = nn.relu(
            nn.Dense(
                self.feat_size,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.zeros,
                name="g1",
            )(g)
        )
        logits = nn.Dense(
            self.ac_space,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="glast",
        )(g)
        ainvprobs = nn.softmax(logits, axis=-1)

        # Forward model: f(phi1, asample) -> phi2_pred. asample comes from a
        # categorical sample so it's already detached from policy gradients
        f_in = jnp.concatenate([phi1, asample], axis=1)
        f = nn.relu(
            nn.Dense(
                self.feat_size,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.zeros,
                name="f1",
            )(f_in)
        )
        phi2_pred = nn.Dense(
            len_features,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="flast",
        )(f)
        return phi1, phi2, logits, ainvprobs, phi2_pred


def icm_inv_loss(logits, asample):
    # Sparse softmax cross-entropy of inverse model
    aindex = jnp.argmax(asample, axis=1)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, aindex))


def icm_forward_loss(phi2, phi2_pred):
    # 0.5 * sum-over-features MSE (factor of len_features makes hps feat-size-invariant)
    len_features = phi2.shape[-1]
    return 0.5 * jnp.mean(jnp.square(phi2_pred - phi2)) * float(len_features)


def icm_predloss(
    inv_loss,
    forward_loss,
    forward_loss_wt: float = 0.2,
    prediction_lr_scale: float = 10.0,
):
    # Combined predictor loss for the action-prediction variant
    return prediction_lr_scale * (
        (1.0 - forward_loss_wt) * inv_loss + forward_loss_wt * forward_loss
    )


def icm_bonus(phi2, phi2_pred, beta: float = 0.01):
    # Per-sample intrinsic bonus = 0.5 * beta * sum_d (φ2_pred - φ2)^2, shape [B]
    return 0.5 * jnp.sum(jnp.square(phi2_pred - phi2), axis=-1) * beta


class StatePredictor(nn.Module):
    # Pixel-space baseline: encode phi1 then predict s2; unsup='stateAenc' adds AE bonus

    ac_space: int
    obs_hwc: Tuple[int, int, int]
    design_head: str = "universe"
    unsup_type: str = "state"  # 'state' or 'stateAenc'

    @nn.compact
    def __call__(self, s1, s2, asample):
        if self.design_head != "universe" and "tile" not in self.design_head:
            raise NotImplementedError(
                "Only universe/tile designHead implemented for state prediction."
            )
        n_convs = 2 if "tile" in self.design_head else 4
        head = UniverseHead(n_convs=n_convs)
        phi1 = head(s1)
        len_features = phi1.shape[-1]

        f = jnp.concatenate([phi1, asample], axis=1)
        f = nn.relu(
            nn.Dense(
                len_features,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.zeros,
                name="f1",
            )(f)
        )
        s2_pred = InverseUniverseHead(final_hwc=self.obs_hwc, n_convs=n_convs)(f)
        forward_loss = 0.5 * jnp.mean(jnp.square(s2_pred - s2))

        # phi2_aenc is needed for per-sample autoencoding bonus (see compute_bonus
        # dispatch in main); we return it instead of folding into a scalar
        phi2_aenc = None
        aenc_bonus = None
        if self.unsup_type == "stateAenc":
            phi2_aenc = head(s2)  # shared encoder, params reused
            aenc_bonus = 0.5 * jnp.mean(jnp.square(phi1 - phi2_aenc))
        return phi1, s2_pred, forward_loss, phi2_aenc, aenc_bonus


@dataclass
class Config:
    # ICM hyperparameters (constants.py)
    prediction_beta: float = 0.01  # 0.5 for unsup=state
    prediction_lr_scale: float = 10.0  # 30-50 for unsup=state
    forward_loss_wt: float = 0.2  # predloss = (1-wt)*inv_loss + wt*forward_loss

    # optimizer (constants.py)
    lr: float = 1e-4
    grad_norm_clip: float = 40.0

    # rollout (constants.py)
    rollout_maxlen: int = 20  # also the "batch=20" gradient scale factor (a3c.py:299)
    gamma: float = 0.99
    lambda_: float = 1.0
    entropy_beta: float = 0.01
    reward_clip: float = 1.0
    policy_no_backprop_steps: int = 0

    # architecture
    design_head: str = "universe"
    unsup_type: str = "action"  # 'action' (ICM), 'state', 'stateAenc'
    lstm_size: int = 256

    # env (VizDoom DoomMyWayHome — original used Fixed=sparse, Fixed15=verySparse)
    env: str = "doomMyWayHomeFixed"  # one of: doomMyWayHomeFixed, doomMyWayHomeFixed15
    no_life_reward: bool = False  # NoNegativeRewardEnv (env_wrapper.py:99)
    frame_skip: int = 4  # SkipEnv + BufferedObsEnv skip (envs.py:75-80)
    frame_stack: int = 4  # BufferedObsEnv n=4 (env_wrapper.py:30)
    frame_shape: int = 42  # BufferedObsEnv shape=(42,42) (envs.py:74)
    no_reward: bool = False  # train.py noReward flag (a3c.py:178)
    wad_dir: str = ""  # override; default = external/noreward-rl/doomFiles/wads

    # training
    num_envs: int = 20
    total_timesteps: int = 100_000_000
    seed: int = 0
    wandb_project: str = "icm-jax"
    log_every_n_updates: int = 10


def create_predictor(
    rng,
    obs_shape: Sequence[int],
    ac_space: int,
    design_head: str = "universe",
    unsup_type: str = "action",
    lr: float = 1e-4,
    grad_norm_clip: float = 40.0,
):
    """Initialize the predictor and its optimizer chain.

    Adam at lr=1e-4 with grads clipped at 40
    (a3c.py:353,367). The "*20.0" gradient scaling in a3c.py:308 is the
    rollout length factored out of the loss — the helper functions here
    (icm_predloss, state_action_predictor_losses) return the unscaled value
    matching a3c.py:306-307; the *ROLLOUT_MAXLEN multiplier is applied at the
    grad call site (see make_predictor_loss_fn / policy_loss_fn in main).
    """
    if unsup_type == "action":
        model = StateActionPredictor(ac_space=ac_space, design_head=design_head)
        init_inputs = (
            jnp.zeros((1,) + tuple(obs_shape)),
            jnp.zeros((1,) + tuple(obs_shape)),
            jnp.zeros((1, ac_space)),
        )
    elif unsup_type in ("state", "stateAenc"):
        if len(obs_shape) != 3:
            raise ValueError("StatePredictor requires (H, W, C) obs_shape.")
        model = StatePredictor(
            ac_space=ac_space,
            obs_hwc=tuple(obs_shape),
            design_head=design_head,
            unsup_type=unsup_type,
        )
        init_inputs = (
            jnp.zeros((1,) + tuple(obs_shape)),
            jnp.zeros((1,) + tuple(obs_shape)),
            jnp.zeros((1, ac_space)),
        )
    else:
        raise ValueError(f"Unknown unsup_type: {unsup_type}")

    params = model.init(rng, *init_inputs)
    tx = optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.adam(lr),
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


def state_action_predictor_losses(
    params,
    model,
    s1,
    s2,
    asample,
    forward_loss_wt: float = 0.2,
    prediction_lr_scale: float = 10.0,
    prediction_beta: float = 0.01,
):
    """One-shot evaluator: returns (predloss, aux) where aux carries all the
    component scalars and the per-sample bonus tensor. Mirrors the bookkeeping
    in a3c.py:302-308 + the bonus return path in model.py:299-312.
    """
    _, phi2, logits, ainvprobs, phi2_pred = model.apply(params, s1, s2, asample)
    inv_loss = icm_inv_loss(logits, asample)
    forward_loss = icm_forward_loss(phi2, phi2_pred)
    predloss = icm_predloss(
        inv_loss,
        forward_loss,
        forward_loss_wt=forward_loss_wt,
        prediction_lr_scale=prediction_lr_scale,
    )
    bonus = icm_bonus(phi2, phi2_pred, beta=prediction_beta)
    return predloss, {
        "inv_loss": inv_loss,
        "forward_loss": forward_loss,
        "predloss": predloss,
        "bonus": bonus,
        "ainvprobs": ainvprobs,
    }


class LSTMPolicy(nn.Module):
    """Faithful port of LSTMPolicy. Encoder -> LSTM(256) -> {logits, value}.

    Operates on time-major batches: obs is [T, B, *obs_shape], dones is [T, B],
    init_done is [B], lstm_state is (c, h) each [B, lstm_size]. Returns
    (logits, value, new_state) where logits is [T, B, ac_space] and value is
    [T, B]. The original passes the full rollout through dynamic_rnn
    (model.py:196) and uses the per-step outputs for both losses; we mirror that.

    a3c.py:211 calls `last_features = policy.get_initial_features()` AFTER an
    episode ends. So at step t the carry should be zeroed iff the previous step
    (t-1) terminated. We track that as a shifted-dones array:
        shifted[0]   = init_done                   (= done at the last step
                                                     of the PREVIOUS rollout)
        shifted[t>0] = dones[t-1]                  (= done at step t-1 of THIS
                                                     rollout)
    `init_done` must be the `prev_done` saved just before the rollout loop.
    """

    ac_space: int
    design_head: str = "universe"
    lstm_size: int = 256

    def setup(self):
        self.encoder = make_head(self.design_head)
        self.cell = nn.OptimizedLSTMCell(features=self.lstm_size)
        self.value_head = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="value",
        )
        self.action_head = nn.Dense(
            self.ac_space,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="action",
        )

    def __call__(self, obs, dones, init_done, lstm_state):
        T, B = obs.shape[:2]
        flat = obs.reshape((T * B,) + obs.shape[2:])
        feat = self.encoder(flat).reshape(T, B, -1)
        # shifted_dones[t] = "did the env terminate at step t-1?", with t=0
        # picking up `init_done` from the previous rollout's last step
        shifted_dones = jnp.concatenate([init_done[None], dones[:-1]], axis=0)
        carry = lstm_state
        outs = []
        for t in range(T):
            notdone = (1.0 - shifted_dones[t])[:, None]
            carry = (carry[0] * notdone, carry[1] * notdone)
            carry, out = self.cell(carry, feat[t])
            outs.append(out)
        core = jnp.stack(outs, axis=0)
        flat_core = core.reshape(T * B, -1)
        logits = self.action_head(flat_core).reshape(T, B, self.ac_space)
        value = jnp.squeeze(self.value_head(flat_core), -1).reshape(T, B)
        return logits, value, carry

    def initial_state(self, batch_size: int):
        return (
            jnp.zeros((batch_size, self.lstm_size)),
            jnp.zeros((batch_size, self.lstm_size)),
        )


def create_policy(
    rng,
    obs_shape: Sequence[int],
    ac_space: int,
    design_head: str = "universe",
    lstm_size: int = 256,
    lr: float = 1e-4,
    grad_norm_clip: float = 40.0,
):
    model = LSTMPolicy(ac_space=ac_space, design_head=design_head, lstm_size=lstm_size)
    init_obs = jnp.zeros((1, 1) + tuple(obs_shape))
    init_dones = jnp.zeros((1, 1))
    init_done0 = jnp.zeros((1,))
    init_state = model.initial_state(1)
    params = model.init(rng, init_obs, init_dones, init_done0, init_state)
    # tf.contrib.rnn.BasicLSTMCell defaults forget_bias=1.0; Flax's LSTMCell
    # uses 0, so set f-gate bias to ones at init to match
    params["params"]["cell"]["hf"]["bias"] = jnp.ones_like(
        params["params"]["cell"]["hf"]["bias"]
    )
    tx = optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.adam(lr),
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


if __name__ == "__main__":
    """The DoomMyWayHomeFixed (sparse) and DoomMyWayHomeFixed15 (very sparse) WADs
    live in external/noreward-rl/doomFiles/wads/. The base scenario .cfg ships with
    the vizdoom pip package (vizdoom.scenarios.my_way_home.cfg). We override
    set_doom_scenario_path to point at the project-local sparse WAD.

    Original env composition (envs.py:35-87, env_wrapper.py:30-95):
        gym.make('ppaquette/DoomMyWayHomeFixed-v0')  # sparse
          -> SetPlayingMode('algo')
          -> SetResolution('160x120')
          -> ToDiscrete('minimal')                   # NOOP, FORWARD, RIGHT, LEFT
          -> BufferedObsEnv(skip=4, shape=(42,42), n=4, maxFrames=True)
          -> SkipEnv(skip=4)                         # action repeat 4

    We reproduce all of that on top of the modern `vizdoom` package directly,
    skipping the long-dead ppaquette gym wrappers. The mathematical content of
    the env wrapping is preserved: max-pool over 2 raw frames -> RGB->Y -> 42x42
    bilinear -> stack 4 frames -> /255 -> action repeat 4.
    """
    import os
    import time
    import urllib.request
    from collections import deque
    from pathlib import Path

    import numpy as np
    import tyro
    import vizdoom as vzd
    import wandb
    from PIL import Image

    # --- env -------------------------------------------------------------

    DOOM_FILES_DIR = Path(__file__).resolve().parents[1] / "doom_files"
    DOOM_WAD_URLS = {
        "my_way_home_sparse.wad": "https://raw.githubusercontent.com/pathak22/noreward-rl/master/doomFiles/wads/my_way_home_sparse.wad",
        "my_way_home_verySparse.wad": "https://raw.githubusercontent.com/pathak22/noreward-rl/master/doomFiles/wads/my_way_home_verySparse.wad",
    }

    DOOM_VARIANTS = {
        # name -> (wad_filename, env_id_for_logs)
        "doomMyWayHomeFixed": ("my_way_home_sparse.wad", "DoomMyWayHomeFixed-v0"),
        "doomMyWayHomeFixed15": (
            "my_way_home_verySparse.wad",
            "DoomMyWayHomeFixed15-v0",
        ),
    }

    def resolve_wad(wad_name: str, wad_dir_override: str = "") -> Path:
        # Locate a Doom WAD or fetch it from upstream into <repo>/doom_files/
        base = Path(wad_dir_override) if wad_dir_override else DOOM_FILES_DIR
        base.mkdir(parents=True, exist_ok=True)
        path = base / wad_name
        if path.exists():
            return path
        url = DOOM_WAD_URLS.get(wad_name)
        if url is None:
            raise FileNotFoundError(
                f"No WAD at {path} and no upstream URL registered for {wad_name}."
            )
        print(f"Downloading {wad_name} from {url} -> {path}")
        urllib.request.urlretrieve(url, path)
        return path

    class DoomMyWayHome:
        # Single VizDoom env mirroring the original ICM Doom config

        BUTTONS = (
            vzd.Button.MOVE_FORWARD,  # action_idx 13
            vzd.Button.TURN_RIGHT,  # action_idx 14
            vzd.Button.TURN_LEFT,  # action_idx 15
        )
        ACTIONS = [
            [0, 0, 0],  # NOOP
            [1, 0, 0],  # FORWARD
            [0, 1, 0],  # TURN_RIGHT
            [0, 0, 1],  # TURN_LEFT
        ]
        N_ACTIONS = 4

        def __init__(
            self,
            wad_path: Path,
            frame_skip: int = 4,
            frame_stack: int = 4,
            frame_shape: int = 42,
            no_life_reward: bool = False,
            no_reward: bool = False,
            seed: int = 0,
        ):
            game = vzd.DoomGame()
            # Base config: my_way_home.cfg ships with the vizdoom pip package
            game.load_config(os.path.join(vzd.scenarios_path, "my_way_home.cfg"))
            # Override scenario WAD to the sparse / verySparse variant
            game.set_doom_scenario_path(str(wad_path))
            game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
            game.set_screen_format(vzd.ScreenFormat.GRAY8)
            game.set_window_visible(False)
            game.set_mode(vzd.Mode.PLAYER)
            game.add_game_args("-nomonsters 1")  # doom_env.py:145
            game.set_doom_skill(5)  # DOOM_SETTINGS[9][DIFFICULTY]
            game.set_episode_timeout(2100)  # 1 min @ 35 fps, doc string
            # Set the buttons explicitly (cfg already declares them but be safe)
            game.set_available_buttons(list(self.BUTTONS))
            game.set_seed(seed)
            game.init()
            self.game = game

            self.frame_skip = frame_skip
            self.frame_stack = frame_stack
            self.frame_shape = (frame_shape, frame_shape)
            self.no_life_reward = no_life_reward
            self.no_reward = no_reward

            # obs_buffer is only populated on skip=1 paths; for skip>=2 the
            # max-pool pair is captured within a single step() call
            self.obs_buffer = deque(maxlen=2)
            self.stack = deque(maxlen=frame_stack)

        def _resize_normalize(self, gray):
            # Pillow BILINEAR + /255
            small = np.array(
                Image.fromarray(gray).resize(self.frame_shape, resample=Image.BILINEAR),
                dtype=np.uint8,
            )
            return small.astype(np.float32) / 255.0

        def _stacked(self):
            return np.stack(list(self.stack), axis=-1)  # (H, W, n)

        def reset(self):
            self.game.new_episode()
            self.obs_buffer.clear()
            self.stack.clear()
            raw = np.asarray(self.game.get_state().screen_buffer, dtype=np.uint8)
            obs = self._resize_normalize(raw)
            for _ in range(self.frame_stack - 1):
                self.stack.append(np.zeros_like(obs))
            self.stack.append(obs)
            return self._stacked()

        def step(self, action_idx: int):
            buttons = self.ACTIONS[int(action_idx)]
            skip = self.frame_skip

            if skip >= 2:
                # Match BufferedObsEnv: appended frame is max-pool over
                # (raw_{skip-1}, raw_{skip}); advance skip-1 then 1 more tic
                reward = self.game.make_action(buttons, skip - 1)
                done = self.game.is_episode_finished()
                if done:
                    frame_pair = None
                else:
                    frame_a = np.asarray(
                        self.game.get_state().screen_buffer, dtype=np.uint8
                    )
                    reward = reward + self.game.make_action(buttons, 1)
                    done = self.game.is_episode_finished()
                    if done:
                        frame_pair = None
                    else:
                        frame_b = np.asarray(
                            self.game.get_state().screen_buffer, dtype=np.uint8
                        )
                        frame_pair = np.maximum(frame_a, frame_b)
            else:
                # skip=1: max-pool with the previous outer step's raw frame
                # via obs_buffer, mirroring BufferedObsEnv at skip=1
                reward = self.game.make_action(buttons, 1)
                done = self.game.is_episode_finished()
                if done:
                    frame_pair = None
                else:
                    raw = np.asarray(
                        self.game.get_state().screen_buffer, dtype=np.uint8
                    )
                    self.obs_buffer.append(raw)
                    frame_pair = np.max(np.stack(self.obs_buffer), axis=0)

            if done:
                # Append a zero frame on done for shape parity and auto-reset;
                # GAE `last_value * (1 - done)` masks the terminal obs
                obs_small = np.zeros(self.frame_shape, dtype=np.float32)
                self.stack.append(obs_small)
                stacked = self._stacked()
                self.reset()
            else:
                obs_small = self._resize_normalize(frame_pair)
                self.stack.append(obs_small)
                stacked = self._stacked()

            if self.no_life_reward and reward < 0:
                reward = 0.0
            if self.no_reward:
                reward = 0.0
            return stacked, float(reward), bool(done)

        def close(self):
            self.game.close()

    class VecDoom:
        def __init__(self, num_envs: int, **kwargs):
            seed = kwargs.pop("seed", 0)
            self.envs = [
                DoomMyWayHome(seed=seed + i, **kwargs) for i in range(num_envs)
            ]
            self.num_envs = num_envs

        def reset(self):
            return np.stack([e.reset() for e in self.envs], axis=0)

        def step(self, actions):
            obs, rew, done = [], [], []
            for e, a in zip(self.envs, actions):
                o, r, d = e.step(int(a))
                obs.append(o)
                rew.append(r)
                done.append(d)
            return (
                np.stack(obs, 0),
                np.array(rew, dtype=np.float32),
                np.array(done, dtype=np.float32),
            )

        def close(self):
            for e in self.envs:
                e.close()

    # --- losses (jit-compiled) ------------------------------------------

    def discount_scan(x, gamma):
        """Equivalent to a3c.py:13-21 scipy.signal.lfilter discount.
        x: [T, B]; returns [T, B] with G_t = sum_{k>=0} gamma^k * x_{t+k}.
        """

        def step(carry, x_t):
            new = x_t + gamma * carry
            return new, new

        _, out = jax.lax.scan(step, jnp.zeros_like(x[0]), x, reverse=True)
        return out

    def policy_loss_fn(
        policy_params, model, batch, init_lstm, ent_beta, rollout_maxlen
    ):
        # batch: dict with obs, dones, init_done, actions_oh, advantages, returns;
        # obs/dones/actions_oh/advantages/returns shaped [T, B, ...], init_done [B]
        logits, values, _ = model.apply(
            policy_params,
            batch["obs"],
            batch["dones"],
            batch["init_done"],
            init_lstm,
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1)

        # all reductions are tf.reduce_mean over the flattened
        # [T*B, ...] dim. We mirror that with `mean()` (mean over T and B)
        log_pi_a = jnp.sum(log_probs * batch["actions_oh"], axis=-1)
        pi_loss = -(log_pi_a * batch["advantages"]).mean()
        vf_loss = 0.5 * ((values - batch["returns"]) ** 2).mean()
        entropy = -(probs * log_probs).sum(axis=-1).mean()

        # total = pi_loss + 0.5*vf_loss - entropy*ENTROPY_BETA
        total = pi_loss + 0.5 * vf_loss - entropy * ent_beta
        # *rollout_maxlen (=20) is a per-rollout gradient scale; applied to the
        # differentiated value so unscaled `total` is still reported in aux
        return total * rollout_maxlen, {
            "pi_loss": pi_loss,
            "vf_loss": vf_loss,
            "entropy": entropy,
            "policy_total": total,
        }

    def make_predictor_loss_fn(unsup_type: str):
        """a3c.py:302-308 dispatched on unsupType.

        For 'action' (ICM): predloss = PREDICTION_LR_SCALE *
            ((1-FORWARD_LOSS_WT)*invloss + FORWARD_LOSS_WT*forwardloss).
        For 'state'/'stateAenc' (state prediction): predloss =
            PREDICTION_LR_SCALE * forwardloss; aencBonus is NOT part of the
            loss (it's only the bonus, per StatePredictor.pred_bonus).
        Both branches multiply by ROLLOUT_MAXLEN to match a3c.py:308's *20.
        """
        if unsup_type == "action":

            def predictor_loss_fn(
                predictor_params,
                predictor_model,
                s1,
                s2,
                asample,
                forward_loss_wt,
                prediction_lr_scale,
                rollout_maxlen,
            ):
                s1f = s1.reshape((-1,) + s1.shape[2:])
                s2f = s2.reshape((-1,) + s2.shape[2:])
                af = asample.reshape((-1, asample.shape[-1]))
                _, phi2, logits, _, phi2_pred = predictor_model.apply(
                    predictor_params, s1f, s2f, af
                )
                inv = icm_inv_loss(logits, af)
                fwd = icm_forward_loss(phi2, phi2_pred)
                pred = icm_predloss(
                    inv,
                    fwd,
                    forward_loss_wt=forward_loss_wt,
                    prediction_lr_scale=prediction_lr_scale,
                )
                return pred * rollout_maxlen, {
                    "inv_loss": inv,
                    "forward_loss": fwd,
                    "predloss": pred,
                }

            return predictor_loss_fn
        if unsup_type in ("state", "stateAenc"):

            def predictor_loss_fn(
                predictor_params,
                predictor_model,
                s1,
                s2,
                asample,
                forward_loss_wt,  # unused in state-prediction branch (kept for sig parity with action branch)
                prediction_lr_scale,
                rollout_maxlen,
            ):
                del forward_loss_wt
                s1f = s1.reshape((-1,) + s1.shape[2:])
                s2f = s2.reshape((-1,) + s2.shape[2:])
                af = asample.reshape((-1, asample.shape[-1]))
                _, _, forward_loss, _, _ = predictor_model.apply(
                    predictor_params, s1f, s2f, af
                )
                pred = prediction_lr_scale * forward_loss
                return pred * rollout_maxlen, {
                    "inv_loss": jnp.zeros(()),
                    "forward_loss": forward_loss,
                    "predloss": pred,
                }

            return predictor_loss_fn
        raise ValueError(f"Unknown unsup_type: {unsup_type}")

    def main(cfg: Config):
        if cfg.env not in DOOM_VARIANTS:
            raise ValueError(
                f"Unknown env {cfg.env}; choose from {list(DOOM_VARIANTS)}"
            )
        wad_name, env_id = DOOM_VARIANTS[cfg.env]
        wad_path = resolve_wad(wad_name, cfg.wad_dir)

        wandb.init(
            project=cfg.wandb_project, config={**vars(cfg), "vizdoom_env_id": env_id}
        )

        envs = VecDoom(
            cfg.num_envs,
            wad_path=wad_path,
            frame_skip=cfg.frame_skip,
            frame_stack=cfg.frame_stack,
            frame_shape=cfg.frame_shape,
            no_life_reward=cfg.no_life_reward,
            no_reward=cfg.no_reward,
            seed=cfg.seed,
        )

        ac_space = DoomMyWayHome.N_ACTIONS
        obs_shape = (cfg.frame_shape, cfg.frame_shape, cfg.frame_stack)

        rng = jax.random.PRNGKey(cfg.seed)
        rng, pkey, qkey = jax.random.split(rng, 3)
        policy_model, policy_state = create_policy(
            pkey,
            obs_shape,
            ac_space,
            design_head=cfg.design_head,
            lstm_size=cfg.lstm_size,
            lr=cfg.lr,
            grad_norm_clip=cfg.grad_norm_clip,
        )
        predictor_model, predictor_state = create_predictor(
            qkey,
            obs_shape,
            ac_space,
            design_head=cfg.design_head,
            unsup_type=cfg.unsup_type,
            lr=cfg.lr,
            grad_norm_clip=cfg.grad_norm_clip,
        )

        @jax.jit
        def act(policy_params, obs, prev_done, lstm_state, key):
            # obs: [B, *obs_shape]; prev_done: [B]. Single-step (T=1) — the
            # `dones` slot is unused (only init_done drives the carry mask)
            dummy_dones = jnp.zeros((1,) + prev_done.shape, dtype=prev_done.dtype)
            logits, value, new_state = policy_model.apply(
                policy_params, obs[None], dummy_dones, prev_done, lstm_state
            )
            # categorical_sample returns (idx, one-hot)
            action, _ = categorical_sample(logits[0], ac_space, key)
            return action, value[0], new_state

        # bonus dispatch: action=inverse-encoded forward error, state=pixel MSE,
        # stateAenc=feature-space autoencoding error against a shared encoder
        if cfg.unsup_type == "action":

            @jax.jit
            def compute_bonus(predictor_params, s1, s2, action_oh):
                _, phi2, _, _, phi2_pred = predictor_model.apply(
                    predictor_params, s1, s2, action_oh
                )
                return icm_bonus(phi2, phi2_pred, beta=cfg.prediction_beta)

        elif cfg.unsup_type == "state":

            @jax.jit
            def compute_bonus(predictor_params, s1, s2, action_oh):
                _, s2_pred, _, _, _ = predictor_model.apply(
                    predictor_params, s1, s2, action_oh
                )
                # Per-sample 0.5 * mean(square) over spatial dims, * beta
                spatial_axes = tuple(range(1, s2.ndim))
                return (
                    0.5
                    * jnp.mean(jnp.square(s2_pred - s2), axis=spatial_axes)
                    * cfg.prediction_beta
                )

        elif cfg.unsup_type == "stateAenc":

            @jax.jit
            def compute_bonus(predictor_params, s1, s2, action_oh):
                phi1, _, _, phi2_aenc, _ = predictor_model.apply(
                    predictor_params, s1, s2, action_oh
                )
                # Per-sample 0.5 * mean(square) over feature dim, * beta
                return (
                    0.5
                    * jnp.mean(jnp.square(phi1 - phi2_aenc), axis=-1)
                    * cfg.prediction_beta
                )

        else:
            raise ValueError(f"Unknown unsup_type: {cfg.unsup_type}")

        # Bind the unsup-type-specific predictor loss fn so update() picks up
        # the right branch via closure
        predictor_loss_fn_local = make_predictor_loss_fn(cfg.unsup_type)

        @jax.jit
        def update(policy_state, predictor_state, batch, init_lstm, policy_no_backprop):
            (_, p_info), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(
                policy_state.params,
                policy_model,
                batch,
                init_lstm,
                cfg.entropy_beta,
                cfg.rollout_maxlen,
            )
            # zero policy grads while predictor warms up
            p_grads = jax.tree_util.tree_map(lambda g: g * policy_no_backprop, p_grads)
            policy_state = policy_state.apply_gradients(grads=p_grads)

            (_, q_info), q_grads = jax.value_and_grad(
                predictor_loss_fn_local, has_aux=True
            )(
                predictor_state.params,
                predictor_model,
                batch["obs_s1"],
                batch["obs_s2"],
                batch["actions_oh"],
                cfg.forward_loss_wt,
                cfg.prediction_lr_scale,
                cfg.rollout_maxlen,
            )
            predictor_state = predictor_state.apply_gradients(grads=q_grads)
            return policy_state, predictor_state, {**p_info, **q_info}

        T = cfg.rollout_maxlen
        B = cfg.num_envs

        obs = envs.reset()  # (B, H, W, C)
        prev_done = np.zeros(B, dtype=np.float32)
        lstm_state = policy_model.initial_state(B)
        ep_return = np.zeros(B, dtype=np.float32)
        ep_history = deque(maxlen=200)
        bonus_history = deque(maxlen=200)

        # Buffers
        buf_obs = np.zeros((T, B) + obs_shape, dtype=np.float32)
        buf_obs_next = np.zeros((T, B) + obs_shape, dtype=np.float32)
        buf_action = np.zeros((T, B), dtype=np.int32)
        buf_value = np.zeros((T, B), dtype=np.float32)
        buf_reward = np.zeros((T, B), dtype=np.float32)
        buf_bonus = np.zeros((T, B), dtype=np.float32)
        buf_done = np.zeros((T, B), dtype=np.float32)

        num_updates = cfg.total_timesteps // (T * B * cfg.frame_skip)
        global_step = 0
        start = time.time()

        for upd in range(num_updates):
            init_lstm = lstm_state  # for re-running policy on the rollout
            # `init_done` is the done flag at the step BEFORE this rollout; it
            # masks the LSTM carry going into step 0
            init_done = jnp.asarray(prev_done)

            for t in range(T):
                rng, key = jax.random.split(rng)
                obs_j = jnp.asarray(obs)
                pd_j = jnp.asarray(prev_done)
                action, value, lstm_state = act(
                    policy_state.params, obs_j, pd_j, lstm_state, key
                )
                action_np = np.asarray(action)
                next_obs, reward, done = envs.step(action_np)

                # ICM bonus on (s_t, s_{t+1}, a_t) — predictor.pred_bonus convention
                action_oh = jax.nn.one_hot(jnp.asarray(action_np), ac_space)
                bonus = compute_bonus(
                    predictor_state.params, obs_j, jnp.asarray(next_obs), action_oh
                )
                bonus_np = np.asarray(bonus)

                buf_obs[t] = obs
                buf_obs_next[t] = next_obs
                buf_action[t] = action_np
                buf_value[t] = np.asarray(value)
                buf_reward[t] = reward
                buf_bonus[t] = bonus_np
                buf_done[t] = done

                ep_return += reward
                for i, d in enumerate(done):
                    if d:
                        ep_history.append(float(ep_return[i]))
                        ep_return[i] = 0.0
                bonus_history.extend(bonus_np.tolist())

                obs = next_obs
                prev_done = done.astype(np.float32)
                global_step += B * cfg.frame_skip

            # bootstrap value for non-terminal rollout end
            obs_j = jnp.asarray(obs)
            pd_j = jnp.asarray(prev_done)
            dummy_dones = jnp.zeros((1,) + pd_j.shape, dtype=pd_j.dtype)
            _, last_value, _ = policy_model.apply(
                policy_state.params, obs_j[None], dummy_dones, pd_j, lstm_state
            )
            last_value = np.asarray(last_value[0])
            # zero-out bootstrap for envs that just terminated (rollout.r=0)
            last_value = last_value * (1.0 - prev_done)

            # === reward processing (process_rollout) ===
            # rewards += bonuses
            shaped = buf_reward + buf_bonus
            # value-target rewards include bootstrapped tail
            rewards_plus_v_tail = np.concatenate([shaped, last_value[None]], axis=0)
            # always clip — envWrap=True in their MyWayHome config
            np.clip(
                rewards_plus_v_tail[:-1],
                -cfg.reward_clip,
                cfg.reward_clip,
                out=rewards_plus_v_tail[:-1],
            )
            shaped = rewards_plus_v_tail[:-1]
            # discount with done-aware mask (the original A3C breaks the rollout at
            # terminal, equivalent to multiplying future returns by (1-done))
            mask = 1.0 - buf_done
            returns = np.zeros_like(shaped)
            G = last_value.copy()
            for t in range(T - 1, -1, -1):
                G = shaped[t] + cfg.gamma * G * mask[t]
                returns[t] = G

            # advantages: GAE with lambda
            values_tp1 = np.concatenate([buf_value[1:], last_value[None]], axis=0)
            deltas = shaped + cfg.gamma * values_tp1 * mask - buf_value
            adv = np.zeros_like(deltas)
            A = np.zeros(B, dtype=np.float32)
            for t in range(T - 1, -1, -1):
                A = deltas[t] + cfg.gamma * cfg.lambda_ * mask[t] * A
                adv[t] = A

            actions_oh = np.eye(ac_space, dtype=np.float32)[buf_action]
            batch = {
                "obs": jnp.asarray(buf_obs),
                "obs_s1": jnp.asarray(buf_obs),
                "obs_s2": jnp.asarray(buf_obs_next),
                "dones": jnp.asarray(buf_done),
                # init_done feeds LSTMPolicy's shifted-dones masking — it's the
                # done at the step BEFORE this rollout (saved above as init_done)
                "init_done": init_done,
                "actions_oh": jnp.asarray(actions_oh),
                "advantages": jnp.asarray(adv),
                "returns": jnp.asarray(returns),
            }

            # POLICY_NO_BACKPROP_STEPS gate
            policy_active = jnp.asarray(
                1.0 if global_step > cfg.policy_no_backprop_steps else 0.0
            )

            policy_state, predictor_state, info = update(
                policy_state,
                predictor_state,
                batch,
                init_lstm,
                policy_active,
            )

            if (upd + 1) % cfg.log_every_n_updates == 0:
                ep_recent = float(np.mean(list(ep_history))) if ep_history else 0.0
                bonus_mean = (
                    float(np.mean(list(bonus_history))) if bonus_history else 0.0
                )
                sps = global_step / max(time.time() - start, 1e-9)
                wandb.log(
                    {
                        "train/pi_loss": float(info["pi_loss"]),
                        "train/vf_loss": float(info["vf_loss"]),
                        "train/entropy": float(info["entropy"]),
                        "train/policy_total": float(info["policy_total"]),
                        "train/inv_loss": float(info["inv_loss"]),
                        "train/forward_loss": float(info["forward_loss"]),
                        "train/predloss": float(info["predloss"]),
                        "train/ep_return_recent": ep_recent,
                        "train/bonus_mean": bonus_mean,
                        "train/sps": sps,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
                print(
                    f"upd={upd + 1}  steps={global_step}  ep_ret={ep_recent:.3f}  "
                    f"bonus={bonus_mean:.4f}  sps={sps:.0f}"
                )

        envs.close()
        wandb.finish()

    main(tyro.cli(Config))