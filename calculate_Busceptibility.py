from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import optax
import jax
import jax.numpy as jnp
import jax.random as jrnd
from flax.training import train_state
from flax import serialization

from rnn_model import RNN, sample_rnn, scheduled_sampling_nll


def load_samples_for_length(data_dir: Path, Lx: int, Ly: int = 1) -> Tuple[List[np.ndarray], List[dict]]:
    pattern = f"hubbard_{Lx}x{Ly}_UoverT_*_samples.npz"
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files found for pattern {pattern} in {data_dir}")

    samples_list: List[np.ndarray] = []
    meta_list: List[dict] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            samples = data["samples"]
            meta = data["meta"].item()
        samples_list.append(samples)
        meta_list.append(meta)

    # sort by U_over_t
    order = np.argsort([m["U_over_t"] for m in meta_list])
    samples_list = [samples_list[i] for i in order]
    meta_list = [meta_list[i] for i in order]
    return samples_list, meta_list


def compute_struct_fac(samples_list: List[np.ndarray], Lx: int) -> np.ndarray:
    stagger = (-1) ** np.arange(Lx)
    out = []
    for samples in samples_list:
        val = (((samples * stagger).sum(-1)) ** 2 / samples.shape[0]).sum(-1)
        out.append(val)
    return np.array(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Hubbard samples and compare to RNN outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--Lx", type=int, default=12)
    parser.add_argument("--Ly", type=int, default=1)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# --------------------------
# Monte Carlo estimator of B
# --------------------------
def make_B_and_second_derivative(
    apply_fn: Callable,
    sampler_fn: Callable,
    params: Dict[str, Any],
    L: int,
    LocalHilDim: int = 2,
):
    """
    Returns two JIT-able callables:
      - estimate_B(key, U, delta, n_samples) -> scalar B_hat
      - estimate_d2B_dU2(key, U, delta, n_samples) -> scalar d2B/dU2 (hat)

    sampler_fn must have signature:
      s = sampler_fn(apply_fn, params, key, num_samples, L, LocalHilDim, g)
    and return (num_samples, L) int tokens.
    """

    def B_hat_from_fixed_samples(samples: Array, U1: Array, U2: Array) -> Array:
        """
        Compute Monte Carlo estimate of B(U, U+delta) using samples ~ p(.|U).
        Treat samples as constant for differentiation.
        """
        samples = jax.lax.stop_gradient(samples)
        U1 = jnp.full((samples.shape[0],), U1, dtype=jnp.float32)
        lp_U = apply_fn({"params": params}, samples, U1 )         # log p(s|U)
        U2 = jnp.full((samples.shape[0],), U2, dtype=jnp.float32)
        lp_Up = apply_fn({"params": params}, samples, U2)        # log p(s|U+δ)

        # integrand: sqrt( p(s|U+δ) / p(s|U) ) = exp(0.5*(lp_Up - lp_U))
        w = jnp.exp(0.5 * (lp_Up - lp_U))
        return jnp.mean(w)

    # JIT wrapper that samples then evaluates
    @partial(jax.jit, static_argnames=("n_samples",))
    def estimate_B(key: Array, U: Array, delta: Array, n_samples: int) -> Array:
        s = sampler_fn(apply_fn, params, key, n_samples, L, LocalHilDim, jnp.asarray(U, dtype=jnp.float32))
        B1 = B_hat_from_fixed_samples(s, U, U + delta)
        s = sampler_fn(apply_fn, params, key, n_samples, L, LocalHilDim, jnp.asarray(U + delta, dtype=jnp.float32))
        B2 = B_hat_from_fixed_samples(s, U + delta, U)
        return 0.5 * (B1 + B2)
        
    return estimate_B

# ------------------------
# Mini-batching
# ------------------------
def batch_iterator(
    key: Array,
    rng: np.random.Generator,
    samples_per_U: "np.ndarray",
    batch_size: int,
    g_value_per_U: float,
    *,
    shuffle: bool = True,
    pbc: bool = True
) -> Iterator[Dict[str, Any]]:
    import numpy as np

    N = samples_per_U[0].shape[0]
    L = samples_per_U[0].shape[1]
    idx = np.arange(N)
    g_idx = np.arange(len(g_value_per_U))
    # enforcing translation symmetry in sampling
    if pbc:
        rolling_shift = jrnd.randint(key, (N,), 0, L)
        cols = (jnp.arange(L)[None, :] - rolling_shift[:, None]) % L

    if shuffle:
        rng.shuffle(idx)
        rng.shuffle(g_idx)

    for start in range(0, N, batch_size):
        for i, _ in enumerate(g_value_per_U):
            batch_g_value = g_value_per_U[g_idx[i]]
            samples = samples_per_U[g_idx[i]]
            if pbc:
                samples = jnp.take_along_axis(samples, cols, axis=1)
            bidx = idx[start : start + batch_size]
            s = samples[bidx].astype(np.int32)
            g = np.full((s.shape[0],), batch_g_value, dtype=np.float32)
            yield {"s": s, "g": g}

# ------------------------
# eps_teacher schedule
# ------------------------
def eps_teacher(epoch, T, eps_min=0.7, eps_max=1.0):
    x = min(max(epoch / T, 0.0), 1.0)
    return eps_min + 0.5*(eps_max-eps_min)*(1.0 + np.cos(jnp.pi * x))


# ------------------------
# Training state
# ------------------------
class TrainState(train_state.TrainState):
    pass


def save_params(params: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))


def load_params(params_template: Dict[str, Any], path: str) -> Dict[str, Any]:
    path = Path(path)
    return serialization.from_bytes(params_template, path.read_bytes())


def create_train_state(
    key: Array,
    model: RNN,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
) -> TrainState:
    # dummy init
    B = 2
    dummy_s = jnp.zeros((B, model.L), dtype=jnp.int32)
    dummy_g = jnp.zeros((B,), dtype=jnp.float32)
    params = model.init(key, dummy_s, dummy_g)["params"]

    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def set_learning_rate(state: TrainState, lr: float, weight_decay: float = 0.0) -> TrainState:
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    # Reset optimizer state when changing the transformation.
    return state.replace(tx=tx, opt_state=tx.init(state.params))


@partial(jax.jit, static_argnames=("scheduled_sampling"))
def train_step(state: TrainState, batch: Dict[str, Array], eps_teacher: float, key: Array, scheduled_sampling = True) -> Tuple[TrainState, Dict[str, Array]]:

    if scheduled_sampling:
        def loss_fn(params):
            logp = scheduled_sampling_nll(state.apply_fn,params, key, batch["s"], batch["g"], eps_teacher, LocalHilDim=2,)
            nll = -jnp.mean(logp)
            return nll, {"nll": nll, "logp_mean": jnp.mean(logp)}
    else:
        def loss_fn(params):
            logp = state.apply_fn({"params": params}, batch["s"], batch["g"])  # (B,)
            nll = -jnp.mean(logp)
            return nll, {"nll": nll, "logp_mean": jnp.mean(logp)}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(grads)])
    # jax.debug.print("{x}", x=flat.sum())
    state = state.apply_gradients(grads=grads)
    return state, metrics


@jax.jit
def eval_step(state: TrainState, batch: Dict[str, Array]) -> Dict[str, Array]:
    logp = state.apply_fn({"params": state.params}, batch["s"], batch["g"])
    return {"nll": -jnp.mean(logp), "logp_mean": jnp.mean(logp)}


# ------------------------
# Training loop
# ------------------------
def train_rnn_on_born_samples(
    samples_per_U: "np.ndarray",
    *,
    g_values: list,
    L: int,
    features: int = 64,
    batch_size: int = 512,
    epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    seed: int = 0,
    val_frac: float = 0.1,
    eps_min=0.7,
    eps_max=1.0,
    scheduled_sampling=True,
    pbc = True,
    state = None,
    model = None,
    load_params_path: Optional[str] = None,
    save_params_path: Optional[str] = None
) -> Tuple[RNN, TrainState, Dict[str, Any]]:

    number_of_samples = [_sample.shape for _sample in samples_per_U]
    if max(number_of_samples) != min(number_of_samples):
        print("All samples must have the same shape")
        return None 

    rng = np.random.default_rng(seed)
    N = samples_per_U[0].shape[0]
    perm = rng.permutation(N)
    n_val = int(val_frac * N)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    train_samples = [samples_bits[tr_idx] for samples_bits in samples_per_U]
    val_samples = [samples_bits[val_idx] for samples_bits in samples_per_U]

    key = jrnd.PRNGKey(seed)

    if model:
        if not state:
            print("State missing for provided Model!")
            return None
        print("Model and state were provided")
        state = set_learning_rate(state, lr, weight_decay=weight_decay)
    elif state:
        if not model:
            print("Model missing for provided State!")
            return None
        print("Model and state were provided.")
    else:
        model = RNN(L=L, LocalHilDim=2, features=features, dtype=jnp.float32, logProbFactor=1.0)
        state = create_train_state(key, model, lr=lr, weight_decay=weight_decay)

    if load_params_path is not None:
        state = state.replace(params=load_params(state.params, load_params_path))
        print(f"Loaded params from {load_params_path}")

    history = {"train_nll": [], "val_nll": []}

    for ep in range(1, epochs + 1):
        key_pbc, key_tr, key, _ = jrnd.split(key,4)
        # --- train ---
        tr_losses = []
        for batch_np in batch_iterator(key_pbc, rng, train_samples, batch_size, g_value_per_U=g_values, shuffle=True, pbc = pbc):
            batch = {
                "s": jnp.asarray(batch_np["s"]),
                "g": jnp.asarray(batch_np["g"]),
            }
            eps_tr = eps_teacher(ep, epochs, eps_min=eps_min, eps_max=eps_max)

            state, metrics = train_step(state, batch, eps_tr, key_tr, scheduled_sampling = scheduled_sampling)
            tr_losses.append(float(metrics["nll"]))

        # --- val ---
        val_losses = []
        for batch_np in batch_iterator(key, rng, val_samples, batch_size, g_value_per_U=g_values, shuffle=False, pbc = False):
            batch = {
                "s": jnp.asarray(batch_np["s"]),
                "g": jnp.asarray(batch_np["g"]),
            }
            m = eval_step(state, batch)
            val_losses.append(float(m["nll"]))

        tr_nll = float(np.mean(tr_losses))
        va_nll = float(np.mean(val_losses))
        history["train_nll"].append(tr_nll)
        history["val_nll"].append(va_nll)

        print(f"epoch {ep:03d} | train NLL {tr_nll:.6f} | val NLL {va_nll:.6f}")

    if save_params_path is not None:
        save_params(state.params, save_params_path)
        print(f"Saved params to {save_params_path}")

    info = {
        #"g_value": float(g_value),
        "L": int(L),
        "features": int(features),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "val_frac": float(val_frac),
        "history": history,
    }
    return model, state, info



def main() -> None:
    args = parse_args()

    samples_list, meta_list = load_samples_for_length(args.data_dir, args.Lx, args.Ly)
    u_values = np.array([m["U_over_t"] for m in meta_list])

    model, state, info = train_rnn_on_born_samples(
        samples_list,
        g_values=u_values,
        L=args.Lx,
        features=256,
        batch_size=2**10,
        epochs=20,
        lr=1e-5,
        seed=0,
        eps_min = 1.,
        pbc=True,
        scheduled_sampling=False,
        # load_params_path="checkpoints/rnn_params.msgpack",
        # save_params_path="checkpoints/rnn_params.msgpack",
    )

    # constant eps teacher schedule 
    model, state, info = train_rnn_on_born_samples(
        samples_list,
        g_values=u_values,
        L=args.Lx,
        features=0,
        batch_size=2**10,
        epochs=40,
        lr=1e-5,
        seed=0,
        # load_params_path="checkpoints/rnn_params.msgpack",
        state=state,
        model=model,
        eps_min=0.9,
        eps_max=0.9,
    )

        # --- plug in your objects ---
    apply_fn = model.apply
    params   = state.params
    sampler_fn = sample_rnn

    # Sample the network and compare to ground truth curve
    net_struc_fac = []
    key = jrnd.PRNGKey(args.seed + 1)
    for u in u_values:
        key, subkey = jrnd.split(key)
        s_gen = sample_rnn(
            apply_fn,
            params,
            subkey,
            num_samples=min(args.num_samples, samples_list[0].shape[0]),
            L=args.Lx,
            LocalHilDim=2,
            g=float(u),
        )
        val = (((s_gen * (-1) ** np.arange(args.Lx)).sum(-1)) ** 2 / s_gen.shape[0]).sum(-1)
        net_struc_fac.append(val)

    net_struc_fac = np.array(net_struc_fac)
    struc_fac = compute_struct_fac(samples_list, args.Lx)

    plt.figure()
    plt.plot(u_values, struc_fac, label="ground truth")
    plt.plot(u_values, net_struc_fac, label="RNN samples")
    plt.xlabel("U/t")
    plt.ylabel(r"$S(\pi)$")
    plt.legend()
    plt.savefig("1.png")


    # Build estimators from the earlier function
    estimate_B = make_B_and_second_derivative(
        apply_fn=apply_fn,
        sampler_fn=sampler_fn,
        params=params,
        L=args.Lx,
        LocalHilDim=model.LocalHilDim,
    )

    # Grid settings
    Umin, Umax = 1., 5.0
    deltaU = 0.1
    interval_step_U = 1e-2
    n_samples = 2**12
    seed = 0

    U_grid = jnp.arange(Umin, Umax, interval_step_U, dtype=jnp.float32)  # stop at Umax-deltaU
    delta = jnp.array(deltaU, dtype=jnp.float32)

    # RNG keys per U
    base_key = jrnd.PRNGKey(seed)
    keys = jrnd.split(base_key, U_grid.shape[0])

    # Vectorized evaluation over the grid
    B_grid = []
    for _U, _key in zip(U_grid,keys):
        B = estimate_B(_key, _U, delta, n_samples)
        B_grid.append(B)
    B_grid = np.array(B_grid)

    chi_grid = (-2.0 * jnp.log(B_grid)) / (delta**2)

    # --- Plot chi_B(U) ---
    plt.figure()
    plt.plot(U_grid, chi_grid/np.max(abs(chi_grid)),label=r"$\chi_B(U)$")
    diff_StrucFac = (struc_fac[1:]-struc_fac[:-1])/(u_values[1:]-u_values[:-1])
    plt.plot(u_values[:-1],diff_StrucFac / np.max(abs(diff_StrucFac)),label=r"dS/dU")
    plt.xlabel("U")
    # plt.ylabel(r"$\chi_B(U)$")
    # plt.yscale("log")
    plt.title(r"Classical fidelity susceptibility from $B_{\rm sym}$")
    plt.legend(loc='best')
    plt.savefig("2.png")


if __name__ == "__main__":
    main()