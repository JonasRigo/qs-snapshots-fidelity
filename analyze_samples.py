# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrnd

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


def main() -> None:
    args = parse_args()

    samples_list, meta_list = load_samples_for_length(args.data_dir, args.Lx, args.Ly)
    u_values = np.array([m["U_over_t"] for m in meta_list])

    struc_fac = compute_struct_fac(samples_list, args.Lx)
    # plt.figure()
    # plt.plot(u_values, struc_fac, label="ground truth")
    # plt.xlabel("U/t")
    # plt.ylabel("struc_fac")
    # plt.legend()
    # plt.show()

    model = RNN(L=args.Lx, LocalHilDim=2, features=args.features, dtype=jnp.float32, logProbFactor=1.0)
    key = jrnd.PRNGKey(args.seed)
    dummy_s = jnp.zeros((2, args.Lx), dtype=jnp.int32)
    dummy_g = jnp.zeros((2,), dtype=jnp.float32)
    params = model.init(key, dummy_s, dummy_g)["params"]

    # Example NLL evaluation for one U/t
    idx = 0
    samples = samples_list[idx]
    g_val = float(meta_list[idx]["U_over_t"])
    s_batch = jnp.asarray(samples[: min(1024, samples.shape[0])], dtype=jnp.int32)
    g_batch = jnp.full((s_batch.shape[0],), g_val, dtype=jnp.float32)
    batch = {"s": s_batch, "g": g_batch}
    
    # compute the negative log-likelihood (teacher-forced)
    logp = model.apply({"params": params}, batch["s"], batch["g"])
    nll = -jnp.mean(logp)
    print(f"Example teacher-forced NLL at U/t={g_val:.3f}: {float(nll):.6f}")

    # comptue the scheduled sampling negative log-likelihood
    eps_teacher = 0.5
    logp = scheduled_sampling_nll(model.apply, params, key, batch["s"], batch["g"], eps_teacher, LocalHilDim=2,)
    nll = -jnp.mean(logp)
    print(f"Example teacher-forced NLL with scheduled sampling at U/t={g_val:.3f}: {float(nll):.6f}")

    # Sample the network and compare to ground truth curve
    net_struc_fac = []
    key = jrnd.PRNGKey(args.seed + 1)
    for u in u_values:
        key, subkey = jrnd.split(key)
        s_gen = sample_rnn(
            model.apply,
            params,
            subkey,
            num_samples=min(args.num_samples, samples.shape[0]),
            L=args.Lx,
            LocalHilDim=2,
            g=float(u),
        )
        val = (((s_gen * (-1) ** np.arange(args.Lx)).sum(-1)) ** 2 / s_gen.shape[0]).sum(-1)
        net_struc_fac.append(val)

    net_struc_fac = np.array(net_struc_fac)

    plt.figure()
    plt.plot(u_values, struc_fac, label="ground truth")
    plt.plot(u_values, net_struc_fac, label="RNN samples")
    plt.xlabel("U/t")
    plt.ylabel("double_occ")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()