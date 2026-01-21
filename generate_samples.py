# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

import numpy as np

from quspin.basis import spinless_fermion_basis_1d
from quspin.operators import hamiltonian
from scipy.sparse.linalg import eigsh


# -----------------------------
# Lattice utilities
# -----------------------------

def lattice_edges_rect(
    Lx: int,
    Ly: int,
    pbcx: bool = False,
    pbcy: bool = False,
) -> List[Tuple[int, int]]:
    """
    Return undirected nearest-neighbor edges (i, j) with i < j
    for a rectangular Lx x Ly lattice mapped to 1D indices:
        idx(x, y) = x + Lx*y
    """
    def idx(x: int, y: int) -> int:
        return x + Lx * y

    edges = set()

    for y in range(Ly):
        for x in range(Lx):
            i = idx(x, y)

            # +x neighbor
            if x + 1 < Lx:
                j = idx(x + 1, y)
                edges.add((min(i, j), max(i, j)))
            elif pbcx and Lx > 1:
                j = idx(0, y)
                edges.add((min(i, j), max(i, j)))

            # +y neighbor
            if y + 1 < Ly:
                j = idx(x, y + 1)
                edges.add((min(i, j), max(i, j)))
            elif pbcy and Ly > 1:
                j = idx(x, 0)
                edges.add((min(i, j), max(i, j)))

    return sorted(edges)


# -----------------------------
# Hubbard model builder
# -----------------------------
@dataclass(frozen=True)
class HubbardParams:
    Lx: int
    Ly: int
    t: float
    U: float
    Nup: int
    Ndown: int
    pbcx: bool = False
    pbcy: bool = False
    dtype: type = np.float64


def build_hubbard_quspin(params: HubbardParams):
    """
    Build QuSpin Hamiltonian and basis for the spinful Hubbard model
    on a rectangular lattice (mapped to 1D indices).
    """
    L = params.Lx * params.Ly
    edges = lattice_edges_rect(params.Lx, params.Ly, params.pbcx, params.pbcy)

    basis = spinless_fermion_basis_1d(L, Nf=params.Nup)

    # Hopping lists for QuSpin operator strings.
    hop_up_pm = [[-params.t, i, j] for (i, j) in edges]
    hop_up_mp = [[-params.t, j, i] for (i, j) in edges]

    # On-site U n_up n_down
    int_list = [[params.U, i, j] for (i, j) in edges]

    static = [
        ["+-", hop_up_pm],
        ["+-", hop_up_mp],
        ["nn", int_list],
    ]

    H = hamiltonian(static, [], basis=basis, dtype=params.dtype)
    return H, basis


# -----------------------------
# Ground state + sampling
# -----------------------------

def ground_state_eigsh(
    H,
    k: int = 1,
    which: str = "SA",
    tol: float = 1e-10,
    maxiter: Optional[int] = None,
):
    """Compute ground state with sparse eigensolver."""
    Hsp = H.tocsr()
    evals, evecs = eigsh(Hsp, k=k, which=which, tol=tol, maxiter=maxiter)
    idx = np.argmin(evals)
    E0 = float(evals[idx])
    psi0 = np.array(evecs[:, idx])
    psi0 = psi0 / np.linalg.norm(psi0)
    return E0, psi0


def sample_born_indices(psi: np.ndarray, nsamples: int, seed: int = 0) -> np.ndarray:
    """Sample basis indices i ~ |psi_i|^2."""
    rng = np.random.default_rng(seed)
    p = np.abs(psi) ** 2
    p = p / p.sum()
    return rng.choice(len(p), size=nsamples, replace=True, p=p)


def int_to_occupation_bits(basis, state_int: int) -> np.ndarray:
    """
    Convert a QuSpin basis integer into a 0/1 occupation vector.
    Returns a flat (L,) or (2L,) vector depending on basis.
    """
    st = basis.int_to_state(state_int)

    if isinstance(st, str):
        bits = np.fromiter((1 if c == "1" else 0 for c in st if c in "01"), dtype=np.int8)
        return bits

    if isinstance(st, (list, tuple)) and len(st) == 2 and all(isinstance(x, str) for x in st):
        up_str, dn_str = st
        up = np.fromiter((1 if c == "1" else 0 for c in up_str), dtype=np.int8)
        dn = np.fromiter((1 if c == "1" else 0 for c in dn_str), dtype=np.int8)
        arr = np.concatenate([up, dn], axis=0)
        return arr

    arr = np.array(st)
    if arr.ndim == 2 and arr.shape[0] == 2:
        up = arr[0].astype(np.int8)
        dn = arr[1].astype(np.int8)
        arr = np.concatenate([up, dn], axis=0)
        return arr
    if arr.ndim == 1:
        return arr.astype(np.int8)

    raise TypeError(f"Unexpected int_to_state return type/shape: {type(st)} / {np.array(st).shape}")


def generate_training_set(
    params: HubbardParams,
    nsamples: int,
    seed: int = 0,
    return_wavefunction: bool = False,
):
    """
    Build H, compute ground state, sample and return:
      - samples_bits: (nsamples, L) occupation vectors
      - meta dict including E0, lattice, (U/t), etc.
    """
    H, basis = build_hubbard_quspin(params)
    E0, psi0 = ground_state_eigsh(H, k=1, which="SA")

    basis_states_int = basis.states

    hilb_idx = sample_born_indices(psi0, nsamples=nsamples, seed=seed)
    conf_int = basis_states_int[hilb_idx]

    L = params.Lx * params.Ly
    samples_bits = np.empty((nsamples, L), dtype=np.int8)
    for n in range(nsamples):
        samples_bits[n] = int_to_occupation_bits(basis, int(conf_int[n]))

    meta = {
        "Lx": params.Lx,
        "Ly": params.Ly,
        "L": L,
        "t": params.t,
        "U": params.U,
        "U_over_t": params.U / params.t,
        "Nup": params.Nup,
        "Ndown": params.Ndown,
        "pbcx": params.pbcx,
        "pbcy": params.pbcy,
        "E0": E0,
        "hilbert_dim": int(basis.Ns),
        "seed": int(seed),
        "nsamples": int(nsamples),
    }

    if return_wavefunction:
        return samples_bits, meta, psi0, basis
    return samples_bits, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and save Hubbard training samples.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 12, 14, 16])
    parser.add_argument("--ly", type=int, default=1)
    parser.add_argument("--u-min", type=float, default=0.1)
    parser.add_argument("--u-max", type=float, default=6.0)
    parser.add_argument("--u-step", type=float, default=0.25)
    parser.add_argument("--nsamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    u_range = np.arange(args.u_min, args.u_max, args.u_step)

    for Lx in args.sizes:
        for U in u_range:
            params = HubbardParams(
                Lx=Lx,
                Ly=args.ly,
                t=1.0,
                U=float(U),
                Nup=Lx // 2,
                Ndown=1,
                pbcx=True,
                pbcy=False,
                dtype=np.float64,
            )
            samples, meta = generate_training_set(params, nsamples=args.nsamples, seed=args.seed)
            filename = f"hubbard_{meta['Lx']}x{meta['Ly']}_UoverT_{meta['U_over_t']:.3f}_samples.npz"
            out_path = args.out_dir / filename
            np.savez_compressed(out_path, samples=samples, meta=meta)
            print(f"Saved {out_path} | samples {samples.shape} | U/t {meta['U_over_t']:.3f}")


if __name__ == "__main__":
    main()
