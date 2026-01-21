# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax import vmap

import flax.linen as nn

Array = jnp.ndarray


# ------------------------
# Conditional Recurrent Autoregressive Network
# ------------------------
class RNN(nn.Module):
    L: int
    LocalHilDim: int = 2
    dtype: Any = jnp.float32
    features: int = 64
    logProbFactor: float = 1.0

    # g-conditioning width
    g_hidden: int = 32

    def setup(self):
        self.embedding = nn.Dense(self.features, dtype=self.dtype, param_dtype=self.dtype)

        self.g_mlp = nn.Sequential([
            nn.Dense(self.g_hidden, dtype=self.dtype, param_dtype=self.dtype),
            nn.tanh,
            nn.Dense(self.features, dtype=self.dtype, param_dtype=self.dtype),
        ])

        self.g_to_h = nn.Dense(self.features, dtype=self.dtype, param_dtype=self.dtype)
        self.g_to_c = nn.Dense(self.features, dtype=self.dtype, param_dtype=self.dtype)

        self.cell = nn.RNN(
            nn.LSTMCell(features=self.features, dtype=self.dtype, param_dtype=self.dtype),
            return_carry=True,
        )

        self.amplitude = nn.Dense(self.LocalHilDim, dtype=self.dtype, param_dtype=self.dtype)

    def init_carry(self, g: Array) -> Tuple[Array, Array]:
        g = jnp.asarray(g, dtype=self.dtype)
        if g.ndim == 0:
            g = g[None]
        if g.ndim == 1:
            g = g[:, None]

        h0 = self.g_to_h(g)
        c0 = self.g_to_c(g)
        return (h0, c0)

    def __call__(
        self,
        s: Array,
        g: Array,
        carry_state: Optional[Tuple[Array, Array]] = None,
        output_state: bool = False,
    ) -> Array:
        s = jnp.asarray(s)
        g = jnp.asarray(g, dtype=self.dtype)

        if g.ndim == 0:
            g_in = g[None, None]
        elif g.ndim == 1:
            g_in = g[:, None]
        else:
            g_in = g
        g_vec = self.g_mlp(g_in)

        if output_state:
            seqy = jax.nn.one_hot(s.squeeze(-1), self.LocalHilDim, dtype=self.dtype)
            emb = self.embedding(seqy) + g_vec

            if carry_state is None:
                carry_state = self.init_carry(g)

            carry_state, y = self.cell(emb[:, None, :], initial_carry=carry_state)
            logits = self.amplitude(y[:, 0, :])
            return logits, carry_state

        s_pad = jnp.pad(s, ((0, 0), (1, 0)), mode="constant", constant_values=0)
        seqy = jax.nn.one_hot(s_pad, self.LocalHilDim, dtype=self.dtype)
        emb = vmap(vmap(self.embedding))(seqy)
        emb = emb + g_vec[:, None, :]

        carry0 = self.init_carry(g)
        carry, y = self.cell(emb, initial_carry=carry0)

        logits = vmap(vmap(self.amplitude))(y[:, :-1, :])
        logp_all = nn.log_softmax(logits) * self.logProbFactor

        token_logp = jnp.take_along_axis(logp_all, s[..., None], axis=-1)[..., 0]
        return token_logp.sum(axis=-1)


# ------------------------
# Sampling
# ------------------------
@partial(jax.jit, static_argnames=("L", "LocalHilDim", "apply_fn", "num_samples"))
def sample_rnn(
    apply_fn,
    params: Dict[str, Any],
    key: Array,
    num_samples: int,
    L: int,
    LocalHilDim: int,
    g: float,
) -> Array:
    """Autoregressive sampling from p_theta(s|g)."""
    g_batch = jnp.full((num_samples,), jnp.asarray(g, dtype=jnp.float32))

    def step(carry, t_key):
        prev_tok, lstm_carry = carry
        logits, lstm_carry = apply_fn({"params": params}, prev_tok, g_batch, carry_state=lstm_carry, output_state=True)
        next_tok = jrnd.categorical(t_key, logits)
        next_tok = next_tok.astype(jnp.int32)
        next_tok = next_tok[:, None]
        return (next_tok, lstm_carry), next_tok

    init_tok = jnp.zeros((num_samples, 1), dtype=jnp.int32)
    _, carry0 = apply_fn({"params": params}, init_tok, g_batch, carry_state=None, output_state=True)

    keys = jrnd.split(key, L)
    (_, _), toks = jax.lax.scan(step, (init_tok, carry0), keys)
    return toks[:, :, 0].T


# ------------------------
# Teacher-Forcing with scheduled sampling
# ------------------------
def scheduled_sampling_nll(
    apply_fn,
    params,
    key,
    s_true,
    g_batch,
    eps_teacher,
    LocalHilDim=2,
):
    """
    Returns per-sample log-prob sums over sequence using scheduled sampling.
    """
    B, L = s_true.shape

    x_prev = jnp.zeros((B, 1), dtype=jnp.int32)

    _, carry0 = apply_fn({"params": params}, x_prev, g_batch, carry_state=None, output_state=True)

    keys = jrnd.split(key, L)

    def step(carry, inp):
        x_prev, carry_state = carry
        key_t, s_t = inp

        logits, carry_state = apply_fn(
            {"params": params},
            x_prev,
            g_batch,
            carry_state=carry_state,
            output_state=True,
        )

        logp = jax.nn.log_softmax(logits)
        logp_t = jnp.take_along_axis(logp, s_t[:, None], axis=-1)[:, 0]

        s_sample = jrnd.categorical(key_t, logits).astype(jnp.int32)

        key_flip = jrnd.fold_in(key_t, 12345)
        use_teacher = jrnd.bernoulli(key_flip, p=eps_teacher, shape=(B,))

        x_next = jnp.where(use_teacher, s_t, jax.lax.stop_gradient(s_sample))
        x_next = x_next[:, None]

        return (x_next, carry_state), logp_t

    sT = jnp.swapaxes(s_true, 0, 1)
    scan_inp = (keys, sT)

    (_, _), logp_seq = jax.lax.scan(step, (x_prev, carry0), scan_inp)
    return jnp.sum(logp_seq, axis=0)
