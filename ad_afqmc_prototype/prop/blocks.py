from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy
from ..core.system import System
from .types import PropOps, PropState, QmcParams


class BlockFn(Protocol):
    def __call__(
        self,
        state: PropState,
        *,
        sys: System,
        params: QmcParams,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ops: PropOps,
        prop_ctx: Any,
    ) -> tuple[PropState, BlockObs]: ...


class BlockObs(NamedTuple):
    scalars: dict[str, jax.Array]


def block(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: Any,
    prop_ops: PropOps,
    prop_ctx: Any,
) -> tuple[PropState, BlockObs]:
    """
    propagation + measurement
    """
    step = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps_new = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, trial_data)
    state = state._replace(walkers=walkers_new, overlaps=overlaps_new)

    e_kernel = meas_ops.require_kernel(k_energy)
    e_samples = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data)
    e_samples = jnp.real(e_samples)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    e_samples = jnp.where(jnp.abs(e_samples - e_ref) > thresh, e_ref, e_samples)

    weights = state.weights
    w_sum = jnp.sum(weights)
    w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
    e_block = jnp.sum(weights * e_samples) / w_sum_safe
    e_block = jnp.where(w_sum == 0, e_ref, e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
    state = state._replace(
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * e_block
    )

    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    w_sr, weights_sr = wk.stochastic_reconfiguration(
        state.walkers, state.weights, zeta, sys.walker_kind
    )
    overlaps_sr = wk.vmap_chunked(
        meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(w_sr, trial_data)
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key,
    )

    obs = BlockObs(scalars={"energy": e_block, "weight": w_sum})
    return state, obs
