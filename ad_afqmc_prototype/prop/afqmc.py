from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .. import walkers as wk
from ..core.ops import k_energy, k_force_bias, meas_ops, trial_ops
from ..core.system import system
from ..ham.chol import ham_chol
from ..walkers import init_walkers
from .chol_afqmc_ops import (
    _build_prop_ctx,
    chol_afqmc_ctx,
    make_trotter_ops,
    trotter_ops,
)
from .types import prop_ops, prop_state, qmc_params


def init_prop_state(
    *,
    sys: system,
    n_walkers: int,
    seed: int,
    ham_data: ham_chol,
    trial_ops: trial_ops,
    trial_data: Any,
    meas_ops: meas_ops,
    params: qmc_params,
    initial_walkers: Any | None = None,
    initial_e_estimate: jax.Array | None = None,
) -> prop_state:
    """
    Initialize AFQMC propagation state.
    """
    key = jax.random.PRNGKey(int(seed))
    weights = jnp.ones((n_walkers,))

    if initial_walkers is None:
        initial_walkers = init_walkers(
            sys=sys, rdm1=trial_ops.get_rdm1(trial_data), n_walkers=n_walkers
        )

    overlaps = wk.apply_chunked(
        initial_walkers,
        lambda w: meas_ops.overlap(w, trial_data),
        n_chunks=params.n_chunks,
    )

    e_est = None
    if initial_e_estimate is not None:
        e_est = jnp.asarray(initial_e_estimate)
    else:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        e_kernel = meas_ops.require_kernel(k_energy)
        e_samples = jnp.real(
            wk.apply_chunked(
                initial_walkers,
                lambda w: e_kernel(w, ham_data, meas_ctx, trial_data),
                n_chunks=params.n_chunks,
            )
        )
        e_est = jnp.mean(e_samples)

    pop_shift = e_est

    node_encounters = jnp.asarray(0)

    return prop_state(
        walkers=initial_walkers,
        weights=weights,
        overlaps=overlaps,
        rng_key=key,
        pop_control_ene_shift=pop_shift,
        e_estimate=e_est,
        node_encounters=node_encounters,
    )


def afqmc_step(
    state: prop_state,
    *,
    params: qmc_params,
    ham_data: ham_chol,
    trial_data: Any,
    meas_ops: meas_ops,
    trotter_ops: trotter_ops,
    prop_ctx: chol_afqmc_ctx,
    meas_ctx: Any,
) -> prop_state:

    key, subkey = jax.random.split(state.rng_key)
    nw = wk.n_walkers(state.walkers)
    fields = jax.random.normal(subkey, (nw, trotter_ops.n_fields()))

    fb_kernel = meas_ops.require_kernel(k_force_bias)
    force_bias = wk.apply_chunked(
        state.walkers,
        lambda w: fb_kernel(w, ham_data, meas_ctx, trial_data),
        n_chunks=params.n_chunks,
    )

    field_shifts = -prop_ctx.sqrt_dt * (1.0j * force_bias - prop_ctx.mf_shifts)
    shifted_fields = fields - field_shifts

    shift_term = jnp.sum(shifted_fields * prop_ctx.mf_shifts, axis=1)
    fb_term = jnp.sum(fields * field_shifts - 0.5 * field_shifts * field_shifts, axis=1)

    walkers_new = wk.apply_chunked_prop(
        state.walkers,
        shifted_fields,
        lambda w, f: trotter_ops.apply_trotter(w, f, prop_ctx, params.n_exp_terms),
        n_chunks=params.n_chunks,
    )

    overlaps_new = wk.apply_chunked(
        walkers_new, lambda w: meas_ops.overlap(w, trial_data), n_chunks=params.n_chunks
    )

    ratio = overlaps_new / state.overlaps
    exponent = (
        -prop_ctx.sqrt_dt * shift_term
        + fb_term
        + prop_ctx.dt * (state.pop_control_ene_shift + prop_ctx.h0_prop)
    )
    imp_fun = jnp.exp(exponent) * ratio

    theta = jnp.angle(jnp.exp(-prop_ctx.sqrt_dt * shift_term) * ratio)
    imp_ph = jnp.abs(imp_fun) * jnp.cos(theta)

    w_floor = float(getattr(params, "weight_floor", 1.0e-3))
    w_cap = float(getattr(params, "weight_cap", 100.0))

    imp_ph = jnp.where(imp_ph < w_floor, 0.0, imp_ph)
    node_encounters_new = state.node_encounters + jnp.sum(imp_ph <= 0.0)
    imp_ph = jnp.where(imp_ph > w_cap, 0.0, imp_ph)

    weights_new = state.weights * imp_ph
    weights_new = jnp.where(weights_new > w_cap, 0.0, weights_new)

    damping = float(getattr(params, "pop_control_damping", 0.1))
    avg_w = jnp.clip(jnp.mean(weights_new), min=1.0e-300)
    pop_shift_new = state.e_estimate - damping * (jnp.log(avg_w) / prop_ctx.dt)

    return prop_state(
        walkers=walkers_new,
        weights=weights_new,
        overlaps=overlaps_new,
        rng_key=key,
        pop_control_ene_shift=pop_shift_new,
        e_estimate=state.e_estimate,
        node_encounters=node_encounters_new,
    )


def make_prop_ops(ham_data: ham_chol, walker_kind: str) -> prop_ops:
    trotter_ops = make_trotter_ops(ham_data, walker_kind)

    def step(
        state: prop_state,
        *,
        params: qmc_params,
        ham_data: Any,
        trial_data: Any,
        meas_ops: meas_ops,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> prop_state:
        return afqmc_step(
            state,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            trotter_ops=trotter_ops,
        )

    def build_prop_ctx(
        ham_data: Any, trial_data: Any, params: qmc_params
    ) -> chol_afqmc_ctx:
        return _build_prop_ctx(ham_data, trial_data, params.dt)

    return prop_ops(
        init_prop_state=init_prop_state, build_prop_ctx=build_prop_ctx, step=step
    )
