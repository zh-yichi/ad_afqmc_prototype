from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .core.ops import meas_ops, trial_ops
from .core.system import system
from .ham.chol import ham_chol
from .prop.afqmc import init_prop_state
from .prop.blocks import block_obs
from .prop.chol_afqmc_ops import make_chol_afqmc_ops
from .prop.types import afqmc_params, prop_state
from .stat_utils import blocking_analysis_ratio, reject_outliers

print = partial(print, flush=True)


def run_afqmc_energy(
    *,
    sys: system,
    params: afqmc_params,
    ham_data: ham_chol,
    trial_data: Any,
    meas_ops: meas_ops,
    trial_ops: trial_ops,
    block_fn: Callable[..., tuple[prop_state, block_obs]],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    equilibration blocks then sampling blocks.

    Returns:
      (mean_energy, stderr, block_energies, block_weights)
    """
    prop_ops = make_chol_afqmc_ops(ham_data, sys.walker_kind)
    prop_ctx = prop_ops.build_prop_ctx(trial_ops.get_rdm1(trial_data), params.dt)
    meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    state = init_prop_state(
        sys=sys,
        n_walkers=params.n_walkers,
        seed=params.seed,
        ham_data=ham_data,
        trial_ops_=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
    )

    def block(state_in):
        return block_fn(
            state_in,
            sys=sys,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            prop_ctx=prop_ctx,
            meas_ctx=meas_ctx,
        )

    block_jit = jax.jit(block)

    t0 = time.perf_counter()
    t_mark = t0

    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    block_e_eq = []
    block_w_eq = []
    block_e_eq.append(state.e_estimate)
    block_w_eq.append(jnp.sum(state.weights))
    print("\nEquilibration:\n")
    if print_every:
        print(
            f"{'':4s}"
            f"{'block':>9s}  "
            f"{'E_blk':>14s}  "
            f"{'W':>12s}  "
            f"{'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{float(state.e_estimate):14.10f}  "
        f"{float(jnp.sum(state.weights)):12.6e}  "
        f"{0.0:8.1f}"
    )
    for ib in range(params.n_eql_blocks):
        state, obs = block_jit(state)
        block_e_eq.append(obs.scalars["energy"])
        block_w_eq.append(obs.scalars["weight"])
        if print_every and ((ib + 1) % print_every == 0):
            elapsed = time.perf_counter() - t0
            print(
                f"[eql {ib+1:4d}/{params.n_eql_blocks}]  "
                f"{float(obs.scalars['energy']):14.10f}  "
                f"{float(obs.scalars['weight']):12.6e}  "
                f"{elapsed:8.1f}"
            )
    block_e_eq = jnp.asarray(block_e_eq)
    block_w_eq = jnp.asarray(block_w_eq)

    # sampling
    print("\nSampling:\n")
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    block_e_s = []
    block_w_s = []
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'E_avg':>14s}  {'E_err':>10s}  {'E_block':>14s}  "
            f"{'W':>12s}  {'dt[s/bl]':>9s}  {'t[s]':>8s}"
        )

    for ib in range(params.n_blocks):
        state, obs = block_jit(state)
        block_e_s.append(obs.scalars["energy"])
        block_w_s.append(obs.scalars["weight"])

        if print_every and ((ib + 1) % print_every == 0):
            e_arr = jnp.asarray(block_e_s)
            w_arr = jnp.asarray(block_w_s)
            stats = blocking_analysis_ratio(e_arr, w_arr, print_q=False)

            now = time.perf_counter()
            elapsed = now - t0
            dt_per_block = (now - t_mark) / float(print_every)
            t_mark = now

            mu = float(stats["mu"])
            se = float(stats["se_star"])
            w_last = float(obs.scalars["weight"])
            print(
                f"[blk {ib+1:4d}/{params.n_blocks}]  "
                f"{mu:14.10f}  "
                f"{se:10.3e}  "
                f"{float(obs.scalars['energy']):14.10f}  "
                f"{w_last:12.6e}  "
                f"{dt_per_block:9.3f}  "
                f"{elapsed:8.1f}"
            )

    block_e_s = jnp.asarray(block_e_s)
    block_w_s = jnp.asarray(block_w_s)
    data_clean, _ = reject_outliers(jnp.column_stack((block_e_s, block_w_s)), obs=0)
    print(f"\nRejected {block_e_s.shape[0] - data_clean.shape[0]} outlier blocks.")
    block_e_s = jnp.asarray(data_clean[:, 0])
    block_w_s = jnp.asarray(data_clean[:, 1])
    print("\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(block_e_s, block_w_s, print_q=True)
    mean, err = stats["mu"], stats["se_star"]

    block_e_all = jnp.concatenate([block_e_eq, block_e_s])
    block_w_all = jnp.concatenate([block_w_eq, block_w_s])

    return mean, err, block_e_all, block_w_all
