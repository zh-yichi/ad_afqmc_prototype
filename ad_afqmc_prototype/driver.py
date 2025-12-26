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


def make_run_blocks(block):  # block: state -> (state, obs)
    """
    To keep things on GPU over multiple blocks.
    """

    def one_block(state, _):
        state, obs = block(state)
        return state, (obs.scalars["energy"], obs.scalars["weight"])

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_blocks(
        state0: prop_state, *, n_blocks: int
    ) -> tuple[prop_state, jax.Array, jax.Array]:
        stateN, (e, w) = jax.lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, e, w

    return run_blocks


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

    run_blocks = make_run_blocks(block)

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
            f"{'W':>12s}   "
            f"{'nodes':>10s}  "
            f"{'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{float(state.e_estimate):14.10f}  "
        f"{float(jnp.sum(state.weights)):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, e_chunk, w_chunk = run_blocks(state, n_blocks=n)
        block_e_eq.extend(e_chunk.tolist())
        block_w_eq.extend(w_chunk.tolist())
        e_chunk_avg = jnp.mean(e_chunk)
        w_chunk_avg = jnp.mean(w_chunk)
        elapsed = time.perf_counter() - t0
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{int(state.node_encounters):10d}  "
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
            f"{'W':>12s}    {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, e_chunk, w_chunk = run_blocks(state, n_blocks=n)
        block_e_s.extend(e_chunk.tolist())
        block_w_s.extend(w_chunk.tolist())
        e_chunk_avg = jnp.mean(e_chunk)
        w_chunk_avg = jnp.mean(w_chunk)
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(block_e_s), jnp.asarray(block_w_s), print_q=False
        )
        mu = float(stats["mu"])
        se = float(stats["se_star"])
        nodes = int(state.node_encounters)
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{mu:14.10f}  "
            f"{se:10.3e}  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{nodes:10d}  "
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
