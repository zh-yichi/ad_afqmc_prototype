from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import ad_afqmc_prototype.walkers as wk
from ad_afqmc_prototype.core.ops import MeasOps, TrialOps, k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.prop.afqmc import afqmc_step, init_prop_state
from ad_afqmc_prototype.prop.blocks import block
from ad_afqmc_prototype.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops
from ad_afqmc_prototype.prop.types import PropOps, QmcParams
from ad_afqmc_prototype.sharding import make_data_mesh, shard_prop_state
from ad_afqmc_prototype import testing

def _dummy_meas_ops() -> MeasOps:
    def build_meas_ctx(_ham, _trial):
        return None

    def overlap(walker, trial_data):
        return jnp.sum(walker)

    def force_bias_kernel(walker, ham_data, meas_ctx, trial_data):
        n_fields = int(ham_data.chol.shape[0])
        return jnp.zeros((n_fields,), dtype=walker.dtype)

    def energy_kernel(walker, ham_data, meas_ctx, trial_data):
        return jnp.real(jnp.vdot(walker, walker))

    return MeasOps(
        overlap=overlap,
        build_meas_ctx=build_meas_ctx,
        kernels={k_force_bias: force_bias_kernel, k_energy: energy_kernel},
        observables={},
    )


def _make_prop_ops_for_afqmc_step(*, trotter_ops) -> PropOps:
    def step(
        state,
        *,
        params,
        ham_data,
        trial_data,
        trial_ops,
        meas_ops,
        prop_ctx,
        meas_ctx,
        **_kw,
    ):
        return afqmc_step(
            state=state,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            trotter_ops=trotter_ops,
            prop_ctx=prop_ctx,
            meas_ctx=meas_ctx,
        )

    def build_prop_ctx(*_a, **_k):
        raise NotImplementedError

    def init_state(*_a, **_k):
        raise NotImplementedError

    return PropOps(step=step, build_prop_ctx=build_prop_ctx, init_prop_state=init_state)


@pytest.mark.parametrize("n_per_dev", [4])
def test_block_runs_under_sharding(n_per_dev):
    if jax.local_device_count() < 2:
        pytest.skip(
            "Need multiple local devices (use XLA_FLAGS=--xla_force_host_platform_device_count=8)."
        )

    mesh = make_data_mesh()
    ndev = mesh.size

    norb, nocc, n_fields = 6, 3, 2
    nw = ndev * n_per_dev

    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0, dtype=jnp.float32),
        h1=jnp.zeros((norb, norb), dtype=jnp.float32),
        chol=jnp.zeros((n_fields, norb, norb), dtype=jnp.float32),
    )
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")

    trial_ops = testing.dummy_trial_ops()
    meas_ops = _dummy_meas_ops()
    trial_data = {"rdm1": jnp.zeros((2, norb, norb), dtype=jnp.float32)}
    meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)

    params = QmcParams(
        dt=0.1,
        n_chunks=1,
        n_exp_terms=4,
        n_prop_steps=1,
        shift_ema=0.5,
        n_walkers=nw,
        seed=0,
        pop_control_damping=0.1,
    )

    trotter_ops = make_trotter_ops(ham.basis, sys.walker_kind)
    prop_ops = _make_prop_ops_for_afqmc_step(trotter_ops=trotter_ops)
    prop_ctx = _build_prop_ctx(ham, trial_data["rdm1"], params.dt)
    initial_walkers = jnp.ones((nw, norb, nocc), dtype=jnp.complex64)

    # unsharded reference
    state0_u = init_prop_state(
        sys=sys,
        ham_data=ham,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        params=params,
        initial_walkers=initial_walkers,
        mesh=None,
    )

    # sharded version of the exact same state
    state0_s = shard_prop_state(state0_u, mesh)

    def _assert_sharded_first_axis(a):
        assert isinstance(a.sharding, NamedSharding)
        assert a.sharding.spec == P("data")

    def _assert_replicated(a):
        assert isinstance(a.sharding, NamedSharding)
        assert a.sharding.spec == P()

    for leaf in tree_util.tree_leaves(state0_s.walkers):
        _assert_sharded_first_axis(leaf)
    _assert_sharded_first_axis(state0_s.weights)
    _assert_sharded_first_axis(state0_s.overlaps)
    _assert_replicated(state0_s.rng_key)
    _assert_replicated(state0_s.e_estimate)
    _assert_replicated(state0_s.pop_control_ene_shift)

    wsum_direct = jnp.sum(state0_s.weights)
    assert float(jax.device_get(wsum_direct)) == pytest.approx(float(nw), abs=1e-5)

    data_sh = NamedSharding(mesh, P("data"))
    sr_sharded = partial(wk.stochastic_reconfiguration, data_sharding=data_sh)

    def _call_block(st):
        return block(
            st,
            sys=sys,
            params=params,
            ham_data=ham,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ops=prop_ops,
            prop_ctx=prop_ctx,
            sr_fun=sr_sharded,
        )

    run_block_u = jax.jit(_call_block)
    run_block_s = jax.jit(_call_block)

    state1_u, obs_u = run_block_u(state0_u)
    state1_s, obs_s = run_block_s(state0_s)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(obs_s.scalars["weight"])),
        np.asarray(jax.device_get(obs_u.scalars["weight"])),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(obs_s.scalars["energy"])),
        np.asarray(jax.device_get(obs_u.scalars["energy"])),
        rtol=1e-5,
        atol=1e-5,
    )

    for leaf in tree_util.tree_leaves(state1_s.walkers):
        _assert_sharded_first_axis(leaf)
    _assert_sharded_first_axis(state1_s.weights)
    _assert_sharded_first_axis(state1_s.overlaps)
    _assert_replicated(state1_s.rng_key)
    _assert_replicated(state1_s.e_estimate)
    _assert_replicated(state1_s.pop_control_ene_shift)
