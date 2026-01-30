from ad_afqmc_prototype import config

config.setup_jax()

import jax
from jax import lax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.uhf import make_uhf_meas_ops, build_meas_ctx
from ad_afqmc_prototype.meas.uhf import energy_kernel_rw_rh, energy_kernel_uw_rh, energy_kernel_gw_rh
from ad_afqmc_prototype.meas.uhf import force_bias_kernel_rw_rh, force_bias_kernel_uw_rh, force_bias_kernel_gw_rh
from ad_afqmc_prototype.trial.uhf import UhfTrial, make_uhf_trial_ops
from ad_afqmc_prototype import testing

def _make_uhf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> UhfTrial:
    ka, kb = jax.random.split(key)
    ca = testing.rand_orthonormal_cols(ka, norb, nup, dtype=dtype)
    cb = testing.rand_orthonormal_cols(kb, norb, ndn, dtype=dtype)
    return UhfTrial(mo_coeff_a=ca, mo_coeff_b=cb)

@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_force_bias_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(0)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        make_meas_ops_fn=make_uhf_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(v_a, v_m, rtol=5e-6, atol=5e-7), (v_a, v_m)

@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_energy_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        make_meas_ops_fn=make_uhf_meas_ops,
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=5e-6, atol=5e-7), (ea, em)

def test_force_bias_equal_when_wu_eq_wr():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "restricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        fbr = force_bias_kernel_rw_rh(wi, ham, ctx, trial)
        fbu = force_bias_kernel_uw_rh((wi, wi), ham, ctx, trial)

        assert jnp.allclose(fbr, fbu, atol=1e-12), (fbr, fbu)

def test_force_bias_equal_when_wg_eq_wu():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        fbu = force_bias_kernel_uw_rh(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2*norb, nup+ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0,0))
        wi = lax.dynamic_update_slice(wi, wb, (norb,nup))
        fbg = force_bias_kernel_gw_rh(wi, ham, ctx, trial)

        assert jnp.allclose(fbu, fbg, atol=1e-12), (fbu, fbg)

def test_energy_equal_when_wu_eq_wr():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "restricted"
    
    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        er = energy_kernel_rw_rh(wi, ham, ctx, trial)
        eu = energy_kernel_uw_rh((wi, wi), ham, ctx, trial)

        assert jnp.allclose(er, eu, atol=1e-12), (er, eu)

def test_energy_equal_when_wg_eq_wu():
    norb = 6
    nup, ndn = 2, 1
    n_chol = 8
    walker_kind = "unrestricted"
    
    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        eu = energy_kernel_uw_rh(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2*norb, nup+ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0,0))
        wi = lax.dynamic_update_slice(wi, wb, (norb,nup))
        eg = energy_kernel_gw_rh(wi, ham, ctx, trial)

        assert jnp.allclose(eu, eg, atol=1e-12), (eu, eg)

if __name__ == "__main__":
    pytest.main([__file__])
