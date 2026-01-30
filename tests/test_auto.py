from ad_afqmc_prototype import config

config.setup_jax()

import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.rhf import make_rhf_meas_ops
from ad_afqmc_prototype.trial.rhf import RhfTrial, make_rhf_trial_ops
from ad_afqmc_prototype import testing

def _make_random_rhf_trial(key, norb, nocc):
    return RhfTrial(mo_coeff=testing.rand_orthonormal_cols(key, norb, nocc))

@pytest.mark.parametrize("walker_kind", ["restricted", "unrestricted", "generalized"])
def test_auto_force_bias_matches_manual_rhf(walker_kind):
    norb = 5
    nocc = 2
    n_chol = 7

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
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_random_rhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=make_rhf_trial_ops,
        make_meas_ops_fn=make_rhf_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(v_a, v_m, rtol=1e-7, atol=1e-8), (v_a, v_m)


@pytest.mark.parametrize("walker_kind", ["restricted", "unrestricted"])
def test_auto_energy_matches_manual_rhf(walker_kind):
    norb = 5
    nocc = 2
    n_chol = 7

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
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_random_rhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=make_rhf_trial_ops,
        make_meas_ops_fn=make_rhf_meas_ops,
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        emr = jnp.real(em)
        ear = jnp.real(ea)

        assert jnp.allclose(ear, emr, rtol=5e-3, atol=5e-4), (ear, emr)


def test_auto_force_bias_matches_manual_rhf_generalized():
    walker_kind = "generalized"
    norb = 5
    nocc = 2
    n_chol = 7

    key = jax.random.PRNGKey(2)
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
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_random_rhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=make_rhf_trial_ops,
        make_meas_ops_fn=make_rhf_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(v_a, v_m, rtol=1e-7, atol=1e-8), (v_a, v_m)


if __name__ == "__main__":
    pytest.main([__file__])
