from ad_afqmc_prototype import config

config.setup_jax()

from typing import Literal

import jax
from jax import lax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.gcisd import make_gcisd_meas_ops, build_meas_ctx
from ad_afqmc_prototype.meas.gcisd import energy_kernel_gw_gh
from ad_afqmc_prototype.meas.gcisd import force_bias_kernel_gw_gh
from ad_afqmc_prototype.trial.gcisd import GcisdTrial, make_gcisd_trial_ops
from ad_afqmc_prototype import testing

def _make_gcisd_trial(
    key,
    norb: int,
    nup: int,
    ndn: int,
    *,
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
    scale_ci2: float = 0.02,
) -> GcisdTrial:
    """
    Random GCISD coefficients in the MO basis where the reference occupies
    [0..nocc-1].

    We keep coefficients modest in magnitude to reduce catastrophic cancellation
    when comparing against overlap-based finite differences.
    """
    norb = 2*norb
    nocc = nup + ndn
    nvir = norb - nocc
    k1, k2 = jax.random.split(key)

    c1 = scale_ci1 * jax.random.normal(k1, (nocc, nvir), dtype=dtype)
    c2 = scale_ci2 * jax.random.normal(k2, (nocc, nvir, nocc, nvir), dtype=dtype)

    # Antisymmetry
    c2 = 0.25 * (c2 - jnp.einsum("iajb->jaib", c2) - jnp.einsum("iajb->ibja", c2) + jnp.einsum("iajb->jbia", c2))

    c = jnp.eye(norb, norb)

    return GcisdTrial(
        mo_coeff=c,
        c1=c1,
        c2=c2,
    )

@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("generalized", 6, 3, 2, 12),
        ("generalized", 10, 4, 3, 12),
    ],
)
def test_auto_force_bias_matches_manual_gcisd(walker_kind, norb, nup, ndn, n_chol):
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
        make_trial_fn=_make_gcisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_gcisd_trial_ops,
        make_meas_ops_fn=make_gcisd_meas_ops,
        ham_basis="generalized",
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(1):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(v_a, v_m, atol=1e-12), (v_a, v_m)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("generalized", 6, 3, 2, 12),
        ("generalized", 10, 4, 3, 12),
    ],
)
def test_auto_energy_matches_manual_gcisd(walker_kind, norb, nup, ndn, n_chol):
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
        make_trial_fn=_make_gcisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_gcisd_trial_ops,
        make_meas_ops_fn=make_gcisd_meas_ops,
        ham_basis="generalized",
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(1):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        e_m = e_manual(wi, ham, ctx_manual, trial)
        e_a = e_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(e_a, e_m, rtol=5e-6, atol=5e-7), (e_a, e_m)

if __name__ == "__main__":
    pytest.main([__file__])
