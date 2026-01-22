from ad_afqmc_prototype import config

config.setup_jax()

from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.cisd import make_cisd_meas_ops
from ad_afqmc_prototype.trial.cisd import CisdTrial, make_cisd_trial_ops
from ad_afqmc_prototype import testing

def _make_cisd_trial(
    key,
    norb: int,
    nocc: int,
    *,
    memory_mode: Literal["low", "high"] = "low",
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
    scale_ci2: float = 0.02,
) -> CisdTrial:
    """
    Random CISD coefficients in the MO basis where the reference occupies [0..nocc-1].

    We keep coefficients modest in magnitude to reduce catastrophic cancellation
    when comparing against overlap-based finite differences.
    """
    nvir = norb - nocc
    k1, k2 = jax.random.split(key)

    ci1 = scale_ci1 * jax.random.normal(k1, (nocc, nvir), dtype=dtype)
    ci2 = scale_ci2 * jax.random.normal(k2, (nocc, nvir, nocc, nvir), dtype=dtype)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    # Use high precision for the "testing" dtypes so the manual kernel is not
    # artificially noisy from float32/complex64 paths.
    return CisdTrial(ci1=ci1, ci2=ci2)


@pytest.mark.parametrize(
    "norb,nocc,n_chol,memory_mode",
    [
        (8, 3, 10, "low"),
        (8, 3, 10, "high"),
        (10, 4, 12, "low"),
        (10, 4, 12, "high"),
    ],
)
def test_auto_force_bias_matches_manual_cisd(norb, nocc, n_chol, memory_mode):
    walker_kind="restricted"
    key = jax.random.PRNGKey(123)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

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
        make_trial_fn=_make_cisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
            memory_mode=memory_mode,
        ),
        make_trial_ops_fn=make_cisd_trial_ops,
        make_meas_ops_fn=make_cisd_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        # CISD overlap is more structured than RHF; auto (finite-diff / overlap-derivative)
        # can need a slightly looser tolerance.
        assert jnp.allclose(v_a, v_m, rtol=2e-5, atol=2e-6), (v_a, v_m)


@pytest.mark.parametrize(
    "norb,nocc,n_chol,memory_mode",
    [
        (8, 3, 10, "low"),
        (8, 3, 10, "high"),
        (10, 4, 12, "low"),
        (10, 4, 12, "high"),
    ],
)
def test_auto_energy_matches_manual_cisd(norb, nocc, n_chol, memory_mode):
    walker_kind="restricted"
    key = jax.random.PRNGKey(456)
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
        make_trial_fn=_make_cisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
            memory_mode=memory_mode,
        ),
        make_trial_ops_fn=make_cisd_trial_ops,
        make_meas_ops_fn=make_cisd_meas_ops,
    )

    if not meas_manual.has_kernel(k_energy):
        pytest.skip("manual CISD meas does not provide k_energy")

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=2e-5, atol=2e-6), (ea, em)


if __name__ == "__main__":
    pytest.main([__file__])
