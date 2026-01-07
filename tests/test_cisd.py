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


def _make_restricted_walker_near_ref(
    key, norb: int, nocc: int, *, mix: float = 0.2, dtype=jnp.complex128
) -> jax.Array:
    """
    Make a restricted walker (norb, nocc) whose occupied block isn't near-singular.

    Start from the reference [I;0] and add a small random perturbation, then QR.
    This avoids tiny det(w[:nocc,:]) which can make overlap-based finite differences noisy.
    """
    k1, k2 = jax.random.split(key)
    w0 = jnp.zeros((norb, nocc), dtype=jnp.complex128)
    w0 = w0.at[:nocc, :].set(jnp.eye(nocc, dtype=jnp.complex128))
    noise = jax.random.normal(
        k1, (norb, nocc), dtype=jnp.float64
    ) + 1.0j * jax.random.normal(k2, (norb, nocc), dtype=jnp.float64)
    w = w0 + mix * noise
    q, _ = jnp.linalg.qr(w, mode="reduced")
    return q.astype(dtype)


def _make_random_ham_chol(key, norb, n_chol, dtype=jnp.float64) -> HamChol:
    """
    Build a small 'restricted' HamChol with:
      - symmetric real h1
      - symmetric real chol[g]
    """
    k1, k2, k3 = jax.random.split(key, 3)

    a = jax.random.normal(k1, (norb, norb), dtype=dtype)
    h1 = 0.5 * (a + a.T)

    b = jax.random.normal(k2, (n_chol, norb, norb), dtype=dtype)
    chol = 0.5 * (b + jnp.swapaxes(b, 1, 2))

    h0 = jax.random.normal(k3, (), dtype=dtype)

    return HamChol(basis="restricted", h0=h0, h1=h1, chol=chol)


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
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")

    key = jax.random.PRNGKey(123)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = _make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = _make_cisd_trial(k_trial, norb=norb, nocc=nocc, memory_mode=memory_mode)

    t_ops = make_cisd_trial_ops(sys)
    meas_manual = make_cisd_meas_ops(
        sys, memory_mode=memory_mode, mixed_precision=False, testing=True
    )
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = _make_restricted_walker_near_ref(
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
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")

    key = jax.random.PRNGKey(456)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = _make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = _make_cisd_trial(k_trial, norb=norb, nocc=nocc, memory_mode=memory_mode)

    t_ops = make_cisd_trial_ops(sys)
    meas_manual = make_cisd_meas_ops(
        sys, memory_mode=memory_mode, mixed_precision=False, testing=True
    )
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    if not meas_manual.has_kernel(k_energy):
        pytest.skip("manual CISD meas does not provide k_energy")

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = _make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=2e-5, atol=2e-6), (ea, em)


if __name__ == "__main__":
    pytest.main([__file__])
