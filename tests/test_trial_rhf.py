from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.rhf import (
    build_meas_ctx,
    energy_kernel_rw_rh,
    energy_kernel_uw_rh,
    force_bias_kernel_rw_rh,
    force_bias_kernel_uw_rh,
    make_rhf_meas_ops,
)
from ad_afqmc_prototype.trial.rhf import RhfTrial


def _rand_orthonormal(key: jax.Array, n: int, k: int) -> jax.Array:
    a = jax.random.normal(key, (n, k)) + 1.0j * jax.random.normal(key, (n, k))
    q, _ = jnp.linalg.qr(a)
    return q[:, :k]


def _make_small_ham(key: jax.Array, norb: int, n_chol: int) -> HamChol:
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (norb, norb))
    h1 = 0.1 * (a + a.T)
    chol = 0.05 * jax.random.normal(k2, (n_chol, norb, norb))
    return HamChol(basis="restricted", h0=jnp.asarray(0.7), h1=h1, chol=chol)


def test_build_meas_ctx_shapes():
    key = jax.random.PRNGKey(0)
    norb, nocc, n_chol = 6, 2, 5

    ham = _make_small_ham(key, norb, n_chol)
    c = _rand_orthonormal(key, norb, nocc)
    tr = RhfTrial(mo_coeff=c)

    ctx = build_meas_ctx(ham, tr)
    assert ctx.rot_h1.shape == (nocc, norb)
    assert ctx.rot_chol.shape == (n_chol, nocc, norb)
    assert ctx.rot_chol_flat.shape == (n_chol, nocc * norb)


def test_force_bias_matches_reference_restricted():
    key = jax.random.PRNGKey(1)
    norb, nocc, n_chol = 5, 2, 4
    ham = _make_small_ham(key, norb, n_chol)

    c = _rand_orthonormal(key, norb, nocc).astype(jnp.complex64)
    tr = RhfTrial(mo_coeff=c)
    ctx = build_meas_ctx(ham, tr)

    A = jnp.array([[1.2 + 0.1j, -0.2 + 0.3j], [0.4 - 0.1j, 0.9 + 0.0j]], dtype=c.dtype)
    w = c @ A  # (norb, nocc)

    fb = force_bias_kernel_rw_rh(w, None, ctx, tr)

    m = tr.mo_coeff.conj().T @ w
    g_half = jnp.linalg.solve(m.T, w.T)
    fb_ref = 2.0 * (ctx.rot_chol_flat @ g_half.reshape(-1))
    assert jnp.allclose(fb, fb_ref)


def test_force_bias_unrestricted_equals_restricted_when_wu_eq_wd():
    key = jax.random.PRNGKey(2)
    norb, nocc, n_chol = 6, 3, 3
    ham = _make_small_ham(key, norb, n_chol)

    c = _rand_orthonormal(key, norb, nocc).astype(jnp.complex64)
    tr = RhfTrial(mo_coeff=c)
    ctx = build_meas_ctx(ham, tr)

    key, sub = jax.random.split(key)
    A = jax.random.normal(sub, (nocc, nocc)).astype(jnp.complex64)
    A = A + 1.5 * jnp.eye(nocc, dtype=A.dtype)

    w = c @ A
    fb_r = force_bias_kernel_rw_rh(w, None, ctx, tr)
    fb_u = force_bias_kernel_uw_rh((w, w), None, ctx, tr)
    assert jnp.allclose(fb_u, fb_r)


def test_energy_is_h0_when_h1_and_chol_zero_restricted_and_unrestricted():
    key = jax.random.PRNGKey(3)
    norb, nocc, n_chol = 5, 2, 4

    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(1.234),
        h1=jnp.zeros((norb, norb)),
        chol=jnp.zeros((n_chol, norb, norb)),
    )

    c = _rand_orthonormal(key, norb, nocc).astype(jnp.complex64)
    tr = RhfTrial(mo_coeff=c)
    ctx = build_meas_ctx(ham, tr)

    A = jnp.eye(nocc, dtype=c.dtype) * (1.1 + 0.2j)
    w = c @ A

    e_r = energy_kernel_rw_rh(w, ham, ctx, tr)
    e_u = energy_kernel_uw_rh((w, w), ham, ctx, tr)
    assert jnp.allclose(e_r, ham.h0)
    assert jnp.allclose(e_u, ham.h0)


def test_make_rhf_meas_ops_dispatch_and_kernels_exist():
    norb, nocc = 4, 2
    sys_r = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    sys_u = System(norb=norb, nelec=(nocc, nocc), walker_kind="unrestricted")

    meas_r = make_rhf_meas_ops(sys_r)
    meas_u = make_rhf_meas_ops(sys_u)

    assert k_force_bias in meas_r.kernels
    assert k_energy in meas_r.kernels
    assert k_force_bias in meas_u.kernels
    assert k_energy in meas_u.kernels


if __name__ == "__main__":
    pytest.main([__file__])
