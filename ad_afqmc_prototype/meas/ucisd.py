from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ucisd import UcisdTrial, overlap_u


def _half_green_from_overlap_matrix(w: jax.Array, ovlp_mat: jax.Array) -> jax.Array:
    """
    green_half = (w @ inv(ovlp_mat)).T
    """
    return jnp.linalg.solve(ovlp_mat.T, w.T)

@dataclass(frozen=True)
class UcisdMeasCfg:
    memory_mode: str = "low"  # or Literal["low","high"]
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64

@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UCisdMeasCtx:
    h1_b: jax.Array  # (norb, norb)
    chol_b: jax.Array  # (n_chol, norb, norb)

    # half-rotated:
    rot_h1_a: jax.Array  # (nocc[0], norb)
    rot_h1_b: jax.Array  # (nocc[1], norb)
    rot_chol_a: jax.Array  # (n_chol, nocc[0], norb)
    rot_chol_b: jax.Array  # (n_chol, nocc[1], norb)
    rot_chol_flat_a: jax.Array  # (n_chol, nocc[0]*norb)
    rot_chol_flat_b: jax.Array  # (n_chol, nocc[1]*norb)

    lci1_a: jax.Array  # (n_chol, norb, nocc[0])
    lci1_b: jax.Array  # (n_chol, norb, nocc[1])

    cfg: UcisdMeasCfg

    def tree_flatten(self):
        return (
            self.h1_b,
            self.chol_b,
            self.rot_h1_a,
            self.rot_h1_b,
            self.rot_chol_a,
            self.rot_chol_b,
            self.rot_chol_flat_a,
            self.rot_chol_flat_b,
            self.lci1_a,
            self.lci1_b,
            self.cfg,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            h1_b,
            chol_b,
            rot_h1_a,
            rot_h1_b,
            rot_chol_a,
            rot_chol_b,
            rot_chol_flat_a,
            rot_chol_flat_b,
            lci1_a,
            lci1_b,
            cfg,
        ) = children
        return cls(
            h1_b=h1_b,
            chol_b=chol_b,
            rot_h1_a=rot_h1_a,
            rot_h1_b=rot_h1_b,
            rot_chol_a=rot_chol_a,
            rot_chol_b=rot_chol_b,
            rot_chol_flat_a=rot_chol_flat_a,
            rot_chol_flat_b=rot_chol_flat_b,
            lci1_a=lci1_a,
            lci1_b=lci1_b,
            cfg=cfg,
        )

def force_bias_kernel_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: UcisdMeasCtx,
    trial_data: UcisdTrial,
) -> jax.Array:
    """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    n_va, n_vb = trial_data.nvir
    ci1_a = trial_data.c1a
    ci1_b = trial_data.c1b
    ci2_aa = trial_data.c2aa
    ci2_ab = trial_data.c2ab
    ci2_bb = trial_data.c2bb
    c_b = trial_data.mo_coeff_b

    wb = c_b.T.dot(wb[:, :n_ob])
    woa = wa[:n_oa, :]  # (n_oa, n_oa)
    wob = wb[:n_ob, :]  # (n_ob, n_ob)

    green_a = jnp.linalg.solve(woa.T, wa.T)  # (n_oa, norb)
    green_b = jnp.linalg.solve(wob.T, wb.T)  # (n_ob, norb)

    green_occ_a = green_a[:, n_oa:].copy()
    green_occ_b = green_b[:, n_ob:].copy()
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(n_va)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(n_vb)))

    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b

    # ref
    fb_0 = lg_a + lg_b

    # single excitations
    ci1g_a = jnp.einsum("pt,pt->", ci1_a, green_occ_a, optimize="optimal")
    ci1g_b = jnp.einsum("pt,pt->", ci1_b, green_occ_b, optimize="optimal")
    ci1g = ci1g_a + ci1g_b
    fb_1_1 = ci1g * lg
    ci1gp_a = jnp.einsum("pt,it->pi", ci1_a, greenp_a, optimize="optimal")
    ci1gp_b = jnp.einsum("pt,it->pi", ci1_b, greenp_b, optimize="optimal")
    gci1gp_a = jnp.einsum("pj,pi->ij", green_a, ci1gp_a, optimize="optimal")
    gci1gp_b = jnp.einsum("pj,pi->ij", green_b, ci1gp_b, optimize="optimal")
    fb_1_2 = -jnp.einsum(
        "gij,ij->g",
        chol_a.astype(meas_ctx.cfg.mixed_real_dtype),
        gci1gp_a.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    ) - jnp.einsum(
        "gij,ij->g",
        chol_b.astype(meas_ctx.cfg.mixed_real_dtype),
        gci1gp_b.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_1 = fb_1_1 + fb_1_2

    # double excitations
    ci2g_a = jnp.einsum(
        "ptqu,pt->qu",
        ci2_aa.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ_a.astype(meas_ctx.cfg.mixed_complex_dtype),
    )
    ci2g_b = jnp.einsum(
        "ptqu,pt->qu",
        ci2_bb.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ_b.astype(meas_ctx.cfg.mixed_complex_dtype),
    )
    ci2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        ci2_ab.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ_b.astype(meas_ctx.cfg.mixed_complex_dtype),
    )
    ci2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        ci2_ab.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ_a.astype(meas_ctx.cfg.mixed_complex_dtype),
    )
    gci2g_a = 0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
    gci2g_b = 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
    gci2g = gci2g_a + gci2g_b + gci2g_ab
    fb_2_1 = lg * gci2g
    ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
    ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
    fb_2_2_a = -jnp.einsum(
        "gij,ij->g",
        chol_a.astype(meas_ctx.cfg.mixed_real_dtype),
        ci2_green_a.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_2_2_b = -jnp.einsum(
        "gij,ij->g",
        chol_b.astype(meas_ctx.cfg.mixed_real_dtype),
        ci2_green_b.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_2_2 = fb_2_2_a + fb_2_2_b
    fb_2 = fb_2_1 + fb_2_2

    # overlap
    overlap_1 = ci1g
    overlap_2 = gci2g
    overlap = 1.0 + overlap_1 + overlap_2

    return (fb_0 + fb_1 + fb_2) / overlap


def energy_kernel_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UcisdMeasCtx,
    trial_data: UcisdTrial,
) -> jax.Array:
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    n_va, n_vb = trial_data.nvir
    ci1_a = trial_data.c1a
    ci1_b = trial_data.c1b
    ci2_aa = trial_data.c2aa
    ci2_ab = trial_data.c2ab
    ci2_bb = trial_data.c2bb
    c_b = trial_data.mo_coeff_b

    wb = c_b.T.dot(wb[:, :n_ob])
    woa = wa[:n_oa, :]  # (n_oa, n_oa)
    wob = wb[:n_ob, :]  # (n_ob, n_ob)

    green_a = jnp.linalg.solve(woa.T, wa.T)  # (n_oa, norb)
    green_b = jnp.linalg.solve(wob.T, wb.T)  # (n_ob, norb)

    green_occ_a = green_a[:, n_oa:].copy()
    green_occ_b = green_b[:, n_ob:].copy()
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(n_va)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(n_vb)))

    lci1_a = meas_ctx.lci1_a
    lci1_b = meas_ctx.lci1_b

    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    h1_a = ham_data.h1 
    h1_b = meas_ctx.h1_b
    hg_a = jnp.einsum("pj,pj->", h1_a[:n_oa, :], green_a)
    hg_b = jnp.einsum("pj,pj->", h1_b[:n_ob, :], green_b)
    hg = hg_a + hg_b

    # 0 body energy
    e0 = ham_data.h0

    # 1 body energy
    # ref
    e1_0 = hg

    # single excitations
    ci1g_a = jnp.einsum("pt,pt->", ci1_a, green_occ_a, optimize="optimal")
    ci1g_b = jnp.einsum("pt,pt->", ci1_b, green_occ_b, optimize="optimal")
    ci1g = ci1g_a + ci1g_b
    e1_1_1 = ci1g * hg
    gpci1_a = greenp_a @ ci1_a.T
    gpci1_b = greenp_b @ ci1_b.T
    ci1_green_a = gpci1_a @ green_a
    ci1_green_b = gpci1_b @ green_b
    e1_1_2 = -(
        jnp.einsum("ij,ij->", h1_a, ci1_green_a, optimize="optimal")
        + jnp.einsum("ij,ij->", h1_b, ci1_green_b, optimize="optimal")
    )
    e1_1 = e1_1_1 + e1_1_2

    # double excitations
    ci2g_a = (
        jnp.einsum(
            "ptqu,pt->qu",
            ci2_aa.astype(meas_ctx.cfg.mixed_real_dtype),
            green_occ_a.astype(meas_ctx.cfg.mixed_complex_dtype),
        )
        / 4
    )
    ci2g_b = (
        jnp.einsum(
            "ptqu,pt->qu",
            ci2_bb.astype(meas_ctx.cfg.mixed_real_dtype),
            green_occ_b.astype(meas_ctx.cfg.mixed_complex_dtype),
        )
        / 4
    )
    ci2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        ci2_ab.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ_b.astype(meas_ctx.cfg.mixed_complex_dtype),
    )
    ci2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        ci2_ab.astype(meas_ctx.cfg.mixed_real_dtype),
        green_occ_a.astype(meas_ctx.cfg.mixed_complex_dtype),
    )
    gci2g_a = jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
    gci2g_b = jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
    gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
    e1_2_1 = hg * gci2g
    ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
    ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
    ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
    ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
    e1_2_2_a = -jnp.einsum(
        "ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a, optimize="optimal"
    )
    e1_2_2_b = -jnp.einsum(
        "ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b, optimize="optimal"
    )
    e1_2_2 = e1_2_2_a + e1_2_2_b
    e1_2 = e1_2_1 + e1_2_2

    e1 = e1_0 + e1_1 + e1_2

    # two body energy
    # ref
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
    lg1_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
    lg1_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
    e2_0_2 = (
        -(
            jnp.sum(jax.vmap(lambda x: x * x.T)(lg1_a))
            + jnp.sum(jax.vmap(lambda x: x * x.T)(lg1_b))
        )
        / 2.0
    )
    e2_0 = e2_0_1 + e2_0_2

    # single excitations
    e2_1_1 = e2_0 * ci1g
    lci1g_a = jnp.einsum(
        "gij,ij->g",
        chol_a.astype(meas_ctx.cfg.mixed_real_dtype),
        ci1_green_a.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    lci1g_b = jnp.einsum(
        "gij,ij->g",
        chol_b.astype(meas_ctx.cfg.mixed_real_dtype),
        ci1_green_b.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
    ci1g1_a = ci1_a @ green_occ_a.T
    ci1g1_b = ci1_b @ green_occ_b.T
    e2_1_3_1 = jnp.einsum(
        "gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, optimize="optimal"
    ) + jnp.einsum("gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, optimize="optimal")
    lci1g_a = jnp.einsum(
        "gip,qi->gpq", lci1_a, green_a, optimize="optimal"
    )
    lci1g_b = jnp.einsum(
        "gip,qi->gpq", lci1_b, green_b, optimize="optimal"
    )
    e2_1_3_2 = -jnp.einsum(
        "gpq,gqp->", lci1g_a, lg1_a, optimize="optimal"
    ) - jnp.einsum("gpq,gqp->", lci1g_b, lg1_b, optimize="optimal")
    e2_1_3 = e2_1_3_1 + e2_1_3_2
    e2_1 = e2_1_1 + e2_1_2 + e2_1_3

    # double excitations
    e2_2_1 = e2_0 * gci2g
    lci2g_a = jnp.einsum(
        "gij,ij->g",
        chol_a.astype(meas_ctx.cfg.mixed_real_dtype),
        8 * ci2_green_a.astype(meas_ctx.cfg.mixed_complex_dtype)
        + 2 * ci2_green_ab_a.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    lci2g_b = jnp.einsum(
        "gij,ij->g",
        chol_b.astype(meas_ctx.cfg.mixed_real_dtype),
        8 * ci2_green_b.astype(meas_ctx.cfg.mixed_complex_dtype)
        + 2 * ci2_green_ab_b.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

    if meas_ctx.cfg.memory_mode == "low":

        def scan_over_chol(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = jnp.einsum("pj,ji->pi", green_a, chol_a_i, optimize="optimal")
            gl_b_i = jnp.einsum("pj,ji->pi", green_b, chol_b_i, optimize="optimal")
            lci2_green_a_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_a_i,
                8 * ci2_green_a + 2 * ci2_green_ab_a,
                optimize="optimal",
            )
            lci2_green_b_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_b_i,
                8 * ci2_green_b + 2 * ci2_green_ab_b,
                optimize="optimal",
            )
            carry[0] += 0.5 * (
                jnp.einsum("pi,pi->", gl_a_i, lci2_green_a_i, optimize="optimal")
                + jnp.einsum("pi,pi->", gl_b_i, lci2_green_b_i, optimize="optimal")
            )
            glgp_a_i = jnp.einsum(
                "pi,it->pt", gl_a_i, greenp_a, optimize="optimal"
            ).astype(meas_ctx.cfg.mixed_complex_dtype_testing)
            glgp_b_i = jnp.einsum(
                "pi,it->pt", gl_b_i, greenp_b, optimize="optimal"
            ).astype(meas_ctx.cfg.mixed_complex_dtype_testing)
            l2ci2_a = 0.5 * jnp.einsum(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_a_i,
                ci2_aa.astype(meas_ctx.cfg.mixed_real_dtype_testing),
                optimize="optimal",
            )
            l2ci2_b = 0.5 * jnp.einsum(
                "pt,qu,ptqu->",
                glgp_b_i,
                glgp_b_i,
                ci2_bb.astype(meas_ctx.cfg.mixed_real_dtype_testing),
                optimize="optimal",
            )
            l2ci2_ab = jnp.einsum(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_b_i,
                ci2_ab.astype(meas_ctx.cfg.mixed_real_dtype_testing),
                optimize="optimal",
            )
            carry[1] += l2ci2_a + l2ci2_b + l2ci2_ab
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = jax.lax.scan(
            scan_over_chol, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
    else:
        gl_a = jnp.einsum(
            "pj,gji->gpi",
            green_a.astype(meas_ctx.cfg.mixed_complex_dtype),
            chol_a.astype(meas_ctx.cfg.mixed_real_dtype),
            optimize="optimal",
        )
        gl_b = jnp.einsum(
            "pj,gji->gpi",
            green_b.astype(meas_ctx.cfg.mixed_complex_dtype),
            chol_b.astype(meas_ctx.cfg.mixed_real_dtype),
            optimize="optimal",
        )
        lci2_green_a = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_a,
            8 * ci2_green_a + 2 * ci2_green_ab_a,
            optimize="optimal",
        )
        lci2_green_b = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_b,
            8 * ci2_green_b + 2 * ci2_green_ab_b,
            optimize="optimal",
        )
        e2_2_2_2 = 0.5 * (
            jnp.einsum("gpi,gpi->", gl_a, lci2_green_a, optimize="optimal")
            + jnp.einsum("gpi,gpi->", gl_b, lci2_green_b, optimize="optimal")
        )
        glgp_a = jnp.einsum(
            "gpi,it->gpt", gl_a, greenp_a, optimize="optimal"
        ).astype(meas_ctx.cfg.mixed_complex_dtype_testing)
        glgp_b = jnp.einsum(
            "gpi,it->gpt", gl_b, greenp_b, optimize="optimal"
        ).astype(meas_ctx.cfg.mixed_complex_dtype_testing)
        l2ci2_a = 0.5 * jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_a,
            glgp_a,
            ci2_aa.astype(meas_ctx.cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2ci2_b = 0.5 * jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_b,
            glgp_b,
            ci2_bb.astype(meas_ctx.cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2ci2_ab = jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_a,
            glgp_b,
            ci2_ab.astype(meas_ctx.cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        e2_2_3 = l2ci2_a.sum() + l2ci2_b.sum() + l2ci2_ab.sum()

    e2_2_2 = e2_2_2_1 + e2_2_2_2
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    e2 = e2_0 + e2_1 + e2_2

    # overlap
    overlap_1 = ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
    overlap_2 = gci2g
    overlap = 1.0 + overlap_1 + overlap_2
    return (e1 + e2) / overlap + e0


def build_meas_ctx(ham_data: HamChol, trial_data: UcisdTrial) -> UcisdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("UCISD MeasOps currently assumes HamChol.basis == 'restricted'.")
    n_oa, n_ob = trial_data.nocc
    cb  = trial_data.mo_coeff_b  # (norb, nocc[1])
    cbH = trial_data.mo_coeff_b.conj().T  # (nocc[1], norb)
    h1_b = 0.5 * (cbH @ (ham_data.h1 + ham_data.h1.T) @ cb)
    chol_b = jnp.einsum("pi,gij,jq->gpq", cbH, ham_data.chol, cb)
    rot_h1_a = ham_data.h1[:n_oa,:]  # (nocc[0], norb)
    rot_h1_b = ham_data.h1[:n_ob,:]  # (nocc[1], norb)
    rot_chol_a = ham_data.chol[:, :n_oa, :]
    rot_chol_b = chol_b[:, :n_ob, :]
    rot_chol_flat_a = rot_chol_a.reshape(rot_chol_a.shape[0], -1)
    rot_chol_flat_b = rot_chol_b.reshape(rot_chol_b.shape[0], -1)
    
    lci1_a = jnp.einsum(
        "git,pt->gip",
        ham_data.chol[:, :, n_oa:],
        trial_data.c1a,
        optimize="optimal",
    )
    lci1_b = jnp.einsum(
        "git,pt->gip",
        chol_b[:, :, n_ob:],
        trial_data.c1b,
        optimize="optimal",
    )
    return UCisdMeasCtx(
        h1_b=h1_b,
        chol_b=chol_b,
        rot_h1_a=rot_h1_a,
        rot_h1_b=rot_h1_b,
        rot_chol_a=rot_chol_a,
        rot_chol_b=rot_chol_b,
        rot_chol_flat_a=rot_chol_flat_a,
        rot_chol_flat_b=rot_chol_flat_b,
        lci1_a=lci1_a,
        lci1_b=lci1_b,
        cfg=UcisdMeasCfg(),
    )


def make_ucisd_meas_ops(sys: System) -> MeasOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        raise NotImplementedError

    if wk == "unrestricted":
        return MeasOps(
            overlap=overlap_u,
            build_meas_ctx=build_meas_ctx,
            kernels={k_force_bias: force_bias_kernel_u, k_energy: energy_kernel_u},
        )

    if wk == "generalized":
        raise NotImplementedError

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
