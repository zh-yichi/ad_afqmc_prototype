import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from pyscf import ao2mo

from . import integrals

from pyscf.scf.hf import RHF
from pyscf.scf.rohf import ROHF
from pyscf.scf.uhf import UHF
from pyscf.scf.ghf import GHF

from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf.cc.gccsd import GCCSD

# Temporary
def get_integrals(mf):
    if not isinstance(mf, (RHF, ROHF, UHF, GHF)):
        raise TypeError(f"Expected RHF, ROHF, UHF, or GHF but found {type(mf)}.")
    if not hasattr(mf, "mo_coeff"):
        raise ValueError(f"mo_coeff not found, you may not have run the scf kernel.")

    mol = mf.mol
    h0 = mf.energy_nuc()

    if isinstance(mf, (RHF, ROHF, GHF)):
        c = mf.mo_coeff
    else:
        c = mf.mo_coeff[0]

    h1 = np.array(c.T.conj() @ mf.get_hcore() @ c)

    #eri = np.array(ao2mo.kernel(mol, mf.mo_coeff))
    #eri = ao2mo.restore(4, eri, mol.nao)
    #chol = integrals.modified_cholesky(eri, max_error=1e-6)
    chol_vec = integrals.chunked_cholesky(mol)  

    nchol = chol_vec.shape[0]

    if not isinstance(mf, GHF):
        norb = c.shape[0]
        chol = np.zeros((nchol, norb, norb), dtype=c.dtype)
        for i in range(nchol):
            chol_i = chol_vec[i].reshape(norb, norb)
            chol[i] = c.T.conj() @ chol_i @ c
    else:
        norb = c.shape[0]//2
        chol = np.zeros((nchol, 2*norb, 2*norb), dtype=c.dtype)
        for i in range(nchol):
            chol_i = chol_vec[i].reshape(norb, norb)
            bchol_i = la.block_diag(chol_i, chol_i)
            chol[i] = c.T.conj() @ bchol_i @ c

    h0=jnp.array(h0)
    h1=jnp.array(h1)
    chol=jnp.array(chol)

    return h0, h1, chol


def get_integrals_unrestricted(mf, chol_cut=1e-6):
    if not isinstance(mf, UHF):
        raise TypeError(f"Expected UHF, but found {type(mf)}.")
    if not hasattr(mf, "mo_coeff"):
        raise ValueError(f"mo_coeff not found, you may not have run the scf kernel.")

    mol = mf.mol
    h0 = jnp.array(mf.energy_nuc())

    mo = mf.mo_coeff
    h1 = mf.get_hcore()
    h1[0] = np.array(mo[0].T.conj() @ h1[0] @ mo[0])
    h1[1] = np.array(mo[1].T.conj() @ h1[1] @ mo[1])

    print("Calculating Cholesky integrals")
    chol = integrals.chunked_cholesky(mol, max_error=chol_cut)  
    chola = jnp.einsum('pr,grs,sq->gpq', mo[0].T.conj(), chol, mo[0], optimize="optimal")
    cholb = jnp.einsum('pr,grs,sq->gpq', mo[1].T.conj(), chol, mo[1], optimize="optimal")
    chol = [chola, cholb]

    return h0, h1, chol


def get_trial_coeff(mf):
    if not isinstance(mf, (RHF, ROHF, UHF, GHF)):
        raise TypeError(f"Expected RHF, ROHF, UHF, or GHF but found {type(mf)}.")
    if not hasattr(mf, "mo_coeff"):
        raise ValueError(f"mo_coeff not found, you may not have run the scf kernel.")

    mol = mf.mol

    if  isinstance(mf, UHF):
        ca = mf.mo_coeff[0]
        cb = mf.mo_coeff[1]
        overlap = mf.get_ovlp(mol)

        q, r = np.linalg.qr(ca.T @ overlap @ ca)
        sgn = np.sign(r.diagonal())
        moa = jnp.einsum("ij,j->ij", q, sgn)

        q, r = np.linalg.qr(ca.T @ overlap @ cb)
        sgn = np.sign(r.diagonal())
        mob = jnp.einsum("ij,j->ij", q, sgn)

        mo = (moa, mob)
    else:
        c = mf.mo_coeff
        overlap = mf.get_ovlp(mol)

        q, r = np.linalg.qr(c.T @ overlap @ c)
        sgn = np.sign(r.diagonal())
        mo = jnp.einsum("ij,j->ij", q, sgn)

    return mo


def get_cisd(cc):
    if not isinstance(cc, (CCSD, GCCSD)):
        raise TypeError(f"Expected CCSD or GCCSD, but found {type(cc)}.")
    if not hasattr(cc, "t1") or not hasattr(cc, "t2"):
        raise ValueError(f"amplitudes not found, you may not have run the cc kernel.")

    ci2 = cc.t2 + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
    ci2 = ci2.transpose(0, 2, 1, 3)

    ci2 = jnp.array(ci2)
    ci1 = jnp.array(cc.t1)

    return ci1, ci2

def get_ucisd(cc):
    if not isinstance(cc, UCCSD):
        raise TypeError(f"Expected UCCSD, but found {type(cc)}.")
    if not hasattr(cc, "t1") or not hasattr(cc, "t2"):
        raise ValueError(f"amplitudes not found, you may not have run the cc kernel.")

    ci2aa = cc.t2[0] + 2 * np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[0])
    ci2aa = (ci2aa - ci2aa.transpose(0, 1, 3, 2)) / 2
    ci2aa = ci2aa.transpose(0, 2, 1, 3)

    ci2bb = cc.t2[2] + 2 * np.einsum("ia,jb->ijab", cc.t1[1], cc.t1[1])
    ci2bb = (ci2bb - ci2bb.transpose(0, 1, 3, 2)) / 2
    ci2bb = ci2bb.transpose(0, 2, 1, 3)

    ci2ab = cc.t2[1] + np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[1])
    ci2ab = ci2ab.transpose(0, 2, 1, 3)

    ci1a = jnp.array(cc.t1[0])
    ci1b = jnp.array(cc.t1[1])
    ci2aa = jnp.array(ci2aa)
    ci2ab = jnp.array(ci2ab)
    ci2bb = jnp.array(ci2bb)

    return ci1a, ci1b, ci2aa, ci2ab, ci2bb
