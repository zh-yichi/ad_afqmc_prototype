from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
from jax import tree_util
from typing import Tuple, Union

ham_basis = Literal["restricted", "unrestricted", "generalized"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamChol:
    """
    cholesky hamiltonian.

    basis="restricted":
      h1:   (norb, norb)
      chol: (n_fields, norb, norb)

    basis="generalized":
      h1:   (nso, nso)   where nso = 2*norb
      chol: (n_fields, nso, nso)
    """

    h0: jax.Array
    h1: jax.Array
    chol: jax.Array
    basis: ham_basis = "restricted"

    def __post_init__(self):
        if self.basis not in ("restricted", "generalized"):
            raise ValueError(f"unknown basis: {self.basis}")

    def tree_flatten(self):
        children = (self.h0, self.h1, self.chol)
        aux = self.basis
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1, chol = children
        basis = aux
        return cls(h0=h0, h1=h1, chol=chol, basis=basis)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamCholU:
    """
    unrestricted cholesky hamiltonian.
    the number of alpha and beta orbital can be different
    the dimension of the auxiliary fields has to be the same for both alpha and beta

    basis="unrestricted":
      h1 = [h1a, h1b]:   [(norba, norba), (norbb, norbb)]
      chol = [chola, cholb]: [(n_fields, norba, norba), (n_fields, norbb, norbb)]
    """

    h0: jax.Array
    h1: Tuple[jax.Array, jax.Array]
    chol: Tuple[jax.Array, jax.Array]
    basis: ham_basis = "unrestricted"

    def __post_init__(self):
        if self.basis != "unrestricted":
            raise ValueError(
                f"unrestricted hamiltonian does not support : {self.basis} basis!")

    def tree_flatten(self):
        children = (self.h0, self.h1, self.chol)
        aux = self.basis
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1, chol = children
        basis = aux
        return cls(h0=h0, h1=h1, chol=chol, basis=basis)


def n_fields(ham: Union[HamChol, HamCholU]) -> int:
    return int(ham.chol.shape[-3])


def slice_ham_level(
    ham: Union[HamChol, HamCholU], 
    *, 
    norb_keep: Union[int,Tuple] | None, 
    nchol_keep: int | None
    ) -> Union[HamChol, HamCholU]:
    """
    Build a HamChol view for measurement in MLMC:
      - slice orbitals as a prefix [:norb_keep]
      - slice chol as a prefix [:nchol_keep]
    """
    h0 = ham.h0
    h1 = ham.h1
    chol = ham.chol

    if ham.basis in ("restricted", "generalized"):
        if norb_keep is not None:
            h1 = h1[:norb_keep, :norb_keep]
            chol = chol[:, :norb_keep, :norb_keep]

        if nchol_keep is not None:
            chol = chol[:nchol_keep]

        return HamChol(h0=h0, h1=h1, chol=chol, basis=ham.basis)
    
    elif ham.basis == "unrestricted":
        if norb_keep is not None:
            h1[0] = h1[0][:norb_keep[0], :norb_keep[0]]
            h1[1] = h1[1][:norb_keep[1], :norb_keep[1]]
            chol[0] = chol[0][:, :norb_keep[0], :norb_keep[0]]
            chol[1] = chol[1][:, :norb_keep[1], :norb_keep[1]]

        if nchol_keep is not None:
            chol = chol[:,:nchol_keep,:,:]

        return HamCholU(h0=h0, h1=h1, chol=chol, basis=ham.basis)