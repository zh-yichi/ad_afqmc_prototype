import jax.numpy as jnp
from . import trial, meas, prop, driver, ham, integrals, core
import numpy as np
from pyscf import ao2mo

class Rhf:
    def __init__(self, mf):
        mol = mf.mol
        h0 = mf.energy_nuc()
        h1 = np.array(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff)
        eri = np.array(ao2mo.kernel(mol, mf.mo_coeff))
        eri = ao2mo.restore(4, eri, mol.nao)
        chol = integrals.modified_cholesky(eri, max_error=1e-6)

        sys = core.system.system(norb=mol.nao, nelec=mol.nelec, walker_kind="restricted")
        ham_data = ham.chol.ham_chol(h0=jnp.array(h0), h1=jnp.array(h1), chol=jnp.array(chol))
        self.trial_data = trial.rhf.rhf_trial(jnp.eye(mol.nao, mol.nelectron // 2))
        self.trial_ops = trial.rhf.make_rhf_trial_ops(sys=sys)
        self.meas_ops = meas.rhf.make_rhf_meas_ops(sys=sys)
        self.prop_ops = prop.chol_afqmc_ops.make_chol_afqmc_ops(ham_data, sys.walker_kind)
        self.params = prop.types.afqmc_params(
            n_eql_blocks=20, n_blocks=200, seed=np.random.randint(0, int(1e6))
        )
        self.block_fn = prop.blocks.afqmc_block

        self.sys = sys
        self.ham_data = ham_data

    def kernel(self):
        return driver.run_afqmc_energy(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
        )
