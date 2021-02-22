#import psijax
import psi4
import jax
from jax.config import config; config.update("jax_enable_x64", True)
from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array
from psijax.methods.hartree_fock import restricted_hartree_fock
import jax.numpy as np
import numpy as onp
np.set_printoptions(linewidth=800)

molecule = psi4.geometry("""
                         0 1
                         H  0.0  0.0 -0.8
                         H  0.0  0.0  0.8
                         units bohr
                         """)

# NOTE flattened geometry
geom = onp.asarray(molecule.geometry())
geomflat = np.asarray(geom.flatten())

basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)

charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
#E = restricted_hartree_fock(geomflat, basis_dict, mints, nuclear_charges, charge, SCF_MAX_ITER=50,return_aux_data=False)
#grad = jax.jacfwd(restricted_hartree_fock, 0)(geomflat, basis_dict, mints, nuclear_charges, charge, SCF_MAX_ITER=50,return_aux_data=False)
#print(grad.reshape(-1,3))

psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': False, 'puream': 0})
print("Psi4 RHF: ", psi4.energy('hf' + '/' + basis_name))
#psi_deriv = onp.round(onp.asarray(psi4.gradient('hf' + '/' + basis_name)), 10)

#print("Psi4 grad:", psi_deriv)

print(mints.ao_oei_deriv1("OVERLAP", 0))


# Wrap TEI array with a flattened geometry
# so we get TEI derivatives of shape (n,n,n,n,ncart,ncart...)
#def wrap(geomflat):
#    geom = geomflat.reshape(-1,3)
#    return tei_array(geom, basis_dict) 
#
