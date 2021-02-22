import psijax
import psi4
import jax
from jax.config import config; config.update("jax_enable_x64", True)
from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array
from psijax.integrals.oei import oei_arrays
from psijax.methods.hartree_fock import restricted_hartree_fock
import jax.numpy as np
import numpy as onp
import os
np.set_printoptions(linewidth=800)

molecule = psi4.geometry("""
                         0 1
                         H  0.0  0.0  0.8
                         H  0.0  0.0 -0.8
                         units ang 
                         """)
# NOTE flattened geometry
geom = onp.asarray(molecule.geometry())
geomflat = np.asarray(geom.flatten())
basis_name = 'cc-pvdz'
xyz_file_name = "geom.xyz"
# Save xyz file, get path
molecule.save_xyz_file(xyz_file_name, True)
xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name

basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])

# My libint interface integrals
eri1 = psijax.external_integrals.libint_interface.eri(xyz_path, basis_name)
dim = int(onp.sqrt(onp.sqrt(eri1.shape[0])))
eri1 = eri1.reshape(dim,dim,dim,dim)
eri2 = onp.asarray(mints.ao_eri())
print("Two electron integrals match:", onp.allclose(eri1,eri2))

# My Libint interface integral derivative
deriv_vec = np.array([0,0,0,0,0,1])
grad = psijax.external_integrals.libint_interface.eri_deriv(xyz_path, basis_name, deriv_vec)
grad = grad.reshape(dim,dim,dim,dim)

def wrap(geomflat):
    geom = geomflat.reshape(-1,3)
    return tei_array(geom, basis_dict) 
tei_grad = jax.jacfwd(wrap)(geomflat)
print("gradients match?",onp.allclose(tei_grad[:,:,:,:,5], grad))

deriv_vec = np.array([0,0,0,0,0,2])
hess1 = psijax.external_integrals.libint_interface.eri_deriv(xyz_path, basis_name, deriv_vec)
hess1 = hess1.reshape(dim,dim,dim,dim)

deriv_vec = np.array([0,0,0,0,1,1])
hess2 = psijax.external_integrals.libint_interface.eri_deriv(xyz_path, basis_name, deriv_vec)
hess2 = hess2.reshape(dim,dim,dim,dim)

deriv_vec = np.array([0,0,1,0,0,1])
hess3 = psijax.external_integrals.libint_interface.eri_deriv(xyz_path, basis_name, deriv_vec)
hess3 = hess3.reshape(dim,dim,dim,dim)

tei_hess = jax.jacfwd(jax.jacfwd(wrap))(geomflat)
print("diag hessian match?",onp.allclose(tei_hess[:,:,:,:,5,5], hess1))
print("offdiag same atom hessian match?",onp.allclose(tei_hess[:,:,:,:,4,5], hess2))
print("offdiag diff atom hessian match?",onp.allclose(tei_hess[:,:,:,:,2,5], hess3))


#print(tei_grad[:,:,:,:,0].reshape(-1))
#print(tei_grad[:,:,:,:,1].reshape(-1))
#print(tei_grad[:,:,:,:,2].reshape(-1))
#print(tei_grad[:,:,:,:,3].reshape(-1))
#print(tei_grad[:,:,:,:,4].reshape(-1))
#print(tei_grad[:,:,:,:,5].reshape(-1))
#
