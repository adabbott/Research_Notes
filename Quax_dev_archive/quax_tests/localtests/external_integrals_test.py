import psijax
import psi4
import jax
from jax.config import config; config.update("jax_enable_x64", True)
from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array
from psijax.integrals.oei import oei_arrays
from psijax.methods.hartree_fock import restricted_hartree_fock
from psijax.methods.ccsd_t import rccsd_t
import jax.numpy as np
import numpy as onp
import os
np.set_printoptions(linewidth=800)

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=true "
                           "intra_op_parallelism_threads=8 "
                           "inter_op_parallelism_threads=8 ")

molecule = psi4.geometry("""
                         0 1
                         H  0.0  0.0  0.8
                         H  0.0  0.0 -0.8
                         units ang 
                         """)

# NOTE flattened geometry
geom = onp.asarray(molecule.geometry())
geomflat = np.asarray(geom.flatten())
basis_name = 'sto-3g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])

#print("Number of basis functions", mints.nbf()) 
# For this fixed basis set, generate TEI hessian, overlap hessian, kinetic hessian, potential hessian

print(tei_array(np.asarray(geom), basis_dict))

def wrap(geomflat):
    geom = geomflat.reshape(-1,3)
    return tei_array(geom, basis_dict) 
tei_grad = jax.jacfwd(wrap)(geomflat)
print(tei_grad[:,:,:,:,0].reshape(-1))
print(tei_grad[:,:,:,:,1].reshape(-1))
print(tei_grad[:,:,:,:,2].reshape(-1))
print(tei_grad[:,:,:,:,3].reshape(-1))
print(tei_grad[:,:,:,:,4].reshape(-1))
print(tei_grad[:,:,:,:,5].reshape(-1))


#tei_hess = jax.jacfwd(jax.jacfwd(wrap))(geomflat)
#tei_hess = onp.asarray(tei_hess)
#onp.save('tei_hess_h2_dz_1p6.npy', tei_hess)

#tei_cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(wrap)))(geomflat)
#tei_cube = onp.asarray(tei_cube)
#onp.save('tei_cube_h2_dz_1p6.npy', tei_cube)
#
#def wrap(geomflat):
#    geom = geomflat.reshape(-1,3)
#    return oei_arrays(geom, basis_dict, nuclear_charges) 
#overlap_hess, kinetic_hess, potential_hess = jax.jacfwd(jax.jacfwd(wrap))(geomflat)
#onp.save('overlap_hess_h2_dz_1p6.npy', overlap_hess)
#onp.save('kinetic_hess_h2_dz_1p6.npy', kinetic_hess)
#onp.save('potential_hess_h2_dz_1p6.npy', potential_hess)

#overlap_cube, kinetic_cube, potential_cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(wrap)))(geomflat)
#onp.save('overlap_cube_h2_dz_1p6.npy', overlap_cube)
#onp.save('kinetic_cube_h2_dz_1p6.npy', kinetic_cube)
#onp.save('potential_cube_h2_dz_1p6.npy', potential_cube)


## TEST
#E = restricted_hartree_fock(geomflat, basis_dict, mints, nuclear_charges, charge, SCF_MAX_ITER=50,return_aux_data=False)
#grad = jax.jacfwd(restricted_hartree_fock, 0)(geomflat, basis_dict, mints, nuclear_charges, charge, SCF_MAX_ITER=50,return_aux_data=False)
#hess = jax.jacfwd(jax.jacfwd(restricted_hartree_fock))(geomflat, basis_dict, mints, nuclear_charges, charge, SCF_MAX_ITER=50,return_aux_data=False)
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(restricted_hartree_fock)))(geomflat, basis_dict, mints, nuclear_charges, charge, SCF_MAX_ITER=50,return_aux_data=False)

#print(E)
#print(onp.round(grad.reshape(-1,3), 10))
#print(onp.round(hess, 10))

#psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': False, 'puream': 0})
#psi_E = psi4.energy('hf' + '/' + basis_name)
#psi_grad = onp.round(onp.asarray(psi4.gradient('hf' + '/' + basis_name)), 10)
#psi_hess = onp.round(onp.asarray(psi4.hessian('hf' + '/' + basis_name)), 10)
#print("Psi4 RHF: ")
#print(psi_E)
#print("Psi4 grad:")
#print(psi_grad)
#print("Psi4 hess:") 
#print(psi_hess)

# Test CCSD(T): hessian matches CFOUR exactly
#E = rccsd_t(geomflat, basis_dict, mints, nuclear_charges, charge)
#grad = jax.jacfwd(rccsd_t, 0)(geomflat, basis_dict, mints, nuclear_charges, charge)
#hess = jax.jacfwd(jax.jacfwd(rccsd_t))(geomflat, basis_dict, mints, nuclear_charges, charge)
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(rccsd_t)))(geomflat, basis_dict, mints, nuclear_charges, charge)
#print(cube)
#print(E)
#print(grad.reshape(-1,3))
#print(hess)

#psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': False, 'puream': 0})
#psi_E = psi4.energy('ccsd(t)' + '/' + basis_name)
#psi_grad = onp.round(onp.asarray(psi4.gradient('ccsd(t)' + '/' + basis_name)), 10)
#psi_hess = onp.round(onp.asarray(psi4.hessian('ccsd(t)' + '/' + basis_name)), 10)
#print("Psi4 CCSD(T) ")
#print(psi_E)
#print("Psi4 grad:")
#print(psi_grad)
#print("Psi4 hess:") 
#print(psi_hess)
#

# Test partial derivs

#geom_list = onp.asarray(molecule.geometry()).reshape(-1).tolist()
#
#def scf_partial_wrapper(*args, **kwargs):
#    geom = np.asarray(args)
#    E_scf = restricted_hartree_fock(geom, basis_dict, mints, nuclear_charges, charge, SCF_MAX_ITER=50,return_aux_data=False)
#    return E_scf

#partial_grad = jax.jacfwd(scf_partial_wrapper, 2)(*geom_list, basis_dict=basis_dict, mints=mints, nuclear_charges=nuclear_charges, charge=charge)
#print(partial_grad)

#geom_list = onp.asarray(molecule.geometry()).reshape(-1).tolist()
#def ccsdt_partial_wrapper(*args, **kwargs):
#    geom = np.asarray(args)
#    E = rccsd_t(geom, basis_dict, mints, nuclear_charges, charge)
#    return E
#
#partial_grad = jax.jacfwd(ccsdt_partial_wrapper, 2)(*geom_list, basis_dict=basis_dict, mints=mints, nuclear_charges=nuclear_charges, charge=charge)
#print(partial_grad)

#partial_quartic 


