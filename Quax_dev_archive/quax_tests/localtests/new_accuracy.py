import psijax
import numpy as onp
import jax.numpy as np
onp.set_printoptions(linewidth=800, precision=10)
from psijax.methods.hartree_fock import restricted_hartree_fock  
from psijax.methods.mp2 import restricted_mp2
import jax

import psi4
import time
import os
psi4.core.be_quiet()

print("Running Accuracy Test: Testing Against Psi4 energies, gradients and Hessians")
#print("Test system: N2 cc-pvdz")
molecule = psi4.geometry("""
                         0 1
                         N 0.0 0.0 -0.80000000000
                         N 0.0 0.0  0.80000000000
                         symmetry c1
                         units bohr
                         """)

# NOTE flattened geometry                                                         
geom = onp.asarray(molecule.geometry())                                           
geomflat = np.asarray(geom.flatten())                                             
basis_name = 'sto-3g'
xyz_file_name = "geom.xyz"                                                        

# Save xyz file, get path                                                         
molecule.save_xyz_file(xyz_file_name, True)                                       
xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name                     

# Psi4
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)     
mints = psi4.core.MintsHelper(basis_set)                                          
charge = molecule.molecular_charge()                                              
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])  
psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': False, 'puream': 0})

print("Testing Energies")
E_hf = restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
print(E_hf)
method = 'scf'
print("Psi4 RHF:       ",psi4.energy(method + '/' + basis_name))
E_mp2 = restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge)
print("MP2",E_mp2)
method = 'mp2'
print("Psi4 MP2:       ",psi4.energy(method + '/' + basis_name))

print("Testing Gradients")
method = 'scf'
psi_deriv = onp.round(onp.asarray(psi4.gradient(method + '/' + basis_name)), 10)
psijax_deriv = jax.jacfwd(restricted_hartree_fock)(geomflat, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
print(psijax_deriv.reshape(-1,3))
print(psi_deriv)
print("CORRECT? ",onp.allclose(psijax_deriv.reshape(-1,3), psi_deriv))

print("Testing Hessians")
psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name)), 10)
psijax_deriv = jax.jacfwd(jax.jacfwd(restricted_hartree_fock))(geomflat, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv))

#print("MP2")
#method = 'mp2'
#psi_deriv = onp.round(onp.asarray(psi4.gradient(method + '/' + basis_name)), 10)
#psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=1))
#print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv))
#
#print("CCSD(T)")
#method = 'ccsd(t)'
#psi_deriv = onp.round(onp.asarray(psi4.gradient(method + '/' + basis_name)), 10)
#psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=1))
#print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv))
#
#print("Testing Hessians")
#print("SCF")
#method = 'scf'
#psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name)), 10)
##print(psi_deriv)
#psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
##print(psijax_deriv)
#print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv))
#
#print("MP2")
#method = 'mp2'
#psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name, dertype='gradient')), 10)
##print(psi_deriv)
#psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
##print(psijax_deriv)
#print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv,rtol=1e-4,atol=1.e-4))
#
#print("CCSD(T)")
#method = 'ccsd(t)'
#psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name, dertype='gradient')), 10)
##print(psi_deriv)
#psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
##print(psijax_deriv)
#print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv,rtol=1e-4,atol=1.e-4))


