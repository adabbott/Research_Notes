import psijax
import numpy as onp
import jax.numpy as np
onp.set_printoptions(linewidth=800, precision=10)

import psi4
import time
psi4.core.be_quiet()

print("Running Accuracy Test: Testing Against Psi4 energies, gradients and Hessians")
print("Test system: N2 cc-pvdz")
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.80000000000
                         H 0.0 0.0  0.80000000000
                         symmetry c1
                         units bohr
                         """)
#basis_name = 'cc-pvdz'
basis_name = 'sto-3g'
#basis_name = 'sto-3g'
psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': False, 'puream': 0})

print("Testing Energies")
method = 'scf'
print("Psi4 RHF:       ",psi4.energy(method + '/' + basis_name))
#print("PsiJax RHF:     ",psijax.core.energy(molecule, basis_name, 'scf'))
psi_deriv = onp.round(onp.asarray(psi4.gradient(method + '/' + basis_name)), 10)
print(psi_deriv)
psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name)), 10)
print(psi_deriv)

#method = 'mp2'
#print("Psi4 MP2:       ",psi4.energy(method + '/' + basis_name))
#print("PsiJax MP2:     ",psijax.core.energy(molecule, basis_name, 'mp2'))


#method = 'ccsd(t)'
#print("Psi4 CCSD(T):   ",psi4.energy(method + '/' + basis_name))
#print("PsiJax CCSD(T): ",psijax.core.energy(molecule, basis_name, 'ccsd(t)'))
#
#
#print("Testing Gradients")
#print("SCF")
#method = 'scf'
##print(psi_deriv)
#psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=1))
##print(psijax_deriv)
#print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv))
# 
#print("MP2")
#method = 'mp2'
#psi_deriv = onp.round(onp.asarray(psi4.gradient(method + '/' + basis_name)), 10)
##print(psi_deriv)
#psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=1))
##print(psijax_deriv)
#print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv))
#
#print("CCSD(T)")
#method = 'ccsd(t)'
#psi_deriv = onp.round(onp.asarray(psi4.gradient(method + '/' + basis_name)), 10)
##print(psi_deriv)
#psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=1))
##print(psijax_deriv)
#print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv))
##
##print("Testing Hessians")
##print("SCF")
##method = 'scf'
##psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name)), 10)
###print(psi_deriv)
##psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
###print(psijax_deriv)
##print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv))
##
##print("MP2")
##method = 'mp2'
##psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name, dertype='gradient')), 10)
###print(psi_deriv)
##psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
###print(psijax_deriv)
##print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv,rtol=1e-4,atol=1.e-4))
##
##print("CCSD(T)")
##method = 'ccsd(t)'
##psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name, dertype='gradient')), 10)
###print(psi_deriv)
##psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
###print(psijax_deriv)
##print("CORRECT? ",onp.allclose(psijax_deriv, psi_deriv,rtol=1e-4,atol=1.e-4))
##
##
