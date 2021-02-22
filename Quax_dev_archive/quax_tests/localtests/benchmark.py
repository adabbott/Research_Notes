import psijax
import psi4
import time
psi4.core.be_quiet()
import numpy as onp


def benchmark(label, molecule, basis_sets, methods):
    """Given label (job type:energy,gradient...), molecule, and list of basis sets and methods, benchmark"""
    print("Molecule: ", end=' ')
    mol_string = ' '
    for i in range(molecule.natom()):
        mol_string += molecule.symbol(i) + ' '
    print(mol_string)

    if label == 'energy':
        for basis in basis_sets:
            for method in methods:
                a = time.time()
                K = psijax.core.energy(molecule, basis, method)
                b = time.time()
                print(mol_string + " " + method + "/" + basis + " " + label + " ", onp.round(b-a,3))

    if label == 'gradient':
        for basis in basis_sets:
            for method in methods:
                a = time.time()
                K = psijax.core.derivative(molecule, basis, method, order=1)
                b = time.time()
                print(mol_string + " " + method + "/" + basis + " " + label + " ", onp.round(b-a,3))

    if label == 'hessian':
        for basis in basis_sets:
            for method in methods:
                a = time.time()
                K = psijax.core.derivative(molecule, basis, method, order=2)
                b = time.time()
                print(mol_string + " " + method + "/" + basis + " " + label + " ", onp.round(b-a,3))

print("Begin benchmark...")
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.80000000000
                         H 0.0 0.0  0.80000000000
                         symmetry c1
                         units bohr
                         """)
#benchmark('energy', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])
#benchmark('gradient', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])
#benchmark('hessian', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])

molecule = psi4.geometry("""
                         0 1
                         N 0.0 0.0 -0.80000000000
                         N 0.0 0.0  0.80000000000
                         symmetry c1
                         units bohr
                         """)
#benchmark('energy', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])
#benchmark('gradient', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])
benchmark('hessian', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])

molecule = psi4.geometry("""
                         0 1
                         O
                         H 1 r1
                         H 1 r2 2 a1
                       
                         r1 = 1.0
                         r2 = 1.0
                         a1 = 104.5
                         units ang
                         """)
benchmark('energy', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])
benchmark('gradient', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])
benchmark('hessian', molecule, ['cc-pvdz', 'cc-pvtz'], ['scf', 'mp2', 'ccsd', 'ccsd(t)'])




#print("Energy Timings")
#method = 'scf'
#basis_name = 'cc-pvdz'
#a = time.time()
#E_scf = psijax.core.energy(molecule, basis_name, method)
#b = time.time()
#print("H2 RHF/DZ:  ", onp.round(b-a,3))
#
#method = 'scf'
#basis_name = 'cc-pvtz'
#a = time.time()
#E_scf = psijax.core.energy(molecule, basis_name, method)
#b = time.time()
#print("H2 SCF/TZ:  ", onp.round(b-a,3))
#
#
#method = 'mp2'
#basis_name = 'cc-pvdz'
#a = time.time()
#E_scf = psijax.core.energy(molecule, basis_name, method)
#b = time.time()
#print("H2 MP2/DZ:  ", onp.round(b-a,3))
#
#method = 'mp2'
#basis_name = 'cc-pvtz'
#a = time.time()
#E_scf = psijax.core.energy(molecule, basis_name, method)
#b = time.time()
#print("H2 MP2/TZ:  ", onp.round(b-a,3))
#
#method = 'ccsd'
#basis_name = 'cc-pvdz'
#a = time.time()
#E_scf = psijax.core.energy(molecule, basis_name, method)
#b = time.time()
#print("H2 CCSD/DZ:  ", onp.round(b-a,3))
#
#method = 'ccsd'
#basis_name = 'cc-pvtz'
#a = time.time()
#E_scf = psijax.core.energy(molecule, basis_name, method)
#b = time.time()
#print("H2 CCSD/TZ:  ", onp.round(b-a,3))
#
#

