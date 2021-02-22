import psijax
import psi4


molecule = psi4.geometry('''
                         0 1
                         N 0.0 0.0 -0.80000000000
                         N 0.0 0.0  0.80000000000
                         symmetry c1
                         units bohr
                         ''')
     

print('n2 scf/cc-pvtz partial quartic')
#print(psijax.core.energy(molecule, 'sto-3g', 'scf'))
psijax.core.partial_derivative(molecule, 'cc-pvtz', 'scf', order=4, address=(5,5,5,5))


