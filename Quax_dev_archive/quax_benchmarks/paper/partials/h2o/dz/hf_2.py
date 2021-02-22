import psi4
import psijax

from data import molecule, basis_name

method = 'rhf'
d = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(2,2))
print("Hartree Fock ", basis_name, ' partial hessian')
print(d)
