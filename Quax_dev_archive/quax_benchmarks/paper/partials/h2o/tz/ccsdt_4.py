import psi4
import psijax

from data import molecule, basis_name

method = 'ccsd(t)'
d = psijax.core.partial_derivative(molecule, basis_name, method, order=4, address=(2,2,2,2))
print("CCSD(T) ", basis_name, ' partial quartic')
print(d)
