import psi4
import psijax

from data import molecule, basis_name

print("Writing integrals up to order 6")
psijax.core.write_integrals(molecule, basis_name, order=1, address=(2,))
psijax.core.write_integrals(molecule, basis_name, order=2, address=(2,2))
psijax.core.write_integrals(molecule, basis_name, order=3, address=(2,2,2))
psijax.core.write_integrals(molecule, basis_name, order=4, address=(2,2,2,2))
psijax.core.write_integrals(molecule, basis_name, order=5, address=(2,2,2,2,2))
psijax.core.write_integrals(molecule, basis_name, order=6, address=(2,2,2,2,2,2))
