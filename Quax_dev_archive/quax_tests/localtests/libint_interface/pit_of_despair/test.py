import psijax
import psi4
from psijax.external_integrals import libint_interface

# Test simple psijax --> c++ interface
a = libint_interface.add(25,50)
print(a)

# Now lets try accessing libint.
# Build molecule
molecule = psi4.geometry("""
                         0 1
                         N 0.0 0.0 -0.80000000000
                         N 0.0 0.0  0.80000000000
                         symmetry c1
                         units bohr
                         """)

molecule.save_xyz_file("tmp.xyz", True)
basis_name = 'cc-pvdz'

