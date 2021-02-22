import psi4

basis_name = 'cc-pvdz'

# N2 CCSD(T)/DZ geometry
molecule = psi4.geometry("""
                         0 1
                         N 0.000000000000     0.000000000000    -1.056883600544
                         N 0.000000000000     0.000000000000     1.056883600544
                         symmetry c1
                         units bohr
                         """)

