import psi4

basis_name = 'cc-pvtz'

# N2 CCSD(T)/TZ geometry
molecule = psi4.geometry("""
                         0 1
                         N  0.000000000000     0.000000000000    -1.040129860737
                         N  0.000000000000     0.000000000000     1.040129860737
                         symmetry c1
                         units bohr
                         """)

