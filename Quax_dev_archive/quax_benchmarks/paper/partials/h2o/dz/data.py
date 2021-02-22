import psi4

basis_name = 'cc-pvdz'

# H2O CCSD(T)/DZ geometry
molecule = psi4.geometry("""
                         0 1
                         O            0.000000436927     0.128648202865     0.000000000000
                         H           -1.417747097703    -1.020872149778     0.000000000000
                         H            1.417740163355    -1.020868187071     0.000000000000
                         symmetry c1
                         units bohr
                         """)

