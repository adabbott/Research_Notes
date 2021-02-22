import psi4

basis_name = 'cc-pvtz'

# H2O CCSD(T)/TZ geometry
molecule = psi4.geometry("""
                         0 1
                         O           -0.000007070942     0.125146536460     0.000000000000
                         H           -1.424097055410    -0.993053750648     0.000000000000
                         H            1.424209276385    -0.993112599269     0.000000000000
                         symmetry c1
                         units bohr
                         """)

