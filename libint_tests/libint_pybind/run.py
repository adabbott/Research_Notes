import pylibint
import numpy as np
np.set_printoptions(linewidth=500)

pylibint.initialize("/home/adabbott/h2.xyz", 'cc-pvdz')
S = pylibint.overlap().reshape(10,10)
print(np.round(S,4))
pylibint.finalize()

