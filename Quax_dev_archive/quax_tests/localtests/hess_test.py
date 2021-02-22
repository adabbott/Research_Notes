from psijax.external_integrals import tei
from psijax.external_integrals import overlap
from psijax.external_integrals import potential
from psijax.external_integrals import libint_initialize
from psijax.external_integrals import libint_finalize
import jax
import psi4
import numpy as onp
import jax.numpy as np
import os

molecule = psi4.geometry("""
                         0 1
                         N 0.0 0.0 -0.80000000000
                         N 0.0 0.0  0.80000000000
                         N 0.0 0.0  0.40000000000
                         N 0.0 0.0  0.10000000000
                         symmetry c1
                         units bohr
                         """)

geom = onp.asarray(molecule.geometry())
geomflat = np.asarray(geom.flatten())

basis_name = "sto-3g"
xyz_file_name = "geom.xyz"                                                        
molecule.save_xyz_file(xyz_file_name, True)
xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name

libint_initialize(xyz_path, basis_name)
G = jax.jacfwd(jax.jacfwd(tei))(geomflat)
#G = jax.jacfwd(jax.jacfwd(potential))(geomflat)
libint_finalize()
print(G.shape)
