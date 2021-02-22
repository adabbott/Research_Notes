import psi4
import jax
from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array
import jax.numpy as np
import numpy as onp

molecule = psi4.geometry("""
                         0 1
                         H -0.0 -0.0 -0.8
                         H  0.0  0.0  0.8
                         units bohr
                         """)

geom = np.asarray(onp.asarray(molecule.geometry()))
basis_name = 'sto-3g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)

# Wrap TEI array with a flattened geometry
# so we get TEI derivatives of shape (n,n,n,n,ncart,ncart...)
def wrap(geomflat):
    geom = geomflat.reshape(-1,3)
    return tei_array(geom, basis_dict) 

tei_hessian = jax.jacfwd(jax.jacfwd(wrap))(geom.reshape(-1))
print(tei_hessian.shape)
# Save to disk?
#np.save('tei_hess.npy',tei_hessian)


