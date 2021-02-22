#import psijax
import psi4
import jax
from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array
import jax.numpy as np
import numpy as onp
np.set_printoptions(linewidth=800)

molecule = psi4.geometry("""
                         0 1
                         H -1.0 -2.0 -3.0
                         H  1.0  2.0 3.0
                         units bohr
                         """)

geom = np.asarray(onp.asarray(molecule.geometry()))
basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)

#G = tei_array(geom, basis_dict)
#psi_G = onp.asarray(mints.ao_eri())
#print("psijax G matches psi4 G", onp.allclose(G, psi_G))

# Wrap TEI array with a flattened geometry
# so we get TEI derivatives of shape (n,n,n,n,ncart,ncart...)
def wrap(geomflat):
    geom = geomflat.reshape(-1,3)
    return tei_array(geom, basis_dict) 

x1, y1, z1 = mints.ao_tei_deriv1(0)
x2, y2, z2 = mints.ao_tei_deriv1(1)
G_grad = jax.jacfwd(wrap)(geom.reshape(-1))
print(onp.allclose(G_grad[:,:,:,:,0], x1))
print(onp.allclose(G_grad[:,:,:,:,1], y1))
print(onp.allclose(G_grad[:,:,:,:,2], z1))
print(onp.allclose(G_grad[:,:,:,:,3], x2))
print(onp.allclose(G_grad[:,:,:,:,4], y2))
print(onp.allclose(G_grad[:,:,:,:,5], z2))



# Second derivative
#G_hess = jax.jacfwd(jax.jacfwd(wrap))(geom.reshape(-1))

#x1x1, x1y1, x1z1, y1x1, y1y1, y1z1, z1x1, z1y1, z1z1  = mints.ao_tei_deriv2(0,0)
#x1x2, x1y2, x1z2, y1x2, y1y2, y1z2, z1x2, z1y2, z1z2  = mints.ao_tei_deriv2(0,1)
#x2x1, x2y1, x2z1, y2x1, y2y1, y2z1, z2x1, z2y1, z2z1  = mints.ao_tei_deriv2(1,0)
#x2x2, x2y2, x2z2, y2x2, y2y2, y2z2, z2x2, z2y2, z2z2  = mints.ao_tei_deriv2(1,1)
#
#psi_G_hess = onp.zeros((2,2,2,2,6,6))
#
#psi_G_hess[:,:,:,:,0,0] = x1x1
#
#psi_G_hess[:,:,:,:,0,0] = x1x1
#psi_G_hess[:,:,:,:,0,1] =
#psi_G_hess[:,:,:,:,0,2]
#psi_G_hess[:,:,:,:,1,0]
#psi_G_hess[:,:,:,:,1,1]
#psi_G_hess[:,:,:,:,1,2]
#psi_G_hess[:,:,:,:,2,0]
#psi_G_hess[:,:,:,:,2,1]
#psi_G_hess[:,:,:,:,2,2]
#psi_G_hess[:,:,:,:,3,3]
#psi_G_hess[:,:,:,:,3,4]
#psi_G_hess[:,:,:,:,3,5]
#psi_G_hess[:,:,:,:,4,3]
#psi_G_hess[:,:,:,:,4,4]
#psi_G_hess[:,:,:,:,4,5]
#psi_G_hess[:,:,:,:,5,3]
#psi_G_hess[:,:,:,:,5,4]
#psi_G_hess[:,:,:,:,5,5]


#print('x1x1', onp.allclose(x1x1, ))
#print('x1y1', onp.allclose(x1y1, ))
#print('x1z1', onp.allclose(x1z1, ))
#print('y1x1', onp.allclose(y1x1, ))
#print('y1y1', onp.allclose(y1y1, ))
#print('y1z1', onp.allclose(y1z1, ))
#print('z1x1', onp.allclose(z1x1, ))
#print('z1y1', onp.allclose(z1y1, ))
#print('z1z1', onp.allclose(z1z1, ))
#print('x2x2', onp.allclose(x2x2, ))
#print('x2y2', onp.allclose(x2y2, ))
#print('x2z2', onp.allclose(x2z2, ))
#print('y2x2', onp.allclose(y2x2, ))
#print('y2y2', onp.allclose(y2y2, ))
#print('y2z2', onp.allclose(y2z2, ))
#print('z2x2', onp.allclose(z2x2, ))
#print('z2y2', onp.allclose(z2y2, ))
#print('z2z2', onp.allclose(z2z2, ))



# atom 1 second derivatives: these are correct
#print(onp.asarray(x1x1))
#print(G_hess[:,:,:,:,0,0])
#
#print('x1x1', onp.allclose(x1x1, G_hess[:,:,:,:,0,0]))
#print('x1y1', onp.allclose(x1y1, G_hess[:,:,:,:,0,1]))
#print('x1z1', onp.allclose(x1z1, G_hess[:,:,:,:,0,2]))
#print('y1x1', onp.allclose(y1x1, G_hess[:,:,:,:,1,0]))
#print('y1y1', onp.allclose(y1y1, G_hess[:,:,:,:,1,1]))
#print('y1z1', onp.allclose(y1z1, G_hess[:,:,:,:,1,2]))
#print('z1x1', onp.allclose(z1x1, G_hess[:,:,:,:,2,0]))
#print('z1y1', onp.allclose(z1y1, G_hess[:,:,:,:,2,1]))
#print('z1z1', onp.allclose(z1z1, G_hess[:,:,:,:,2,2]))
#
## atom 2 second derivatives: these are correct
#print('x2x2', onp.allclose(x2x2, G_hess[:,:,:,:,3,3]))
#print('x2y2', onp.allclose(x2y2, G_hess[:,:,:,:,3,4]))
#print('x2z2', onp.allclose(x2z2, G_hess[:,:,:,:,3,5]))
#print('y2x2', onp.allclose(y2x2, G_hess[:,:,:,:,4,3]))
#print('y2y2', onp.allclose(y2y2, G_hess[:,:,:,:,4,4]))
#print('y2z2', onp.allclose(y2z2, G_hess[:,:,:,:,4,5]))
#print('z2x2', onp.allclose(z2x2, G_hess[:,:,:,:,5,3]))
#print('z2y2', onp.allclose(z2y2, G_hess[:,:,:,:,5,4]))
#print('z2z2', onp.allclose(z2z2, G_hess[:,:,:,:,5,5]))
#
## Cross derivatives: no match
#print('x1x2', onp.allclose(x1x2, G_hess[:,:,:,:,0,3]))
#
#print('x1x2', onp.allclose(x1x2, G_hess[:,:,:,:,0,3]))
#
