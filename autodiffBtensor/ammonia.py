import psi4
import torch
import ad_intcos
import ad_v3d
import numpy as np
torch.set_printoptions(threshold=5000, linewidth=400, precision=6)
np.set_printoptions(threshold=5000, linewidth=400, precision=6)
bohr2ang = 0.529177249

psi4.core.be_quiet()
h2o = psi4.geometry(
'''
 N  0.000000  0.0       0.0 
 H  1.584222  0.0       1.12022
 H  0.0      -1.58422  -1.12022
 H -1.584222  0.0       1.12022
 H  0.0       1.58422  -1.12022
 unit au
''')
# Load Cartesian geometry. Keep graph of derivative for all operations on geometry.
#npgeom = np.array(h2o.geometry()) * bohr2ang
npgeom = np.array(h2o.geometry())
geom = torch.tensor(npgeom, requires_grad=True)
print(geom)

def get_interatomics(geom):
    """ 
    Creates internal coordinates in terms of a list of OptKing interatomic distances.
    The order of the interatomic distances is the 
    lower triangle of interatomic distance matrix, in row-wise order.
    Parameters
    ----------
    geom : torch.tensor
        Cartesian coordinates array (N x 3)
    """
    natoms = geom.shape[0]
    indices = np.asarray(np.tril_indices(natoms,-1)).transpose(1,0)
    interatomics = []
    for i in indices:
        idx1, idx2 = i
        interatomics.append(ad_intcos.STRE(idx1, idx2))
    return interatomics

def build_autodiff_B(intcos, geom, order=1):
    internal_coordinate_values = ad_intcos.qValues(intcos, geom) 
    grads = []
    for val in internal_coordinate_values:
        g = torch.autograd.grad(val, geom, create_graph=True)[0].reshape(3*geom.shape[0])
        grads.append(g)
    B = torch.stack(grads) 
    return B
    
#interatomics = get_interatomics(geom)
interatomics = [ad_intcos.STRE(0,1), ad_intcos.STRE(0,2), ad_intcos.STRE(0,3), ad_intcos.STRE(0,4), ad_intcos.BEND(1,0,2), ad_intcos.BEND(1,0,3), ad_intcos.BEND(1,0,4), ad_intcos.BEND(2,0,3), ad_intcos.BEND(2,0,4), ad_intcos.BEND(3,0,4)]
B = build_autodiff_B(interatomics, geom)
print(B)

#intcos = ad_intcos.qValues(interatomics, geom)
#for i in intcos:
#    print(i)
#print(intcos)
#g1 = torch.autograd.grad(intcos[0], geom, create_graph=True)[0]
#g2 = torch.autograd.grad(intcos[1], geom, create_graph=True)[0]
#g3 = torch.autograd.grad(intcos[2], geom, create_graph=True)[0]
#g = torch.autograd.grad(intcos, geom, create_graph=True)
#print(g)
#tmpB = torch.stack([g1.reshape(9),g2.reshape(9), g3.reshape(9)])
#print(tmpB)

# Now do the same with original
from helper import helper
def old_get_interatomics(geom):
    # Build OptKing internal coordinates of interatomic distances
    natoms = geom.shape[0]
    # Indices of unique interatomic distances, lower triangle row-wise order
    indices = np.asarray(np.tril_indices(natoms,-1)).transpose(1,0)
    interatomics = []
    for i in indices:
        idx1, idx2 = i
        interatomics.append(helper.STRE(idx1, idx2))
    return interatomics

#old_interatomics = old_get_interatomics(npgeom)
old_interatomics = [helper.STRE(2,1), helper.STRE(2,0), helper.BEND(1,2,0)]
old_intcos = helper.qValues(old_interatomics, npgeom)
realB = helper.Bmat(old_interatomics, npgeom)
print(realB)
