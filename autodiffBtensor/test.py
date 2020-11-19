import psi4
import torch
import ad_intcos
import ad_v3d
import numpy as np
torch.set_printoptions(threshold=5000, linewidth=200, precision=10)
np.set_printoptions(threshold=5000, linewidth=200, precision=10)
bohr2ang = 0.529177249

psi4.core.be_quiet()
h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     0.950000000000 
H            0.000000000000     0.872305301500    -0.376275777700 
O            0.000000000000     0.000000000000     0.000000000000 
''')
# Load Cartesian geometry. Keep graph of derivative for all operations on geometry.
npgeom = np.array(h2o.geometry()) * bohr2ang
geom = torch.tensor(npgeom, requires_grad=True)
print(geom)

def get_interatomics(geom):
    """ 
    Creates internal coordinates in terms of a list of OptKing interatomic distances.
    The order of the interatomic distances is the 
    lower triangle of interatomic distance matrix, in row-wise order.
    Parameters
    ----------
    geom : torch.tensor or numpy.ndarray
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
    nint = internal_coordinate_values.shape[0]
    ncart = 3 * geom.shape[0]
    first = []
    for val in internal_coordinate_values:
        g = torch.autograd.grad(val, geom, create_graph=True)[0].reshape(ncart)
        first.append(g)
    B1 = torch.stack(first) 

    if order > 1:
        second = []
        for Bval in B1.flatten():
            g = torch.autograd.grad(Bval, geom, create_graph=True)[0].reshape(ncart)
            second.append(g)
        B2 = torch.stack(second).reshape(nint, ncart, ncart)

        if order > 2:
            third = []
            for Bval in B2.flatten():
                g = torch.autograd.grad(Bval, geom, create_graph=True)[0].reshape(ncart)
                third.append(g)
            B3 = torch.stack(third).reshape(nint, ncart, ncart, ncart)
            return B3
        return B2
    return B1
    
# TEST FIRST ORDER B TENSOR
#interatomics = get_interatomics(geom)
interatomics = [ad_intcos.STRE(2,1), ad_intcos.STRE(2,0), ad_intcos.BEND(1,2,0)]
B = build_autodiff_B(interatomics, geom, order=2)
print(B[2])

import optking
trueinteratomics = [optking.Stre(2,1), optking.Stre(2,0), optking.Bend(1,2,0)]
trueB = optking.intcosMisc.Bmat(trueinteratomics, npgeom)
print(trueB)
#print(torch.allclose(B, torch.tensor(trueB)))

boolean, B2 = optking.testB.testDerivativeB(trueinteratomics, npgeom)
print(B2)
print(torch.allclose(B[2], torch.tensor(B2)))


