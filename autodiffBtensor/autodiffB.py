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
# Load Cartesian geometry. Convert to Torch Tensor to keep graph of derivative for all operations on the Cartesian geometry.
npgeom = np.array(h2o.geometry()) * bohr2ang
geom = torch.tensor(npgeom, requires_grad=True)

def get_interatomics(geom):
    """ 
    Convenience function. Creates internal coordinates in terms of a list of OptKing interatomic distances.
    The order of the interatomic distances is the lower triangle of interatomic distance matrix, in row-wise order.

    Parameters
    ----------
    geom : torch.tensor or numpy.ndarray
        Cartesian coordinates array (N x 3)
    Returns
    -------
    interatomics : list
        A list of OptKing STRE internal coordinate objects
    """
    natoms = geom.shape[0]
    indices = np.asarray(np.tril_indices(natoms,-1)).transpose(1,0)
    interatomics = []
    for i in indices:
        idx1, idx2 = i
        interatomics.append(ad_intcos.STRE(idx1, idx2))
    return interatomics

def autodiff_Btensor(intcos, geom, order=1):
    """
    Given internal coordinate definitions and Cartesian geometry, compute the order'th B Tensor
    with PyTorch automatic differentiation. Beyond third order gets really expensive. 
    """
    B = ad_intcos.qValues(intcos, geom)   # Generate internal coordinates from cartesians
    nint = B.shape[0]
    ncart = 3 * geom.shape[0]
    count = 0
    shape = [nint, ncart]
    while count < order:
        gradients = []
        for val in B.flatten():
            if count + 1 != order: 
                g = torch.autograd.grad(val, geom, create_graph=True)[0].reshape(ncart)
            else: # Save time: for last B tensor, don't create derivative graph.
                g = torch.autograd.grad(val, geom, retain_graph=True)[0].reshape(ncart)
            gradients.append(g)
        B = torch.stack(gradients).reshape(tuple(shape)) # shape into final B tensor
        shape.append(ncart)                              # update shape for next B tensor
        count += 1                              
    return B

# TEST FIRST ORDER B TENSOR
#interatomics = get_interatomics(geom)
interatomics = [ad_intcos.STRE(2,1), ad_intcos.STRE(2,0), ad_intcos.BEND(1,2,0)]
B1 = autodiff_Btensor(interatomics, geom, order=1)
#TODO TODO test against optking, more systems

#import optking
#trueinteratomics = [optking.Stre(2,1), optking.Stre(2,0), optking.Bend(1,2,0)]
#trueB = optking.intcosMisc.Bmat(trueinteratomics, npgeom)
#print(trueB)
##print(torch.allclose(B, torch.tensor(trueB)))
#
#boolean, B2 = optking.testB.testDerivativeB(trueinteratomics, npgeom)
#print(B2)
#print(torch.allclose(B[2], torch.tensor(B2)))
#



def oldold(intcos, geom, order=1):
    """
    Given internal coordinate definitions and Cartesian geometry, compute the order'th B Tensor
    with PyTorch automatic differentiation. Only up to third order is supported, but this is easily extensible.
    """
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
                g = torch.autograd.grad(Bval, geom, retain_graph=True)[0].reshape(ncart)
                third.append(g)
            B3 = torch.stack(third).reshape(nint, ncart, ncart, ncart)
            return B3
        return B2
    return B1

def old(intcos, geom, order=1):
    if order > 3:
        raise Exception("Too expensive.")
    internal_coordinate_values = ad_intcos.qValues(intcos, geom) 
    nint = internal_coordinate_values.shape[0]
    ncart = 3 * geom.shape[0]
    first = []
    for val in internal_coordinate_values:
        g = torch.autograd.grad(val, geom, create_graph=True)[0].reshape(ncart)
        first.append(g)
    B = torch.stack(first) 

    count = 1
    shape = [nint, ncart, ncart]
    while count < order:
        gradients = []
        for Bval in B.flatten():
            if count + 1 == order: # Save time: on last B tensor, don't create derivative graph.
                g = torch.autograd.grad(Bval, geom, retain_graph=True)[0].reshape(ncart)
            else:
                g = torch.autograd.grad(Bval, geom, create_graph=True)[0].reshape(ncart)
            gradients.append(g)
        B = torch.stack(gradients).reshape(tuple(shape))
        shape.append(ncart)
        count += 1
    return B
