import psi4
import torch
import ad_intcos
import ad_v3d
import numpy as np
torch.set_printoptions(threshold=5000, linewidth=200, precision=10)
np.set_printoptions(threshold=5000, linewidth=200, precision=10)
bohr2ang = 0.529177249


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

def dummy_computation(p):
    inp1 = p + 2 * p ** 2 
    inp2 = p ** 3
    inp3 = torch.cross(inp1, inp2)
    inp4 = torch.acos(inp2)
    inp5 = torch.cross(inp4, inp3)
    final = torch.sum(inp5, 1) 
    return final

def derivatives(inp, order=1):
    """Computes the n'th order derivative"""
    out = dummy_computation(inp)
    dim1 = out.shape[0]
    dim2 = inp.flatten().shape[0]
    shape = [dim1, dim2]
    count = 0
    while count < order:
        derivatives = []
        for val in out.flatten():
            d = torch.autograd.grad(val, inp, create_graph=True)[0].reshape(dim2)
            derivatives.append(d)
        out = torch.stack(derivatives).reshape(tuple(shape))
        shape.append(dim2)
        count += 1
    return out

#    for param in out.flatten():
#        torch.autograd.backward(param, create_graph=True)
#        first = inp.grad.clone()
#        print('first', inp.grad)
#        inp.grad.zero_()
#    for first_derivative in first.flatten():
#        torch.autograd.backward(first_derivative, create_graph=True)
#        print('second',inp.grad)
#

def experiment(intcos, geom):
    B = ad_intcos.qValues(intcos, geom)   # Generate internal coordinates from cartesians
    for param in B.flatten():
        torch.autograd.backward(param, create_graph=True)
        first = geom.grad.clone()
        geom.grad.zero_()
        #for first_derivative in geom.grad.flatten():
        for first_derivative in first.flatten():
            torch.autograd.backward(first_derivative, create_graph=True)
            print('second',geom.grad)

        geom.grad.zero_()
    nint = B.shape[0]
    ncart = 3 * geom.shape[0]
    count = 0
    shape = [nint, ncart]


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
