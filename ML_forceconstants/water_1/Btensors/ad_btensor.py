import torch
from . import ad_intcos
from . import ad_v3d
#import numpy as np
torch.set_printoptions(threshold=5000, linewidth=200, precision=10)
#np.set_printoptions(threshold=5000, linewidth=200, precision=10)
bohr2ang = 0.529177249

def autodiff_Btensor(intcos, geom, order=1):
    """
    Given internal coordinate definitions and Cartesian geometry, compute the order'th B Tensor
    with PyTorch automatic differentiation. Beyond second order gets really expensive for larger systems. 
    Simple, but inefficient.
    """
    B = ad_intcos.qValues(intcos, geom)   # Generate internal coordinates from cartesians
    nint = B.shape[0]
    ncart = 3 * geom.shape[0]
    count = 0
    shape = [nint, ncart]
    count2 = 0
    while count < order:
        gradients = []
        for val in B.flatten():
            if count + 1 != order: 
                g = torch.autograd.grad(val, geom, create_graph=True)[0].reshape(ncart)
                #count2 += 1
                #print("Derivative {}".format(count2))
            else: # Save time: for last B tensor, don't create derivative graph.
                g = torch.autograd.grad(val, geom, retain_graph=True)[0].reshape(ncart)
                #count2 += 1
                #print("Derivative {}".format(count2))
            gradients.append(g)
        B = torch.stack(gradients).reshape(tuple(shape)) # shape into final B tensor
        shape.append(ncart)                              # update shape for next B tensor
        count += 1                              
    return B

def fast_B(intcos, geom):
    """
    Computes first and second order B tensors more efficiently by ignoring trivial derivatives
    """
    cartesians = list(geom)
    for cart in cartesians:
        cart.requires_grad = True
    # Get internal coordinate types, atom indices, and find the coordinate values
    indices = []
    coords = []
    for i in intcos:
        name = i.__class__.__name__
        if name == 'STRE':
            indices.append([i.A, i.B])
            coord = i.new_q(cartesians[i.A], cartesians[i.B])
        if name == 'BEND':
            indices.append([i.A, i.B, i.C])
            coord = i.new_q(cartesians[i.A], cartesians[i.B], cartesians[i.C])
        if name == 'TORS':
            indices.append([i.A, i.B, i.C, i.D])
            coord = i.new_q(cartesians[i.A], cartesians[i.B], cartesians[i.C], cartesians[i.D])
        coords.append(coord)

    ncart = geom.shape[0] * geom.shape[1]
    natoms = len(cartesians)
    ncoords = len(coords)

    # Compute first order B tensor, while avoiding unnecessary derivatives.
    # For each internal coordinate, and for each cartesian which contributes to defining that internal coordinate, compute the gradient 
    # otherwise, if the cartesian coordinate does not contribute to that internal coordinate, set the gradient to 0
    Brows = []
    for i,intco in enumerate(coords):
        tensorlist = []
        for j in range(natoms):
            if j in indices[i]:
                g = torch.autograd.grad(intco, cartesians[j], create_graph=True, allow_unused=False)[0] 
                tensorlist.append(g)
            else:
                tensorlist.append(torch.zeros((3), dtype=torch.float64))
        Brow = torch.stack(tensorlist, dim=0).flatten()
        Brows.append(Brow)
    B1 = torch.stack(Brows)

    B = B1.reshape((ncoords, natoms, 3))
    c = []
    for i,intco in enumerate(B):
        tensorlist1 = []
        for j,svec in enumerate(intco):
            tensorlist2 = []
            for val in svec:
                if j in indices[i]:
                    for k in range(len(cartesians)):
                        if k in indices[i]:
                            h = torch.autograd.grad(val, cartesians[k], create_graph=True, allow_unused=False)[0] 
                            tensorlist2.append(h)
                        else:
                            tensorlist2.append(torch.zeros((3), dtype=torch.float64))
                else:
                    for l in range(len(cartesians)):
                        tensorlist2.append(torch.zeros((3), dtype=torch.float64))
            temp = torch.stack(tensorlist2, 0).reshape((3,ncart))
            tensorlist1.append(temp)
        c.append(torch.stack(tensorlist1))
    B2 = torch.stack(c).reshape((ncoords,ncart,ncart))
    return B1, B2

