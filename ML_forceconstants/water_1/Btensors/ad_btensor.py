import torch
from . import ad_intcos
#import .ad_intcos
#import .ad_v3d
from . import ad_v3d
#import numpy as np
torch.set_printoptions(threshold=5000, linewidth=200, precision=10)
#np.set_printoptions(threshold=5000, linewidth=200, precision=10)
bohr2ang = 0.529177249

def compute_btensors(intcos, geom, order=1):
    """Computes and returns all B tensors up to order'th order. Max order is 3rd order"""
    if order > 3:
        raise Exception("Only up to 3rd order is allowed. Too expensive after that!")
    #TODO remove trivial derivative computations
    cart_vec = geom.flatten()
    # Generate internal coordinates from cartesians. This constructs a computation graph which can be differentiated.
    internals = ad_intcos.qValues(intcos, cart_vec)   
    nint = internals.shape[0]
    ncart = 3 * geom.shape[0]
    count = 0
    shape = [nint, ncart]
    count2 = 0

    g1, h1, c1 = [], [], []
    for d0 in internals:
        g = torch.autograd.grad(d0, cart_vec, create_graph=True)[0]
        g1.append(g)
        if order > 1:
            h2, c2 = [], []
            for d1 in g:
                h = torch.autograd.grad(d1, cart_vec, create_graph=True)[0]
                h2.append(h)
                if order > 2:
                    c3 = []
                    for d2 in h:
                        c = torch.autograd.grad(d2, cart_vec, create_graph=True)[0]
                        c3.append(c)
                    c2.append(torch.stack(c3))
                else:
                    continue
            h1.append(torch.stack(h2))
            if order > 2: c1.append(torch.stack(c2))
        else:
            continue
    B1 = torch.stack(g1)
    if order > 1:
        B2 = torch.stack(h1)
        if order > 2:
            B3 = torch.stack(c1)
            return B1, B2, B3
        else:
            return B1, B2
    else:
        return B1

def fast_B(intcos, geom):
    #TODO this is actually slower than above function
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


def old_autodiff_Btensor(intcos, geom, order=1):
    """
    DEPRECATED. Very slow because objects are overwritten, computation graphs destroyed, causes memory leaks at high order
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
            else: # Save time: for last B tensor, don't create derivative graph.
                g = torch.autograd.grad(val, geom, retain_graph=True)[0].reshape(ncart)
            gradients.append(g)
        B = torch.stack(gradients).reshape(tuple(shape)) # shape into final B tensor
        shape.append(ncart)                              # update shape for next B tensor
        count += 1                              
    return B


#def compute_btensors(intcos, geom, order=1):
#    """Computes and returns all B tensors up to order'th order. Max order is 5th order"""
#    cart_vec = geom.flatten()
#    # Generate internal coordinates from cartesians. This constructs a computation graph which can be differentiated.
#    internals = ad_intcos.qValues(intcos, cart_vec)   
#    nint = internals.shape[0]
#    ncart = 3 * geom.shape[0]
#    count = 0
#    shape = [nint, ncart]
#    count2 = 0
#
#    g1, h1, c1, q1, f1 = [], [], [], [], []
#    for d0 in internals:
#        g = torch.autograd.grad(d0, cart_vec, create_graph=True)[0]
#        g1.append(g)
#        h2, c2, q2, f2 = [], [], [], []
#        for d1 in g:
#            h = torch.autograd.grad(d1, cart_vec, create_graph=True)[0]
#            h2.append(h)
#            c3, q3, f3 = [], [], []
#            for d2 in h:
#                c = torch.autograd.grad(d2, cart_vec, create_graph=True)[0]
#                c3.append(c)
#                q4, f4 = [], []
#                for d3 in c:
#                    q = torch.autograd.grad(d3, cart_vec, create_graph=True)[0]
#                    q4.append(q)
#                    f5 = []
#                    for d4 in q:
#                        f = torch.autograd.grad(d4, cart_vec, create_graph=True)[0]
#                        f5.append(f)
#                    f4.append(torch.stack(f5))
#                f3.append(torch.stack(f4))
#                q3.append(torch.stack(q4))
#            c2.append(torch.stack(c3))
#            q2.append(torch.stack(q3))
#            f2.append(torch.stack(f3))
#        h1.append(torch.stack(h2))
#        c1.append(torch.stack(c2))
#        q1.append(torch.stack(q2))
#        f1.append(torch.stack(f2))
#    B1 = torch.stack(g1)
#    B2 = torch.stack(h1)
#    B3 = torch.stack(c1)
#    B4 = torch.stack(q1)
#    B5 = torch.stack(f1)
#    return B1, B2, B3, B4, B5

