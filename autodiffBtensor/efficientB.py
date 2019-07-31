import psi4
import torch
import ad_intcos
import ad_v3d
import numpy as np
torch.set_printoptions(threshold=5000, linewidth=200, precision=5)
np.set_printoptions(threshold=5000, linewidth=200, precision=5)
bohr2ang = 0.529177249

def fast_Btensor(intcos, geom):
    cartesians = list(geom)
    natoms = len(cartesians)
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
    B = torch.stack(Brows)
    return B

def fast_B2(intcos, geom):
    cartesians = list(geom)
    natoms = len(cartesians)
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
    B = torch.stack(Brows)
    for i in B:
        print(i)
    return B


def autodiff_Btensor(intcos, geom, order=1):
    """
    Given internal coordinate definitions and Cartesian geometry, compute the order'th B Tensor
    with PyTorch automatic differentiation. Beyond third order gets really expensive. 
    """
    #geom.requires_grad = True
    cartesians = list(geom)
    natoms = len(cartesians)
    for cart in cartesians:
        cart.requires_grad = True
    # Get internal coordinate types, atom indices, and find the coordinate values
    types = []
    indices = []
    coords = []
    for i in intcos:
        name = i.__class__.__name__
        types.append(name)
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

    #print(types)
    print(indices)
    print(coords)
    #coords = ad_intcos.qValues(intcos, geom)   # Generate internal coordinates from cartesians
    #print(coords)
    #flatxyz = geom.flatten()
    #for i,intco in enumerate(coords):
    #    for cart_idx in indices[i]:
    #        # Compute gradient of this internal coordinates wrt the cartesian coordinates of this atom
    #        g = torch.autograd.grad(intco, cartesians[cart_idx], create_graph=True, allow_unused=False)[0] 
    #        print(g)
    Bs = []
    for i,intco in enumerate(coords):
        tensorlist = []
        for j in range(natoms):
            if j in indices[i]:
                g = torch.autograd.grad(intco, cartesians[j], create_graph=True, allow_unused=False)[0] 
                tensorlist.append(g)
            else:
                tensorlist.append(torch.zeros((3), dtype=torch.float64))
        intB = torch.stack(tensorlist, dim=0).flatten()
        Bs.append(intB)
    B = torch.stack(Bs)
    print(B)
#.reshape(tuple(shape)) # shape into final B tensor
        #for cart_idx in indices[i]:
            # Compute gradient of this internal coordinate wrt the cartesian coordinates of this atom
        #    g = torch.autograd.grad(intco, cartesians[cart_idx], create_graph=True, allow_unused=False)[0] 

    #test = intcos[0].q(geom)# cartesians[1], cartesians[2])
    #print(intcos[0].q(geom))
    #test = intcos[0].newq(cartesians[1], cartesians[2])
    #g = torch.autograd.grad(test, cartesians[1], create_graph=True, allow_unused=False)[0] 
    #print(g)
    

    # Compute first order B tensor. Avoid unnecessary derivatives.
    # For each internal coordinate, for each cartesian defining that internal coordinate, compute the gradient 
    #for i, intcoord in enumerate(coords):
    #    for j, intcoord in enumerate(coords):
        #g = torch.autograd.grad(intcoord, geom[indices[i]].flatten(), create_graph=True, allow_unused=False)[0] 
    #    g = torch.autograd.grad(intcoord, geom, create_graph=True, allow_unused=True)[0] 
    #    print(g)
    

    #print(intcos[0].__class__.__name__)
    #print(B[0].A)
    #print(B[0].B)
    #nint = B.shape[0]
    #ncart = 3 * geom.shape[0]
    #count = 0
    #shape = [nint, ncart]
    #while count < order:
    #    gradients = []
    #    for val in B.flatten():
    #        if count + 1 != order: 
    #            g = torch.autograd.grad(val, geom, create_graph=True)[0].reshape(ncart)
    #        else: # Save time: for last B tensor, don't create derivative graph.
    #            g = torch.autograd.grad(val, geom, retain_graph=True)[0].reshape(ncart)
    #        gradients.append(g)
    #    B = torch.stack(gradients).reshape(tuple(shape)) # shape into final B tensor
    #    shape.append(ncart)                              # update shape for next B tensor
    #    count += 1                              
#    return B



h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     0.950000000000 
H            0.000000000000     0.872305301500    -0.376275777700 
O            0.000000000000     0.000000000000     0.000000000000 
''')

h2o_autodiff = [ad_intcos.STRE(2,1),ad_intcos.STRE(2,0),ad_intcos.BEND(1,2,0)]
npgeom = np.array(h2o.geometry()) * bohr2ang
geom = torch.tensor(npgeom,requires_grad=False)

#B = fast_Btensor(h2o_autodiff, geom)
#print(coords)


def OLD(intcos, geom, order=1):
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

temp = torch.tensor(npgeom,requires_grad=True)

print(fast_B2(h2o_autodiff, geom))
#print("Fast B Tensor RESULT")
#print(fast_Btensor(h2o_autodiff, geom))
#print("OLD RESULT")
#print(OLD(h2o_autodiff, temp, order=1))
