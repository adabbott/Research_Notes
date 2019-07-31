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
                #g = torch.squeeze(g)
                #for first in g:
                #    h = torch.autograd.grad(first, cartesians[j], create_graph=True, allow_unused=False)[0] 
                #    print('H', h)
                #print('G',g)
                tensorlist.append(g)
            else:
                tensorlist.append(torch.zeros((3), dtype=torch.float64))
        Brow = torch.stack(tensorlist, dim=0).flatten()
        Brows.append(Brow)
    B = torch.stack(Brows)

    print("ORIGINAL B TENSOR")
    print(B)
    print("RESHAPED B TENSOR")
    B = B.reshape((ncoords, natoms, 3))
    print(B)
    c = []
    for i,intco in enumerate(B):
        tensorlist1 = []
        for j,svec in enumerate(intco):
            tensorlist2 = []
            for val in svec:
                if j in indices[i]:
                #TODO still inefficiency, dont need derivatives wrt all k in cartesians, just ones which match j
                # smth like if k == j: else just apend tensor zeros(3)
                    for k in range(len(cartesians)):
                        h = torch.autograd.grad(val, cartesians[k], create_graph=True, allow_unused=False)[0] 
                        tensorlist2.append(h)
                else:
                    tensorlist2.append(torch.zeros((3), dtype=torch.float64))
                    tensorlist2.append(torch.zeros((3), dtype=torch.float64))
                    tensorlist2.append(torch.zeros((3), dtype=torch.float64))
            temp = torch.stack(tensorlist2)
            tensorlist1.append(temp)
        c.append(torch.stack(tensorlist1))
    print(torch.stack(c).reshape((3,9,9)))
    #print(torch.stack(tensorlist))
    return B

#h2o = psi4.geometry(
#'''
#H            0.000000000000     0.000000000000     0.950000000000 
#H            0.000000000000     0.872305301500    -0.376275777700 
#O            0.000000000000     0.000000000000     0.000000000000 
#''')
h2o = psi4.geometry(
'''
H            3.000000000000    -2.000000000000     0.950000000000 
H            2.000000000000     0.872305301500    -0.376275777700 
O            1.000000000000     1.000000000000    -2.000000000000 
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

B = fast_B2(h2o_autodiff, geom)
#print(fast_B2(h2o_autodiff, geom))
#print("Fast B Tensor RESULT")
#print(fast_Btensor(h2o_autodiff, geom))
#print("OLD RESULT")
print(OLD(h2o_autodiff, temp, order=2))
