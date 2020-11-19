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
                tensorlist.append(g)
            else:
                tensorlist.append(torch.zeros((3), dtype=torch.float64))
        Brow = torch.stack(tensorlist, dim=0).flatten()
        Brows.append(Brow)
    B = torch.stack(Brows)

    B = B.reshape((ncoords, natoms, 3))
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
    B = torch.stack(c).reshape((ncoords,ncart,ncart))
    return B

def FASTER_B2(intcos, geom):
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
    B = torch.stack(Brows)
    print(B)
    #print(B.reshape((ncoords,natoms,3)))
    #print(B.reshape((ncoords,natoms,3)).nonzero())
    tmp = B.reshape((ncoords,natoms,3))
    print("Zero values",tmp)
    print((tmp == 0).nonzero())

    all_values = (B == B).nonzero()
    nontrivial_derivatives = []

    # Filter through all addresses of values in B tensor, find out if derivative is needed based on if s-vector is all zero.
    for Bidx in all_values:
        print(tmp[list(Bidx)])
        k = tmp[list(Bidx)[:1]]
        for svec in k:
#if any([(a == c_).all() for c_ in c]):V
            print(any([(val == 0).all() for val in svec ])) #svec.nonzero().all())



    # TODO improvement: rather than iterating over ALL values, you only need to differentiate those that have one or more nonzero values in s vector
    # Distinguish between trivial and nontrivial derivatives. If nontrivial, then check which cartesian coordinates to differentiate wrt to.
    # Differentiate wrt 3 cartesian coordinates at a time, or just return three 0.0 derivatives
    tlist = []
    for a in all_values:
        for j in range(natoms):
            if j in indices[a[0]]:
                g = torch.autograd.grad(B[list(a)], cartesians[j], create_graph=True, allow_unused=False)[0]
                tlist.append(g)
            else:
                g = torch.zeros((3), dtype=torch.float64)
                tlist.append(g)
    B2 = torch.stack(tlist).reshape(ncoords,ncart,ncart)

    ## Try 3rd order?
    #all_values = (B2 == B2).nonzero()
    #tlist = []
    #count = 0
    #for a in all_values:
    #    for j in range(natoms):
    #        if j in indices[a[0]]:
    #            g = torch.autograd.grad(B2[list(a)], cartesians[j], create_graph=True, allow_unused=False)[0]
    #            tlist.append(g)
    #            count +=1
    #        else:
    #            g = torch.zeros((3), dtype=torch.float64)
    #            tlist.append(g)
    #            count +=1
    #        print(count)
    #B3 = torch.stack(tlist).reshape(ncoords,ncart,ncart,ncart)
    #print(B3)

    return B2
    



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



def test(psigeom, autodiff):
    """A Psi4 molecule.geometry(), list of internal coordinates"""
    npgeom = np.array(psigeom) * bohr2ang
    geom = torch.tensor(npgeom,requires_grad=False)
    temp = torch.tensor(npgeom,requires_grad=True)
    B = fast_Btensor(autodiff, geom)
    oldB = OLD(autodiff, temp, order=1)
    print("1st order B tensors match")
    print(torch.allclose(B.detach(),oldB))
    oldB2 = OLD(autodiff, temp, order=2)
    B2 = fast_B2(autodiff, geom)
    print("2nd order B tensors match")
    print(torch.allclose(B2.detach(),oldB2))


# TESTS
h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     0.950000000000 
H            0.000000000000     0.872305301500    -0.376275777700 
O            0.000000000000     0.000000000000     0.000000000000 
''')
h2o_autodiff = [ad_intcos.STRE(2,1),ad_intcos.STRE(2,0),ad_intcos.BEND(1,2,0)]


h2co = psi4.geometry(
'''
C            0.000000000000     0.000000000000    -0.607835855018 
O            0.000000000000     0.000000000000     0.608048883261 
H            0.000000000000     0.942350938995    -1.206389817026 
H            0.000000000000    -0.942350938995    -1.206389817026 
''')
h2co_autodiff = [ad_intcos.STRE(0,1),ad_intcos.STRE(0,2),ad_intcos.BEND(2,0,1),ad_intcos.STRE(0,3),ad_intcos.BEND(3,0,1),ad_intcos.TORS(3,0,1,2)]

allene = psi4.geometry(
"""
H  0.0  -0.92   -1.8
H  0.0   0.92   -1.8
C  0.0   0.00   -1.3
C  0.0   0.00    0.0
C  0.0   0.00    1.3
H  0.92  0.00    1.8
H -0.92  0.00    1.8
""")
allene_autodiff = [ad_intcos.STRE(0, 2),ad_intcos.STRE(1, 2),ad_intcos.STRE(2, 3),ad_intcos.STRE(3, 4),ad_intcos.STRE(4, 5),ad_intcos.STRE(4, 6),ad_intcos.BEND(0, 2, 1),ad_intcos.BEND(0, 2, 3),ad_intcos.BEND(1, 2, 3),ad_intcos.BEND(2, 3, 4),ad_intcos.BEND(2, 3, 4),ad_intcos.BEND(3, 4, 5),ad_intcos.BEND(3, 4, 6),ad_intcos.BEND(5, 4, 6),ad_intcos.TORS(0, 2, 4, 5),ad_intcos.TORS(0, 2, 4, 6),ad_intcos.TORS(1, 2, 4, 5),ad_intcos.TORS(1, 2, 4, 6)]


big = psi4.geometry( 
'''
 C  0.00000000 0.00000000 0.00000000
 Cl 0.19771002 -0.99671665 -1.43703398
 C  1.06037767 1.11678073 0.00000000
 C  2.55772698 0.75685710 0.00000000
 H  3.15117939 1.67114056 0.00000000
 H  2.79090687 0.17233980 0.88998127
 H  2.79090687 0.17233980 -0.88998127
 H  0.75109254 2.16198057 0.00000000
 H -0.99541786 0.44412079 0.00000000
 H  0.12244541 -0.61728474 0.88998127
'''
)

big_autodiff = [ad_intcos.STRE(0, 1),ad_intcos.STRE(0, 2),ad_intcos.STRE(0, 8),ad_intcos.STRE(0, 9),ad_intcos.STRE(2, 3),ad_intcos.STRE(2, 7),ad_intcos.STRE(3, 4),ad_intcos.STRE(3, 5),ad_intcos.STRE(3, 6),ad_intcos.BEND(0, 2, 3),ad_intcos.BEND(0, 2, 7),ad_intcos.BEND(1, 0, 2),ad_intcos.BEND(1, 0, 8),ad_intcos.BEND(1, 0, 9),ad_intcos.BEND(2, 0, 8),ad_intcos.BEND(2, 0, 9),ad_intcos.BEND(2, 3, 4),ad_intcos.BEND(2, 3, 5),ad_intcos.BEND(2, 3, 6),ad_intcos.BEND(3, 2, 7),ad_intcos.BEND(4, 3, 5),ad_intcos.BEND(4, 3, 6),ad_intcos.BEND(5, 3, 6),ad_intcos.BEND(8, 0, 9),ad_intcos.TORS(0, 2, 3, 4),ad_intcos.TORS(0, 2, 3, 5),ad_intcos.TORS(0, 2, 3, 6),ad_intcos.TORS(1, 0, 2, 3),ad_intcos.TORS(1, 0, 2, 7),ad_intcos.TORS(3, 2, 0, 8),ad_intcos.TORS(3, 2, 0, 9),ad_intcos.TORS(4, 3, 2, 7),ad_intcos.TORS(5, 3, 2, 7),ad_intcos.TORS(6, 3, 2, 7),ad_intcos.TORS(7, 2, 0, 8),ad_intcos.TORS(7, 2, 0, 9)]



#test(h2o.geometry(), h2o_autodiff)
#test(h2co.geometry(), h2co_autodiff)
#test(allene.geometry(), allene_autodiff)
npgeom = np.array(h2o.geometry()) * bohr2ang
geom = torch.tensor(npgeom,requires_grad=False)
B2fast = FASTER_B2(h2o_autodiff, geom)
print(B2fast)
#B2 = fast_B2(h2o_autodiff, geom)
#print(B2)
#print(torch.allclose(B2fast,B2))

#npgeom = np.array(allene.geometry()) * bohr2ang
#geom = torch.tensor(npgeom,requires_grad=False)
#B2 = fast_B2(allene_autodiff, geom)
#print(torch.nonzero(B2).size())

#print(fast_Btensor(allene_autodiff, geom))
#print(fast_B2(allene_autodiff, geom))
#npgeom = np.array(big.geometry()) * bohr2ang
#geom = torch.tensor(npgeom,requires_grad=False)
#print(fast_Btensor(big_autodiff, geom))
#print(fast_B2(big_autodiff, geom))



