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
    count2 = 0
    while count < order:
        gradients = []
        for val in B.flatten():
            if count + 1 != order: 
                g = torch.autograd.grad(val, geom, create_graph=True)[0].reshape(ncart)
                count2 += 1
                print("Derivative {}".format(count2))
            else: # Save time: for last B tensor, don't create derivative graph.
                g = torch.autograd.grad(val, geom, retain_graph=True)[0].reshape(ncart)
                count2 += 1
                print("Derivative {}".format(count2))
            gradients.append(g)
        B = torch.stack(gradients).reshape(tuple(shape)) # shape into final B tensor
        shape.append(ncart)                              # update shape for next B tensor
        count += 1                              
    return B

