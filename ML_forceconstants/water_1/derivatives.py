import psi4
from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch
from compute_energy import pes
import Btensors
np.set_printoptions(threshold=5000, linewidth=200, precision=5)
torch.set_printoptions(threshold=5000, linewidth=200, precision=5)

# Load NN model
nn = NeuralNetwork('model_data/PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (64, 64), 'morse_transform': {'morse': True, 'morse_alpha': 1.2000000000000002}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'}, 'scale_y': 'std', 'lr': 0.8}
X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model_data/model.pt')

def transform(interatomics):
    """ Takes Torch Tensor (requires_grad=True) of interatomic distances, manually transforms geometry to track gradients, computes energy
        Hard-coded based on hyperparameters above. Returns: energy in units the NN model was trained on"""
    inp2 = -interatomics / 1.2
    inp3 = torch.exp(inp2)
    inp4 = torch.stack((inp3[0], inp3[1] + inp3[2], torch.sum(torch.pow(inp3[1:],2))), dim=0) # Careful! Degree reduce?
    inp5 = (inp4 * torch.tensor(Xscaler.scale_, dtype=torch.float64)) + torch.tensor(Xscaler.min_, dtype=torch.float64)
    out1 = model(inp5)
    energy = (out1 * torch.tensor(yscaler.scale_, dtype=torch.float64)) + torch.tensor(yscaler.mean_, dtype=torch.float64)
    return energy

def cart2distances(cart):
    """Transforms cartesian coordinate torch Tensor (requires_grad=True) into interatomic distances"""
    natom = cart.size()[0]
    ndistances = int((natom**2 - natom) / 2)
    distances = torch.zeros((ndistances), requires_grad=True, dtype=torch.float64)
    count = 0
    for i,atom1 in enumerate(cart):
        for j,atom2 in enumerate(cart):
            if j > i:
                distances[count] = torch.norm(cart[i]- cart[j])
                count += 1
    return distances

def differentiate_nn(energy, geometry, order=1):
    # The grad_tensor starts of as a single element, the energy. Then it becomes the gradient, hessian, cubic ... 
    # depeneding on value of 'order'
    grad_tensor = energy
    # The number of geometry parameters. Returned gradient tensor will have this size in all dimensions
    nparams = torch.numel(geometry)
    # The shape of first derivative. Will be adjusted at higher order
    shape = [nparams]
    count = 0
    while count < order:
        gradients = []
        for value in grad_tensor.flatten():
            g = torch.autograd.grad(value, geometry, create_graph=True)[0].reshape(nparams)
            gradients.append(g)
        grad_tensor = torch.stack(gradients).reshape(tuple(shape))
        shape.append(nparams)
        count += 1
    return grad_tensor
    
# this is optimized tightly, but apparently model was not trained on true equilibrium, just approximate. Next geom is true geom the model was trained on
#cartesians = torch.tensor([[0.0000000000,  1.0144292965, -0.0959637982],
#                           [0.0000000000, -0.0959637982,  1.0144292965],
#                           [0.0000000000,  0.0815344978,  0.0815344978]], dtype=torch.float64, requires_grad=True)
cartesians = torch.tensor([[ 0.0000000000,0.0000000000,0.9496765298],
                           [ 0.0000000000,0.8834024755,-0.3485478124],
                           [ 0.0000000000,0.0000000000,0.0000000000]], dtype=torch.float64, requires_grad=True)

# HH H1O H2O
eq_geom = [1.570282260121,0.949676529800,0.949676529800]
distances = torch.tensor(eq_geom, dtype=torch.float64, requires_grad=True)
print("Equilibrium geometry and energy: ", eq_geom, pes(eq_geom,cartesian=False)) # Test computation

# Compute internal coordinate Hessian wrt starting from both interatomic distances and cartesian coordinates, this works
E = transform(distances)
g =  differentiate_nn(E, distances, order=1)
h =  differentiate_nn(E, distances, order=2)
print("Pytorch internal coordinate Hessian:\n", h)

computed_distances = cart2distances(cartesians)
E = transform(computed_distances)
g =  differentiate_nn(E, computed_distances, order=1)
h2 =  differentiate_nn(E, computed_distances, order=2)
print("Pytorch internal coordinate Hessian:\n", h2)

hcart =  differentiate_nn(E, cartesians, order=2)
print("Pytorch Cartesian Hessian:\n", hcart)

psi4.core.be_quiet()
h2o = psi4.geometry(
'''
H 0.0000000000 0.0000000000 0.9496765298
H 0.0000000000 0.8834024755 -0.3485478124
O 0.0000000000 0.0000000000 0.0000000000
no_com
no_reorient
''')

psi4.set_options({'scf_type': 'pk'})
h, wfn = psi4.hessian('scf/6-31g', return_wfn = True)
hess = np.array(h)
# convert hessian to angstrom
hess /= 0.529177249**2
print(hess)



def get_interatomics(xyz):
    # Build autodiff-OptKing internal coordinates of interatomic distances
    natoms = xyz.shape[0]
    # Indices of unique interatomic distances, lower triangle row-wise order
    indices = np.asarray(np.tril_indices(natoms,-1)).transpose(1,0)
    interatomics = []
    for i in indices:
        idx1, idx2 = i
        interatomics.append(Btensors.ad_intcos.STRE(idx1, idx2))
    return interatomics

