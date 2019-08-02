from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch
from compute_energy import pes
import Btensors

nn = NeuralNetwork('model_data/PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (64, 64), 'morse_transform': {'morse': True, 'morse_alpha': 1.2000000000000002}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'}, 'scale_y': 'std', 'lr': 0.8}
X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model_data/model.pt')

def transform(interatomics):
    """ Takes Torch Tensor (requires_grad=True) of interatomic distances, manually transforms geometry to track gradients, computes energy
        Hard-coded based on hyperparameters """
    #inp1 = torch.tensor(interatomics, dtype=torch.float64, requires_grad=True)
    inp2 = -interatomics / 1.2
    inp3 = torch.exp(inp2)
    inp4 = torch.stack((inp3[0], inp3[1] + inp3[2], torch.sum(torch.pow(inp3[1:],2))), dim=0) # Careful! Degree reduce?
    inp5 = (inp4 * torch.tensor(Xscaler.scale_, dtype=torch.float64)) + torch.tensor(Xscaler.min_, dtype=torch.float64)
    out1 = model(inp5)
    out2 = (out1 * torch.tensor(yscaler.scale_, dtype=torch.float64)) + torch.tensor(yscaler.mean_, dtype=torch.float64)
    return out2

def cart2distances(cart):
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
g = torch.autograd.grad(E, distances, create_graph=True)[0]
h1 = torch.autograd.grad(g[0], distances, create_graph=True)[0]
h2 = torch.autograd.grad(g[1], distances, create_graph=True)[0]
h3 = torch.autograd.grad(g[2], distances, create_graph=True)[0]
F = torch.stack([h1,h2,h3])
print("Pytorch internal coordinate Hessian:\n", F)

computed_distances = cart2distances(cartesians)
print(computed_distances)
E = transform(computed_distances)
g = torch.autograd.grad(E, computed_distances, create_graph=True)[0]
h1 = torch.autograd.grad(g[0], computed_distances, create_graph=True)[0]
h2 = torch.autograd.grad(g[1], computed_distances, create_graph=True)[0]
h3 = torch.autograd.grad(g[2], computed_distances, create_graph=True)[0]
F = torch.stack([h1,h2,h3])
print("Pytorch internal coordinate Hessian:\n", F)

# Compute Cartesian hessian
computed_distances = cart2distances(cartesians)
E = transform(computed_distances)
g = torch.autograd.grad(E, cartesians, create_graph=True)[0]
print('Cartesian gradient', g)

#test = torch.autograd.grad(tmp_distances[0], cartesians, create_graph=True)[0]
#print(test)

#g = torch.autograd.grad(out2, inp1, create_graph=True)[0]
#print("Pytorch Gradient:\n", g)
##
## The second derivatives of the derivative of the first input.
#h1 = torch.autograd.grad(g[0], inp1, create_graph=True)[0]
## The second derivatives of the derivative of the second input
#h2 = torch.autograd.grad(g[1], inp1, create_graph=True)[0]
#h3 = torch.autograd.grad(g[2], inp1, create_graph=True)[0]
### The Hessian, [d^2y/dx1^2, d^2y/dx1dx2]
###              [d^2y/dx2dx1, d^2y/dx2^2]
#F = torch.stack([h1,h2,h3])
#print("Pytorch internal coordinate Hessian:\n", F)

# H H O
# Transform cartesians into interatomic distances

#G = torch.tensor([[ 1.05475559, 0.05223459, 0.82900085],
#                  [ 0.05223459, 1.05475559,-0.39300449],
#                  [ 0.82900085,-0.39300449, 1.98447145]], dtype=torch.float64)
#
#GF = torch.matmul(G, F * 4.35974)
#
#lamda = torch.symeig(GF)[0]
#lamda = lamda  * 1e-18 * (1 / 1.66054e-27) * (1e10)**2
#hertz = torch.sqrt(lamda) / (2* np.pi)
#hz2cm = 3.33565e-11
#frequencies = hertz*hz2cm
#print(frequencies)


## Manually tranform to track gradients: 1. Initial geometry 2. Morse transform 3. PIP transform 4. Scale transform
## Compute energy, and reverse scaling 
#inp1 = torch.tensor(eq_geom, dtype=torch.float64, requires_grad=True)
#inp2 = -inp1 / 1.2
#inp3 = torch.exp(inp2)
## Careful! Degree reduce?
#inp4 = torch.stack((inp3[0], inp3[1] + inp3[2], torch.sum(torch.pow(inp3[1:],2))), dim=0)
#inp5 = (inp4 * torch.tensor(Xscaler.scale_, dtype=torch.float64)) + torch.tensor(Xscaler.min_, dtype=torch.float64)
#out1 = model(inp5)
#out2 = (out1 * torch.tensor(yscaler.scale_, dtype=torch.float64)) + torch.tensor(yscaler.mean_, dtype=torch.float64)
#print("Manual transformation energy: ",out2)

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

