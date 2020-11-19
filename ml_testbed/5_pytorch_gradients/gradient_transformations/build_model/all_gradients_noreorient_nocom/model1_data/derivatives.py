from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch

from compute_energy import pes
print(pes([1.587426841685,0.950000000000,0.950000000000], cartesian=False))

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (64, 64), 'morse_transform': {'morse': True, 'morse_alpha': 1.2000000000000002}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'}, 'scale_y': 'std', 'lr': 0.8}
X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')

# Track gradients: 1. Initial geometry 2. Morse transform 3. PIP transform 4. Scale transform
# Compute energy, and reverse scaling 
# point 161
inp1 = torch.tensor([1.587426841685,0.950000000000,0.950000000000], dtype=torch.float64, requires_grad=True)
inp2 = -inp1 / 1.2
inp3 = torch.exp(inp2)
# Careful! Degree reduce?
inp4 = torch.stack((inp3[0], inp3[1] + inp3[2], torch.sum(torch.pow(inp3[1:],2))), dim=0)
inp5 = (inp4 * torch.tensor(Xscaler.scale_, dtype=torch.float64)) + torch.tensor(Xscaler.min_, dtype=torch.float64)
out1 = model(inp5)
out2 = (out1 * torch.tensor(yscaler.scale_, dtype=torch.float64)) + torch.tensor(yscaler.mean_, dtype=torch.float64)
print(out2)

g = torch.autograd.grad(out2, inp1, create_graph=True)[0]
print("Pytorch Gradient:\n", g)

# The second derivatives of the derivative of the first input.
h1 = torch.autograd.grad(g[0], inp1, create_graph=True)[0]
# The second derivatives of the derivative of the second input
h2 = torch.autograd.grad(g[1], inp1, create_graph=True)[0]
h3 = torch.autograd.grad(g[2], inp1, create_graph=True)[0]
## The Hessian, [d^2y/dx1^2, d^2y/dx1dx2]
##              [d^2y/dx2dx1, d^2y/dx2^2]
F = torch.stack([h1,h2,h3])
print("Pytorch Hessian:\n", F)
G = torch.tensor([[ 1.05475559, 0.05223459, 0.82900085],
                  [ 0.05223459, 1.05475559,-0.39300449],
                  [ 0.82900085,-0.39300449, 1.98447145]], dtype=torch.float64)

GF = torch.matmul(G, F * 4.35974)

lamda = torch.symeig(GF)[0]
lamda = lamda  * 1e-18 * (1 / 1.66054e-27) * (1e10)**2
hertz = torch.sqrt(lamda) / (2* np.pi)
hz2cm = 3.33565e-11
frequencies = hertz*hz2cm
print(frequencies)




