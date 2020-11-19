from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch

from compute_energy import pes
print(pes([1.532010010387,0.900000000000,0.900000000000], cartesian=False))

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (16, 16), 'morse_transform': {'morse': True, 'morse_alpha': 1.0}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'std'}, 'scale_y': 'mm01', 'lr': 0.8}
X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')

# 255
#tmp1 = np.array([1.532010010387,0.900000000000,0.900000000000])
#tmp2 = nn.transform_new_X(tmp1, params, Xscaler)
#inp = torch.tensor(tmp2, dtype=torch.float64, requires_grad=True)
#
##with torch.no_grad():
#out = model(inp)

# Track gradients: 1. Initial geometry 2. Morse transform 3. PIP transform 4. Scale transform
# Compute energy, and reverse scaling 
inp1 = torch.tensor([1.532010010387,0.900000000000,0.900000000000], dtype=torch.float64, requires_grad=True)
inp2 = -inp1 / 1.0
inp3 = torch.exp(inp2)
#inp4 = torch.tensor([inp3[0], inp3[1] + inp3[2], torch.sum(torch.pow(inp3[1:],2))**0.5], dtype=torch.float64)
inp4 = torch.stack((inp3[0], inp3[1] + inp3[2], torch.sum(torch.pow(inp3[1:],2))**0.5), dim=0)
inp5 = (inp4 - torch.tensor(Xscaler.mean_, dtype=torch.float64)) / torch.tensor(Xscaler.scale_, dtype=torch.float64)
out1 = model(inp5)
out2 = (out1 - torch.tensor(yscaler.min_, dtype=torch.float64)) / torch.tensor(yscaler.scale_, dtype=torch.float64)
print(out2)
print('Error',out2 - -75.979540876231)

g = torch.autograd.grad(out2, inp1, create_graph=True)[0]
print("Pytorch Gradient:", g)



# TODO test
## The second derivatives of the derivative of the first input.
#h1 = torch.autograd.grad(g[0], x, create_graph=True)[0]
## The second derivatives of the derivative of the second input
#h2 = torch.autograd.grad(g[1], x, create_graph=True)[0]
## The Hessian, [d^2y/dx1^2, d^2y/dx1dx2]
##              [d^2y/dx2dx1, d^2y/dx2^2]
#h = torch.stack([h1,h2])
#print("Pytorch Hessian:", h)




