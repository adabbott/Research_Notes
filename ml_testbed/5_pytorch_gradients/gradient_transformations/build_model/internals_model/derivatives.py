from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch

from compute_energy import pes
print(pes([0.900000000000,0.900000000000,116.666666666667], cartesian=False))

nn = NeuralNetwork('PES.dat', InputProcessor(''))
params = {'layers': (16, 16, 16, 16), 'morse_transform': {'morse': False}, 'pip': {'pip': False}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'}, 'scale_y': 'std', 'lr': 1.0}
X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')

#tmp1 = np.array([0.900000000000,0.900000000000,116.666666666667])
#-75.979540876231
#tmp2 = nn.transform_new_X(tmp1, params, Xscaler)
#inp = torch.tensor(tmp2, dtype=torch.float64, requires_grad=True)

#with torch.no_grad():
#out = model(inp)

# Track gradients: 1. Initial geometry 2. Morse transform 3. Scale transform
# Compute energy, and reverse scaling 
# 298 E = -75.979540876231
#inp = torch.tensor([0.900000000000,0.900000000000,116.666666666667], dtype=torch.float64, requires_grad=True)
#inp = inp * torch.tensor(Xscaler.scale_, dtype=torch.float64)
#inp = inp + torch.tensor(Xscaler.min_, dtype=torch.float64)
#out = model(inp)
#out = (out * torch.tensor(yscaler.scale_, dtype=torch.float64)) + torch.tensor(yscaler.mean_, dtype=torch.float64) 
inp1 = torch.tensor([0.900000000000,0.900000000000,116.666666666667], dtype=torch.float64, requires_grad=True)
inp2 = inp1 * torch.tensor(Xscaler.scale_, dtype=torch.float64)
inp3 = inp2 + torch.tensor(Xscaler.min_, dtype=torch.float64)
out1 = model(inp3)
out2 = (out1 * torch.tensor(yscaler.scale_, dtype=torch.float64)) + torch.tensor(yscaler.mean_, dtype=torch.float64) 

#print(nn.inverse_transform_new_y(out, yscaler))
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




