from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch

nn = NeuralNetwork('PES.dat', InputProcessor(''))
params = {'layers': (32,), 'morse_transform': {'morse': True, 'morse_alpha': 1.6}, 'pip': {'pip': False}, 'scale_X': {'activation': 'tanh', 'scale_X': 'std'}, 'scale_y': 'mm11', 'lr': 0.5}
X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')

## 298
#tmp1 = np.array([1.532010010387,0.900000000000,0.900000000000])
#tmp2 = nn.transform_new_X(tmp1, params, Xscaler)
#inp = torch.tensor(tmp2, dtype=torch.float64, requires_grad=True)
#
##with torch.no_grad():
#out = model(inp)

# Track gradients: 1. Initial geometry 2. Morse transform 3. Scale transform
# Compute energy, and reverse scaling 
inp1 = torch.tensor([1.532010010387,0.900000000000,0.900000000000], dtype=torch.float64, requires_grad=True)
inp2 = -inp1 / 1.6
inp3 = torch.exp(inp2)
inp4 = (inp3 - torch.tensor(Xscaler.mean_, dtype=torch.float64)) / torch.tensor(Xscaler.scale_, dtype=torch.float64)
out1 = model(inp4)
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




