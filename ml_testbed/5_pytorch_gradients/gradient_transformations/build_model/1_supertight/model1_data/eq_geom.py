import numpy as np
from helper import helper
from itertools import combinations
import psi4
import analytic_derivative as ad

#psi4.core.be_quiet()
h2o = psi4.geometry(
'''
H 0.0000000000 0.0000000000 0.9496306473
H 0.0000000000 0.8832771478 -0.3487403711
O 0.0000000000 0.0000000000 0.0000000000
''')
psi4.set_options({'scf_type': 'pk'})
e, wfn = psi4.frequencies('scf/6-31g', return_wfn = True)
geom = np.array(h2o.geometry())
grad = np.array(wfn.gradient())
hess = np.array(wfn.hessian())
dforces = ad.interatomic_forces(geom, grad) / 0.529177249
hess /= 0.529177249**2

interatomics = [helper.STRE(0,1), helper.STRE(0,2), helper.BEND(2,0,1)]
#TODO
#interatomics = ad.get_interatomics(geom) #TODO
#TODO
inthess = helper.convertHessianToInternals(hess, interatomics, geom)
print("Psi4 Gradient (Hartrees/Angs)\n", dforces)
print("Psi4 Hessian (Hartrees/Angs^2)\n",inthess)

# Compute frequencies
B = helper.Bmat(interatomics, geom) 
invmass = 1 / np.array([1.007825032230, 1.007825032230,1.007825032230, 1.007825032230,1.007825032230, 1.007825032230, 15.994914619570,15.994914619570,15.994914619570])
G = np.einsum('in,jn,n->ij', B, B, invmass)
# Hartrees, Angstrom, amu 
GF = G.dot(inthess)
print(GF)
#print(inthess * 4.3597482 * 1e-18 * (1 / 1.6605402e-27) * (1e10)**2 )
lamda, L = np.linalg.eig(GF)
# Converts Hartree to attojoule to Joule, then amu to kg, then Ang to Meters. All SI. Then convert to Hertz
#print(lamda)
#print(L)
lamda = lamda * 4.3597482 * 1e-18 * (1 / 1.6605402e-27) * (1e10)**2 
hertz = np.sqrt(lamda) / (2* np.pi)
hz2cm = 1 / 2.99792458e10
frequencies = hertz*hz2cm
print("Psi4 Frequencies\n",np.array(wfn.frequencies()))
print("Psi4/PyOPTKING Frequencies\n",frequencies)


from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (128, 128, 128), 'morse_transform': {'morse': False}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'std'}, 'scale_y': 'mm01', 'lr': 0.2}

X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')

# Track gradients: 1. Initial geometry 2. Morse transform 3. PIP transform 4. Scale transform
# Compute energy, and reverse scaling 
# point 161
inp1 = torch.tensor([1.570333028514,0.949630647150,0.949630647150], dtype=torch.float64, requires_grad=True)
# Careful! Degree reduce?
inp4 = torch.stack((inp1[0], inp1[1] + inp1[2], torch.sum(torch.pow(inp1[1:],2))**0.5), dim=0)
inp5 = (inp4 - torch.tensor(Xscaler.mean_, dtype=torch.float64)) / torch.tensor(Xscaler.scale_, dtype=torch.float64)
out1 = model(inp5)
out2 = (out1 - torch.tensor(yscaler.min_, dtype=torch.float64)) / torch.tensor(yscaler.scale_, dtype=torch.float64)
print(out2)

g = torch.autograd.grad(out2, inp1, create_graph=True)[0]
print("Pytorch Gradient:\n", g)
h1 = torch.autograd.grad(g[0], inp1, create_graph=True)[0]
h2 = torch.autograd.grad(g[1], inp1, create_graph=True)[0]
h3 = torch.autograd.grad(g[2], inp1, create_graph=True)[0]
F = torch.stack([h1,h2,h3])
print("Pytorch Hessian:\n", F)

GF = G.dot(F.detach().numpy())
lamda, L = np.linalg.eig(GF)
print(L)
# Converts Hartree to attojoule to Joule, then amu to kg, then Ang to Meters. All SI. Then convert to Hertz
lamda = lamda * 4.3597482 * 1e-18 * (1 / 1.6605402e-27) * (1e10)**2 
hertz = np.sqrt(lamda) / (2* np.pi)
hz2cm = 1 / 2.99792458e10
frequencies = hertz*hz2cm
print("NN Frequencies\n", frequencies)

# We must have 'create_graph=True' up until our last derivative we desire, which just needs retain_graph=True
# Cubic 'Hessian'
c1 = torch.autograd.grad(h1[0], inp1, retain_graph=True)[0]  
c2 = torch.autograd.grad(h1[1], inp1, retain_graph=True)[0]  
c3 = torch.autograd.grad(h1[2], inp1, retain_graph=True)[0]  
c4 = torch.autograd.grad(h2[0], inp1, retain_graph=True)[0]  
c5 = torch.autograd.grad(h2[1], inp1, retain_graph=True)[0]  
c6 = torch.autograd.grad(h2[2], inp1, retain_graph=True)[0]  
c7 = torch.autograd.grad(h3[0], inp1, retain_graph=True)[0]  
c8 = torch.autograd.grad(h3[1], inp1, retain_graph=True)[0]  
c9 = torch.autograd.grad(h3[2], inp1, retain_graph=True)[0]  
tmp1 = torch.stack([c1, c2, c3])
tmp2 = torch.stack([c4, c5, c6])
tmp3 = torch.stack([c7, c8, c9])
cubic = torch.stack([tmp1, tmp2, tmp3], dim=1)
#print("Pytorch Third Derivative Tensor:\n", cubic)
#
#
