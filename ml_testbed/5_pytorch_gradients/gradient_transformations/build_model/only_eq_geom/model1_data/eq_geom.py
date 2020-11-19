import numpy as np
from helper import helper
from itertools import combinations
import psi4
import analytic_derivative as ad

#psi4.core.be_quiet()
#h2o = psi4.geometry(
#'''
#H            0.000000000000     0.000000000000     0.949676529800    
#H            0.000000000000     0.883402475500    -0.348547812400    
#O            0.000000000000     0.000000000000     0.000000000000    
#''')
#
#psi4.set_options({'scf_type': 'pk'})
#g, wfn = psi4.gradient('scf/6-31g', return_wfn = True)
## geometry in Bohr, grad in Hartrees/Bohr, Hessian in Hartrees/Bohr/Bohr
#geom = np.array(h2o.geometry())
#grad = np.array(g)
#h, wfn = psi4.hessian('scf/6-31g', return_wfn = True)
#hess = np.array(h)
#
## convert hessian to angstrom, use interatomic distances
#hess /= 0.529177249**2
#
##interatomics = [helper.STRE(0,1), helper.STRE(0,2), helper.BEND(2,0,1)]
#interatomics = ad.get_interatomics(geom)
#inthess = helper.convertHessianToInternals(hess, interatomics, geom)
#
## Compute frequencies
#B = helper.Bmat(interatomics, geom) 
#invmass = 1 / np.array([1.007825032230, 1.007825032230,1.007825032230, 1.007825032230,1.007825032230, 1.007825032230, 15.994914619570,15.994914619570,15.994914619570])
#G = np.einsum('in,jn,n->ij', B, B, invmass)
## Hartrees, Angstrom, amu 
#GF = G.dot(inthess)
#lamda, L = np.linalg.eig(GF)
## Converts Hartree to attojoule to Joule, then amu to kg, then Ang to Meters. All SI. Then convert to Hertz
#lamda = lamda * 4.3597482 * 1e-18 * (1 / 1.6605402e-27) * (1e10)**2 
#hertz = np.sqrt(lamda) / (2* np.pi)
#hz2cm = 1 / 2.99792458e10
#frequencies = hertz*hz2cm
#print(frequencies)


#dforces = ad.interatomic_forces(geom, grad)
#print("Psi4 Gradient (Hartrees/Angs)\n", dforces/0.529177249)
#print("Psi4 Hessian (Hartrees/Angs^2)\n",inthess)

from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (32, 32), 'morse_transform': {'morse': False}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'std'}, 'scale_y': 'std', 'lr': 0.8}
X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')

# Track gradients: 1. Initial geometry 2. Morse transform 3. PIP transform 4. Scale transform
# Compute energy, and reverse scaling 
# point 161
inp1 = torch.tensor([1.570282260121,0.949676529800,0.949676529800], dtype=torch.float64, requires_grad=True)
# Careful! Degree reduce?
inp4 = torch.stack((inp1[0], inp1[1] + inp1[2], torch.sum(torch.pow(inp1[1:],2))**0.5), dim=0)
inp5 = (inp4 - torch.tensor(Xscaler.mean_, dtype=torch.float64)) / torch.tensor(Xscaler.scale_, dtype=torch.float64)
out1 = model(inp5)
out2 = (out1 * torch.tensor(yscaler.scale_, dtype=torch.float64)) + torch.tensor(yscaler.mean_, dtype=torch.float64)

g = torch.autograd.grad(out2, inp1, create_graph=True)[0]
print("Pytorch Gradient:\n", g)
h1 = torch.autograd.grad(g[0], inp1, create_graph=True)[0]
h2 = torch.autograd.grad(g[1], inp1, create_graph=True)[0]
h3 = torch.autograd.grad(g[2], inp1, create_graph=True)[0]
F = torch.stack([h1,h2,h3])
print("Pytorch Hessian:\n", F)
#
#GF = G.dot(F.detach().numpy())
#lamda, L = np.linalg.eig(GF)
## Converts Hartree to attojoule to Joule, then amu to kg, then Ang to Meters. All SI. Then convert to Hertz
#lamda = lamda * 4.3597482 * 1e-18 * (1 / 1.6605402e-27) * (1e10)**2 
#hertz = np.sqrt(lamda) / (2* np.pi)
#hz2cm = 1 / 2.99792458e10
#frequencies = hertz*hz2cm
#print(frequencies)

