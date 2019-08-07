import psi4
from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import scipy 
import torch
import sympy
from compute_energy import pes
import Btensors
np.set_printoptions(threshold=5000, linewidth=200, precision=5, suppress=True)
torch.set_printoptions(threshold=5000, linewidth=200, precision=5)
bohr2ang = 0.529177249
hartree2J = 4.3597443e-18
hartree2cm = 219474.63
amu2kg = 1.6605389e-27
ang2m = 1e-10
h = 6.6260701510e-34   # Plancks in J s
hbar = 1.054571817e-34 # Reduced Plancks constant J s
hbarcm = (hbar / hartree2J) * hartree2cm
c = 29979245800.0 # speed of light in cm/s
cmeter = 299792458 # speed of light in cm/s
hz2cm = 3.33565e-11

convert = np.sqrt(hartree2J/(amu2kg*ang2m*ang2m))/(c*2*np.pi)

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

def cart2internals(cart, internals):
    values = Btensors.ad_intcos.qValues(internals, cart)   # Generate internal coordinates from cartesians
    return values
      
def differentiate_nn(energy, geometry, order=1):
    '''Very slow since nothing is held in memory and graphs need to keep being recreated. also causes memory leaks'''
    grad_tensor = energy
    nparams = torch.numel(geometry)
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


def new_general(geometry, cartesian=False, order=4):
    """Takes in list of interatomic distances or cartesian coordinates, computes energy and derivatives
       Keeps all intermediate derivative graphs in memory as their own object for optimal performance.
    """
    # Initialize variables, transform geometry, compute energy.
    tmpgeom = []
    for i in geometry:
        tmpgeom.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
    geom = torch.stack(tmpgeom)
    E = transform(geom)
    #if cartesian: #TODO handle torch.clone.detach(), make sure its corect
    #    geom = cart2distances(tmpgeom2)
    #    E = transform(geom)

    # Compute derivatives. Build up higher order tensors one dimension at a time.
    gradient = torch.autograd.grad(E, geom, create_graph=True)[0]
    h1, c1, q1, f1, s1 = [], [], [], [], []
    for d1 in gradient:
        h = torch.autograd.grad(d1, geom, create_graph=True)[0]
        h1.append(h) 
        c2, q2, f2, s2 = [], [], [], []
        for d2 in h: 
            c = torch.autograd.grad(d2, geom, create_graph=True)[0]
            c2.append(c)
            q3, f3, s3 = [], [], []
            for d3 in c: 
                q = torch.autograd.grad(d3, geom, create_graph=True)[0]
                q3.append(q)
                f4, s4 = [], []
                for d4 in q:
                    f = torch.autograd.grad(d4, geom, create_graph=True)[0]
                    f4.append(f)
                    s5 = []
                    for d5 in f:
                        s = torch.autograd.grad(d5, geom, create_graph=True)[0]
                        s5.append(s)
                    s4.append(torch.stack(s5))
                f3.append(torch.stack(f4))
                s3.append(torch.stack(s4))
            q2.append(torch.stack(q3))
            f2.append(torch.stack(f3))
            s2.append(torch.stack(s3))
        c1.append(torch.stack(c2))
        q1.append(torch.stack(q2))
        f1.append(torch.stack(f2))
        s1.append(torch.stack(s2))

    hessian = torch.stack(h1)
    cubic = torch.stack(c1)
    quartic = torch.stack(q1)
    quintic = torch.stack(f1)
    sextic = torch.stack(s1)
    return hessian, cubic, quartic, quintic, sextic
    


def new(geometry):
    tmpgeom = []
    for i in geometry:
        tmpgeom.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
    geom = torch.stack(tmpgeom)
    E = transform(geom)
    gradient = torch.autograd.grad(E, geom, create_graph=True)[0]
    print("gradient done")
    h1 = torch.autograd.grad(gradient[0], geom, create_graph=True)[0]  # Each h1 is three elements
    h2 = torch.autograd.grad(gradient[1], geom, create_graph=True)[0]
    h3 = torch.autograd.grad(gradient[2], geom, create_graph=True)[0]
    print("hessian done")
    c1 = torch.autograd.grad(h1[0], geom, create_graph=True)[0] # Each c1 is three elements
    c2 = torch.autograd.grad(h1[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h1[2], geom, create_graph=True)[0]
    c4 = torch.autograd.grad(h2[0], geom, create_graph=True)[0]
    c5 = torch.autograd.grad(h2[1], geom, create_graph=True)[0]
    c6 = torch.autograd.grad(h2[2], geom, create_graph=True)[0]
    c7 = torch.autograd.grad(h3[0], geom, create_graph=True)[0]
    c8 = torch.autograd.grad(h3[1], geom, create_graph=True)[0]
    c9 = torch.autograd.grad(h3[2], geom, create_graph=True)[0]
    print("cubic done")
    q1  = torch.autograd.grad(c1[0], geom, create_graph=True)[0]
    q2  = torch.autograd.grad(c1[1], geom, create_graph=True)[0]
    q3  = torch.autograd.grad(c1[2], geom, create_graph=True)[0]
    q4  = torch.autograd.grad(c2[0], geom, create_graph=True)[0]
    q5  = torch.autograd.grad(c2[1], geom, create_graph=True)[0]
    q6  = torch.autograd.grad(c2[2], geom, create_graph=True)[0]
    q7  = torch.autograd.grad(c3[0], geom, create_graph=True)[0]
    q8  = torch.autograd.grad(c3[1], geom, create_graph=True)[0]
    q9  = torch.autograd.grad(c3[2], geom, create_graph=True)[0]
    q10 = torch.autograd.grad(c4[0], geom, create_graph=True)[0]
    q11 = torch.autograd.grad(c4[1], geom, create_graph=True)[0]
    q12 = torch.autograd.grad(c4[2], geom, create_graph=True)[0]
    q13 = torch.autograd.grad(c5[0], geom, create_graph=True)[0]
    q14 = torch.autograd.grad(c5[1], geom, create_graph=True)[0]
    q15 = torch.autograd.grad(c5[2], geom, create_graph=True)[0]
    q16 = torch.autograd.grad(c6[0], geom, create_graph=True)[0]
    q17 = torch.autograd.grad(c6[1], geom, create_graph=True)[0]
    q18 = torch.autograd.grad(c6[2], geom, create_graph=True)[0]
    q19 = torch.autograd.grad(c7[0], geom, create_graph=True)[0]
    q20 = torch.autograd.grad(c7[1], geom, create_graph=True)[0]
    q21 = torch.autograd.grad(c7[2], geom, create_graph=True)[0]
    q22 = torch.autograd.grad(c8[0], geom, create_graph=True)[0]
    q23 = torch.autograd.grad(c8[1], geom, create_graph=True)[0]
    q24 = torch.autograd.grad(c8[2], geom, create_graph=True)[0]
    q25 = torch.autograd.grad(c9[0], geom, create_graph=True)[0]
    q26 = torch.autograd.grad(c9[1], geom, create_graph=True)[0]
    q27 = torch.autograd.grad(c9[2], geom, create_graph=True)[0]
    print("quartic done")
    f1 = torch.autograd.grad(q1[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q1[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q1[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q2[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q2[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q2[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q3[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q3[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q3[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q4[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q4[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q4[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q5[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q5[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q5[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q6[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q6[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q6[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q7[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q7[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q7[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q8[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q8[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q8[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q9[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q9[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q9[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q10[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q10[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q10[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q11[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q11[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q11[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q12[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q12[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q12[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q13[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q13[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q13[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q14[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q14[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q14[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q15[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q15[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q15[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q16[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q16[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q16[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q17[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q17[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q17[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q18[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q18[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q18[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q19[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q19[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q19[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q20[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q20[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q20[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q21[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q21[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q21[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q22[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q22[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q22[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q23[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q23[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q23[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q24[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q24[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q24[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q25[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q25[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q25[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q26[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q26[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q26[2], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q27[0], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q27[1], geom, create_graph=True)[0]
    f1 = torch.autograd.grad(q27[2], geom, create_graph=True)[0]
    print("quintic done")
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    s1 = torch.autograd.grad(f1[2], geom, create_graph=True)[0]
    print("sextic done")


def new_cart(geometry):
    tmpgeom = []
    for i in geometry:
        tmpgeom.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
    geom = torch.stack(tmpgeom)
    computed_distances = cart2distances(geom)
    E = transform(computed_distances)
    x_dim = torch.numel(geom)   
    gradient = torch.autograd.grad(E, geom, create_graph=True)[0]
    print("gradient done")
    h1 = torch.autograd.grad(gradient[0], geom, create_graph=True)[0]  # Each h1 is three elements
    h2 = torch.autograd.grad(gradient[1], geom, create_graph=True)[0]
    h3 = torch.autograd.grad(gradient[2], geom, create_graph=True)[0]
    h4 = torch.autograd.grad(gradient[3], geom, create_graph=True)[0]  # Each h1 is three elements
    h5 = torch.autograd.grad(gradient[4], geom, create_graph=True)[0]
    h6 = torch.autograd.grad(gradient[5], geom, create_graph=True)[0]
    h7 = torch.autograd.grad(gradient[6], geom, create_graph=True)[0]  # Each h1 is three elements
    h8 = torch.autograd.grad(gradient[7], geom, create_graph=True)[0]
    h9 = torch.autograd.grad(gradient[8], geom, create_graph=True)[0]
    print("hessian done")
    c1 = torch.autograd.grad(h1[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h1[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h1[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h1[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h1[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h1[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h1[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h1[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h1[8], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h2[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h2[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h2[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h2[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h2[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h2[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h2[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h2[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h2[8], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h3[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h3[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h3[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h3[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h3[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h3[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h3[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h3[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h3[8], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h4[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h4[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h4[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h4[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h4[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h4[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h4[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h4[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h4[8], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h5[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h5[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h5[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h5[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h5[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h5[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h5[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h5[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h5[8], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h6[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h6[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h6[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h6[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h6[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h6[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h6[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h6[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h6[8], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h7[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h7[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h7[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h7[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h7[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h7[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h7[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h7[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h7[8], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h8[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h8[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h8[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h8[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h8[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h8[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h8[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h8[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h8[8], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h9[0], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h9[1], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h9[2], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h9[3], geom, create_graph=True)[0] 
    c2 = torch.autograd.grad(h9[4], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h9[5], geom, create_graph=True)[0]
    c1 = torch.autograd.grad(h9[6], geom, create_graph=True)[0]
    c2 = torch.autograd.grad(h9[7], geom, create_graph=True)[0]
    c3 = torch.autograd.grad(h9[8], geom, create_graph=True)[0]
    print("cubic done")




def get_derivatives(energy, params):
    """
    Returns all derivatives involving E and members of params. d^nE/d(params). 
    For example, if params are x,y, and z, 
    this will return dE/dx, dE/dy, dE/dz, d2E/dx2, d2E/dy2, d2E/dz2, d2E/dxdy, d2E/dxdz, d2E/dydz, d3E/dxdydz
    """
    grad_tensor = energy
    nparams = torch.numel(params)
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
    return

def get_derivative(energy, params):
    """
    Returns d^nE/d(params) for n params. 
    """
    return

def get_interatomics(natoms):
    # Build autodiff-OptKing internal coordinates of interatomic distances
    #natoms = xyz.shape[0]
    # Indices of unique interatomic distances, lower triangle row-wise order
    indices = np.asarray(np.tril_indices(natoms,-1)).transpose(1,0)
    interatomics = []
    for i in indices:
        idx1, idx2 = i
        interatomics.append(Btensors.ad_intcos.STRE(idx1, idx2))
    return interatomics

def cartesian_freq(hess, m):
    """
    Do normal coordinate analysis in Cartesian coordinates
    
    Parameters
    ----------
    hess : 2d array
        The Hessian in Cartesian coordinates (not mass weighted)
    m : 1d array
        The masses of the atoms in amu. Has size = natoms.
    Returns
    -------
    freqs : Harmonic frequencies in wavenumbers (cm-1).
    LMW   : Normal coordinate eigenvectors from mass-weighted Hessian
    Lmw   : Normal coordinate eigenvectors with massweighting partially removed: Lmw = m^-1/2 * LMW

Get harmonic frequencies in wavenumbers from Cartesian Hessian in Hartree/ang^2. 
    m is a numpy array containing mass in amu for each atom, same order as Hessian. size:natom"""
    m = np.repeat(m,3)
    M = 1 / np.sqrt(m)
    diagM = np.diag(M)
    Hmw = diagM.dot(hess).dot(diagM)
    lamda, LMW = np.linalg.eig(Hmw)
    idx = lamda.argsort()
    lamda = lamda[idx]
    LMW = LMW[:,idx]
    freqs = np.sqrt(lamda) * convert
    Lmw = np.einsum('i,ir->ir', M,LMW)
    #Lmw = np.einsum('i,ir->ir', np.sqrt(m),LMW)
    return freqs[6:], LMW[:,6:], Lmw[:,6:]

def internal_freq(hess, B1, m):
    """
    Do normal coordinate analysis with GF method. 

    Parameters
    ----------
    hess : ndarray 
        NumPy array of Hessian in internal coordinates, Hartrees/Angstrom^2
    B1 : ndarray
        Numpy array 1st order B tensor corresponding to internal coordinate definitions in Hessian 
    m : ndarray
        Numpy array of masses of each atom in amu. Size is number of atoms. 
    Returns 
    -------
    Frequencies in wavenumbers, normalized normal coordinates, and mass-weighted (1/sqrt(amu)) normal coordinates 
    All values sorted to be in order of increasing energy of the frequencies.
    """
    m = np.repeat(m,3)
    M = 1 / m
    G = np.einsum('in,jn,n->ij', B1, B1, M)
    Gt = scipy.linalg.fractional_matrix_power(G, 0.5)
    Fp = Gt.dot(hess).dot(Gt)
    lamda, L = np.linalg.eig(Fp)
    mwL = Gt.dot(L)
    # Return Frequencies and 'L matrix' (mass weighted) in increasing order
    idx = lamda.argsort()
    lamda = lamda[idx]
    freqs = np.sqrt(lamda) * convert
    return freqs, L[:,idx], mwL[:,idx], 

def cartcubic2intcubic(cart_cubic, int_hess, B1, B2):
    G = np.dot(B1, B1.T)
    Ginv = np.linalg.inv(G)
    A = np.dot(Ginv, B1)
    tmp1 = np.einsum('ia,jb,kc,abc->ijk', A, A, A, cart_cubic)
    tmp2 = np.einsum('lmn,il,jm,kn->ijk', B2, int_hess, A, A)
    tmp3 = np.einsum('lmn,jl,im,kn->ijk', B2, int_hess, A, A)
    tmp4 = np.einsum('lmn,kl,im,jn->ijk', B2, int_hess, A, A)
    int_cubic = tmp1 - tmp2 - tmp3 - tmp4
    return int_cubic

def cartderiv2intderiv(derivative_tensor, B1):
    """
    Converts cartesian derivative tensor (gradient, Hessian, cubic, quartic ...) into internal coordinates
    Only valid at stationary points for Hessians and above.

    Parameters
    ----------   
    derivative_tensor : np.ndarray
        Tensor of nth derivative in Cartesian coordinates 
    B1 : np.ndarray
        B-matrix converting Cartesian coordinates to internal coordinates
    """
    G = np.dot(B1, B1.T)
    Ginv = np.linalg.inv(G)
    A = np.dot(Ginv, B1)
    dim = len(derivative_tensor.shape)
    if dim == 1:
        int_tensor = np.einsum('ia,a->i', A, derivative_tensor)
    elif dim == 2:
        int_tensor = np.einsum('ia,jb,ab->ij', A, A, derivative_tensor)
    elif dim == 3:
        int_tensor = np.einsum('ia,jb,kc,abc->ijk', A, A, A, derivative_tensor)
    elif dim == 4:
        int_tensor = np.einsum('ia,jb,kc,ld,abcd->ijkl', A, A, A, A, derivative_tensor)
    else:
        raise Exception("Too many dimensions. Add code to function")
    return int_tensor

def intcubic2cartcubic(intcubic, inthess, B1, B2):
    tmp1 = np.einsum('ia,jb,kc,ijk->abc', B1, B1, B1, intcubic)
    tmp2 = np.einsum('iab,jc,ij->abc', B2, B1, inthess)
    tmp3 = np.einsum('ica,jb,ij->abc', B2, B1, inthess)
    tmp4 = np.einsum('ibc,ja,ij->abc', B2, B1, inthess)
    cart_cubic = tmp1 + tmp2 + tmp3 + tmp4
    return cart_cubic
    
def intderiv2cartderiv(derivative_tensor, B1):
    """ 
    Converts cartesian derivative tensor (gradient, Hessian, cubic, quartic ...) into internal coordinates
    Only valid at stationary points for Hessians and above.

    Parameters
    ----------   
    derivative_tensor : np.ndarray
        Tensor of nth derivative in internal coordinates 
    B1 : np.ndarray
        B-matrix converting internal coordinates to Cartesian coordinates
    """
    dim = len(derivative_tensor.shape)
    if dim == 1:
        cart_tensor = np.einsum('ia,i->a', B1, derivative_tensor)
    elif dim == 2:
        cart_tensor = np.einsum('ia,jb,ij->ab', B1, B1, derivative_tensor)
    elif dim == 3:
        cart_tensor = np.einsum('ia,jb,kc,ijk->abc', B1, B1, B1, derivative_tensor)
    elif dim == 4:
        cart_tensor = np.einsum('ia,jb,kc,ld,ijkl->abcd', B1, B1, B1, B1, derivative_tensor)
    else:
        raise Exception("Too many dimensions. Add code to function to compute")
    return cart_tensor

def cubic_from_internals(hess, cubic, m, L, B1, B2):
    """
    Computes cubic normal coordinate force constants in cm-1 from internal coordinate derivatives.
    Parameters
    ----------
    hess : 2d array
        Internal coordinate Hessian in Hartree/Angstrom^2
    cubic : 3d array
        Internal coordinate third derivative tensor (analogue of Hessian) Hartree/Angstrom^3
    m : 1d array
        Masses of the atoms in amu (length is number of atoms)
    L : 2d array
        The 'L Matrix', mass-weighted (1/sqrt(amu)) normal coordinates wrt internals. 
        These are the eigenvectors from GF method weighted by the G matrix : G^(1/2)L
    B1 : 2d array
        1st order B tensor which relates internal coordinates to cartesian coordinates
    B2 : 3d array
        2nd order B tensor which relates internal coordinates to cartesian coordinates
    """
    M = np.sqrt(1 / np.repeat(m,3))
    inv_trans_L = np.linalg.inv(L).T 
    little_l = np.einsum('a,ia,ir->ar', M, B1, inv_trans_L)
    L1 = np.einsum('ia,a,ar->ir', B1, M, little_l)
    L2 = np.einsum('iab,a,ar,b,bs->irs', B2, M, little_l, M, little_l)
    term1 = np.einsum('ijk,ir,js,kt->rst', cubic, L1, L1, L1)
    term2 = np.einsum('ij, irs, jt->rst', hess, L2, L1)
    term3 = np.einsum('ij, irt, js->rst', hess, L2, L1)
    term4 = np.einsum('ij,ist,jr->rst', hess, L2, L1)
    nc_cubic = term1 + term2 + term3 + term4                             # UNITS: Hartree / Ang^3 amu^3/2
    frac = (hbar / (2*np.pi*cmeter))**(3/2) 
    nc_cubic *= (1 / (ang2m**3 * amu2kg**(3/2)))                         # UNITS: Hartree / m^3 kg^3/2
    nc_cubic *= frac                                                     # UNITS: Hartree / m^(3/2) 
    nc_cubic *= (1 / 100**(3/2))                                         # UNITS: Hartree cm-1 ^ (3/2)
    # Multiply each element by appropriate 3 harmonic frequencies
    omega = np.array([1737.31536,3987.9131,4144.72382])**(-1/2)
    nc_cubic = np.einsum('ijk,i,j,k->ijk', nc_cubic, omega, omega, omega)# UNITS: Hartree
    # add minus sign and convert to cm-1
    nc_cubic *= -hartree2cm                                              # UNITS: cm-1
    return nc_cubic

def cubic_from_cartesians(cubic, L):
    """ 
    Parameters
    ----------
    cubic : 3d array
        Cartesian coordinate third derivative tensor in Hartree/Angstrom^3
    L : 2d array
        The 'L Matrix', mass-weighted (1/sqrt(amu)) normal coordinates wrt cartesians. 
        These are the eigenvectors of the Mass-weighted cartesian Hessian, multiplied by 1/sqrt(amu). 
    """
    nc_cubic = np.einsum('ir,js,kt,ijk->rst', L, L, L, cubic)            # UNITS: Hartree/ Ang^3 amu^(3/2)
    frac = (hbar / (2*np.pi*cmeter))**(3/2) 
    nc_cubic *= (1 / (ang2m**3 * amu2kg**(3/2)))                         # UNITS: Hartree / m^3 kg^3/2
    nc_cubic *= frac                                                     # UNITS: Hartree / m^(3/2) 
    nc_cubic *= (1 / 100**(3/2))                                         # UNITS: Hartree cm-1 ^ (3/2)
    # Multiply each element by appropriate 3 harmonic frequencies
    omega = np.array([1737.31536,3987.9131,4144.72382])**(-1/2)
    nc_cubic = np.einsum('ijk,i,j,k->ijk', nc_cubic, omega, omega, omega)# UNITS: Hartree
    nc_cubic *= -hartree2cm                                              # UNITS: cm-1
    return nc_cubic


cartesians = torch.tensor([[ 0.0000000000,0.0000000000,0.9496765298],
                           [ 0.0000000000,0.8834024755,-0.3485478124],
                           [ 0.0000000000,0.0000000000,0.0000000000]], dtype=torch.float64, requires_grad=True)
cartesians_nograd = torch.tensor([[ 0.0000000000,0.0000000000,0.9496765298],
                           [ 0.0000000000,0.8834024755,-0.3485478124],
                           [ 0.0000000000,0.0000000000,0.0000000000]], dtype=torch.float64, requires_grad=False)
#
# HH H1O H2O
eq_geom = [1.570282260121,0.949676529800,0.949676529800]
distances = torch.tensor(eq_geom, dtype=torch.float64, requires_grad=True)
print("Equilibrium geometry and energy: ", eq_geom, pes(eq_geom,cartesian=False)) # Test computation

# Compute internal coordinate Hessian wrt starting from both interatomic distances and cartesian coordinates, this works
#computed_distances = cart2distances(cartesians)
#E = transform(computed_distances)
#hint =  differentiate_nn(E, computed_distances, order=2)
#hcart =  differentiate_nn(E, cartesians, order=2)

# Psi4 hessian and frequencies
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
freq, wfn = psi4.frequencies('scf/6-31g', return_wfn = True)
print("Psi4 analytic frequencies")
print(np.sort(np.array(wfn.frequencies()))[::-1])
psihess = np.array(wfn.hessian())
psihess /= 0.529177249**2


def quadratic(hess, m, L, B):
    """internal coordinate hessian, Mass of each atom, G 1/2 dotted with eigenvectors of hessian, and B tensor"""
    M = np.sqrt(1 / np.repeat(m,3))
    inv_trans_L = np.linalg.inv(L).T 
    little_l = np.einsum('a,ia,ir->ar', M, B, inv_trans_L)
    L1_tensor = np.einsum('ia,a,ar->ir', B, M, little_l)
    quadratic = np.einsum('ij,ir,js->rs', hess, L1_tensor, L1_tensor)
    print('quad',np.diagonal(np.sqrt(quadratic) * convert))
    return np.diagonal(np.sqrt(quadratic) * convert)
    

internals = [Btensors.ad_intcos.STRE(0,2), Btensors.ad_intcos.STRE(1,2), Btensors.ad_intcos.BEND(0,2,1)]
interatomics = get_interatomics(3)
B1, B2 = Btensors.ad_btensor.fast_B(internals, cartesians_nograd)
B1, B2 = B1.detach().numpy(), B2.detach().numpy()
B1_idm, B2_idm = Btensors.ad_btensor.fast_B(interatomics, cartesians_nograd)
B1_idm, B2_idm = B1_idm.detach().numpy(), B2_idm.detach().numpy()
 
m = np.array([1.007825032230, 1.007825032230, 15.994914619570])
distances = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
computed_distances = cart2distances(cartesians)
E = transform(computed_distances)

# TESTBED for faster derivatives
# Create single parameter tensors and combine them together for convenience
eq_geom = [1.570282260121,0.949676529800,0.949676529800]
tmpdistances = []
for i in eq_geom:
    tmpdistances.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
distances = torch.stack(tmpdistances)
E = transform(distances)

#cart_eq_geom = [ 0.0000000000,0.0000000000,0.9496765298, 0.0000000000,0.8834024755,-0.3485478124, 0.0000000000,0.0000000000,0.0000000000]
#tmp = []
#for i in cart_eq_geom:
#    tmp.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
#full_cart = torch.stack(tmp)
#print(full_cart)
#new_cart(full_cart)

h, c, q, f, s = new_general(eq_geom)
test = differentiate_nn(E, distances, order=2)
print(torch.allclose(test, h))
test = differentiate_nn(E, distances, order=3)
print(torch.allclose(test, c))
test = differentiate_nn(E, distances, order=4)
print(torch.allclose(test, q))
test = differentiate_nn(E, distances, order=5)
print(torch.allclose(test, f))


#test = differentiate_nn(E, distances, order=2)
#print(test)
#test = differentiate_nn(E, distances, order=3)
#print(test)
##new(eq_geom)
#
#test = differentiate_nn(E, distances, order=4)
#print(test)
#test = differentiate_nn(E, distances, order=5)
#print(test)

## Force constant stuff
#interatomic_hess = differentiate_nn(E, computed_distances, order=2).detach().numpy()
#cart_hess = intderiv2cartderiv(interatomic_hess, B1_idm)
#curvilinear_hess = cartderiv2intderiv(cart_hess, B1)
#psi4_curvihess = cartderiv2intderiv(psihess, B1)
#
#interatomic_cubic = differentiate_nn(E, computed_distances, order=3).detach().numpy()
#cart_cubic = intcubic2cartcubic(interatomic_cubic, interatomic_hess, B1_idm, B2_idm)
#int_cubic = cartcubic2intcubic(cart_cubic, interatomic_hess, B1_idm, B2_idm)
#curvilinear_cubic = cartcubic2intcubic(cart_cubic, curvilinear_hess, B1, B2)
#
#f, L, mwL = internal_freq(curvilinear_hess, B1, m)
#b = quadratic(curvilinear_hess, m, mwL, B1)
#cubic_from_internals(curvilinear_hess, curvilinear_cubic, m, mwL, B1, B2)
#
#f, L, mwL = cartesian_freq(cart_hess, m)
#cubic_from_cartesians(cart_cubic, mwL)



