import psi4
from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch
from compute_energy import pes
import Btensors
np.set_printoptions(threshold=5000, linewidth=200, precision=5)
torch.set_printoptions(threshold=5000, linewidth=200, precision=5)
bohr2ang = 0.529177249
hartree2J = 4.3597443e-18
amu2kg = 1.6605389e-27
ang2m = 1e-10
c = 29979245800.0 # speed of light in cm/s
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

def differentiate_nn(energy, geometry, order=1):
    # The grad_tensor starts of as a single element, the energy. Then it becomes the gradient, hessian, cubic ... 
    # depending on value of 'order'
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
    """Get harmonic frequencies in wavenumbers from Cartesian Hessian in Hartree/ang^2. 
    m is a numpy array containing mass in amu for each atom, same order as Hessian. size:natom"""
    m = np.repeat(m,3)
    M = 1 / np.sqrt(m)
    M = np.diag(M)
    Hmw = M.dot(hess).dot(M)
    cartlamda, cartL = np.linalg.eig(Hmw)
    idx = cartlamda.argsort()[::-1]
    cartlamda = cartlamda[idx]
    freqs = np.sqrt(cartlamda) * convert
    print(freqs[:-6])
    return freqs[:-6]

def internal_freq(hess, internals, geom, m):
    """
    Get harmonic frequencies with GF method. 
    hess : numpy array of hessian in internal coordinates Hartree/ang^2
    internals : list of internal coordinate definitions from ad_intcos, STRE, BEND, TORS objects same order as in Hessian
    geom : Torch Tensor of cartesian coordinates in angstrom, requires_grad=True
    m : numpy array of masses in amu
    Returns 
    -------
    Frequencies in wavenumbers
    """
    m = np.repeat(m,3)
    M = 1 / m
    B = Btensors.ad_btensor.autodiff_Btensor(internals, geom, order=1)
    B = B.detach().numpy()
    G = np.einsum('in,jn,n->ij', B, B, M)
    GF = G.dot(hess)
    intlamda, intL = np.linalg.eig(GF)
    idx = intlamda.argsort()[::-1]
    intlamda = intlamda[idx]
    freqs = np.sqrt(intlamda) * convert
    print(freqs)
    return freqs

def cartHess2intHess(H, intcos, geom):
    """
    Converts Cartesian hessian to internal coordinates Hessian.
    H is a numpy array of the hessian
    intcos is a list of internal coordinate definitions from Btensors.ad_intcos
    geom is a torch.Tensor of cartesian coordinates, requires_grad=True
    """
    B = Btensors.ad_btensor.autodiff_Btensor(intcos, geom, order=1)
    B = B.detach().numpy()
    G = np.dot(B, B.T)
    Ginv = np.linalg.inv(G)
    Atranspose = np.dot(Ginv, B)
    Hq = np.dot(Atranspose, np.dot(H, Atranspose.T))
    return Hq




# this is optimized tightly, but apparently model was not trained on true equilibrium, just approximate. Next geom is true geom the model was trained on
#cartesians = torch.tensor([[0.0000000000,  1.0144292965, -0.0959637982],
#                           [0.0000000000, -0.0959637982,  1.0144292965],
#                           [0.0000000000,  0.0815344978,  0.0815344978]], dtype=torch.float64, requires_grad=True)
cartesians = torch.tensor([[ 0.0000000000,0.0000000000,0.9496765298],
                           [ 0.0000000000,0.8834024755,-0.3485478124],
                           [ 0.0000000000,0.0000000000,0.0000000000]], dtype=torch.float64, requires_grad=True)
#
# HH H1O H2O
eq_geom = [1.570282260121,0.949676529800,0.949676529800]
distances = torch.tensor(eq_geom, dtype=torch.float64, requires_grad=True)
print("Equilibrium geometry and energy: ", eq_geom, pes(eq_geom,cartesian=False)) # Test computation

# Compute internal coordinate Hessian wrt starting from both interatomic distances and cartesian coordinates, this works
computed_distances = cart2distances(cartesians)
E = transform(computed_distances)
hint =  differentiate_nn(E, computed_distances, order=2)
hcart =  differentiate_nn(E, cartesians, order=2)

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

m = np.array([1.007825032230, 1.007825032230, 15.994914619570])
print("Manually computed frequencies with Psi4 Hessian")
psi4freq = cartesian_freq(psihess, m)
print("Manually computed frequencies with NN with Cartesian Hessian")
nnfreq = cartesian_freq(hcart.detach().numpy(), m)
print("Manually computed frequencies with NN with interatomic distance coordinate Hessian")
nnfreq2 = internal_freq(hint.detach().numpy(), get_interatomics(3), cartesians, m)
print("Manually computed frequencies with NN with curvilinear internal coordinate Hessian")
internals = [Btensors.ad_intcos.STRE(0,2), Btensors.ad_intcos.STRE(1,2), Btensors.ad_intcos.BEND(0,2,1)]
curvi_inthess = cartHess2intHess(hcart.detach().numpy(), internals, cartesians)
nnfreq2 = internal_freq(curvi_inthess, internals, cartesians, m)



