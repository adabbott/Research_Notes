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
convert2 = np.sqrt(hartree2J/(amu2kg*amu2kg*ang2m*ang2m))/(c*2*np.pi)

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
    return freqs[:-6], cartL

def internal_freq(hess, B1, m):
    """
    Get harmonic frequencies with GF method. 
    hess : numpy array of hessian in internal coordinates Hartree/ang^2
    B1 : 1st order B tensor in terms of internal coordinates in Hessian 
    m : numpy array of masses in amu
    Returns 
    -------
    Frequencies in wavenumbers, normal coordinates
    """
    m = np.repeat(m,3)
    M = 1 / m
    #B = Btensors.ad_btensor.autodiff_Btensor(internals, geom, order=1)
    #B = B.detach().numpy()
    G = np.einsum('in,jn,n->ij', B1, B1, M)
    GF = G.dot(hess)
    intlamda, intL = np.linalg.eig(GF)
    idx = intlamda.argsort()[::-1]
    intlamda = intlamda[idx]
    freqs = np.sqrt(intlamda) * convert
    return freqs, intL

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

#m = np.array([1.007825032230, 1.007825032230, 15.994914619570])
#
#psi4freq, junk = cartesian_freq(psihess, m)
#print("Manually computed frequencies with Psi4 Hessian", psi4freq)
#
#nnfreq, junk = cartesian_freq(hcart.detach().numpy(), m)
#print("Manually computed frequencies with NN with Cartesian Hessian", nnfreq)
#
#nnfreq2, junk = internal_freq(hint.detach().numpy(), get_interatomics(3), cartesians, m)
#print("Manually computed frequencies with NN with interatomic distance coordinate Hessian", nnfreq2)
#
#internals = [Btensors.ad_intcos.STRE(0,2), Btensors.ad_intcos.STRE(1,2), Btensors.ad_intcos.BEND(0,2,1)]
#curvi_inthess = cartHess2intHess(hcart.detach().numpy(), internals, cartesians)
#nnfreq3, L = internal_freq(curvi_inthess, internals, cartesians, m)
#print("Manually computed frequencies with NN with curvilinear internal coordinate Hessian", nnfreq3)


# Cubic force constants in curvilinear coordinates
#M = np.sqrt(1 / np.repeat(m,3))
#B = Btensors.ad_btensor.autodiff_Btensor(internals, cartesians, order=1)
#B = B.detach().numpy()
# Get curvilinear internal eigenvectors L
#freq, L = internal_freq(curvi_inthess, internals, cartesians, m)

#inv_trans_L = np.linalg.inv(L).T # Maybe dont tranpose? does inverse flip dimension? 
#little_l = np.einsum('a,ia,ir->ar', M, B, inv_trans_L)
#L1_tensor = np.einsum('ia,a,ar->ir', B, M, little_l)
#quadratic = np.einsum('ij,ir,jr->r', curvi_inthess, L1_tensor, L1_tensor)
#print(quadratic)
#print(quadratic * convert)

def quadratic(hess, m, L, B):
    """internal coordinate hessian, Mass of each atom,  eigenvectors of hessian, B tensor"""
    M = np.sqrt(1 / np.repeat(m,3))
    inv_trans_L = np.linalg.inv(L).T # Maybe dont tranpose? does inverse flip dimension? 
    little_l = np.einsum('a,ia,ir->ar', M, B, inv_trans_L)
    L1_tensor = np.einsum('ia,a,ar->ir', B, M, little_l)
    # NORMALIZE 
    L1 = L1_tensor / np.linalg.norm(L1_tensor, axis=0)
    # MASS WEIGHT HESSIAN w/G THEN USE FORCE CONSTANT EQUATION
    G = np.einsum('in,jn,n,n->ij', B, B, M, M)
    GF = np.einsum('ij,jk->ik', G,hess)
    quadratic = np.einsum('ij,ir,js->rs', GF, L1, L1)
    print('quad',np.sqrt(quadratic) * convert)

def cubic(hess, cubic, m, L, B1, B2):
    M = np.sqrt(1 / np.repeat(m,3))
    inv_trans_L = np.linalg.inv(L).T # Maybe dont tranpose? does inverse flip dimension? 
    little_l = np.einsum('a,ia,ir->ar', M, B1, inv_trans_L)

    L1_tensor = np.einsum('ia,a,ar->ir', B1, M, little_l)
    L1 = L1_tensor / np.linalg.norm(L1_tensor, axis=0)

    L2_tensor = np.einsum('iab,a,ar,b,bs->irs', B2, M, little_l, M, little_l)
    invnorm = np.reciprocal(np.linalg.norm(L2_tensor, axis=1))
    # This normalization may be wrong.. but what else could it be?
    L2 = np.einsum('ijk,ik->ijk', L2_tensor, invnorm)

    # TODO: Mass weight Cubic tensor???
    G1 = np.einsum('in,jn,n,n->ij', B1, B1, M, M)
    mwhess = np.einsum('ij,jk->ik', G1,hess)
    G2 = np.einsum('inm, jnm, knm, n, m->ijk', B2, B2, B2, M, M)
    #G2 = np.einsum('in,jn,kn,n,n->ijk', B1, B1, B1, M, M)
    #mwcubic = np.einsum('ijk,jkl->ikl', G2, cubic)
    mwcubic = np.einsum('ijk,jkl->ikl', G2, cubic)
    #mwcubic = G2.dot(cubic)
    print(mwcubic.shape)
    #GF = np.einsum('ij,jk->ik', G,hess)

    term1 = np.einsum('ijk,ir,js,kt->rst', mwcubic, L1, L1, L1)
    term2 = np.einsum('ij, irs, jt->rst', mwhess, L2, L1)
    term3 = np.einsum('ij, irt, js->rst', mwhess, L2, L1)
    term4 = np.einsum('ij,ist,jr->rst', mwhess, L2, L1)
    fc_3 = term1 + term2 + term3 + term4

    print(fc_3)
    print(fc_3 * convert2)
    print(np.sqrt(fc_3) * convert2)

# Use Psi4 data first
#internals = [Btensors.ad_intcos.STRE(0,2), Btensors.ad_intcos.STRE(1,2), Btensors.ad_intcos.BEND(0,2,1)]
#psi4_inthess = cartHess2intHess(psihess, internals, cartesians)

#m = np.array([1.007825032230, 1.007825032230, 15.994914619570])
#f, L = internal_freq(psi4_inthess, internals, cartesians, m)
#print(f)

#B1 = Btensors.ad_btensor.autodiff_Btensor(internals, cartesians, order=1)
#B1 = B1.detach().numpy()
#quadratic(m, psi4_inthess, L, B1)
#cubic(psi4_inthess, L, B1, B2)



# Define curvilinear internal coordinates, 1st, 2nd order B tensors

# Genearte Cartesian Hessian with NN, convert to internal coordinates
#internal_coordinates =  cart2internals(cartesians, internals)
#print(internal_coordinates)
#hess_curvi = differentiate_nn(E, ad_internals, order=2)
#print(hess_curvi)
#hess_cart =  differentiate_nn(E, cartesians, order=2)
#hess_int = cartHess2intHess(hess_cart.detach().numpy(), internals, cartesians)
#m = np.array([1.007825032230, 1.007825032230, 15.994914619570])
#f, L = internal_freq(hess_int, internals, cartesians, m)
#print(f)
# Check: perform normal coordinate analysis, the L-tensor way
#quadratic(hess_int, m, L, B1)

#cubic_cart =  differentiate_nn(E, cartesians, order=3)
#quartic_cart =  differentiate_nn(E, computed_distances, order=4)

#hess_internals =  differentiate_nn(E, computed_distances, order=2)
#cubic_internals =  differentiate_nn(E, computed_distances, order=2)


# CHECK
#G = np.einsum('in,jn,n,n->ij', B, B, M, M)
#lamda = L.T.dot(G.dot(psi4_inthess)).dot(L)
#print(np.sqrt(lamda) * convert)



# Some accuracy as lost here, probably due to the gradient not being exactly zero.
#computed_distances = cart2distances(cartesians)
#E = transform(computed_distances)
#hess = differentiate_nn(E, cartesians, order=2)
#print('cart hess')
#print(hess)
#print(cartesian_freq(hess.detach().numpy(), m))
#print('converted cart hess to int hess')
#inthess = cartderiv2intderiv(hess.detach().numpy(), B1)
#print(inthess)
#print('backtransformed to cart hess')
#newcart = intderiv2cartderiv(inthess, B1)
#print(newcart)
#print(cartesian_freq(newcart, m))

internals = [Btensors.ad_intcos.STRE(0,2), Btensors.ad_intcos.STRE(1,2), Btensors.ad_intcos.BEND(0,2,1)]
interatomics = get_interatomics(3)
B1, B2 = Btensors.ad_btensor.fast_B(internals, cartesians_nograd)
B1, B2 = B1.detach().numpy(), B2.detach().numpy()
B1_idm, B2_idm = Btensors.ad_btensor.fast_B(interatomics, cartesians_nograd)
B1_idm, B2_idm = B1_idm.detach().numpy(), B2_idm.detach().numpy()
 
m = np.array([1.007825032230, 1.007825032230, 15.994914619570])
computed_distances = cart2distances(cartesians)
E = transform(computed_distances)
interatomic_hess = differentiate_nn(E, computed_distances, order=2).detach().numpy()
cart_hess = intderiv2cartderiv(interatomic_hess, B1_idm)
curvilinear_hess = cartderiv2intderiv(cart_hess, B1)

interatomic_cubic = differentiate_nn(E, computed_distances, order=3).detach().numpy()
cart_cubic = intderiv2cartderiv(interatomic_cubic, B1_idm)
curvilinear_cubic = cartderiv2intderiv(cart_cubic, B1)

f, L = internal_freq(curvilinear_hess, B1, m)
quadratic(curvilinear_hess, m, L, B1)
cubic(curvilinear_hess, curvilinear_cubic, m, L, B1, B2)




# Identical frequencies, good!
#idm_f, idm_L = internal_freq(interatomic_hess, B1_idm, m)
#cart_f, cart_L = cartesian_freq(cart_hess, m)
#f, L = internal_freq(curvilinear_hess, B1, m)


