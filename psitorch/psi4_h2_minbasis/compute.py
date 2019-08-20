import torch
import math
torch.set_printoptions(precision=10, linewidth=400)
torch.set_default_dtype(torch.float64)
ang2bohr = 1 / 0.52917720859

# Define coordinates in Angstroms as Torch tensors, turn on gradient tracking. Convert to Bohr. 
tmpgeom1 = [0.0000000000000000,0.0000000000000,0.00000000000000, 0.0000000000000000,0.0000000000000000,0.90000000000000]
tmpgeom2 = []
for i in tmpgeom1:
    tmpgeom2.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
geom = torch.stack(tmpgeom2).reshape(2,3) * ang2bohr
print(geom)
atom1 = geom[0]
atom2 = geom[1]

# Define s atomic orbital basis function exponent. Use 1.0 for contraction coefficents.
a1 = torch.tensor([0.3000000000], requires_grad=False)

def overlap_s(a1, a2, A, B):
    """
    A: vector of atom coordinates
    B: vecotr of atom coordinates
    """
    gamma = a1 + a2
    factor = (math.pi / gamma)**(3/2)
    AB = A - B
    exponent = torch.exp((-a1 * a2 * torch.dot(AB, AB)) / gamma)
    N = math.pi**(-3/4) * (1 / (a1 + a2)**(3/2))**(-0.5)
    S =  N**2 * factor * exponent
    return S

def kinetic_s(a1, a2, A, B):
    ''' Stolen from DiffiQult'''
    gamma = 1.0 / (a1 + a2)
    eta = a1 * a2 * gamma
    AB = A - B
    ab = -1.0 * torch.dot(AB, AB)
    N = math.pi**(-3/4) * (1 / (a1 + a2)**(3/2))**(-0.5)
    S00 = overlap_s(a1, a2, A, B)
    K00 = 3.0 * eta + 2.0 * eta * eta * ab
    return S00 * K00

def torchboys(nu, arg):
    '''Pytorch only has torch.erf, does not have torch.lowergamma. can only compute F0 boys function using the error function relation, the rest would have to
    be determined recursively? Deepmind has a Cephes wrapper for PyTorch which has it. http://deepmind.github.io/torch-cephes/   '''
    if arg < 1e-8:
        boys =  1 / (2 * nu + 1) - arg / (2 * nu + 3)
    else:
        boys = torch.erf(torch.sqrt(arg)) * math.sqrt(math.pi) / (2 * torch.sqrt(arg))
    return boys

def potential_s(a1, a2, A, B, atom, charge):
    g = a1 + a2
    eps = 1 / (4 * g)
    Rp = (a1 * A + a2 * B) / (a1 + a2)
    tmpc = torch.dot(A-B, A-B) * ((-a1 * a2) / (a1 + a2))
    c = torch.exp(tmpc)
    arg = g * torch.dot(Rp - atom, Rp - atom)
    F = torchboys(0, arg)
    N = math.pi**(-3/4) * (1 / (a1 + a2)**(3/2))**(-0.5)
    Vn = -charge * F *  N * N * c * 2 * math.pi / g
    return Vn

def eri_s(a1,a2,a3,a4,A,B,C,D):
    g1 = a1 + a2
    g2 = a3 + a4
    Rp = (a1 * A + a2 * B) / (a1 + a2)
    tmpc = torch.dot(A-B, A-B) * ((-a1 * a2) / (a1 + a2))
    c1 = torch.exp(tmpc)

    Rq = (a3 * C + a4 * D) / (a3 + a4)
    tmpc = torch.dot(C-D, C-D) * ((-a3 * a4) / (a3 + a4))
    c2 = torch.exp(tmpc)

    delta = 1 / (4 * g1) + 1 / (4 * g2)

    arg = torch.dot(Rp - Rq, Rp - Rq) / (4 * delta)
    F = torchboys(0, arg)
    N = math.pi**(-3/4) * (1 / (a1 + a2)**(3/2))**(-0.5)
    G = F * N * N * N * N * c1 * c2 * 2 * math.pi**2 / (g1 * g2) * torch.sqrt(math.pi / (g1 + g2))
    return G
    
def nuclear_repulsion(atom1, atom2):
    ''' warning : hard coded'''
    Za = 1
    Zb = 1
    return Za*Zb / torch.norm(atom1-atom2) 

#def factorial(tensr):
#    return torch.exp(torch.mvlgamma(tensr, 1))

# Compute nuclear repulsion energy, and integrals: overlap, kinetic, nuclear-electron potential, two-electron repulsion 
Enuc = nuclear_repulsion(atom1, atom2)
s1 = overlap_s(a1, a1, atom1, atom1)
s2 = overlap_s(a1, a1, atom1, atom2)
s3 = overlap_s(a1, a1, atom2, atom1)
s4 = overlap_s(a1, a1, atom2, atom2)
S = torch.stack([s1,s2,s2,s1]).reshape(2,2)
t1 = kinetic_s(a1, a1, atom1, atom1)
t2 = kinetic_s(a1, a1, atom1, atom2)
t3 = kinetic_s(a1, a1, atom2, atom1)
t4 = kinetic_s(a1, a1, atom2, atom2)
T = torch.stack([t1,t2,t3,t4]).reshape(2,2)
v1 = potential_s(a1, a1, atom1, atom1, atom1, 1) + potential_s(a1, a1, atom1, atom1, atom2, 1)
v2 = potential_s(a1, a1, atom1, atom2, atom1, 1) + potential_s(a1, a1, atom1, atom2, atom2, 1)
v3 = potential_s(a1, a1, atom1, atom2, atom1, 1) + potential_s(a1, a1, atom1, atom2, atom2, 1)
v4 = potential_s(a1, a1, atom1, atom1, atom1, 1) + potential_s(a1, a1, atom1, atom1, atom2, 1)
V = torch.stack([v1,v2,v3,v4]).reshape(2,2)
g1 = eri_s(a1, a1, a1, a1, atom1, atom1, atom1, atom1)
g2 = eri_s(a1, a1, a1, a1, atom1, atom1, atom1, atom2)
g3 = eri_s(a1, a1, a1, a1, atom1, atom1, atom2, atom2)
g4 = eri_s(a1, a1, a1, a1, atom1, atom2, atom1, atom2)
G = torch.stack([g1, g2, g2, g3, g2, g4, g4, g2, g2, g4, g4, g2, g3, g2, g2, g1]).reshape(2,2,2,2)

# HARTREE FOCK FROM SCRATCH
# For some reason forming orthogonalizer from this method gives NaN hessians 
eigval, eigvec = torch.symeig(S, eigenvectors=True)
#d12 = torch.sqrt(torch.diag(eigval))
d12 = torch.diag(torch.sqrt(eigval))
tmpA = torch.chain_matmul(eigvec, d12, eigvec) 
A = torch.inverse(tmpA)  # Orthogonalizer S^(-1/2)

## This implementation does not give NaN hessians, has stable backward() gradient function
#from sqrtm import sqrtm
#S_1_2 = sqrtm(S)
#A = torch.inverse(S_1_2)
ndocc = 1
H = T + V
# Guess 0 density matrix. (Can alternatively start with *converged* Density, then run through just 2 iteration cycles to
# 'connect' all arrays back to the geometry in a Pytorch computation graph. For example, the J,K, and F arrays must be constructed from a D that is
# determined from F)
D = torch.zeros((2,2))

# Do HF. 
for i in range(10):
    J = torch.einsum('pqrs,rs->pq', G, D)
    K = torch.einsum('prqs,rs->pq', G, D)
    F = H + J * 2 - K
    E = torch.einsum('pq,pq->', F + H, D) + Enuc
    Fp = torch.chain_matmul(A, F, A)
    e, C2 = torch.symeig(Fp, eigenvectors=True)             
    C = torch.matmul(A, C2)
    Cocc = C[:, :ndocc]                                                              
    D = torch.einsum('pi,qi->pq', Cocc, Cocc)

# Compute derivatives
grad = torch.autograd.grad(E, geom, create_graph=True)[0] 
h = []
for g in grad.flatten():
    h1 = torch.autograd.grad(g, geom, create_graph=True)[0]
    h.append(h1)
hess = torch.stack(h).reshape(6,6)


print(E)
print("Gradient")
print(grad)
print("Hessian")
print(hess)
print(torch.eig(hess))


