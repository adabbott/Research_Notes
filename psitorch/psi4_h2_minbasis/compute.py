import torch
import math
torch.set_printoptions(precision=12)
torch.set_default_dtype(torch.float64)

ang2bohr = 1 / 0.52917720859
#def factorial(tensr):
#    return torch.exp(torch.mvlgamma(tensr, 1))

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
    '''Pytorch can only compute F0 boys function using the error function relation, the rest would have to
    be determined recursively'''
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
    return Za*Zb / torch.sqrt(torch.sum((atom1 - atom2)**2))

# Define coordinates in angstroms as Torch tensors, turn on gradient tracking. Convert to Bohr. 
tmpatom1 = torch.tensor([0.0,0.0,0.0], requires_grad=True)
tmpatom2 = torch.tensor([0.0,0.0,0.9], requires_grad=True)
atom1 = ang2bohr * tmpatom1
atom2 = ang2bohr * tmpatom2
# Define basis exponent
a1 = torch.tensor([0.2331359749], requires_grad=True)

# Compute nuclear repulsion energy, and integrals: overlap, kinetic, nuclear-electron potential, two-electron repulsion 
Enuc = nuclear_repulsion(atom1, atom2)

s1 = overlap_s(a1, a1, atom1, atom1)
s2 = overlap_s(a1, a1, atom1, atom2)
S = torch.stack([s1,s2,s2,s1]).reshape(2,2)

t1 = kinetic_s(a1, a1, atom1, atom1)
t2 = kinetic_s(a1, a1, atom1, atom2)
T = torch.stack([t1,t2,t2,t1]).reshape(2,2)

v1 = potential_s(a1, a1, atom1, atom1, atom1, 1) + potential_s(a1, a1, atom1, atom1, atom2, 1)
v2 = potential_s(a1, a1, atom1, atom2, atom1, 1) + potential_s(a1, a1, atom1, atom2, atom2, 1)
V = torch.stack([v1,v2,v2,v1]).reshape(2,2)

g1 = eri_s(a1, a1, a1, a1, atom1, atom1, atom1, atom1)
g2 = eri_s(a1, a1, a1, a1, atom1, atom1, atom1, atom2)
g3 = eri_s(a1, a1, a1, a1, atom1, atom1, atom2, atom2)
g4 = eri_s(a1, a1, a1, a1, atom1, atom2, atom1, atom2)
G = torch.stack([g1, g2, g2, g3, g2, g4, g4, g2, g2, g4, g4, g2, g3, g2, g2, g1]).reshape(2,2,2,2)

# Converged density from Psi4
D = torch.tensor([[0.29175269765654,0.29175269765655],
                  [0.29175269765655,0.29175269765655]])

def hartree_fock_energy(H, G, D, Enuc):
    J = torch.einsum('pqrs,rs->pq', G, D)
    K = torch.einsum('prqs,rs->pq', G, D)
    F = H + J * 2 - K
    e = torch.einsum('pq,pq->', F + H, D) + Enuc
    return e

# Energy in Hartree agrees
print(hartree_fock_energy(T + V, G, D, Enuc))


#print(torch.autograd.grad(k2, tmpatom2))
#print(torch.autograd.grad(v1, tmpatom1))

