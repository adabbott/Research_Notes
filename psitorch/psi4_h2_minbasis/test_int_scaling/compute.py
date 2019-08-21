import torch
import math
torch.set_printoptions(precision=8,linewidth=300)
torch.set_default_dtype(torch.float64)
ang2bohr = 1 / 0.52917720859
#@torch.jit.script
def gp(aa,bb,A,B):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (aa * A + bb * B) / (aa + bb)
    c = torch.exp(torch.dot(A-B,A-B) * (-aa * bb / (aa + bb)))
    return R,c

def normalize(aa):
    '''Normalization constant for s primitive basis functions. Argument is orbital exponent coefficient'''
    N = (2*aa/math.pi)**(3/4)
    return N

def overlap(aa, bb, Ra, Rb):
    Na = normalize(aa)
    Nb = normalize(bb)
    R,c = gp(aa,bb,Ra,Rb)
    S = Na * Nb * c * (math.pi / (aa + bb)) ** (3/2)
    return S

def kinetic(aa,bb,A,B):
    P = (aa * bb) / (aa + bb)
    ab = -1.0 * torch.dot(A-B, A-B)
    K = overlap(aa,bb,A,B) * (3 * P + 2 * P * P * ab)
    return K

def build_oei(basisA, basisB, A, B, mode):
    '''Builds overlap or kinetic matrix of diatomic molecule with s-orbital basis functions'''
    nbfA = torch.numel(basisA)
    nbfB = torch.numel(basisB)
    nbf = nbfA + nbfB
    I = torch.zeros((nbf,nbf))
    basis = torch.cat((basisA,basisB), dim=0).reshape(-1,1)
    An = A.repeat(nbfA).reshape(nbfA, 3)
    Bn = B.repeat(nbfB).reshape(nbfB, 3)
    centers = torch.cat((An,Bn),dim=0)
    if mode == 'kinetic':
        for i,b1 in enumerate(basis):
            for j in range(i+1):
                I[i,j] = kinetic(b1, basis[j], centers[i], centers[j])
                I[j,i] = I[i,j]
    if mode == 'overlap':
        for i,b1 in enumerate(basis):
            for j in range(i+1):
                I[i,j] = overlap(b1, basis[j], centers[i], centers[j])
                I[j,i] = I[i,j]
    if mode == 'potential':
        for i,b1 in enumerate(basis):
            for j in range(i+1):
                I[i,j] = potential(b1, basis[j], centers[i], centers[j], A, torch.tensor(1.0)) + \
                         potential(b1, basis[j], centers[i], centers[j], B, torch.tensor(1.0))
                I[j,i] = I[i,j]
    return I

def torchboys(nu, arg):
    '''Pytorch can only compute F0 boys function using the error function relation, the rest would have to
    be determined recursively'''
    if arg < 1e-8:
        boys =  1 / (2 * nu + 1) - arg / (2 * nu + 3)
    else:
        boys = torch.erf(torch.sqrt(arg)) * math.sqrt(math.pi) / (2 * torch.sqrt(arg))
    return boys

#@torch.jit.script
def potential_s(a1, a2, A, B, atom, charge):
    g = a1 + a2
    eps = 1 / (4 * g)
    Rp = (a1 * A + a2 * B) / (a1 + a2)
    tmpc = torch.dot(A-B, A-B) * ((-a1 * a2) / (a1 + a2))
    c = torch.exp(tmpc)
    arg = g * torch.dot(Rp - atom, Rp - atom)
    #F = torchboys(0, arg)
    F = torchboys(torch.tensor(0.0), arg)
    N = math.pi**(-3/4) * (1 / (a1 + a2)**(3/2))**(-0.5)
    Vn = -charge * F *  N * N * c * 2 * math.pi / g
    return Vn

def potential(aa,bb,A,B,atom,charge):
    g = aa + bb
    eps = 1 / (4 * g)
    P, c = gp(aa,bb,A,B)
    arg = g * torch.dot(P - atom, P - atom)
    Na = normalize(aa)
    Nb = normalize(bb)
    F = torchboys(torch.tensor(0.0), arg)
    V = -charge * F * Na * Nb * c * 2 * math.pi / g
    return V

def build_potential(basisA, basisB, A, B):
    nbfA = torch.numel(basisA)
    nbfB = torch.numel(basisB)
    nbf = nbfA + nbfB
    I = torch.zeros((nbf,nbf))
    basis = torch.cat((basisA,basisB), dim=0).reshape(-1,1)
    An = A.repeat(nbfA).reshape(nbfA, 3)
    Bn = B.repeat(nbfB).reshape(nbfB, 3)
    centers = torch.cat((An,Bn),dim=0)
    for i,b1 in enumerate(basis):
        for j in range(i+1):
            I[i,j] = potential(b1, basis[j], centers[i], centers[j], A, torch.tensor(1.0)) + \
                     potential(b1, basis[j], centers[i], centers[j], B, torch.tensor(1.0))
            I[j,i] = I[i,j]
    return I

#@torch.jit.script
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
    #F = torchboys(0, arg)
    F = torchboys(torch.tensor(0.0), arg)
    N = math.pi**(-3/4) * (1 / (a1 + a2)**(3/2))**(-0.5)
    G = F * N * N * N * N * c1 * c2 * 2 * math.pi**2 / (g1 * g2) * torch.sqrt(math.pi / (g1 + g2))
    return G
    
#@torch.jit.script
def nuclear_repulsion(atom1, atom2):
    ''' warning : hard coded'''
    Za = 1.0
    Zb = 1.0
    return Za*Zb / torch.norm(atom1-atom2) 

#@torch.jit.script
def orthogonalizer(S):
    '''Compute overlap to the negative 1/2 power'''
    eigval, eigvec = torch.symeig(S, eigenvectors=True)
    d12 = torch.diag(torch.sqrt(eigval))
    tmpA = torch.chain_matmul(eigvec, d12, eigvec) 
    A = torch.inverse(tmpA)  # Orthogonalizer S^(-1/2)
    return A

#def factorial(tensr):
#    return torch.exp(torch.mvlgamma(tensr, 1))


#@torch.jit.script
def hf_energy(A,T,V,G,D,Enuc,ndocc):
    '''Given a converged density matrix, construct a PyTorch graph mapping the geometry to the SCF energy'''
    H = T + V
    J = torch.einsum('pqrs,rs->pq', G, D)
    K = torch.einsum('prqs,rs->pq', G, D)
    F = H + J * 2 - K
    Fp = torch.chain_matmul(A, F, A)
    e, C2 = torch.symeig(Fp, eigenvectors=True)             
    C = torch.matmul(A, C2)
    Cocc = C[:, :ndocc]                                                              
    D = torch.einsum('pi,qi->pq', Cocc, Cocc)
    J = torch.einsum('pqrs,rs->pq', G, D)
    K = torch.einsum('prqs,rs->pq', G, D)
    F = H + J * 2 - K
    E = torch.einsum('pq,pq->', F + H, D) + Enuc
    return E

# Define coordinates in Bohr as Torch tensors, turn on gradient tracking.  
tmpgeom1 = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
tmpgeom2 = []
for i in tmpgeom1:
    tmpgeom2.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
geom = torch.stack(tmpgeom2).reshape(2,3)
atom1 = geom[0]
atom2 = geom[1]
# Define s atomic orbital basis function exponent
basis = torch.tensor([0.5000000000,0.4000000], requires_grad=False)

# Compute nuclear repulsion energy, and integrals: overlap, kinetic, nuclear-electron potential, two-electron repulsion 
Enuc = nuclear_repulsion(atom1, atom2)
S = build_oei(basis, basis, atom1, atom2,mode='overlap')
T = build_oei(basis, basis, atom1, atom2,mode='kinetic')
V = build_oei(basis, basis, atom1, atom2,mode='potential')
print(S)
print(T)
print(V)
#V = build_potential(basis,basis,atom1,atom2)



#s2 = overlap_s(a1, a1, atom1, atom2)
#s3 = overlap_s(a1, a1, atom2, atom1)
#s4 = overlap_s(a1, a1, atom2, atom2)
#S = torch.stack([s1,s2,s3,s4]).reshape(2,2)
#S = torch.stack([s1,s2,s2,s1]).reshape(2,2)
#
#t1 = kinetic_s(a1, a1, atom1, atom1)
#t2 = kinetic_s(a1, a1, atom1, atom2)
###t3 = kinetic_s(a1, a1, atom2, atom1)
###t4 = kinetic_s(a1, a1, atom2, atom2)
###T = torch.stack([t1,t2,t3,t4]).reshape(2,2)
#T = torch.stack([t1,t2,t2,t1]).reshape(2,2)
#v1 = potential_s(a1, a1, atom1, atom1, atom1, torch.tensor(1.0)) + potential_s(a1, a1, atom1, atom1, atom2, torch.tensor(1.0))
#v2 = potential_s(a1, a1, atom1, atom2, atom1, torch.tensor(1.0)) + potential_s(a1, a1, atom1, atom2, atom2, torch.tensor(1.0))
###v3 = potential_s(a1, a1, atom1, atom2, atom1, 1) + potential_s(a1, a1, atom1, atom2, atom2, 1)
###v4 = potential_s(a1, a1, atom1, atom1, atom1, 1) + potential_s(a1, a1, atom1, atom1, atom2, 1)
###V = torch.stack([v1,v2,v3,v4]).reshape(2,2)
#V = torch.stack([v1,v2,v2,v1]).reshape(2,2)
#g1 = eri_s(a1, a1, a1, a1, atom1, atom1, atom1, atom1)
#g2 = eri_s(a1, a1, a1, a1, atom1, atom1, atom1, atom2)
#g3 = eri_s(a1, a1, a1, a1, atom1, atom1, atom2, atom2)
#g4 = eri_s(a1, a1, a1, a1, atom1, atom2, atom1, atom2)
#G = torch.stack([g1, g2, g2, g3, g2, g4, g4, g2, g2, g4, g4, g2, g3, g2, g2, g1]).reshape(2,2,2,2)
##print(G)
##
##g1 = eri_s(a1, a1, a1, a1, atom1, atom1, atom1, atom1)
##g2 = eri_s(a1, a1, a1, a1, atom1, atom1, atom2, atom1)
##g2 = eri_s(a1, a1, a1, a1, atom1, atom2, atom1, atom1)
##g2 = eri_s(a1, a1, a1, a1, atom2, atom1, atom1, atom1)
##
##
#A = orthogonalizer(S)
#D = torch.tensor([[0.336432896297142, 0.336432896297142],
#                  [0.336432896297142, 0.336432896297142]])
#E = hf_energy(A, T, V, G, D, Enuc, ndocc=torch.tensor(1))
#
##grad = torch.autograd.grad(E, geom, create_graph=True)[0] 
##h = []
##for g in grad.flatten():
##    h1 = torch.autograd.grad(g, geom, create_graph=True)[0]
##    h.append(h1)
##hess = torch.stack(h).reshape(6,6)
##
##print(E)
##print(grad)
##print(hess)
#
##from pyforce.transforms import differentiate_nn, slow_differentiate_nn
##hess = differentiate_nn(E, tmpgeom2, order=4)
##hess = slow_differentiate_nn(E, tmpgeom2, order=3)
##print(hess)
#
#
#
##print(torch.eig(hess))
##print(torch.symeig(hess))

