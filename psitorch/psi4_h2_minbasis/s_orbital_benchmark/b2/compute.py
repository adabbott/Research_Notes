import torch
import numpy as np
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

def torchboys(nu, arg):
    '''Pytorch can only compute F0 boys function using the error function relation, the rest would have to
    be determined recursively'''
    if arg < 1e-8:
        boys =  1 / (2 * nu + 1) - arg / (2 * nu + 3)
    else:
        boys = torch.erf(torch.sqrt(arg)) * math.sqrt(math.pi) / (2 * torch.sqrt(arg))
    return boys

def overlap(aa, bb, Ra, Rb):
    '''Computes a single overlap integral over two primitive s-orbital basis functions'''
    Na = normalize(aa)
    Nb = normalize(bb)
    R,c = gp(aa,bb,Ra,Rb)
    S = Na * Nb * c * (math.pi / (aa + bb)) ** (3/2)
    return S

def kinetic(aa,bb,A,B):
    '''Computes a single kinetic energy integral over two primitive s-orbital basis functions'''
    P = (aa * bb) / (aa + bb)
    ab = -1.0 * torch.dot(A-B, A-B)
    K = overlap(aa,bb,A,B) * (3 * P + 2 * P * P * ab)
    return K

def potential(aa,bb,A,B,atom,charge):
    '''Computes a single electron-nuclear potential energy integral over two primitive s-orbital basis functions'''
    g = aa + bb
    eps = 1 / (4 * g)
    P, c = gp(aa,bb,A,B)
    arg = g * torch.dot(P - atom, P - atom)
    Na = normalize(aa)
    Nb = normalize(bb)
    F = torchboys(torch.tensor(0.0), arg)
    V = -charge * F * Na * Nb * c * 2 * math.pi / g
    return V

def eri(aa,bb,cc,dd,A,B,C,D):
    '''Computes a single two electron integral over 4 s-orbital basis functions on 4 centers'''
    g1 = aa + bb
    g2 = cc + dd
    Rp = (aa * A + bb * B) / (aa + bb)
    tmpc1 = torch.dot(A-B, A-B) * ((-aa * bb) / (aa + bb))
    c1 = torch.exp(tmpc1)
    Rq = (cc * C + dd * D) / (cc + dd)
    tmpc2 = torch.dot(C-D, C-D) * ((-cc * dd) / (cc + dd))
    c2 = torch.exp(tmpc2)

    Na, Nb, Nc, Nd = normalize(aa), normalize(bb), normalize(cc), normalize(dd)
    delta = 1 / (4 * g1) + 1 / (4 * g2)
    arg = torch.dot(Rp - Rq, Rp - Rq) / (4 * delta)
    F = torchboys(torch.tensor(0.0), arg)
    G = F * Na * Nb * Nc * Nd * c1 * c2 * 2 * math.pi**2 / (g1 * g2) * torch.sqrt(math.pi / (g1 + g2))
    return G

def build_oei(basisA, basisB, A, B, mode):
    '''Builds overlap/kinetic/potential one-electron integral matrix of diatomic molecule with s-orbital basis functions, based on 'mode' argument'''
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

def build_tei(basisA, basisB, basisC, basisD, A, B, C, D):
    '''Builds two-electron integral 4d tensor for a diatomic molecule with s-orbital basis functions'''
    nbfA, nbfB = torch.numel(basisA), torch.numel(basisB)
    nbf = nbfA + nbfB 
    G = torch.zeros((nbf,nbf,nbf,nbf))
    basis = torch.cat((basisA,basisB,basisC,basisD), dim=0).reshape(-1,1)
    An = A.repeat(nbfA).reshape(nbfA, 3)
    Bn = B.repeat(nbfB).reshape(nbfB, 3)
    Cn = C.repeat(nbfA).reshape(nbfA, 3)
    Dn = D.repeat(nbfB).reshape(nbfB, 3)
    centers = torch.cat((An,Bn,Cn,Dn),dim=0)
    # WARNING: 8 fold redundancy
    for i in range(nbf):
        for j in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    val = eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l])
                    G[i,j,k,l] = val
                    # TODO just check to see if tensor position has been modified yet? possible with empty() mayb?
                    #G[j,i,l,k] = G[i,j,k,l] 
                    #G[k,l,i,j] = G[i,j,k,l] 
                    #G[l,k,j,i] = G[i,j,k,l] #TODO theres a typo here?
                    #G[k,j,i,l] = G[i,j,k,l] 
                    #G[l,i,j,k] = G[i,j,k,l] 
                    #G[i,l,k,j] = G[i,j,k,l] 
                    #G[j,k,l,i] = G[i,j,k,l] 
    return G

def nuclear_repulsion(atom1, atom2):
    ''' warning : hard coded'''
    Za = 1.0
    Zb = 1.0
    return Za*Zb / torch.norm(atom1-atom2) 

def orthogonalizer(S):
    '''Compute overlap to the negative 1/2 power'''
    eigval, eigvec = torch.symeig(S, eigenvectors=True)
    d12 = torch.diag(torch.reciprocal(torch.sqrt(eigval)))
    A = torch.chain_matmul(eigvec, d12, torch.t(eigvec))
    return A

#def factorial(tensr):
#    return torch.exp(torch.mvlgamma(tensr, 1))

def hartree_fock(geom, basis):
    '''Computes hartree fock energy from psi4 converged density''' 
    atom1 = geom[0]
    atom2 = geom[1]
    nbf = basis.size()[0] * 2 #hard coded
    ndocc = 1                 #hard coded
    Enuc = nuclear_repulsion(atom1, atom2)
    S = build_oei(basis, basis, atom1, atom2,mode='overlap')
    A = orthogonalizer(S)
    T = build_oei(basis, basis, atom1, atom2,mode='kinetic')
    V = build_oei(basis, basis, atom1, atom2,mode='potential')
    G = build_tei(basis,basis,basis,basis,atom1,atom2,atom1,atom2)
    # Get converged density from psi4, check integrals
    D = torch.from_numpy(np.load('D.npy'))
    S2 = torch.from_numpy(np.load('S.npy'))
    A2 = torch.from_numpy(np.load('A.npy'))
    T2 = torch.from_numpy(np.load('T.npy'))
    V2 = torch.from_numpy(np.load('V.npy'))
    G2 = torch.from_numpy(np.load('G.npy'))
    print(torch.allclose(S2, S), end=' ')
    print(torch.allclose(A2, A), end=' ')
    print(torch.allclose(T2, T), end=' ')
    print(torch.allclose(V2, V), end=' ')
    print(torch.allclose(G2, G))

    H = T + V
    J = torch.einsum('pqrs,rs->pq', G, D)
    K = torch.einsum('prqs,rs->pq', G, D)
    F = H + J * 2 - K
    #E = torch.einsum('pq,pq->', F + H, D) + Enuc
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
tmpgeom2 = [torch.tensor(i, requires_grad=True) for i in tmpgeom1]
geom = torch.stack(tmpgeom2).reshape(2,3)
# Define s atomic orbital basis function exponents
basis = torch.tensor([0.5000000000,0.4000000], requires_grad=False)
# Compute energy
E = hartree_fock(geom, basis)
print(E)
# Compute gradient
grad = torch.autograd.grad(E, geom, create_graph=True)[0] 
print(grad)

##from pyforce.transforms import differentiate_nn, slow_differentiate_nn
##hess = differentiate_nn(E, tmpgeom2, order=4)
##hess = slow_differentiate_nn(E, tmpgeom2, order=3)
