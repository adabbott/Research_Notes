import numpy as np
from helper import helper
from itertools import combinations
import psi4
import analytic_derivative as ad

np.set_printoptions(linewidth=150)
hartree2J = 4.3597443e-18
amu2kg = 1.6605389e-27
ang2m = 1e-10
c = 29979245800.0 # speed of light in cm/s
convert = np.sqrt(hartree2J/(amu2kg*ang2m*ang2m))/(c*2*np.pi)

#psi4.core.be_quiet()
mol = psi4.geometry(
'''
 O            0.000000000000     0.000000000000     0.000000000000    
 H            0.000000000000     0.000000000000     0.949676529800    
 H            0.000000000000     0.883402475500    -0.348547812400    
''')
psi4.set_options({'scf_type': 'pk'})
e, wfn = psi4.frequencies('scf/6-31g', return_wfn = True)
geom = np.array(mol.geometry())
grad = np.array(wfn.gradient())
hess = np.array(wfn.hessian())
hess /= 0.529177249**2

internals = [helper.STRE(0,1), helper.STRE(0,2), helper.BEND(2,0,1)]
inthess = helper.convertHessianToInternals(hess, internals, geom)

# Internal coordinate Vib Analysis
B = helper.Bmat(internals, geom) 
invmass = 1 / np.array([15.994914619570,15.994914619570,15.994914619570, 1.007825032230, 1.007825032230,1.007825032230, 1.007825032230,1.007825032230, 1.007825032230])
G = np.einsum('in,jn,n->ij', B, B, invmass)
GF = G.dot(inthess)
intlamda, intL = np.linalg.eig(GF)
print(np.sqrt(intlamda) * convert)


## Cartesian coordinate Vib Analysis
m = np.array([15.994914619570,15.994914619570,15.994914619570, 1.007825032230, 1.007825032230,1.007825032230, 1.007825032230,1.007825032230, 1.007825032230])
M = 1 / np.sqrt(m)
M = np.diag(M)

Lt_inv = np.linalg.inv(intL).T
little_l = M.dot(B.T).dot(Lt_inv)
print(little_l)


Hmw = M @ hess @ M
cartlamda, cartL = np.linalg.eig(Hmw)
idx = cartlamda.argsort()[::-1]   
cartlamda = cartlamda[idx]
cartL = cartL[:,idx]
print(cartL[:,:mol.natom()])
print(cartL)
#vibrational_normcoords = cartL[:,:mol.natom()]
#print(vibrational_normcoords)
#print(np.sqrt(cartlamda[:mol.natom()]) * convert)

