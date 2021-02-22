import psi4
import jax.numpy as np
from jax.experimental import loops
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True, threshold=100)
from pprint import pprint
import time
from oei_s import * 
from oei_p import * 
from oei_d import * 
from oei_f import * 


# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         H 0.0 0.0  2.000000000000
                         H 0.0 0.0  3.000000000000
                         H 0.0 0.0  4.000000000000
                         H 0.0 0.0  5.000000000000
                         H 0.0 0.0  6.000000000000
                         H 0.0 0.0  7.000000000000
                         H 0.0 0.0  8.000000000000
                         H 0.0 0.0  9.000000000000
                         H 0.0 0.0  10.000000000000
                         H 0.0 0.0  11.000000000000
                         H 0.0 0.0  12.000000000000
                         H 0.0 0.0  13.000000000000
                         H 0.0 0.0  14.000000000000
                         H 0.0 0.0  15.000000000000
                         H 0.0 0.0  16.000000000000
                         H 0.0 0.0  17.000000000000
                         H 0.0 0.0  18.000000000000
                         H 0.0 0.0  19.000000000000
                         units bohr
                         """)

# Get Psi Basis Set and basis set dictionary objects
basis_name = 'cc-pvdz'
#basis_name = '6-31g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)

# hack to make big basis but small system
for key in basis_dict:
    basis_dict[key]['atom'] = 0
molecule = psi4.geometry("""
                         0 1
                         units bohr
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))


nbf = basis_set.nbf()
nshells = len(basis_dict)
max_prim = basis_set.max_nprimitive()
biggest_K = max_prim * max_prim
print("Number of basis functions: ", nbf)
print("Number of shells : ", nshells)
print("Max primitives: ", max_prim)
print("Biggest contraction: ", biggest_K)

def preprocess(geom, basis_dict, nshells):
    basis_data = []
    centers = []
    ang_mom = []
    for i in range(nshells):
        c1 =    onp.asarray(basis_dict[i]['coef'])
        exp1 =  onp.asarray(basis_dict[i]['exp'])
        atom1_idx = basis_dict[i]['atom']
        bra_am = basis_dict[i]['am']
        for j in range(nshells):
            c2 =    onp.asarray(basis_dict[j]['coef'])
            exp2 =  onp.asarray(basis_dict[j]['exp'])
            atom2_idx = basis_dict[j]['atom']
            ket_am = basis_dict[j]['am']

            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            current_K = exp_combos.shape[0] # Size of this contraction
            exp_combos = onp.pad(exp_combos, ((0, biggest_K - current_K), (0,0)))
            coeff_combos = onp.pad(coeff_combos, ((0, biggest_K - current_K), (0,0)))

            size = ((bra_am + 1) * (bra_am + 2) // 2) * ((ket_am + 1) * (ket_am + 2) // 2) 

            if bra_am == 0 and ket_am == 0: am=0
            if bra_am == 1 and ket_am == 0: am=1
            if bra_am == 0 and ket_am == 1: am=4
            if bra_am == 1 and ket_am == 1: am=7

            # every primitive component gets its own basis data row (dumb, ik)
            for component in range(size):
                basis_data.append([exp_combos[:,0], exp_combos[:,1], coeff_combos[:,0], coeff_combos[:,1]])
                centers.append([atom1_idx, atom2_idx])
                ang_mom.append(am)
                am += 1

    return np.asarray(onp.asarray(basis_data)), np.asarray(onp.asarray(centers)), np.asarray(onp.asarray(ang_mom))

basis_data, centers, AM = preprocess(geom, basis_dict, nshells)
print(basis_data.shape)
print(centers.shape)
print(AM.shape)

def build_overlap(geom, centers, basis_data, nbf):
    centers_bra = np.take(geom, centers[:,0], axis=0) 
    centers_ket = np.take(geom, centers[:,1], axis=0)

    with loops.Scope() as s:
        # Computes a primitive ss overlap
        #@jax.jit
        def primitive(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, am):
            '''Geometry parameters, exponents, coefficients, angular momentum index'''
            args = (Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2)
            sgra = (Cx, Cy, Cz, Ax, Ay, Az, e2, e1, c2, c1)
            # Do no computation for dummy padded values
            prim =  np.where(e1 ==  0, 0.0, 
                    np.where(am ==  0, overlap_ss(*args),
                    np.where(am ==  1, overlap_ps(*args)[0],
                    np.where(am ==  2, overlap_ps(*args)[1],
                    np.where(am ==  3, overlap_ps(*args)[2],
                    np.where(am ==  4, overlap_ps(*sgra)[0],
                    np.where(am ==  5, overlap_ps(*sgra)[1],
                    np.where(am ==  6, overlap_ps(*sgra)[2],
                    np.where(am ==  7, overlap_pp(*args)[0,0],
                    np.where(am ==  8, overlap_pp(*args)[0,1],
                    np.where(am ==  9, overlap_pp(*args)[0,2],
                    np.where(am == 10, overlap_pp(*args)[1,0],
                    np.where(am == 11, overlap_pp(*args)[1,1],
                    np.where(am == 12, overlap_pp(*args)[1,2],
                    np.where(am == 13, overlap_pp(*args)[2,0],
                    np.where(am == 14, overlap_pp(*args)[2,1],
                    np.where(am == 15, overlap_pp(*args)[2,2],0.0)))))))))))))))))
            return prim

        # Computes multiple primitive ss overlaps with same center, angular momentum 
        vectorized_primitive = jax.vmap(primitive, (None,None,None,None,None,None,0,0,0,0,None))
        #vectorized_primitive = jax.jit(jax.vmap(primitive, (None,None,None,None,None,None,0,0,0,0,None)))

        # Computes a contracted ss overlap 
        #@jax.jit
        def contraction(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, am):
            primitives = vectorized_primitive(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, am)
            return np.sum(primitives)

        s.overlap = np.zeros(nbf**2)
        for i in s.range(s.overlap.shape[0]):
            exp1,exp2,c1,c2 = basis_data[i]
            Ax, Ay, Az = centers_bra[i]
            Cx, Cy, Cz = centers_ket[i]
            am = AM[i]
            args = (Ax, Ay, Az, Cx, Cy, Cz, exp1, exp2, c1, c2, am)
            val = contraction(*args)
            s.overlap = jax.ops.index_update(s.overlap, i, val)
        return s.overlap

#S = build_overlap(geom, centers, basis_data, nbf)
#print(S.shape)
#grad = jax.jacfwd(build_overlap)(geom, centers, basis_data, nbf)
#print(grad.shape)
#grad = jax.jacfwd(build_overlap)(geom, centers, basis_data, nbf)
#print(grad.shape)
#hess = jax.jacfwd(jax.jacfwd(build_overlap))(geom, centers, basis_data, nbf)
#print(hess)
quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap))))(geom, centers, basis_data, nbf)
print(quar.shape)


#def silly(nbf):
#    with loops.Scope() as s:
#        # All value references must be obtained in scope
#        def compute(i,j):
#            return i**2 + 2*j + i*j
#
#        s.overlap = np.zeros((nbf,nbf))
#        for i in s.range(s.overlap.shape[0]):
#            for j in s.range(s.overlap.shape[1]):
#                val = compute(i,j)
#                s.overlap = jax.ops.index_update(s.overlap, (i,j), val)
#        return s.overlap
#
#test = silly(nbf)




