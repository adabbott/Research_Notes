import psi4
import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import numpy as onp
import os

# Set number of threads for Psi4 and JAX
nthreads = 4
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={} inter_op_parallelism_threads={} ".format(nthreads,nthreads))
psi4.set_num_threads(nthreads)

# Load molecule, basis
molecule = psi4.geometry("""                        
                        0 1                        
                        N  0.0  0.0 -0.8           
                        N  0.0  0.0  0.8           
                        units bohr                 
                        """)                       
#basis_name = 'cc-pvqz'
basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)  
mints = psi4.core.MintsHelper(basis_set)    

# Create TEI's
psi_G = mints.ao_eri()
G = np.asarray(onp.asarray(psi_G))
dim = G.shape[0]
# Create fake MO coefs
C = onp.random.rand(dim,dim)
psi_C = psi4.core.Matrix.from_array(C)
C = np.asarray(C)

# Okay, for whatever reason this is better than transposing in jitted function, doing it all in one einsum, doing it all in one function without tmp_transform
# Other methods seem to create intermediates, this does not
@jax.jit
def tmp_transform(G, C):
    return np.tensordot(C, G, axes=(0,0))

# NOTE number of transpositions matter greatly
#def tei_transformation(G, C):
    #G = tmp_transform(G, C)           # (A,b,c,d)
    #G = np.transpose(G, (1,0,2,3))    # (b,A,c,d)
    #G = tmp_transform(G, C)           # (B,A,c,d)
    #G = np.transpose(G, (2,0,1,3))    # (c,B,A,d)
    #G = tmp_transform(G, C)           # (C,B,A,d)
    #G = np.transpose(G, (3,0,1,2))    # (d,C,B,A)
    #G = tmp_transform(G, C)           # (D,C,B,A)

    #G = np.tensordot(C, G, axes=(0,0))
    #G = np.transpose(G, (1,0,2,3))    # (b,A,c,d)
    #G = np.tensordot(C, G, axes=(0,0))
    #G = np.transpose(G, (2,0,1,3))    # (c,B,A,d)
    #G = np.tensordot(C, G, axes=(0,0))
    #G = np.transpose(G, (3,0,1,2))    # (d,C,B,A)
    #G = np.tensordot(C, G, axes=(0,0))


# new knowledgge: number of transposes matters. OG best func has 1 + 2 + 4 axis changes
@jax.jit
def tmp_transform(G, C):
    return np.tensordot(C, G, axes=(0,0))

def tei_transformation(G, C):
    #G = tmp_transform(G, C)           # (A,b,c,d)
    #G = np.transpose(G, (1,0,2,3))    # (b,A,c,d)
    #G = tmp_transform(G, C)           # (B,A,c,d)
    #G = np.transpose(G, (2,0,1,3))    # (c,B,A,d)
    #G = tmp_transform(G, C)           # (C,B,A,d)
    #G = np.transpose(G, (3,0,1,2))    # (d,C,B,A)
    #G = tmp_transform(G, C)           # (D,C,B,A)

    #G = np.tensordot(C, G, axes=(0,0))
    #G = np.transpose(G, (1,0,2,3))    # (b,A,c,d)
    #G = np.tensordot(C, G, axes=(0,0))
    #G = np.transpose(G, (2,0,1,3))    # (c,B,A,d)
    #G = np.tensordot(C, G, axes=(0,0))
    #G = np.transpose(G, (3,0,1,2))    # (d,C,B,A)
    #G = np.tensordot(C, G, axes=(0,0))

    #G = tmp_transform(G,C)          # A b c d
    #G = np.transpose(G, (1,0,2,3))  # b A c d 1 transpose
    #G = tmp_transform(G,C)          # B A c d 
    #G = np.transpose(G, (2,3,0,1))  # c d B A 2 transposes
    #G = tmp_transform(G,C)          # C d B A
    #G = np.transpose(G, (1,0,2,3))  # d C B A 1 transpose
    #G = tmp_transform(G,C)          # D C B A

    #G = tmp_transform(G,C)          # A b c d
    #G = np.transpose(G, (1,0,2,3))  # b A c d 1 transpose
    #G = tmp_transform(G,C)          # B A c d 
    #G = np.transpose(G, (2,1,0,3))  # c A B d 1 transpose
    #G = tmp_transform(G,C)          # C A B d
    #G = np.transpose(G, (3,1,2,0))  # d A B C 1 transpose
    #G = tmp_transform(G,C)          # D A B C

    # The best: only 3 tranposes, 4 equivlaent tensor contractions which are jit compilable, 
    # and no intermediates by overwriting same tensor each time to save memory
    G = tmp_transform(G,C)          # A b c d
    G = np.transpose(G, (3,1,2,0))  # 1 transpose
    G = tmp_transform(G,C)          # D b c A
    G = np.transpose(G, (1,0,2,3))  # 1 transpose
    G = tmp_transform(G,C)          # b D c A
    G = np.transpose(G, (2,1,0,3))  # 1 transpose
    G = tmp_transform(G,C)          # C D B A 
    return G

def tei_transformation1(G,C):
    G = tmp_transform(G,C)          # A b c d
    G = np.transpose(G, (1,0,2,3))  # b A c d 1 transpose
    G = tmp_transform(G,C)          # B A c d 
    G = np.transpose(G, (2,3,0,1))  # c d B A 2 transposes
    G = tmp_transform(G,C)          # C d B A
    G = np.transpose(G, (1,0,2,3))  # d C B A 1 transpose
    G = tmp_transform(G,C)          # D C B A
    return G

def tei_transformation2(G,C):
    G = tmp_transform(G,C)          # A b c d
    G = np.transpose(G, (3,1,2,0))  # 1 transpose
    G = tmp_transform(G,C)          # D b c A
    G = np.transpose(G, (1,0,2,3))  # 1 transpose
    G = tmp_transform(G,C)          # b D c A
    G = np.transpose(G, (2,1,0,3))  # 1 transpose
    G = tmp_transform(G,C)          # C D B A 
    return G

def tei_transformation3(G,C):
    G = tmp_transform(G, C)           # (A,b,c,d)
    G = np.transpose(G, (1,0,2,3))    # (b,A,c,d) 1 T
    G = tmp_transform(G, C)           # (B,A,c,d)
    G = np.transpose(G, (2,0,1,3))    # (c,B,A,d) 2 T
    G = tmp_transform(G, C)           # (C,B,A,d)
    G = np.transpose(G, (3,0,1,2))    # (d,C,B,A) 3 T
    G = tmp_transform(G, C)           # (D,C,B,A)
    return G

def tei_transformation4(G,C):
    G = tmp_transform(G,C)          # A b c d
    G = np.swapaxes(G,0,3)       # 1 transpose
    G = tmp_transform(G,C)          # D b c A
    G = np.swapaxes(G,0,1)       # 1 transpose
    G = tmp_transform(G,C)          # b D c A
    G = np.swapaxes(G,0,2)       # 1 transpose
    G = tmp_transform(G,C)          # C D B A 
    return G

def tei_transformation5(G,C):
    G = tmp_transform(G,C)          # A b c d
    G = jax.lax.transpose(G, (3,1,2,0))  # 1 transpose
    G = tmp_transform(G,C)          # D b c A
    G = jax.lax.transpose(G, (1,0,2,3))  # 1 transpose
    G = tmp_transform(G,C)          # b D c A
    G = jax.lax.transpose(G, (2,1,0,3))  # 1 transpose
    G = tmp_transform(G,C)          # C D B A 
    return G

def tei_transformation6(G,C):
    G = tmp_transform(G,C)          # A b c d
    G = jax.lax.transpose(G, (1,0,2,3))  # b A c d 1 transpose
    G = tmp_transform(G,C)          # B A c d 
    G = jax.lax.transpose(G, (2,3,0,1))  # c d B A 2 transposes
    G = tmp_transform(G,C)          # C d B A
    G = jax.lax.transpose(G, (1,0,2,3))  # d C B A 1 transpose
    G = tmp_transform(G,C)          # D C B A
    return G

#T1 = tei_transformation1(G, C)
#T2 = tei_transformation5(G, C)
#T2 = tei_transformation2(G, C)
#T3 = tei_transformation3(G, C)
#print(np.allclose(T1,T2))
#print(np.allclose(T2,T3))


#T1 = tei_transformation(G, C)

#def tei_transformation(G, C):
#    G = tmp_transform(G, C)           # (A,b,c,d)
#    G = np.transpose(G, (1,0,2,3))    # (b,A,c,d)
#    G = tmp_transform(G, C)           # (B,A,c,d)
#    G = np.transpose(G, (2,0,1,3))    # (c,B,A,d)
#    G = tmp_transform(G, C)           # (C,B,A,d)
#    G = np.transpose(G, (3,0,1,2))    # (d,C,B,A)
#    G = tmp_transform(G, C)           # (D,C,B,A)
#    return G
#
#
#T2 = tei_transformation(G, C)
#
#print(np.allclose(T1,T2))

# To test in ipython:
# %load mo_transform_benchmark.py
# %timeit mints.mo_transform(psi_G, psi_C, psi_C, psi_C, psi_C)
# %timeit tei_transformation(G, C).block_until_ready()
#tei_transformation(G, C).block_until_ready()


