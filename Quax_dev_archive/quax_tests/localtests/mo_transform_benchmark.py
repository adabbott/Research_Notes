import psi4
import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import numpy as onp
import os

# Set number of threads for Psi4 and JAX
nthreads = 8
#os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={} inter_op_parallelism_threads={} ".format(nthreads,nthreads))
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={} ".format(nthreads))
psi4.set_num_threads(nthreads)

# Load molecule, basis
molecule = psi4.geometry("""                        
                        0 1                        
                        N  0.0  0.0 -0.8           
                        N  0.0  0.0  0.8           
                        units bohr                 
                        """)                       
basis_name = 'cc-pvqz'
#basis_name = 'cc-pvtz'
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
# also : number of tranposes matter. 1 transpose (1,0,2,3) != 4 transposes(1,2,3,0) 

@jax.jit
def tmp_transform(G, C):
    return np.tensordot(C, G, axes=(0,0))

#TEMP
@jax.jit
def tei_transformation(G, C):
    G = tmp_transform(G,C)          # A b c d
    G = jax.lax.transpose(G, (1,0,2,3))  # b A c d 1 transpose
    G = tmp_transform(G,C)          # B A c d 
    G = jax.lax.transpose(G, (2,3,0,1))  # c d B A 2 transposes
    G = tmp_transform(G,C)          # C d B A
    G = jax.lax.transpose(G, (1,0,2,3))  # d C B A 1 transpose
    G = tmp_transform(G,C)          # D C B A
    return G

# To test in ipython:
# %load mo_transform_benchmark.py
# %timeit mints.mo_transform(psi_G, psi_C, psi_C, psi_C, psi_C)
# %timeit tei_transformation(G, C).block_until_ready()

