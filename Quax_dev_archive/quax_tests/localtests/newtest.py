import psijax
import psi4
import jax

 
molecule = psi4.geometry('''
                         0 1
                         H 0.0 0.0 -0.80000000000
                         H 0.0 0.0  0.80000000000
                         symmetry c1
                         units bohr
                         ''')
     
print('h2 ccsd(t)/sto-3g partial quartic')
a = psijax.core.partial_derivative(molecule, 'sto-3g', 'ccsd(t)', order=4,address=(5,5,5,5)).block_until_ready()

#jax.profiler.save_device_memory_profile("memory.prof")
