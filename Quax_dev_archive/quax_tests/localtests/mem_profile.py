import psijax
import jax
import psi4


molecule = psi4.geometry('''
              0 1
              O
              H 1 r1
              H 1 r2 2 a1
              
              r1 = 1.0
              r2 = 1.0
              a1 = 104.5
              units ang
              ''')


#print('h2o scf/cc-pvtz hessian')
#psijax.core.derivative(molecule, 'cc-pvtz', 'scf', order=2)
k = psijax.core.derivative(molecule, 'cc-pvtz', 'scf', order=2).block_until_ready()

jax.profiler.save_device_memory_profile("memory.prof")

#print('h2o scf/sto3g test')
#psijax.core.energy(molecule, 'sto-3g', 'scf')


