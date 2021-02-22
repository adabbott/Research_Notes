import psijax
import psi4
psi4.core.be_quiet()
import jax
from jax.config import config; config.update("jax_enable_x64", True)
from psijax.integrals.basis_utils import build_basis_set
#from psijax.integrals.tei import tei_array
#from psijax.integrals.oei import oei_arrays
from psijax.methods.hartree_fock import restricted_hartree_fock
from psijax.methods.ccsd_t import rccsd_t
import jax.numpy as np
import numpy as onp
import os
np.set_printoptions(linewidth=800)

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=true "
                           "intra_op_parallelism_threads=8 "
                           "inter_op_parallelism_threads=8 ")

molecule = psi4.geometry("""
                         0 1
                         H  0.0  0.0 -0.370424047469
                         H  0.0  0.0  0.370424047469
                         units ang
                         """)

# NOTE flattened geometry
geom = onp.asarray(molecule.geometry())
geomflat = np.asarray(geom.flatten())
basis_name = 'cc-pvdz'
xyz_file_name = "geom.xyz"
# Save xyz file, get path
molecule.save_xyz_file(xyz_file_name, True)
xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name

#print(os.path.abspath(os.getcwd()))
#print(xyzpath)
#print(basis_name)

basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
print("Number of basis functions", mints.nbf()) 


E = restricted_hartree_fock(geomflat, basis_name, xyz_path, nuclear_charges, charge, SCF_MAX_ITER=50,return_aux_data=False)
print(E)


psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': False, 'puream': 0})
psi_E = psi4.energy('hf' + '/' + basis_name)
print(psi_E)
