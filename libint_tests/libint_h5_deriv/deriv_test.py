import pylibint
import h5py
import numpy as np
import psi4
np.set_printoptions(linewidth=500)

basis_name = 'cc-pvtz'
xyzpath = "/home/adabbott/h2.xyz"

pylibint.initialize(xyzpath, basis_name)
print("Writing overlap integrals")
my_S = pylibint.overlap()
print("Writing kinetic and overlap derivatives to disk")
pylibint.oei_deriv_disk(1)
print("Writing eri derivatives to disk")
pylibint.eri_deriv_disk(1)
pylibint.finalize()

with h5py.File('eri_deriv1.h5', 'r') as f:
    eri_grad = f['eri_deriv']         
    my_eri_2 = eri_grad[:,:,:,:,2]

#print("Now reading HDF5 oei derivatives")
with h5py.File('oei_derivs.h5', 'r') as f:
    overlap_grad = f['overlap_deriv1']
    kinetic_grad = f['kinetic_deriv1']

    my_overlap_2 = overlap_grad[:,:,2]
    my_kinetic_2 = kinetic_grad[:,:,2]

with open(xyzpath, 'r') as f:
    tmp = f.read()
molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)

psi_S = mints.ao_overlap()
x1, y1, z1 = mints.ao_tei_deriv1(0,0)
psi_eri_2 = np.asarray(z1)

overlap_0, overlap_1, overlap_2 = mints.ao_oei_deriv1("OVERLAP", 0)
overlap_3, overlap_4, overlap_5 = mints.ao_oei_deriv1("OVERLAP", 1)

kinetic_0, kinetic_1, kinetic_2 = mints.ao_oei_deriv1("KINETIC", 0)
kinetic_3, kinetic_4, kinetic_5 = mints.ao_oei_deriv1("KINETIC", 1)

print("Overlap integrals match?", np.allclose(np.asarray(psi_S).flatten(), my_S))
print("ERI gradient match?", np.allclose(np.asarray(psi_eri_2), my_eri_2))

print("Overlap gradient match?", np.allclose(np.asarray(overlap_2), my_overlap_2))
print("Kinetic gradient match?", np.allclose(np.asarray(kinetic_2), my_kinetic_2))

#print(np.allclose(, overlap_2))# overlap_grad[:,:,2]))
#print(np.allclose(psi_deriv, kinetic_2))# overlap_grad[:,:,2]))
#
##psi_deriv =  np.asarray(z2)

# Hmm. TZ only gradients of overlap, kinetic fail
# Use PSIJAX since psi4 cannot be trusted.
#from psijax.integrals.basis_utils import build_basis_set
#from psijax.integrals.tei import tei_array
#from psijax.integrals.oei import oei_arrays
#import numpy as onp
#import jax.numpy as np
#import jax
#
#geom = onp.asarray(molecule.geometry())
#geomflat = np.asarray(geom.flatten())
#basis_dict = build_basis_set(molecule, basis_name)
#nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
#
#def wrap_oeis(geomflat):
#    geom = geomflat.reshape(-1,3)
#    S, T, V = oei_arrays(geom,basis_dict,nuclear_charges)
#    return S, T, V
#
#print("Computing psijax derivs")
#psijax_overlap_grad, psijax_kinetic_grad, psijax_potential_grad = jax.jacfwd(wrap_oeis)(geomflat)
#
#print("HDF5 and PsiJax agree?")
#print(onp.allclose(my_overlap_2, psijax_overlap_grad[:,:,2]))
#print(onp.allclose(my_kinetic_2, psijax_kinetic_grad[:,:,2]))
#
#print("Psi4 and PsiJax agree?")
#print(onp.allclose(overlap_2, psijax_overlap_grad[:,:,2]))
#print(onp.allclose(kinetic_2, psijax_kinetic_grad[:,:,2]))



