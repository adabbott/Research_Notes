import pylibint
import h5py
import numpy as np
import psi4
np.set_printoptions(linewidth=500)

basis_name = 'cc-pvdz'
xyzpath = "/home/adabbott/h2.xyz"

pylibint.initialize(xyzpath, basis_name)
print("writing eris to disk")
pylibint.eri_deriv_disk(1)
pylibint.eri_deriv_disk(2)
print("writing done")
pylibint.finalize()

print("Now reading HDF5 eri derivatives")

#with h5py.File('eri_deriv.h5', 'r') as f:
#    data_set = f['eri_deriv']
#    # Slice of [0,0,2,0,0,0]
#    data = data_set[:,:,:,:,2]
#    #print(data.shape)
#
#with open(xyzpath, 'r') as f:
#    tmp = f.read()
#molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
#basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
#mints = psi4.core.MintsHelper(basis_set)
#
#x1, y1, z1 = mints.ao_tei_deriv1(0,0)
#psi_deriv =  np.asarray(z1)
#print(np.allclose(psi_deriv, data))
#
#
