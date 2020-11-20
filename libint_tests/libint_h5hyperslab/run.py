import pylibint
import h5py
import numpy as np
np.set_printoptions(linewidth=500)

pylibint.initialize("/home/adabbott/h2.xyz", 'cc-pvdz')
S = pylibint.overlap().reshape(10,10)
#S = pylibint.overlap().reshape(2,2)
print(np.round(S,4))

print("writing overlap to disk")
pylibint.overlap_disk()
print("writing done")
pylibint.finalize()

print("Now reading HDF5 overlap integrals")
with h5py.File('overlap.h5', 'r') as f:
    data_set = f['overlap']
    data = data_set[:]
    print(data.shape)
newS = np.asarray(data).reshape(10,10)
print(np.round(newS,4))


