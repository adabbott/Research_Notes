import pylibint
import h5py
import numpy as np
np.set_printoptions(linewidth=500)

pylibint.initialize("/home/adabbott/h2.xyz", 'cc-pvdz')
S = pylibint.overlap().reshape(10,10)
print(np.round(S,4))

print("writing overlap to disk")
pylibint.overlap_disk()
print("writing done")
pylibint.finalize()

print("Now reading HDF5 overlap integrals")
# Partial load test: question: does f['dataset'] load the whole thing in memory?
# claim is 'yes'  https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
with h5py.File('overlap.h5', 'r') as f:
    data_set = f['overlap']
    data = data_set[:]
    print(data.shape)
#print(data)

S = np.asarray(data).reshape(10,10)
print(np.round(S,4))


