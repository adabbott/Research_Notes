import h5py
import numpy as np


# Partial load test: question: does f['dataset'] load the whole thing in memory?
# claim is 'yes'  https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
with h5py.File('random.hdf5', 'r') as f:
    data_set = f['dataset']
    data = data_set[:,:,:,:,:,:]
    print(data.shape)

