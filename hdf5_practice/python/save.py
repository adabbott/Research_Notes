import h5py
import numpy as np

N = 50
n = 3

# faux hessian
#arr = np.ones((N, N, N, N, n, n))

#use 'a' to add to a file 
#with h5py.File('random.hdf5', 'w') as f:
#    dset = f.create_dataset("dataset", data=arr)

f = h5py.File('eri_derivs.h5', 'w')
dset = f.create_dataset("test", (N,N,N,N,n,n))
K = np.random.rand(N,N,N,N)

for i in range(n):
    for j in range(n):
        dset[:,:,:,:,i,j] = K 

f.close()


