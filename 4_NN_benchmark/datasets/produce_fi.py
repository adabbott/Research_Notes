import peslearn
import numpy as np
import pandas as pd

input_string = ("""
               O
               H 1 r1
               H 1 r2 2 a1
               """)

path = "h2o"

input_obj = peslearn.input_processor.InputProcessor(input_string)
mol = peslearn.molecule.Molecule(input_obj.zmat_string)
gp = peslearn.gaussian_process.GaussianProcess(path, input_obj, mol)
# transform to permutation invariant geometry
params = {'morse_transform': {'morse': False}, 'pip': {'pip': True, 'degree_reduction': False}, 'scale_X': None, 'scale_y': None}
fi_X, y, junk1, junk2 = gp.preprocess(params, gp.raw_X, gp.raw_y)
data = np.hstack((fi_X, y))
col = list(np.arange(fi_X.shape[1]))
col.append('E')
df = pd.DataFrame(data, columns=col)
df.to_csv(path + "_fi", sep=',', index=False, float_format='%12.12f')

