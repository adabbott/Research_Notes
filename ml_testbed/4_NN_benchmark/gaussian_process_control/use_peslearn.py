import peslearn
import numpy as np
import pandas as pd

#input_string = ("""
#               O
#               H 1 r1
#               H 1 r2 2 a1
#               training_points = 950
#               sampling = random
#               hp_max_evals = 10
#               use_pips = true
#               """)
#
#input_obj = peslearn.input_processor.InputProcessor(input_string)
#mol = peslearn.molecule.Molecule(input_obj.zmat_string)
#gp = peslearn.gaussian_process.GaussianProcess("../datasets/h2o", input_obj, mol)
#gp.optimize_model()
#
#
#input_string = ("""
#               O
#               H 1 r1
#               H 1 r2 2 a1
#               H 1 r2 2 a1 3 d1
#               training_points = 950
#               sampling = random
#               hp_max_evals = 10
#               """)
#
#input_obj = peslearn.input_processor.InputProcessor(input_string)
#mol = peslearn.molecule.Molecule(input_obj.zmat_string)
#gp = peslearn.gaussian_process.GaussianProcess("../datasets/h3o", input_obj, mol)
#gp.optimize_model()
#
#
input_string = ("""
               H
               H 1 r1
               C 1 r2 2 a1
               O 1 r2 2 a1 3 d1
               training_points = 950
               sampling = random
               hp_max_evals = 10
               """)

input_obj = peslearn.input_processor.InputProcessor(input_string)
mol = peslearn.molecule.Molecule(input_obj.zmat_string)
gp = peslearn.gaussian_process.GaussianProcess("../datasets/h2co", input_obj, mol)
gp.optimize_model()

#input_string = ("""
#               O
#               C 1 r1
#               H 1 r2 2 a1
#               C 1 r3 2 a2 3 d1
#               O 1 r4 2 a3 3 d2
#               training_points = 950
#               sampling = random
#               hp_max_evals = 10
#               """)
#
#input_obj = peslearn.input_processor.InputProcessor(input_string)
#mol = peslearn.molecule.Molecule(input_obj.zmat_string)
#gp = peslearn.gaussian_process.GaussianProcess("../datasets/ochco", input_obj, mol)
#gp.optimize_model()
