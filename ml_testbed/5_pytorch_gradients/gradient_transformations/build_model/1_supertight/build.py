import peslearn

input_string = ("""
               O 
               H 1 r1
               H 1 r2 2 a2 
            
               # +/- 0.005 from equilibrium, +/- 0.03 for angle
               r1 = [0.944630647,0.9546306473,5] 
               r2 = [0.944630647,0.9546306473,5] 
               a2 = [111.5153723625, 111.5753723625, 5]

               energy = 'regex'
               energy_regex = 'Total Energy\s+=\s+(-\d+\.\d+)'
               hp_maxit = 15
               training_points = 100
               remove_redundancy = true
               use_pips = true 
               sampling = structure_based
               nn_precision = 64
               eq_geom = [0.9496306473, 0.9496306473,111.5453723625]
               """)

input_obj = peslearn.InputProcessor(input_string)
template_obj = peslearn.datagen.Template("./template.dat")
mol = peslearn.datagen.Molecule(input_obj.zmat_string)
config = peslearn.datagen.ConfigurationSpace(mol, input_obj)
config.generate_PES(template_obj)

# run single point energies with Psi4
import os
os.chdir("PES_data")
dirs = [i for i in os.listdir(".") if os.path.isdir(i) ]
for d in dirs:
    os.chdir(d)
    if "output.dat" not in os.listdir('.'):
        print(d, end=', ')
        os.system("psi4 input.dat")
    os.chdir("../")
os.chdir("../")

print('\nParsing ab initio data...')
peslearn.utils.parsing_helper.parse(input_obj, mol)

print('\nBeginning NN optimization...')
nn = peslearn.ml.NeuralNetwork("PES.dat", input_obj, 'A2B')
nn.optimize_model()
