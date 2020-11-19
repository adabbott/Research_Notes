import peslearn

grid_sizes = ['20', '40', '60', '80', '100']

for grid_size in grid_sizes:
    input_string = ("""
                   O 
                   C 1 r1
                   H 2 r2 1 a1
                   H 2 r3 1 a1 3 d1
                
                   r1 = [0.85,1.20, 10]
                   r2 = [0.85,1.20, 10]
                   a2 = [90.0,120.0, 10]
    
                   energy = 'regex'
                   energy_regex = 'Total Energy\s+=\s+(-\d+\.\d+)'
                   hp_maxit = 10
                   use_pips = true
                   sort_pes = true
                   grid_reduction = {}
                   training_points = {}
                   sampling = structure_based
                   """.format(grid_size, round(int(grid_size)*0.8)))

    input_obj = peslearn.InputProcessor(input_string)
    template_obj = peslearn.datagen.Template("./template.dat")
    mol = peslearn.datagen.Molecule(input_obj.zmat_string)
    config = peslearn.datagen.ConfigurationSpace(mol, input_obj)
    config.generate_PES(template_obj)
    # run single point energies with Psi4 which have not been computed already
    import os
    os.chdir("PES_data")
    dirs = [i for i in os.listdir(".") if os.path.isdir(i) ]
    dirs = sorted(dirs, key=lambda x: int(x))
    for d in dirs:
        os.chdir(d)
        print(d)
        if "output.dat" not in os.listdir('.'):
            os.system("psi4 input.dat")
        os.chdir("../")
    os.chdir("../")

    print('\nParsing ab initio data...')
    peslearn.utils.parsing_helper.parse(input_obj, mol)

    gp = peslearn.ml.gaussian_process.GaussianProcess("PES.dat", input_obj, 'A2B')
    gp.optimize_model()
    error_cm = round(219474.63 * gp.test_error,2)
    if error_cm > 5:
        continue
    else:
        break
    
