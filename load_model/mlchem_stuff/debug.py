import MLChem

# Local PES we wish to study
input_string = ("""
               O
               H 1 r1
               H 1 r2 2 a1
            
               r1 = [0.9, 1.2, 7]
               r2 = [0.9, 1.2, 7]
               a1 = [100, 110, 3]

               energy = 'regex'
               energy_regex = 'Final Energy:\s+(-\d+\.\d+)'
               hp_max_evals = 100
               """)

input_obj = MLChem.input_processor.InputProcessor(input_string)
template_obj = MLChem.template_processor.TemplateProcessor("./template.dat")
mol = MLChem.molecule.Molecule(input_obj.zmat_string)

# GENERATE
#config = MLChem.configuration_space.ConfigurationSpace(mol, input_obj)
#config.generate_PES(template_obj)

# PARSE
#MLChem.parsing_helper.parse(input_obj, mol)

# LEARN
gp = MLChem.gaussian_process.GaussianProcess("PES.dat", 25, input_obj, mol)
gp.optimize_model()

