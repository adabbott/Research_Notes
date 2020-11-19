import peslearn

input_string = ("""
               O 
               H 1 r1
               H 1 r2 2 a2 
            
               r1 = [0.90,1.00, 7]
               r2 = [0.90,1.00, 7]
               a2 = [100.0,120.0,7]

               energy = 'regex'
               energy_regex = 'Total Energy\s+=\s+(-\d+\.\d+)'
               gradient = cclib
               """)


input_obj = peslearn.InputProcessor(input_string)
mol = peslearn.datagen.Molecule(input_obj.zmat_string)
peslearn.utils.parsing_helper.parse(input_obj, mol)
