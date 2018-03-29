# driver script for auto generation of magma input files
import induced_permutations as ip
import os

def atomvec_to_molstring(vec):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    string = ''
    for i, a in enumerate(vec):
        string += letters[i]
        if a > 1:
            string += str(a)
    return string



# for 3 atoms up to n atoms 
os.mkdir("fundamental_invariants")
os.chdir("fundamental_invariants")
for system_size in range(3, 7):
    # make a directory for that number of atoms
    dirname = str(system_size) + "_atom_system"
    os.mkdir(dirname)
    atomvectors = ip.atom_combinations(system_size)
    
    os.chdir("./{}".format(dirname))
    for vec in atomvectors:
        bond_indice_permutations = ip.permute_bond_indices(vec)
        IP = ip.induced_permutations(vec, bond_indice_permutations)
        if IP != []:
            magmainput = ip.write_magma_input(sum(vec), IP)
            mol = atomvec_to_molstring(vec) 
            os.mkdir(mol)
            os.chdir(mol)
            with open("input.magma", "w") as f:
                f.write(magmainput)
            os.chdir("../")

    os.chdir("../")
os.chdir("../")
