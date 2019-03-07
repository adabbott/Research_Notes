# driver script for auto generation of magma input files
import induced_permutations as ip
import time
import os
import math
from numpy import prod
import subprocess
import psutil

def atomvec_to_molstring(vec):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    string = ''
    for i, a in enumerate(vec):
        string += letters[i]
        if a > 1:
            string += str(a)
    return string

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


# for 3 atoms up to n atoms 
if not os.path.exists("./fundamental_invariants"):
    os.mkdir("fundamental_invariants")
os.chdir("fundamental_invariants")
b = 1
a = 0
for system_size in range(3, 9):
    # make a directory for that number of atoms
    dirname = str(system_size) + "_atom_system"
    os.mkdir(dirname)
    atomvectors = ip.atom_combinations(system_size)
    
    os.chdir("./{}".format(dirname))
    for vec in atomvectors:
        print(vec)
        #bond_indice_permutations = ip.permute_bond_indices(vec)
        #IP = ip.induced_permutations(vec, bond_indice_permutations)
        #if IP != []:
        #    #permutation_order = prod([math.factorial(i) for i in vec])
        #    #if permutation_order < 10000:
        #    if 1 == 1:
        #        singularinput = ip.write_singular_input(sum(vec), IP)
        #        mol = atomvec_to_molstring(vec) 
        #        os.mkdir(mol)
        #        os.chdir(mol)
        #        with open("singular.inp", "w") as f:
        #            f.write(singularinput)
        #        proc = subprocess.Popen(["Singular -q singular.inp >> output"], shell=True)
        #        b = time.time()
        #        try:
        #            # kill Singular if it takes longer than 1 hour to run
        #            proc.wait(timeout=7200)
        #        except subprocess.TimeoutExpired:
        #            kill(proc.pid)
        #        os.chdir("../")

    os.chdir("../")
os.chdir("../")
