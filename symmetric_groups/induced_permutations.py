# This code aims to take an arbitrary molecular system AnBmCp... with any number of like atoms and:
# 1. Determine the atom permutation operations (cycles) of the permutation groups Sn, Sm, Sp ... 
# 2. Find the induced permutations of the atom permutation operations of Sn, Sm, Sp ...  on the set of interatomic distances
# 3. Export Magma code to derive the fundamental invariants
# Result: a generalized algorithm for obtaining a permutationally invariant basis for geometrical parameters so that the PES is permutation invariant

import numpy as np
import itertools as it
import math
import copy

def generate_permutations(k):
    """
    Generates a list of lists of all possible orderings of k indices
    """
    f_k = math.factorial(k)
    A = []
    for perm in (it.permutations(range(k))):
        A.append(list(perm)) 
    return A


def find_cycles(perm):
    """
    Finds the cycle(s) required to get the permutation. For example,
    the permutation [3,1,2] is obtained by permuting [1,2,3] with the cycle [1,2,3]
    read as "1 goes to 2, 2 goes to 3, 3 goes to 1".
    Sometimes cycles are products of more than one subcycle, e.g. (12)(34)(5678)
    This function is to find them all. Ripped bits and pieces of this off from SE, 
    don't completely understand it but it works :)
    """
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break
        cycles.append(cycle[::-1])

    # only save cycles of size 2 and larger
    cycles[:] = [cyc for cyc in cycles if len(cyc) > 1]
    return cycles



# might need to change this order to column wise for easier sorting. We'll see.
def generate_bond_indices(natoms):
    """
    natoms: int
        The number of atoms
    Finds the array of bond indices of an interatomic distance matrix, in row wise order:
    [[0,1], [0,2], [1,2], [0,3], [1,3], [2,3], ...,[0, natom], ...,[natom-1, natom]]
    """ 
    # initialize j as the number of atoms
    j = natoms - 1
    # now loop backward until you generate all bond indices 
    bond_indices = []
    while j > 0:
        i = j - 1
        while i >= 0:
            new = [i, j]
            bond_indices.insert(0, new)
            i -= 1
        j -= 1 
    return bond_indices

def molecular_cycles(atomtype_vector):
    """
    Finds the complete set of cycles that may act on a molecular system.
    Given an atomtype vector, containing the number of each atom:
         1.  generate the permutations of each atom
         2.  generate the cycles of each atom
         3.  adjust the indices to be nonoverlapping, so that each atom has a unique set of indices.
    For example, For an A2BC system, the indices may be assigned as follows: A 0,1; B 2; C 3; 
    while the methods generate_permutations and find_cycles index from 0 for every atom, so we adjust the indices of every atom appropriately
    """
    permutations_by_atom = [] 
    for atom in atomtype_vector:
        # add the set of permutations for each atom type to permutations_by_atom
        permutations_by_atom.append(generate_permutations(atom)) # an array of permutations is added for atom type X
    cycles_by_atom = [] 
    # each atom has a set of permutations, saved in permutations_by_atom 
    for i, perms in enumerate(permutations_by_atom):
        cycles = []
        # find the cycles of each permutation and append to cycles, then append cycles to cycles_by_atom
        for perm in perms:
            cyc = find_cycles(perm)
            if cyc:  # dont add empty cycles (identity permutation)
                cycles.append(cyc)
        cycles_by_atom.append(cycles)
    # now update the indices of the second atom through the last atom since they are currently indexed from zero
    # to do this we need to know the number of previous atoms, num_prev_atoms
    atomidx = 0
    num_prev_atoms = 0
    for atom in cycles_by_atom[1:]:
        num_prev_atoms += atomtype_vector[atomidx]
        for cycle in atom:
            for subcycle in cycle: # some cycles are composed of two or more subcycles (12)(34) etc.
                for i, idx in enumerate(subcycle): 
                    subcycle[i] = idx + num_prev_atoms
        atomidx += 1
    return cycles_by_atom


def permute_bond(bond, cycle):
    """
    Permutes a bond inidice if the bond indice is affected by the permutation cycle.
    There is certainly a better way to code this. Yikes.
    """
    count0 = 0
    count1 = 0
    # if the bond indice matches the cycle indice, set the bond indice equal to the next indice in the cycle
    # we count so we dont change a bond indice more than once.
    # If the cycle indice is at the end of the list, the bond indice should become the first element of the list since thats how cycles work.
    # theres probably a better way to have a list go back to the beginning
    for i, idx in enumerate(cycle):
        if (bond[0] == idx) and (count0 == 0):
            try:
                bond[0] = cycle[i+1]
            except:
                bond[0] = cycle[0]
            count0 += 1

        if (bond[1] == idx) and (count1 == 0):
            try:
                bond[1] = cycle[i+1]
            except:
                bond[1] = cycle[0]
            count1 += 1
    # sort if the permutation messed up the order. if you convert 1,2 to 2,1, for example    
    bond.sort()
    return bond 
   
def permute_bond_indices(atomtype_vector):
    """
    Permutes the set of bond indices of a molecule according to the complete set of valid molecular permutation cycles
    atomtype_vector: array-like
        A vector of the number of each atoms, the length is the total number of atoms.
        An A3B8C system would be [3, 8, 1]
    Returns many sets permuted bond indices, the number of which equal to the number of cycles
    """
    natoms = sum(atomtype_vector) 
    bond_indices = generate_bond_indices(natoms)    
    cycles_by_atom = molecular_cycles(atomtype_vector)
         
    bond_indice_permutations = [] # interatomic distance matrix permutations
    for atom in cycles_by_atom:
        for cycle in atom:
            tmp_bond_indices = copy.deepcopy(bond_indices) # need a deep copy, list of lists
            for subcycle in cycle:
                for i, bond in enumerate(tmp_bond_indices):
                    tmp_bond_indices[i] = permute_bond(bond, subcycle)
            bond_indice_permutations.append(tmp_bond_indices) 

    return bond_indice_permutations 

def induced_permutations(atomtype_vector, bond_indice_permutations):
    """
    Given the original bond indices list [[0,1],[0,2],[1,2]...] and a permutation of this bond indices list,
    find the permutation vector that maps the original to the permuted list. 
    Do this for all permutations of the bond indices list. 
    Result: The complete set induced interatomic distance matrix permutatations caused by the molecular permutation cycles 
    """
    natoms = sum(atomtype_vector) 
    bond_indices = generate_bond_indices(natoms)    
   
    induced_perms = [] 
    for bip in bond_indice_permutations:
        perm = []
        for bond1 in bond_indices:
            for i, bond2 in enumerate(bip):
                if bond1 == bond2:
                    perm.append(i)
        cycle = find_cycles(perm) 
        induced_perms.append(cycle)
    return induced_perms
                
def write_magma_input(natoms, induced_perms):
    """
    Writes a magma input file which can be copy and pasted 
    to the Magma online code editor for obtaining the fundamental invariants
    This has been tested against every result in the SI of Shao, Chen, Zhao, Zhang, J Chem Phys 145 2016
    https://aip.scitation.org/doi/suppl/10.1063/1.4961454
    For A2B, A2B2, A3B, A4B
    """
    A = []
    nbonds = int((natoms**2 - natoms) / 2)
    for i in range(nbonds):
        A.append(i)

    operators = ''
    for cycle in induced_perms:
        for subcycle in cycle:
            operators += str(tuple(subcycle))
        # add comma and space except at end
        if cycle != induced_perms[-1]:
            operators += ', '
    line1 = "K := RationalField();\n"  
    line2 = "X := {};\n".format(set(A))
    line3 = "G := PermutationGroup<X | {}>;\n".format(operators)
    last = "R := InvariantRing(G,K);\nFundamentalInvariants(R);"
    
    return (line1 + line2 + line3 + last)


def process_magma_output(string):
    """
    Takes as argument a multiline string of the Magma output and creates a Python list
    of fundamental invariants as strings
    """
    # fix exponents, remove brackets if user included brackets
    string = string.replace('^', '**').replace('[', '').replace(']', '')
    # each fundamental invariant is separated by a comma
    tmp = [FI for FI in string.split(',')] 
    # clean up whitespace
    fi_list = [re.sub('\s+', ' ', FI).strip() for FI in tmp]
    return fi_list

             

# use this to test:
atomtype_vector = [9]
bond_indice_permutations = permute_bond_indices(atomtype_vector)
IP  = induced_permutations(atomtype_vector, bond_indice_permutations)
a = write_magma_input(sum(atomtype_vector), IP)
print(a)
