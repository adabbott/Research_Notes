# This code aims to take an arbitrary molecular system AnBmCp... with any number of like atoms and:
# 1. Determine the atom permutation operations of the permutation groups Sn, Sm, Sp ... 
# 2. Find the induced permutations of the atom permutation operations of Sn, Sm, Sp ...  on the set of interatomic distances
# 3. Export Magma or Singular input code to derive the fundamental invariants
# Result: a generalized algorithm for obtaining a permutationally invariant basis for geometrical parameters so that the PES is permutation invariant

import numpy as np
import itertools as it
import math

def generate_permutations(k):
    """
    Generates a list of lists of all possible orderings of k indices
    """
    f_k = math.factorial(k)
    # create an empty array to collect permutations
    #A = np.empty((f_k, k), dtype=int)
    A = []
    for i, perm in enumerate(it.permutations(range(k))):
        #A[i,:] = perm
        A.append(list(perm)) 
    return A

S2 = generate_permutations(2)
S3 = generate_permutations(3)
#print(S3)

def find_cycles(perm):
    """
    Finds the cycle(s) required to get the permutation. For example,
    the permutation [3,1,2] is obtained by permuting [1,2,3] with the cycle [1,2,3]
    read as "1 goes to 2, 2 goes to 3, 3 goes to 1"
    Sometimes cycles are products of more than one subcycle, e.g. (12)(34)(5678)
    This function is to find them all. Ripped bits and pieces of this off from SE, 
    don't completely understand it but it works xD
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
    new_cycles = []
    for cyc in cycles:
        if len(cyc) > 1: 
            new_cycles.append(cyc)
    return new_cycles


# might need to change this order to column wise for easier sorting. We'll see.
def find_bond_indices(natoms):
    """
    natoms: int
        The number of atoms
    Finds the array of bond indices of an interatomic distance matrix, in row wise order:
    [[0,1], [0,2], [1,2], [0,3], [1,3], [2,3], ...,[0, natom], ...,[natom-1, natom]]
    """ 
    # initialize j as the number of atoms
    j = natoms
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

def adjust_permutation_indices(atomtype_vector):
    """
    Given an atomtype vector, containing the number of each atom, generate the permutations of each atom,
    and then generate the cycles of each atom, and finally adjust the indices to be nonoverlapping, so that each atom has a unique set of indices.
    For example, For an A2BC system, the indices may be assigned as follows: A 0,1; B 2; C 3; 
    This needs to be done because the functions generate_permutations and find_cycles index from 0 for every atom.
    This way, we can permute bond distance subscripts according to the correct permutation indices.
    """
    permutations_by_atom = [] 
    for atom in atomtype_vector:
        # add the set of permutations for each atom type to permutations_by_atom
        permutations_by_atom.append(generate_permutations(atom)) # an array of permutations is added for atom type X
    print(permutations_by_atom)
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

# represents an A4B2 system
atomtype_vector = [2,2,2,2]
a = adjust_permutation_indices(atomtype_vector)
print(a)

b = find_bond_indices(8)
print(b)
   
def bond_distance_permutations(atomtype_vector):
    """
    Find the effect permutations on interatomic distances from like atom permutations
    atomtype_vector: array-like
        A vector of the number of each atoms, the length is the total number of atoms.
        An A3B8C system would be [3, 8, 1]
    """
    natoms = sum(atomtype_vector) 
    cycles_by_atom = adjust_permutation_indices(atomtype_vector)
    bond_indices = find_bond_indices(natoms)    
         
    # for every atom permutation cycle
    for atom in cycles_by_atom:
        for cycle in atom:
            for subcycle in cycle:
                for i,idx in enumerate(subcycle): 
                    # for every single bond index of the interatomic distance matrix bond distances... 
                    for bond in bond_indices:
                        for bond_idx in bond:
                        # if the permutation cycle index corresponds to the bond atom index,
                        # perform the permutation
                            if idx == bond_idx:
                                #do something
                                



