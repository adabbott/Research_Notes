# This code aims to take a molecular system AnBmCp... and:
# 1. Determine the permutation operations of the permutation groups Sn, Sm, Sp ... 
# 2. Find the induced permutations of the permutation operations on the set of interatomic distances
# 3. Export Magma or Singular input code to derive the fundamental invariants

import numpy as np
import itertools as it
import math



def generate_permutations(k):
    """
    Generates an array of all possible orderings of k indices
    """
    f_k = math.factorial(k)
    # create an empty array to collect permutations
    A = np.empty((f_k, k), dtype=int)
    for i, perm in enumerate(it.permutations(range(k))):
        A[i,:] = perm
    return A

S2 = generate_permutations(2)
S3 = generate_permutations(3)
print(S2)

def find_cycles(perm):
    """
    Finds the cycle(s) required to get the permutation. For example,
    the permutation [3,1,2] is obtained by permuting [1,2,3] with the cycle [1,2,3]
    read as "1 goes to 2, 2 goes to 3, 3 goes to 1"
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
        if (len(cyc) > 1): 
            new_cycles.append(cyc)
    return new_cycles

for p in S3:
    print(find_cycles(p))
print(find_cycles(S3[0]))


# might need to change this order to column wise for easy sorting
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

def induced_permutations(atomtype_vector):
    """
    Find the induced permutations on interatomic distances from like atom permutations
    atomtype_vector: array-like
        A vector of the number of each atoms, the length is the total number of atoms.
        An A3B8C system would be [3, 8, 1]
    """
    natoms = sum(atomtype_vector)
    permutations_by_atom = [] 
    for atom in atomtype_vector:
        permutations_by_atom.append(generate_permutations(atom))
    print(permutations_by_atom) 


# represents an A2B2 system
atomtype_vector = [2, 2]
a = induced_permutations(atomtype_vector)
#print(a)
    
