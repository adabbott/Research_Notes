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

S3 = generate_permutations(3)
print(S3)

def find_permutation_cycles(perm):
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
        if len(cyc) > 1:
            new_cycles.append(cyc)
    return new_cycles

for p in S3:
    print(find_permutation_cycles(p))

