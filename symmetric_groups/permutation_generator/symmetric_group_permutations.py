# this is a script for computing the permutations of an arbitrary tensor product 
# of the symmetric groups S_n. This is needed for generating fundamental invariants with Singular or Magma.
# For example, for an A_2 B_2 molecular system, we have S_2 x S_2 as the permutation group 
# and internuclear distances r_12 r_13 r_14 r_23 r_24 r_34 denoted as x1 x2 x3 x4 x5 x6.
# x1 and x6 are invariant to permutations, but x2-x5 belong to the permutation group
# The matrix representations of the operations under S_2 are the 2x2 identity matrix and the matrix 
# transformation corresponding to a permutation operation (12) expressed as  [ 0 1 ]
#                                                                            [ 1 0 ]
# taking the tensor product S_2 x S_2 we get 4 different matrix representations, one of which being the 4x4 identity.
# the other three are the matrix representations of the permutation operations (x2 x3)(x4 x5), (x2 x4)(x3 x5), and (x2 x5)(x3 x4)
# the goal of this script is to generalize this generation of permutation operations for obtaining fundamental invariants

import numpy as np
import itertools as it
import math


def number_of_bonds(n_atoms):
    return (n_atoms ** 2 - n_atoms) / 2

def generate_permutations(k):
    """
    Generates an array of all possible orderings of k indices
    """
    f_k=math.factorial(k)
    A=np.empty((f_k,k))
    for i,perm in enumerate(it.permutations(range(k))):
        A[i,:] = perm
    A+=1
    return A

S1 = generate_permutations(1)
S2 = generate_permutations(2)
S3 = generate_permutations(3)

def compute_matrix_representations(S):
    """
    Finds the matrix representations of the operations associated with the symmetric group 
    Returns them as a list of numpy arrays
    """
    matrix_representations = []
    base_permutation = S[0]
    for perm in S[0:]:
        size = len(S[0])
        matrix_rep = np.zeros((size, size))
        for row_idx in range(size):
            for col_idx in range(size):
                if perm[col_idx] == base_permutation[row_idx]:
                    matrix_rep[row_idx,col_idx] = 1 
        matrix_representations.append(matrix_rep)
    return matrix_representations

S1 = compute_matrix_representations(S1)
S2 = compute_matrix_representations(S2)
S3 = compute_matrix_representations(S3)

def kronecker_product(G1, G2):
    """
    Finds the full tensor product of the matrix representations of the operators of two groups
    """
    tensor_product = []
    for matrix_op1 in G1:
        for matrix_op2 in G2:  
            # does order of tensor product matter, as long as its consistent?
            tensor_product.append(np.kron(matrix_op1, matrix_op2))
    return tensor_product

def combine_groups(list_of_groups):
    """
    Take some list of groups Sn Sm Sp Sq... Sz which are each lists of matrix representations of the operations in Sn
    and does a complete tensor product of the entire set into one group
    """
    if len(list_of_groups) == 1:
        raise ValueError("Only one group given as input")

    # number of groups we are combining as a tensor product
    count = len(list_of_groups)
    supergroup = kronecker_product(list_of_groups[0],list_of_groups[1])
    count -= 2
    while count > 0:
        for i in range(2, len(list_of_groups)):
            supergroup = kronecker_product(supergroup, list_of_groups[i])
            count -= 1
    return supergroup 

S2xS2xS2 = combine_groups([S2, S2, S2])
for i in S2xS2xS2:
    print(i)
#def generate_permutation_tuples(group):
#    """
#    Takes a list of the matrix representations of operators of a group and finds the corresponding permutation operator.
#    e.g. the matrix acting on a two level system [0 1] permutes x1 and x2, which we denote as the operation (12), programmatically expressed as (1,2)
#                                                 [1 0]
#    """
#for i in S2:
#    print(i) 
#
#S2xS2 = kronecker_product(S2, S2)
#for i in S2xS2:
#    print(i)
#S3xS3 = kronecker_product(S3, S3)
#for i in S3xS3:
#    print(i)

#S3xS1 = kronecker_product(S3, S1)
#for i in S3xS1:
#    print(i)
#next : generate all possible kronecker products (tensor products) between 2, 3, or n symmetric groups matrix representation operators
#question: do i need to include the identity?
# then: you must convert these matrix representations into tuples representing the permutations  

# we will need to know which bond distances x_n are included in the permutation space.
# as a constraint, we know the number of x_n in the permutation space is equal to the dimension of the permutation group matrix operators

# For a bond distance to be in the permutation space, one of two mutually exclusive conditions must be satisfied:
#   1. Both indices of the bond distance r_nm have only one atom of the each kind (e.g. C--F distance in CH3F)
#   2. The two indices of the bond distance are the same kind of atom, but the only two (e.g. H--H distance in H2CN)

