# Week of March 19th 

* Nearly finished with a generalized algorithm for generating the Magma or Singular code
necessary for obtaining all fundamental invariants of an arbitrary molecular system. 
Algorithm is structured as follows:
    * Input a string of a molecule "A<sub>2</sub>B<sub>2</sub>C<sub>4</sub>B<sub>2</sub>..." and order it as "A<sub>i</sub>B<sub>j</sub>C<sub>k</sub>..."
    * Generate all permutation operations (cycles) of the symmetric groups S<sub>i</sub>, S<sub>j</sub>, S<sub>k</sub>...
    * Generate the interatomic distance matrix elements r<sub>mn</sub> and a list of indice pairs
    * Map the indices of the permutation operations of each atom type to new indice values corresponding to the range of interatomic distance matrix element indices. 
        *  (e.g. atom C might have the permutation operation (1,2) but by assigning 1 through i + j to atoms of type A and B, this operation (1,2) must be mapped to the absolute indices (1+j+k,2+j+k))
    * Operate all permutation cycles in S<sub>i</sub>, S<sub>j</sub>, S<sub>k</sub>... onto the list of bond indices
    * Determine the induced permutations of the interatomic distances
    * Save the variable list r<sub>12</sub>,r<sub>13</sub>, r<sub>23</sub> and induced permutations on these bond distances
    * Write magma or Singular code for finding the fundamental invariant basis of the geometry parameters 

* The result of this algorithm is hopefully temporary. Ideally, the invariants would be derived by either a local implementation or importing a library that can do it.
These computer algebra codes are the only resources that I have found that can derive fundamental invariants. I may need to contact the guy who generalized the method and coded the implementations in Magma and Singular and see if he has a Python implementation laying around.
