# Week of March 19th 

* Nearly finished with a [generalized algorithm](https://github.com/adabbott/Molssi_projectnotes/blob/master/symmetric_groups/induced_permutations.py) for generating the Magma or Singular code
necessary for obtaining all fundamental invariants (or primary and secondary invariants) of an arbitrary molecular system. 

* Once finished, this would seem to be a good contribution. The only other software that I'm aware of that can generally derive a permutation invariant basis for any molecular system is Xie and Bowman's [MSA](https://scholarblogs.emory.edu/bowman/msa/) code, but this does not plug in nicely as a module, as it uses F90, Python 2, and Perl, and requires an input file for using it. Appears to be meant for use as a standalone code for fitting PESs. It also does not determine primary, secondary, or fundamental invariants, but instead uses monomial symmetrization [(see associated paper by Xie and Bowman)](https://pubs.acs.org/doi/abs/10.1021/ct9004917) to derive *some* invariant polynomial basis which appears to be absurdly large compared to the required amount of invariants. For example, for an A2B2 system, the symmetrized monomial basis is nearly 300 polynomials, while alternatively one can use a basis of 6 primary invariants and 2 secondary invariants, or just the 7 fundamental invariants. In terms of computational efficiency and simplicity, the later bases are preferable. I suppose the advantage is that their implementation is easier and does not rely on a computer algebra package. 

Algorithm is structured as follows:
    1. Input a string of a molecule "A<sub>2</sub>B<sub>2</sub>C<sub>4</sub>B<sub>2</sub>..." and order it as "A<sub>i</sub>B<sub>j</sub>C<sub>k</sub>..."
    2. Generate all permutation operations (cycles) of the symmetric groups S<sub>i</sub>, S<sub>j</sub>, S<sub>k</sub>...
    3. Generate the interatomic distance matrix elements r<sub>mn</sub> and a list of indice pairs
    4. Map the indices of the permutation operations of each atom type to new indice values corresponding to the range of interatomic distance matrix element indices. 
      (e.g. atom C might have the permutation operation (1,2) but by assigning 1 through i + j to atoms of type A and B, this operation (1,2) must be mapped to the absolute indices (1+j+k,2+j+k))
    5. Operate all permutation cycles in S<sub>i</sub>, S<sub>j</sub>, S<sub>k</sub>... onto the list of bond indices
    6. Determine the induced permutations of the interatomic distances
    7. Save the variable list r<sub>12</sub>,r<sub>13</sub>, r<sub>23</sub> and induced permutations on these bond distances
    8. Write magma or Singular code for finding the fundamental invariant basis of the geometry parameters, which can be submitted to online code editors on their respective websites.

* The result of this algorithm is hopefully temporary. Ideally, the invariants would be derived by either a local implementation or importing a library that can do it.
These computer algebra codes are the only resources that I have found that can derive fundamental invariants. I may need to contact the guy who generalized the method and coded the implementations in Magma and Singular and see if he has a Python implementation laying around. Also Sage may have this feature, and Sage plays nicely with Python, so this also may be an option.
