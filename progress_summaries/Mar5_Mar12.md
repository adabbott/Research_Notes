# Weeks of March 5th and 12th

* Began tensorflow online tutorial

* Learned how to use Singular and Magma to generate fundamental invariants of a polynomial ring, see [these notes](notes/FI.md)

* Learned how to go from permutation operations on a molecular system A<sub>p</sub>B<sub>q</sub>C<sub>r</sub> ... X<sub>n</sub> to the corresponding induced permutations on bond distances in the interatomic distance matrix by hand. Have yet to generalize programmatically.

* Generated [initial code](Molssi_projectnotes/symmetric_groups/symmetric_group_permutations.py) for fundamental invariants on the project notes repo. Currently is capable of: 
    - Generating permutations of k indices
    - Computing the matrix representations of all operators in an arbitrary permutation group S<sub>n</sub> 
    - Computing the direct product between two permutation groups S<sub>n</sub> and S<sub>m</sub>
    - Computing a direct product between an arbitrary number of permutation groups 

* Began adding output file parsing functionality to main repo

