Notes about permutationally invariant polynomials (PIPs)

PIPs are necessary for neural networks in the context of constructing PESs because of permutational symmetry in the input coordinates describing the molecular system.
As a simple example case, water has two hdyrogen atoms, which if there positions were permuted would yield the exact same molecule and therefore the same energy/gradient on the potential energy surface.
e.g.
O  
H1 1 1.2  
H2 1 1.0 2 103.5  

is the same as 

O  
H2 1 1.2  
H1 1 1.0 2 103.5  

But since neural networks have fixed nodes in the input vector, the NN is not aware of this symmetry. In principle the permutational symmetry can be learned by just expanding the training set (doubling it, in the case of water)
but this can greatly increase the cost of training. 
One approach is to used symmetrized neurons in the first hidden layer, but this is not favorable because it is difficult to generalize, and therefore is system dependent and requires manual specification. 


So instead, the geometry is expressed instead in the basis of permutationally invariant polynomials.
For water, one set of PIPs can be expressed by using the internuclear distance matrix as the coordinate representation, and reduce bond lengths to  

p_ij = exp(r_ij)  

Letting H1 be index 1, H2 be index 2, and O be index 3, the three PIPs can be expressed as  

(p_13 + p_23) / 2  
p_13p_23  
p_12  

which by inspection are all individually invariant if indices 1 and 2 are swapped.

A general method for generating these things is nontrivial. (i.e. find ANY permutation invariant polynomial set for molecular sysetms A_m B_n C_p ... Z_r)
Jiang and Guo 2013 supposedly generalize it but they explain it very poorly and give uninformative and trivial examples.

An alternative is using fundamental invariants, which are essentially the "basis set" of permutationally invariant polynomials,
The code [Singular](https://www.singular.uni-kl.de/index.php) is able to do this using the recently developed 
algorithm by [Simon King](https://www.sciencedirect.com/science/article/pii/S074771711200079X) 
This currently seems to be the most favorable method available. The code is open source and the particular library is [here](https://github.com/Singular/Sources/blob/aad4ca42fd2df029c3026bf93dfc208d7275cbdb/Singular/LIB/finvar.lib)
Interestingly, a lot of the people using PIPs do not mention fundamental invariants. [This work](http://aip.scitation.org/doi/full/10.1063/1.4961454) uses them, but the work of Bowman, Behler, Manzhos, Guo, etc do not even mention fundamental invariants.
