# Fundamental Invariants #


I have successfully been able to replicate the results from the supplementary
material in [this paper](https://aip.scitation.org/doi/suppl/10.1063/1.4961454)
using the [online magma computer algebra system](http://magma.maths.usyd.edu.au/calc/) which is very promising.


For an A~2~B system, such as H~2~O, we construct a permutation group
over the rational field to create a ring containing the fundamental invariants.

The magma code is as follows:  
```K := RationalField();
G := PermutationGroup<3 | (1,2)>;
R := InvariantRing(G,K); 
FundamentalInvariants(R);```
