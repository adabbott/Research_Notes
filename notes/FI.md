# Fundamental Invariants #


I have successfully been able to replicate the results from the supplementary
material in [this paper](https://aip.scitation.org/doi/suppl/10.1063/1.4961454)
using the [online magma computer algebra system](http://magma.maths.usyd.edu.au/calc/) which is very promising.


For an A<sub>2</sub>B system, such as H<sub>2</sub>O, we construct a permutation group
over the rational field to create a ring containing the fundamental invariants.

The magma code is as follows:  
```
K := RationalField();
G := PermutationGroup<3 | (1,2)>;
R := InvariantRing(G,K); 
FundamentalInvariants(R);
```

which generates the fundamental invariant basis for expressing the 
molecular geometry (as an interatomic distance matrix) in a permutationally invariant manner:
```
[
    x1 + x2,
    x3,
    x1^2 + x2^2
]
```

where `x1` and `x2` are the bonds OH<sub>1</sub> and OH<sub>2</sub>.

For an A<sub>2</sub>B<sub>2</sub> system, we have the symmetric group S<sub>2</sub> x S<sub>2</sub> 
