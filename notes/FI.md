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

For an A<sub>2</sub>B<sub>2</sub> system, we have the symmetric group S<sub>2</sub> x S<sub>2</sub> (direct product).
Now we have 6 bonds (r12, r13, r14, r23, r24, r34) which we denote (x1, x2, x3, x4, x5, x6).
x1 and x6 are unaffected by permutations of like atoms, however the subset (x2, x3, x4, x5) have the permutations
(23)(45), (24)(35), (25)(34) (which can be seen by taking the tensor product of the matrix representations of the operators of S2 x S2)  

The magma code is:
```
K := RationalField();
X := {1,2,3,4,5,6};
G := PermutationGroup<X | (2,3)(4,5), (2,4)(3,5), (2,5)(3,4)>;
R := InvariantRing(G,K);
FundamentalInvariants(R);
```
and the fundamental invariants are 
```
[
    x1,
    x2 + x3 + x4 + x5,
    x6,
    x2^2 + x3^2 + x4^2 + x5^2,
    x2*x3 + x4*x5,
    x2*x4 + x3*x5,
    x2^3 + x3^3 + x4^3 + x5^3
]
```

The [Singular code](https://www.singular.uni-kl.de/) (click "Try Online" tab for online code editor) for this operation uses the matrix representations of the operators instead of the parenthetical notation for the permutations:
```
ring R=0,(x2,x3,x4,x5),dp;
matrix M1[4][4]=0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0;
matrix M2[4][4]=0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0;
matrix M3[4][4]=0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0;
list L=group_reynolds(M1, M2, M3);
matrix G = invariant_algebra_reynolds(L[1],L[2],L[3],1);
G;
```
and the output is similar, except singular appears to exclude trivial cases:
```
G[1,1]=x2+x3+x4+x5
G[1,2]=x2^2+x3^2+x4^2+x5^2
G[1,3]=x2*x3+x4*x5
G[1,4]=x2*x4+x3*x5
G[1,5]=x2^3+x3^3+x4^3+x5^3
```
