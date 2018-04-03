# Notes about representation of molecules for ML

So I found [this repo](https://github.com/crcollins/molml) and [this paper](https://arxiv.org/pdf/1701.06649.pdf)
    The former of which is a python library for mapping molecular geometries to various "featurizations" and the latter of which
    is somewhat of a summary of these features.

The idea here is to construct a *feature vector*, which is just a valid input to a machine learning model, from a known molecule with a known geometry and properties,
The size of the feature vector depends only on the diversity of molecules in the data set, such as the number of unique elements and bond types, and not on the number of atoms.

Coulomb matrices are similar to interatomic distance matrices except the diagonal terms are 0.5 * Z ^ 2.4 and the off diagonal elements are nuclear repulsion terms as one would find in HF theory.
There are three main issues with using the Coulomb matrix as a vector input for ML:
 * First, the dimension of the CM depends on the 
number of atoms in the molecule. This is solved by introducing "ghost atoms" which are present in some molecules in the dataset but not in others. This forces all CM's over a dataset to be the exact same size, equal to the size of the CM of the largest molecule. 
 * Second, the ordering of the atoms is undefined. 
 * Third, there is no permutation invariance built in. 

Various ways to overcome these last two issues are nicely outlined in [this paper](https://pdfs.semanticscholar.org/5761/d22bb67798167a832879a473e0ece867df04.pdf)

An expansion of the coulomb matrix is the [Bag of Bonds](https://www.ncbi.nlm.nih.gov/pubmed/26113956) feature.
In this representation, off-diagonal coulomb matrix elements are divided into "bags" based on the type of coulomb interaction
involved (CC, OO, OH, HH) and values in each bag are sorted from high to low. The size of each bag is then adjusted
to the maximum size of each bag across all molecules in the dataset by adding zeroes to them, so all bags are the same length.
The bags are then concatenated to make a single feature vector which is invariant to reordering of atoms, and is valid 
for every molecule in the dataset, as it is always ordered in the same way and is the same size between different molecules.

"Connectivity counts" are a class of features which attempt to illustrate the bonding between atoms by counting 
the bonding patterns. Rank 1 features count atom types and/or coordination numbers, rank 2 features count bond types (single, double, etc), rank 3 count triplets of bonded atoms, rank 4 being 4 atom substructures, and so on.

Connectivity counts summarize bonding patterns but do not give quantitative information about the bonding.
"Encoded distances" allow for adding information about bond lengths to the connectivity information.
