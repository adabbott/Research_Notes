# Week of April 2nd 

* found a [python github project](https://github.com/crcollins/molml) which automates the mapping of molecular geometries to features commonly used in ML. This will be useful.

* found another [python github project](https://github.com/hachmannlab/chemml) which claims to be a library of machine learning methods for chemistry applications. Looks like its underdeveloped and the complete opposite of user-friendly,
but they have some ML modules which may be useful: various featurization classes, and wrappers for Keras/scikitlearn neural networks

* Reviewed featurization methods for chemical machine learning (see notes)

* Designed and began programming part of code relevant to data generation workflows 
    * The user may input a Z-Matrix (internal coordinates) of a molecule with specified ranges and other options in the input file
    * The software will extract the Z-Matrix and create Atom class instances for each atom and a Molecule class instance for the whole molecule
    * can now transform internals to cartesian coordinates 
    * sketched out initial driver and input_processor code

* Finished poster 
