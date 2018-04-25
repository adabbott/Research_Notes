# Week of April 23rd

* Continued writing of phase-II proposal

* Improved on my initial attempt at machine learning O2 potential energy curve. 100 data points, 33 train, 33 validate, and 34 test. Achieved MAE of the test set  on the order of millihartree (i.e., predicted values on the order of 100.000 correct to the third decimal place)

* Enabled keyword usage in input files for specifying parsing routines
    * `energy = cclib` or `= regex`
    * `gradient = cclib` or `= regex`
    * options to input regular expressions and select cclib methods

* Redesigned driver structure
    - driver.py is now in same directory as other modules
    - now uses keywords obtained from input file to choose parsing routine
    - added exceptions for faulty user input  
    - now only saves one file as a csv

* Created documentation for data generation

* Reformatted examples 

* Renamed repository and software to MLChem

* Read about [Machine Learning Automated Algorithm Design](http://www.ml4aad.org/),  looked into software package capabilities.
