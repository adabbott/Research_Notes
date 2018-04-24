# Week of April 23rd


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
