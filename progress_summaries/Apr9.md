# Week of April 9th

* Finished data generation driver, running multiple examples. User-side, it works in the following manner:
    * User is required to supply a `input.dat` which contains internal coordinates for their system, with the desired ranges they wish to sample over in the form `r1 = [start, stop, n_points]` or just a fixed value `a1 = 104.5`.
    * User is also required to supply a template input file, `template.dat` which is an xyz input file to compute an energy and/or gradient from a software of their choosing.
    * From there it is a matter of running the driver `python driver.py` and choosing 'generate' to create input files  or 'parse' to obtain the data

* In terms of how to code works:
    * From the internal coordinate file `input.dat`, it creates an InputProcessor object, which uses regex to extract the displacments, and creates a list displacement dictionaries of the form `"parameter": value` for all internal coordinate parameters.
    * From the template input file `template.dat`, create a TemplateProcessor object, which splits the file string into sections: before the cartesian geometry, the cartesian geometry, and after the cartesian geometry. Each string is saved as a class attribute.  
    * Initialize a Molecule object with the internal coordinates in `input.dat`. 
    * For every displacement, update the Molecule with new internal coordinates, convert to cartesians, and write a new input file in its own directory
    * Once input files are run, for every output file, create an OutputFile object, and parse data either with supplied regular expressions or cclib, store in a pandas DataFrame
    * Write PES to a file

* Debugged a variety of input sensitivities with the data generation code

* Read a new paper, [Deep Learning for Computational Chemistry](https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.24764)
