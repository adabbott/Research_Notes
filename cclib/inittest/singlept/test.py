# cclib cloned from Github, installed with:
# python setup.py build
# python setup.py install 

# attempt to parse a psi4 output file with cclib

# ccio class contains ccwrite, ccread, ccopen, etc, which all claim to automatically determine output file type
import cclib.io.ccio as ccio 


output_object = ccio.ccread('./output.dat')

print(output_object.atomcoords)
print(output_object.atommasses)
print(output_object.atomnos)
#cclib prints energy in eV... why
print(output_object.ccenergies / 27.21138505)
