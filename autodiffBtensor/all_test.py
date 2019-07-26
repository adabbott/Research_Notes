import psi4
import torch
import numpy as np
from autodiffB import autodiff_Btensor
import ad_intcos
import optking

bohr2ang = 0.529177249
psi4.core.be_quiet()

def test_molecule(psi_geom, ad_intcoords, optking_intcoords):
    """ 
    Tests autodiff-OptKing hack against original OptKing by comparing 1st order B tensors

    Parameters: 
    ---------
    psi_geom : Psi4 Matrix, Molecule.geometry(), 
    ad_intcoords : list of new autodiff optking internal coordinates objects STRE, BEND, TORS
    optking_intcoords : list of optking internal coordinates Stre, Bend, Tors 
    """
    npgeom = np.array(psi_geom) * bohr2ang
    geom = torch.tensor(npgeom, requires_grad=True)
    # Autodiff B tensor (specify 1st order only)
    autodiff_coords = ad_intcos.qValues(ad_intcoords, geom)
    #print(autodiff_coords)
    B1 = autodiff_Btensor(ad_intcoords, geom, order=1)
    #print(B1)
    # Original PyOptKing B tensor
    optking_coords = optking.intcosMisc.qValues(optking_intcoords, npgeom)
    #print(optking_coords)
    B2 = optking.intcosMisc.Bmat(optking_intcoords, npgeom)
    #print(B2)
    # Prints True if B-matrices are the same
    print('Same Internal Coordinates...',torch.allclose(autodiff_coords, torch.tensor(optking_coords)))
    print('Same B-Matrix...',torch.allclose(B1, torch.tensor(B2)))

h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     0.950000000000 
H            0.000000000000     0.872305301500    -0.376275777700 
O            0.000000000000     0.000000000000     0.000000000000 
''')
h2o_autodiff = [ad_intcos.STRE(2,1), ad_intcos.STRE(2,0), ad_intcos.BEND(1,2,0)]
h2o_optking  = [optking.Stre(2,1), optking.Stre(2,0), optking.Bend(1,2,0)]

linear_h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     1.950000000000 
H            0.000000000000     0.000000000000     0.050000000000 
O            0.000000000000     0.000000000000     1.000000000000 
''')
linear_h2o_autodiff = [ad_intcos.STRE(2,1), ad_intcos.STRE(2,0), ad_intcos.BEND(1,2,0, bendType='LINEAR')]
linear_h2o_optking  = [optking.Stre(2,1), optking.Stre(2,0), optking.Bend(1,2,0, bendType='LINEAR')]

ammonia = psi4.geometry(
'''
 N  0.000000  0.0       0.0 
 H  1.584222  0.0       1.12022
 H  0.0      -1.58422  -1.12022
 H -1.584222  0.0       1.12022
 H  0.0       1.58422  -1.12022
 unit au
''')

ammonia_autodiff = [ad_intcos.STRE(0,1), ad_intcos.STRE(0,2), ad_intcos.STRE(0,3), ad_intcos.STRE(0,4), 
                   ad_intcos.BEND(1,0,2), ad_intcos.BEND(1,0,3), ad_intcos.BEND(1,0,4), ad_intcos.BEND(2,0,3), 
                   ad_intcos.BEND(2,0,4), ad_intcos.BEND(3,0,4)]

ammonia_optking = [optking.Stre(0,1), optking.Stre(0,2), optking.Stre(0,3), optking.Stre(0,4), 
                   optking.Bend(1,0,2), optking.Bend(1,0,3), optking.Bend(1,0,4), optking.Bend(2,0,3), 
                   optking.Bend(2,0,4), optking.Bend(3,0,4)]

h2co = psi4.geometry(
'''
C            0.000000000000     0.000000000000    -0.607835855018 
O            0.000000000000     0.000000000000     0.608048883261 
H            0.000000000000     0.942350938995    -1.206389817026 
H            0.000000000000    -0.942350938995    -1.206389817026 
'''
)
h2co_autodiff = [ad_intcos.STRE(0,1), ad_intcos.STRE(0,2), ad_intcos.BEND(2,0,1), ad_intcos.STRE(0,3), ad_intcos.BEND(3,0,1), ad_intcos.TORS(3,0,1,2)]
h2co_optking = [optking.Stre(0,1), optking.Stre(0,2), optking.Bend(2,0,1), optking.Stre(0,3), optking.Bend(3,0,1), optking.Tors(3,0,1,2)]

# h2co, but 160 degree dihedral
bent_h2co = psi4.geometry(
'''
C            0.011014420656    -0.636416764906     0.000000000000 
O            0.011014420656     0.628402205849     0.000000000000 
H           -0.152976834267    -1.197746817763     0.930040622624 
H           -0.152976834267    -1.197746817763    -0.930040622624 
'''
)

hooh = psi4.geometry(
'''
H
O 1 0.9
O 2 1.4 1 100.0
H 3 0.9 2 100.0 1 114.0
'''
)
hooh_autodiff = [ad_intcos.STRE(0,1), ad_intcos.STRE(0,2), ad_intcos.BEND(2,1,0), ad_intcos.STRE(3,2), ad_intcos.BEND(3,2,1), ad_intcos.TORS(3,2,1,0)]
hooh_optking = [optking.Stre(0,1), optking.Stre(0,2), optking.Bend(2,1,0), optking.Stre(3,2), optking.Bend(3,2,1), optking.Tors(3,2,1,0)]

sf4 = psi4.geometry(
'''
 S  0.00000000  -0.00000000  -0.30618267
 F -1.50688420  -0.00000000   0.56381732
 F  0.00000000  -1.74000000  -0.30618267
 F -0.00000000   1.74000000  -0.30618267
 F  1.50688420   0.00000000   0.56381732
'''
)
# try to break TORS
sf4_autodiff = [ad_intcos.TORS(0,1,2,3), ad_intcos.TORS(1,3,4,0), ad_intcos.TORS(0,2,1,4)]
sf4_optking = [optking.Tors(0,1,2,3), optking.Tors(1,3,4,0), optking.Tors(0,2,1,4)]


print("Testing water...")
test_molecule(h2o.geometry(), h2o_autodiff, h2o_optking)

print("Testing linear water...")
test_molecule(linear_h2o.geometry(), linear_h2o_autodiff, linear_h2o_optking)

print("Testing ammonia...")
test_molecule(ammonia.geometry(), ammonia_autodiff, ammonia_optking)

print("Testing formaldehyde...")
test_molecule(h2co.geometry(), h2co_autodiff, h2co_optking)

print("Testing bent formaldehyde...")
test_molecule(bent_h2co.geometry(), h2co_autodiff, h2co_optking)

print("Testing hooh...")
test_molecule(hooh.geometry(), hooh_autodiff, hooh_optking)
#
print("Testing nonsense sf4 ...")
test_molecule(sf4.geometry(), sf4_autodiff, sf4_optking)
#    

