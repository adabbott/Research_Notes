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
    B1 = autodiff_Btensor(ad_intcoords, geom, order=1)
    #print(B1)
    # Original PyOptKing B tensor
    B2 = optking.intcosMisc.Bmat(optking_intcoords, npgeom)
    #print(B2)
    # Prints True if B-matrices are the same
    print(torch.allclose(B1, torch.tensor(B2)))

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

print("Testing water...", end=' ')
test_molecule(h2o.geometry(), h2o_autodiff, h2o_optking)

print("Testing linear water...", end=' ')
test_molecule(linear_h2o.geometry(), linear_h2o_autodiff, linear_h2o_optking)

print("Testing ammonia...", end=' ')
test_molecule(ammonia.geometry(), ammonia_autodiff, ammonia_optking)

print("Testing formaldehyde...", end=' ')
test_molecule(h2co.geometry(), h2co_autodiff, h2co_optking)

print("Testing bent formaldehyde...", end=' ')
test_molecule(bent_h2co.geometry(), h2co_autodiff, h2co_optking)
    

