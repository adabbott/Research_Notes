import psi4
import torch
import numpy as np
from autodiffB import autodiff_Btensor
from autodiffB import experiment
from autodiffB import dummy_computation, derivatives
import ad_intcos
import optking
import time
torch.set_printoptions(precision=5)

bohr2ang = 0.529177249
psi4.core.be_quiet()

def test_molecule(psi_geom,ad_intcoords,optking_intcoords):
    """ 
    Tests autodiff-OptKing hack against original OptKing by comparing 1st order B tensors

    Parameters: 
    ---------
    psi_geom : Psi4 Matrix,Molecule.geometry(),
    ad_intcoords : list of new autodiff optking internal coordinates objects STRE,BEND,TORS
    optking_intcoords : list of optking internal coordinates Stre,Bend,Tors 
    """
    npgeom = np.array(psi_geom) * bohr2ang
    geom = torch.tensor(npgeom,requires_grad=True)

    # Autodiff B tensor (specify 1st order only)
    a = time.time()
    autodiff_coords = ad_intcos.qValues(ad_intcoords,geom)
    B1 = autodiff_Btensor(ad_intcoords,geom,order=1)
    b = time.time()

    # Original PyOptKing B tensor
    optking_coords = optking.intcosMisc.qValues(optking_intcoords,npgeom)
    B2 = optking.intcosMisc.Bmat(optking_intcoords,npgeom)
    c = time.time()

    ## Test 2nd order B tensor. Can only do one coord at a time in OPTKING.
    ## Must modify optking to return it in testDerivativeB()
    #A = time.time()
    #B3 = autodiff_Btensor(ad_intcoords,geom,order=2)
    #B = time.time()
    #junk, B4 = optking.testB.testDerivativeB(optking_intcoords, npgeom) 
    #C = time.time()

    print('Same Internal Coordinates...',torch.allclose(autodiff_coords,torch.tensor(optking_coords)))
    print('Same B-Matrix...',torch.allclose(B1,torch.tensor(B2)))
    #print('Same Second Order B-Tensor...',torch.allclose(B3[-1],torch.tensor(B4)))

    print("AutoDiff Optking took {} seconds".format(round((b-a), 2)))
    print("Optking took {} seconds".format(round((c-b), 2)))
    #print("AutoDiff Optking took {} seconds".format(round((B-A) + (b-a), 2)))
    #print("Optking took {} seconds".format(round((C-B) + (c-b), 2)))

h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     0.950000000000 
H            0.000000000000     0.872305301500    -0.376275777700 
O            0.000000000000     0.000000000000     0.000000000000 
''')
h2o_autodiff = [ad_intcos.STRE(2,1),ad_intcos.STRE(2,0),ad_intcos.BEND(1,2,0)]
h2o_optking  = [optking.Stre(2,1),optking.Stre(2,0),optking.Bend(1,2,0)]

linear_h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     1.950000000000 
H            0.000000000000     0.000000000000     0.050000000000 
O            0.000000000000     0.000000000000     1.000000000000 
''')
linear_h2o_autodiff = [ad_intcos.STRE(2,1),ad_intcos.STRE(2,0),ad_intcos.BEND(1,2,0,bendType='LINEAR')]
linear_h2o_optking  = [optking.Stre(2,1),optking.Stre(2,0),optking.Bend(1,2,0,bendType='LINEAR')]

ammonia = psi4.geometry(
'''
 N  0.000000  0.0       0.0 
 H  1.584222  0.0       1.12022
 H  0.0      -1.58422  -1.12022
 H -1.584222  0.0       1.12022
 H  0.0       1.58422  -1.12022
 unit au
''')

ammonia_autodiff = [ad_intcos.STRE(0,1),ad_intcos.STRE(0,2),ad_intcos.STRE(0,3),ad_intcos.STRE(0,4),
                   ad_intcos.BEND(1,0,2),ad_intcos.BEND(1,0,3),ad_intcos.BEND(1,0,4),ad_intcos.BEND(2,0,3),
                   ad_intcos.BEND(2,0,4),ad_intcos.BEND(3,0,4)]

ammonia_optking = [optking.Stre(0,1),optking.Stre(0,2),optking.Stre(0,3),optking.Stre(0,4),
                   optking.Bend(1,0,2),optking.Bend(1,0,3),optking.Bend(1,0,4),optking.Bend(2,0,3),
                   optking.Bend(2,0,4),optking.Bend(3,0,4)]

h2co = psi4.geometry(
'''
C            0.000000000000     0.000000000000    -0.607835855018 
O            0.000000000000     0.000000000000     0.608048883261 
H            0.000000000000     0.942350938995    -1.206389817026 
H            0.000000000000    -0.942350938995    -1.206389817026 
'''
)
h2co_autodiff = [ad_intcos.STRE(0,1),ad_intcos.STRE(0,2),ad_intcos.BEND(2,0,1),ad_intcos.STRE(0,3),ad_intcos.BEND(3,0,1),ad_intcos.TORS(3,0,1,2)]
h2co_optking = [optking.Stre(0,1),optking.Stre(0,2),optking.Bend(2,0,1),optking.Stre(0,3),optking.Bend(3,0,1),optking.Tors(3,0,1,2)]

# h2co,but 160 degree dihedral
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
hooh_autodiff = [ad_intcos.STRE(0,1),ad_intcos.STRE(0,2),ad_intcos.BEND(2,1,0),ad_intcos.STRE(3,2),ad_intcos.BEND(3,2,1),ad_intcos.TORS(3,2,1,0)]
hooh_optking = [optking.Stre(0,1),optking.Stre(0,2),optking.Bend(2,1,0),optking.Stre(3,2),optking.Bend(3,2,1),optking.Tors(3,2,1,0)]

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
sf4_autodiff = [ad_intcos.TORS(0,1,2,3),ad_intcos.TORS(1,3,4,0),ad_intcos.TORS(0,2,1,4)]
sf4_optking = [optking.Tors(0,1,2,3),optking.Tors(1,3,4,0),optking.Tors(0,2,1,4)]

sf4_autodiff2 = [ad_intcos.TORS(0,1,2,3),ad_intcos.TORS(1,3,4,0),ad_intcos.TORS(0,2,1,4),ad_intcos.TORS(4,3,2,1),ad_intcos.TORS(4,3,2,0)]
sf4_optking2 = [optking.Tors(0,1,2,3),optking.Tors(1,3,4,0),optking.Tors(0,2,1,4),optking.Tors(4,3,2,1),optking.Tors(4,3,2,0)]

npgeom = np.array(h2o.geometry()) * bohr2ang
print(npgeom)
geom = torch.tensor(npgeom,requires_grad=True)
#experiment(h2o_autodiff, geom)
#r = fake(geom)
#experiment(h2o_autodiff, geom)

#print("Gradient")
#a = time.time()
#derivatives(geom, order=1)
#b = time.time()
#print(b-a, "seconds")
#
#print("Hessian")
#derivatives(geom, order=2)
#c = time.time()
#print(c-b, "seconds")
#
#print("Third")
#derivatives(geom, order=3)
#d = time.time()
#print(d-c, "seconds")
#

print("Testing water...")
test_molecule(h2o.geometry(),h2o_autodiff,h2o_optking)

print("Testing linear water...")
test_molecule(linear_h2o.geometry(),linear_h2o_autodiff,linear_h2o_optking)
print("Testing ammonia...")
test_molecule(ammonia.geometry(),ammonia_autodiff,ammonia_optking)
print("Testing formaldehyde...")
test_molecule(h2co.geometry(),h2co_autodiff,h2co_optking)
print("Testing bent formaldehyde...")
test_molecule(bent_h2co.geometry(),h2co_autodiff,h2co_optking)
print("Testing hooh...")
test_molecule(hooh.geometry(),hooh_autodiff,hooh_optking)
print("Testing nonsense sf4 ...")
test_molecule(sf4.geometry(),sf4_autodiff,sf4_optking)

# Use intcos_generate_exit autogenerated internal coordinates for a few big molecules
allene = psi4.geometry(
"""
H  0.0  -0.92   -1.8
H  0.0   0.92   -1.8
C  0.0   0.00   -1.3
C  0.0   0.00    0.0
C  0.0   0.00    1.3
H  0.92  0.00    1.8
H -0.92  0.00    1.8
""")
allene_autodiff = [ad_intcos.STRE(0, 2),ad_intcos.STRE(1, 2),ad_intcos.STRE(2, 3),ad_intcos.STRE(3, 4),ad_intcos.STRE(4, 5),ad_intcos.STRE(4, 6),ad_intcos.BEND(0, 2, 1),ad_intcos.BEND(0, 2, 3),ad_intcos.BEND(1, 2, 3),ad_intcos.BEND(2, 3, 4),ad_intcos.BEND(2, 3, 4),ad_intcos.BEND(3, 4, 5),ad_intcos.BEND(3, 4, 6),ad_intcos.BEND(5, 4, 6),ad_intcos.TORS(0, 2, 4, 5),ad_intcos.TORS(0, 2, 4, 6),ad_intcos.TORS(1, 2, 4, 5),ad_intcos.TORS(1, 2, 4, 6)]
allene_optking = [optking.Stre(0, 2),optking.Stre(1, 2),optking.Stre(2, 3),optking.Stre(3, 4),optking.Stre(4, 5),optking.Stre(4, 6),optking.Bend(0, 2, 1),optking.Bend(0, 2, 3),optking.Bend(1, 2, 3),optking.Bend(2, 3, 4, bendType="LINEAR"),optking.Bend(2, 3, 4, bendType="LINEAR"),optking.Bend(3, 4, 5),optking.Bend(3, 4, 6),optking.Bend(5, 4, 6),optking.Tors(0, 2, 4, 5),optking.Tors(0, 2, 4, 6),optking.Tors(1, 2, 4, 5),optking.Tors(1, 2, 4, 6)]

print("Testing allene...")
test_molecule(allene.geometry(),allene_autodiff,allene_optking)

big = psi4.geometry( 
'''
 C  0.00000000 0.00000000 0.00000000
 Cl 0.19771002 -0.99671665 -1.43703398
 C  1.06037767 1.11678073 0.00000000
 C  2.55772698 0.75685710 0.00000000
 H  3.15117939 1.67114056 0.00000000
 H  2.79090687 0.17233980 0.88998127
 H  2.79090687 0.17233980 -0.88998127
 H  0.75109254 2.16198057 0.00000000
 H -0.99541786 0.44412079 0.00000000
 H  0.12244541 -0.61728474 0.88998127
'''
)

big_autodiff = [ad_intcos.STRE(0, 1),ad_intcos.STRE(0, 2),ad_intcos.STRE(0, 8),ad_intcos.STRE(0, 9),ad_intcos.STRE(2, 3),ad_intcos.STRE(2, 7),ad_intcos.STRE(3, 4),ad_intcos.STRE(3, 5),ad_intcos.STRE(3, 6),ad_intcos.BEND(0, 2, 3),ad_intcos.BEND(0, 2, 7),ad_intcos.BEND(1, 0, 2),ad_intcos.BEND(1, 0, 8),ad_intcos.BEND(1, 0, 9),ad_intcos.BEND(2, 0, 8),ad_intcos.BEND(2, 0, 9),ad_intcos.BEND(2, 3, 4),ad_intcos.BEND(2, 3, 5),ad_intcos.BEND(2, 3, 6),ad_intcos.BEND(3, 2, 7),ad_intcos.BEND(4, 3, 5),ad_intcos.BEND(4, 3, 6),ad_intcos.BEND(5, 3, 6),ad_intcos.BEND(8, 0, 9),ad_intcos.TORS(0, 2, 3, 4),ad_intcos.TORS(0, 2, 3, 5),ad_intcos.TORS(0, 2, 3, 6),ad_intcos.TORS(1, 0, 2, 3),ad_intcos.TORS(1, 0, 2, 7),ad_intcos.TORS(3, 2, 0, 8),ad_intcos.TORS(3, 2, 0, 9),ad_intcos.TORS(4, 3, 2, 7),ad_intcos.TORS(5, 3, 2, 7),ad_intcos.TORS(6, 3, 2, 7),ad_intcos.TORS(7, 2, 0, 8),ad_intcos.TORS(7, 2, 0, 9)]
big_optking = [optking.Stre(0, 1),optking.Stre(0, 2),optking.Stre(0, 8),optking.Stre(0, 9),optking.Stre(2, 3),optking.Stre(2, 7),optking.Stre(3, 4),optking.Stre(3, 5),optking.Stre(3, 6),optking.Bend(0, 2, 3),optking.Bend(0, 2, 7),optking.Bend(1, 0, 2),optking.Bend(1, 0, 8),optking.Bend(1, 0, 9),optking.Bend(2, 0, 8),optking.Bend(2, 0, 9),optking.Bend(2, 3, 4),optking.Bend(2, 3, 5),optking.Bend(2, 3, 6),optking.Bend(3, 2, 7),optking.Bend(4, 3, 5),optking.Bend(4, 3, 6),optking.Bend(5, 3, 6),optking.Bend(8, 0, 9),optking.Tors(0, 2, 3, 4),optking.Tors(0, 2, 3, 5),optking.Tors(0, 2, 3, 6),optking.Tors(1, 0, 2, 3),optking.Tors(1, 0, 2, 7),optking.Tors(3, 2, 0, 8),optking.Tors(3, 2, 0, 9),optking.Tors(4, 3, 2, 7),optking.Tors(5, 3, 2, 7),optking.Tors(6, 3, 2, 7),optking.Tors(7, 2, 0, 8),optking.Tors(7, 2, 0, 9)]
print("Testing ch3chch2cl...")
test_molecule(big.geometry(),big_autodiff,big_optking)



