# Pytorch autograd-based functions to compute properties of 3d vectors, including angles, torsions, out-of-plane angles.  
import math
import torch

TORS_ANGLE_LIM = 0.017
TORS_COS_TOL   = 1e-10
DOT_PARALLEL_LIMIT = 1.e-10

def norm(v):
    return torch.norm(v)

def normalize(v1, Rmin=1.0e-8, Rmax=1.0e15):
    """
    Do not normalize in place. Not supported in autograd
    """
    n = norm(v1)
    if n < Rmin or n > Rmax:
        print('vector:', v1)
        print('norm:', n)
        raise Exception("Could not normalize vector. Vector norm beyond tolerance")
    else:
        return v1 / n 

def dot(v1, v2):
    return torch.dot(v1, v2)

def dist(v1, v2):
    return torch.norm(v1 - v2)

def eAB(p1, p2):
    eAB = normalize(p2 - p1)
    return eAB

def cross(u, v):
    return torch.cross(u, v)

def are_parallel(u, v):
    """ Determines if two vectors are parallel within tolerance (1e-10)"""
    if math.fabs(dot(u, v) - 1.0e0) < DOT_PARALLEL_LIMIT:
        return True
    else:
        return False

def are_antiparallel(u, v):
    """ Determines if two vectors are antiparallel within tolerance (1e-10)"""
    if math.fabs(dot(u, v) + 1.0e0) < DOT_PARALLEL_LIMIT:
        return True
    else:
        return False

def are_parallel_or_antiparallel(u, v):
    """
    Determines if two vectors are parallel and or antiparallal
    Returns
    -------
    boolean
        if vectors are either parallel or antiparallel
    """
    return are_parallel(u, v) or are_antiparallel(u, v)

def angle(A, B, C, tol=1.0e-14):
    """ Compute and return angle in radians A-B-C (between vector B->A and vector B->C)
    If points are absurdly close or far apart, returns False
    Parameters
    ----------
    A : int
        number of atom in fragment system. uses 1 indexing
    B : int
    C : int
    Returns
    -------
    float
        angle in radians
    """
    try:
        eBA = eAB(B, A)
    except: 
        raise Exception("Could not normalize eBA in angle()\n")
    try:
        eBC = eAB(B, C)
    except: 
        raise Exception("Could not normalize eBA in angle()\n")

    dotprod = dot(eBA, eBC)
    if dotprod > 1.0 - tol:
        #phi = 0.0
        phi = torch.tensor(0.0, requires_grad=True)
    elif dotprod < -1.0 + tol:
        phi = torch.acos(torch.tensor(-1.0, requires_grad=True))
    else:
        phi = torch.acos(dotprod)
    return phi

    #return _calc_angle(eBA, eBC, tol)

def _calc_angle(vec_1, vec_2, tol=1.0e-14):
    """
    Computes and returns angle in radians A-B_B (between vector B->A and vector B->C
    Should only be called by tors or angle. Error checking and vector creation
    is performed in angle() or tors() previously
    Paramters
    ---------
    vec_1 : ndarray
        first vector of an angle
    vec_2 : ndarray
        second vector on an angle
    tol : float
        nearness of cos to 1/-1 to set angle 0/pi.
    """

    dotprod = dot(vec_1, vec_2)
    # TODO If close to 0 or 180, the gradient will be 0 from the round function
    if dotprod > 1.0 - tol:
        print("ROUND WARNING")
        phi = torch.round(dotprod)
    elif dotprod < -1.0 + tol:
        print("ROUND WARNING")
        phi = torch.round(dotprod)
    else:
        phi = torch.acos(dotprod)
    return phi

# Compute and return angle in dihedral angle in radians A-B-C-D
# returns false if bond angles are too large for good torsion definition
def tors(A, B, C, D):
    phi_lim = TORS_ANGLE_LIM
    tors_cos_tol   = TORS_COS_TOL

    # Form e vectors
    try:
        EBA = eAB(B, A)
        EAB = -1 * EBA
    except AlgError as error:
        #logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(A), str(B)))
        raise Exception("Could not normalize %d, %d vector in tors()\n" % (str(A), str(B)))
    try:
        EBC = eAB(B, C)
    except AlgError as error:
        #logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(B), str(C)))
        raise Exception("Could not normalize %d, %d vector in tors()\n" % (str(B), str(C)))
    try:
        ECB = eAB(C, B)
        EBC = -1 * ECB
    except AlgError as error:
        # This is a bug I think: str(C), str(D) should be C, B
        #logger.warning("Could not normalize %d, %d vector in tors()\n" % (str(C), str(D)))
        raise Exception("Could not normalize %d, %d vector in tors()\n" % (str(C), str(B)))
    try:
        ECD = eAB(C, D)
    except AlgError as error:
        raise Exception("Could not normalize %d, %d vector in tors()\n" % (str(C), str(D)))

    # Compute bond angles
    phi_123 = _calc_angle(EBA, EBC)
    phi_234 = _calc_angle(ECB, ECD)
    # Dr. Allen's notes:
    tau = -torch.asin(dot(EBA, cross(ECB, ECD)) / (torch.sin(phi_123) * torch.sin(phi_234)))
    tau_final = tau

    ## determine sign of torsion ; this convention matches Wilson, Decius and Cross
    #if tau != math.pi: # no torsion will get value of -pi; Range is (-pi,pi].
    #    tmp = cross(EBC, ECD)
    #    tval = dot(EAB, tmp)
    #    if tval < 0:
    #        print('CHANGING SIGN!!!')
    #        tau_final = -1 * tau  # removed inplace operation by creating new variable, tau_final
    #    else:
    #        tau_final = tau
    #else:
    #    tau_final = tau
    return tau_final


#TODO TODO TODO TODO
# Compute and return angle in dihedral angle in radians A-B-C-D
# returns false if bond angles are too large for good torsion definition
def oofp(A, B, C, D):
    check1, eBA = eAB(B, A)
    check2, eBC = eAB(B, C)
    check3, eBD = eAB(B, D)
    if not check1 or not check2 or not check3:
        return False, 0.0

    check1, phi_CBD = angle(C, B, D)
    if not check1:
        return False, 0.0

    # This shouldn't happen unless angle B-C-D -> 0,
    if torch.sin(phi_CBD) < op.Params.v3d_tors_cos_tol:  #reusing parameter
        return False, 0.0

    dotprod = dot(cross(eBC, eBD), eBA) / torch.sin(phi_CBD)

    if dotprod > 1.0: tau = torch.acos(-1.0)
    elif dotprod < -1.0: tau = -1 * torch.acos(-1.0)
    else: tau = torch.asin(dotprod)
    return True, tau
