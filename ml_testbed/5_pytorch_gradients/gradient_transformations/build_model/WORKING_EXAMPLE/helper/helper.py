import numpy as np
from . import v3d
from math import sqrt, cos, fabs
FIX_VAL_NEAR_PI = 1.57

def qValues(intcos, geom):
    """Calculates internal coordinates from cartesian geometry
    Parameters
    ---------
    intcos : list
        (nat) list of stretches, bends, etc...
    geom : ndarray
        (nat, 3) cartesian geometry
    Returns
    -------
    ndarray
        internal coordinate values
    """

    q = np.array([i.q(geom) for i in intcos])
    return q

def qForces(intcos, geom, gradient_x, B=None):
    """Transforms cartesian gradient to internals
    Parameters
    ----------
    intcos : list
        stretches, bends, etc
    geom : ndarray
        (nat, 3) cartesian geometry
    gradient_x :
        (3nat, 1) cartesian gradient
    Returns
    -------
    ndarray
        forces in internal coordinates (-1 * gradient)
    Notes
    -----
    fq = (BuB^T)^(-1)*B*f_x
    """
    if len(intcos) == 0 or len(geom) == 0:
        return np.zeros(0, float)

    if B is None:
        B = Bmat(intcos, geom)

    fx = np.multiply(-1.0, gradient_x)  # gradient -> forces
    G = np.dot(B, B.T)

    Ginv = symmMatInv(G, redundant=True)
    #Ginv = np.linalg.inv(G)
    fq = np.dot(np.dot(Ginv, B), fx)
    return fq

def Bmat(intcos, geom):
    Nint = len(intcos)
    Ncart = geom.size

    B = np.zeros((Nint, Ncart), float)
    for i, intco in enumerate(intcos):
        intco.DqDx(geom, B[i])
    return B

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def convertHessianToInternals(H, intcos, geom, g_x=None):
    """ converts the hessian from cartesian coordinates into internal coordinates 
    
    Parameters
    ----------
    H : ndarray
        Hessian in cartesians
    B : ndarray
        Wilson B matrix
    intcos : list 
        internal coordinates (stretches, bends, etc...)
    geom : ndarray
        nat, 3 cartesian geometry
    
    Returns
    -------
    Hq : ndarray
    """
    B = Bmat(intcos, geom)
    G = np.dot(B, B.T)

    Ginv = symmMatInv(G, redundant=True)
    Atranspose = np.dot(Ginv, B)

    Hworking = H.copy()
    if g_x is None:  # A^t Hxy A
        print("Neglecting force/B-matrix derivative term, only correct at"
                    + "stationary points.\n")
    else:  # A^t (Hxy - Kxy) A;    K_xy = sum_q ( grad_q[I] d^2(q_I)/(dx dy) )
        print("Including force/B-matrix derivative term.\n")

        g_q = np.dot(Atranspose, g_x)
        Ncart = 3 * len(geom)
        dq2dx2 = np.zeros((Ncart, Ncart), float)  # should be cart x cart for fragment ?

        for I, q in enumerate(intcos):
            dq2dx2[:] = 0
            q.Dq2Dx2(geom, dq2dx2)  # d^2(q_I)/ dx_i dx_j

            for a in range(Ncart):
                for b in range(Ncart):
                    # adjust indices for multiple fragments
                    Hworking[a, b] -= g_q[I] * dq2dx2[a, b] 


    
    Hq = np.dot(Atranspose, np.dot(Hworking, Atranspose.T))
    return Hq


class STRE(object):
    def __init__(self, a, b):
        if a < b: 
            atoms = (a, b)
        else: 
            atoms = (b, a)
        self.A = atoms[0]
        self.B = atoms[1]
        self.atoms = (self.A, self.B)

    def q(self, geom):
        return v3d.dist(geom[self.A], geom[self.B])

    def DqDx(self, geom, dqdx):
        check, eAB = v3d.eAB(geom[self.A], geom[self.B])  # A->B
        startA = 3 * self.A
        startB = 3 * self.B
        dqdx[startA:startA + 3] = -1 * eAB[0:3]
        dqdx[startB:startB + 3] = eAB[0:3]

    # Return derivative B matrix elements.  Matrix is cart X cart and passed in.
    def Dq2Dx2(self, geom, dq2dx2):
        #try:
        #    eAB = v3d.eAB(geom[self.A], geom[self.B])  # A->B
        #except AlgError:
        #    raise AlgError("Stre.Dq2Dx2: could not normalize s vector") from error
        eAB = v3d.eAB(geom[self.A], geom[self.B])  # A->B

        length = self.q(geom)

        for a in range(2):
            for a_xyz in range(3):
                for b in range(2):
                    for b_xyz in range(3):
                        tval = (
                            eAB[a_xyz] * eAB[b_xyz] - delta(a_xyz, b_xyz)) / length
                        if a == b:
                            tval *= -1.0
                        print('tval',tval)
                        print('first',3*self.atoms[a]+a_xyz)
                        print('second',3*self.atoms[b]+b_xyz)
                        print(dq2dx2)
                        dq2dx2[3*self.atoms[a]+a_xyz,3*self.atoms[b]+b_xyz] = tval

        return


# Returns eigenvectors as rows?
def symmMatEig(mat):
    try:
        evals, evects = np.linalg.eigh(mat)
        if abs(min(evects[:,0])) > abs(max(evects[:,0])):
            evects[:,0] *= -1.0
    except:
        raise OptError("symmMatEig: could not compute eigenvectors")
        # could be ALG_FAIL ?
    evects = evects.T
    return evals, evects

#  Return the inverse of a real, symmetric matrix.  If "redundant" == true,
#  then a generalized inverse is permitted.
def symmMatInv(A, redundant=False, redundant_eval_tol=1.0e-10):
    dim = A.shape[0]
    if dim == 0:
        return np.zeros((0, 0), float)
    det = 1.0

    try:
        evals, evects = symmMatEig(A)
    except LinAlgError:
        raise OptError("symmMatrixInv: could not compute eigenvectors")
        # could be LinAlgError?

    for i in range(dim):
        det *= evals[i]

    if not redundant and fabs(det) < 1E-10:
        raise OptError(
            "symmMatrixInv: non-generalized inverse failed; very small determinant")
        # could be LinAlgError?

    diagInv = np.zeros((dim, dim), float)

    if redundant:
        for i in range(dim):
            if fabs(evals[i]) > redundant_eval_tol:
                diagInv[i, i] = 1.0 / evals[i]
    else:
        for i in range(dim):
            diagInv[i, i] = 1.0 / evals[i]

    # A^-1 = P^t D^-1 P
    tmpMat = np.dot(diagInv, evects)
    AInv = np.dot(evects.T, tmpMat)
    return AInv


class BEND(object):
    def __init__(self, a, b, c, bendType="REGULAR"):

        if a < c: self.atoms = (a, b, c)
        else: self.atoms = (c, b, a)

        self.A = self.atoms[0]
        self.B = self.atoms[1]
        self.C = self.atoms[2]

        self.bendType = bendType
        self._axes_fixed = False
        self._x = np.zeros(3, float)
        self._w = np.zeros(3, float)

    def compute_axes(self, geom):
        check, u = v3d.eAB(geom[self.B], geom[self.A])  # B->A
        check, v = v3d.eAB(geom[self.B], geom[self.C])  # B->C

        self._w[:] = v3d.cross(u, v)  # orthogonal vector
        v3d.normalize(self._w)
        self._x[:] = u + v  # angle bisector
        v3d.normalize(self._x)

    def q(self, geom):
        if not self._axes_fixed:
            self.compute_axes(geom)
        check, u = v3d.eAB(geom[self.B], geom[self.A])  # B->A
        check, v = v3d.eAB(geom[self.B], geom[self.C])  # B->C

        # linear bend is sum of 2 angles, u.x + v.x
        origin = np.zeros(3, float)
        check, phi = v3d.angle(u, origin, self._x)

        check, phi2 = v3d.angle(self._x, origin, v)
        phi += phi2
        return phi

    @staticmethod
    def zeta(a, m, n):
        if a == m: 
            return 1
        elif a == n: 
            return -1
        else: 
            return 0

    def DqDx(self, geom, dqdx):
        if not self._axes_fixed:
            self.compute_axes(geom)

        u = geom[self.A] - geom[self.B]  # B->A
        v = geom[self.C] - geom[self.B]  # B->C
        Lu = v3d.norm(u)  # RBA
        Lv = v3d.norm(v)  # RBC
        u[:] *= 1.0 / Lu  # u = eBA
        v[:] *= 1.0 / Lv  # v = eBC

        uXw = v3d.cross(u, self._w)
        wXv = v3d.cross(self._w, v)

        # B = overall index of atom; a = 0,1,2 relative index for delta's
        for a, B in enumerate(self.atoms):
            dqdx[3*B : 3*B+3] = BEND.zeta(a,0,1) * uXw[0:3]/Lu + \
                                BEND.zeta(a,2,1) * wXv[0:3]/Lv


class TORS(object):
    def __init__(self, a, b, c, d):

        if a < d: self.atoms = (a, b, c, d)
        else: self.atoms = (d, c, b, a)
        self.A = self.atoms[0]
        self.B = self.atoms[1]
        self.C = self.atoms[2]
        self.D = self.atoms[3]
        self._near180 = 0

    def near180(self):
        return self._near180

    def zeta(a, m, n):
        if a == m: 
            return 1
        elif a == n: 
            return -1
        else: 
            return 0

    # compute angle and return value in radians
    def q(self, geom):
        check, tau = v3d.tors(geom[self.A], geom[self.B], geom[self.C], geom[self.D])

    def DqDx(self, geom, dqdx, mini=False):
        u = geom[self.A] - geom[self.B]  # u=m-o eBA
        v = geom[self.D] - geom[self.C]  # v=n-p eCD
        w = geom[self.C] - geom[self.B]  # w=p-o eBC
        Lu = v3d.norm(u)  # RBA
        Lv = v3d.norm(v)  # RCD
        Lw = v3d.norm(w)  # RBC
        u *= 1.0 / Lu  # eBA
        v *= 1.0 / Lv  # eCD
        w *= 1.0 / Lw  # eBC

        cos_u = v3d.dot(u, w)
        cos_v = -v3d.dot(v, w)

        # abort and leave zero if 0 or 180 angle
        if 1.0 - cos_u * cos_u <= 1.0e-12 or 1.0 - cos_v * cos_v <= 1.0e-12:
            return

        sin_u = sqrt(1.0 - cos_u * cos_u)
        sin_v = sqrt(1.0 - cos_v * cos_v)
        uXw = v3d.cross(u, w)
        vXw = v3d.cross(v, w)

        # a = relative index; B = full index of atom
        for a, B in enumerate(self.atoms):
            for i in range(3):  #i=a_xyz
                tval = 0.0

                if a == 0 or a == 1:
                    tval += TORS.zeta(a, 0, 1) * uXw[i] / (Lu * sin_u * sin_u)

                if a == 2 or a == 3:
                    tval += TORS.zeta(a, 2, 3) * vXw[i] / (Lv * sin_v * sin_v)

                if a == 1 or a == 2:
                    tval += TORS.zeta(a, 1, 2) * uXw[i] * cos_u / (Lw * sin_u * sin_u)

                # "+" sign for zeta(a,2,1)) differs from JCP, 117, 9164 (2002)
                if a == 1 or a == 2:
                    tval += -TORS.zeta(a, 2, 1) * vXw[i] * cos_v / (Lw * sin_v * sin_v)

                if not mini:
                    dqdx[3 * B + i] = tval
                else:
                    dqdx[3 * a + i] = tval


