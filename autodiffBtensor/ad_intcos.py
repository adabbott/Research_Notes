import torch
import ad_v3d

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
    q = torch.stack([i.q(geom) for i in intcos])
    return q

class STRE(object):
    def __init__(self, a, b):
        if a < b: 
            atoms = (a,b) 
        else: 
            atoms = (b,a)
        self.A = atoms[0]
        self.B = atoms[1]

    def q(self, geom):
        return ad_v3d.dist(geom[self.A], geom[self.B])

class BEND(object):
    def __init__(self, a, b, c, bendType="REGULAR"):
        if a < c:
            atoms = (a, b, c)
        else:
            atoms = (c, b, a)
        self.A = atoms[0]
        self.B = atoms[1]
        self.C = atoms[2]
        self.bendType = bendType # just "REGULAR" bends for now, no linear or complimentary
        self._axes_fixed = False

    def compute_axes(self, geom):
        u = ad_v3d.eAB(geom[self.B], geom[self.A])  # B->A
        v = ad_v3d.eAB(geom[self.B], geom[self.C])  # B->C
        self._w = ad_v3d.normalize(ad_v3d.cross(u,v)) # cross product and normalize
        self._x = ad_v3d.normalize(u + v)  # angle bisector
        
    def q(self, geom):
        if not self._axes_fixed:
            self.compute_axes(geom)
        u = ad_v3d.eAB(geom[self.B], geom[self.A])  # B->A
        v = ad_v3d.eAB(geom[self.B], geom[self.C])  # B->C
        origin = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        phi1 = ad_v3d.angle(u, origin, self._x) 
        phi2 = ad_v3d.angle(self._x, origin, v)
        phi = phi1 + phi2
        return phi


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
        check, tau = ad_v3d.tors(geom[self.A], geom[self.B], geom[self.C], geom[self.D])
        return tau


