import jax
from jax.experimental import loops
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np

import numpy as onp 
#from scipy import special

def boys(m,x):
    #return 0.5 * (x + 1e-11)**(-(m + 0.5)) * jax.lax.igamma(m + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(m + 0.5))
    return 0.5 * (x + 1e-11)**(-(m + 0.5)) * jax.scipy.special.gammainc(m + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(m + 0.5))

#def gaussian_product_center(alpha1,A,alpha2,B):
#    #return (alpha1 * A + alpha2 * B) / (alpha1 + alpha2)

def gaussian_product_center(alpha1,Ax,Ay,Az,alpha2,Bx,By,Bz):
    return (alpha1 * Ax + alpha2 * Bx) / (alpha1 + alpha2), (alpha1 * Ay + alpha2 * By) / (alpha1 + alpha2),(alpha1 * Az + alpha2 * Bz) / (alpha1 + alpha2)

#def vrr(la,ma,na,lc,mc,nc,xa,ya,za,xb,yb,zb,xc,yc,zc,xd,yd,zd, alphaa,alphab,alphac,alphad): 

def vrr(superarg):
    la,ma,na,lb,mb,nb,lc,mb,nc,ld,md,nd,xa,ya,za,xb,yb,zb,xc,yc,zc,xd,yd,zd,alphaa,alphab,alphac,alphad = superarg
    lmna = (la,ma,na)
    lmnb = (lb,mb,nb)
    lmnc = (lc,mc,nc)
    lmnd = (ld,md,nd)
    # For first VRR (xs|ss). However, HS algo requires the [(x + 1i) s | y s ] integrals so we add 1 i think TODO
    La = la + lb + lc + ld #+ 1 #TODO
    Ma = ma + mb + mc + md #+ 1 #TODO
    Na = na + nb + nc + nd #+ 1 #TODO

    px,py,pz = gaussian_product_center(alphaa,xa,ya,za,alphab,xb,yb,zb)
    qx,qy,qz = gaussian_product_center(alphac,xc,yc,zc,alphad,xd,yd,zd)
    zeta = alphaa + alphab
    eta = alphac + alphad
    wx,wy,wz = gaussian_product_center(zeta,px,py,pz,eta,qx,qy,qz)
    rab2 = (xa-xb)**2 + (ya-yb)**2 + (za-zb)**2
    Kab = np.sqrt(2) * np.pi**1.25 / (alphaa+alphab) * np.exp(-alphaa * alphab / (alphaa + alphab) * rab2)
    rcd2 = (xc-xd)**2 + (yc-yd)**2 + (zc-zd)**2
    Kcd = np.sqrt(2) * np.pi**1.25 / (alphac + alphad) * np.exp(-alphac * alphad / (alphac + alphad) * rcd2)
    rpq2 = (px-qx)**2 + (py-qy)**2 + (pz-qz)**2
    boys_arg = zeta * eta / (zeta + eta) * rpq2

    deltax = (2 * alphab * (xa - xb) + 2 * alphad * (xc - xd)) 
    deltay = (2 * alphab * (ya - yb) + 2 * alphad * (yc - yd)) 
    deltaz = (2 * alphab * (za - zb) + 2 * alphad * (zc - zd)) 

    oot_eta = 1 / (2 * eta)

    #mtot = la + ma + na + lc + mc + nc + M
    # Static size of boys function values for jittableness
    # Currently supports pppp
    mtot = 4
    boys_indices = np.arange(mtot + 1)
    boys_arg = np.repeat(boys_arg,mtot + 1)
    boys_vals = boys(boys_indices, boys_arg)

    with loops.Scope() as S:
        # Static size for the recursion stack for jittableness
        # Supports only up to p functions at this time.
        S.vrr_terms = np.zeros((3,3,3,3,3,3,mtot+1))

        S.im = 0
        for _ in S.while_range(lambda: S.im < mtot + 1):
            tmp = Kab * Kcd / np.sqrt(zeta + eta) * boys_vals[S.im]
            S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[0,0,0,0,0,0,S.im], tmp)
            S.im += 1
    
        S.i = 0
        for _ in S.while_range(lambda: S.i < La):
            S.im = 0
            for _ in S.while_range(lambda: S.im < mtot - S.i):
                tmp = (px - xa) * S.vrr_terms[S.i,0,0,0,0,0, S.im] + (wx - px) * S.vrr_terms[S.i,0,0,0,0,0, S.im+1]
                tmp += S.i/2./zeta * (S.vrr_terms[S.i-1,0,0,0,0,0,S.im] - eta / (zeta + eta) * S.vrr_terms[S.i-1,0,0,0,0,0, S.im+1])
                S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i+1,0,0,0,0,0, S.im], tmp)
                S.im += 1
            S.i += 1

        S.j = 0
        for _ in S.while_range(lambda: S.j < Ma):
            S.i = 0
            for _ in S.while_range(lambda: S.i < (La + 1)):
                S.im = 0 
                for _ in S.while_range(lambda: S.im < (mtot - S.i - S.j)):
                    tmp =  (py - ya) * S.vrr_terms[S.i,S.j,0,0,0,0,S.im] + (wy - py) * S.vrr_terms[S.i,S.j,0,0,0,0, S.im+1]
                    tmp += S.j / 2. / zeta*(S.vrr_terms[S.i,S.j-1,0,0,0,0, S.im] - eta / (zeta+ eta) * S.vrr_terms[S.i,S.j-1,0,0,0,0,S.im+1])
                    S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j+1,0,0,0,0,S.im], tmp)
                    S.im += 1 
                S.i += 1 
            S.j += 1 


        S.k = 0
        for _ in S.while_range(lambda: S.k < Na):
            S.j = 0
            for _ in S.while_range(lambda: S.j < (Ma + 1)):
                S.i = 0
                for _ in S.while_range(lambda: S.i < (La + 1)):
                    S.im = 0 
                    for _ in S.while_range(lambda: S.im < (mtot - S.i - S.j - S.k)):
                        tmp = (pz - za) * S.vrr_terms[S.i,S.j,S.k,0,0,0, S.im] + (wz - pz) * S.vrr_terms[S.i,S.j,S.k,0,0,0, S.im+1]
                        tmp += S.k /2. / zeta * (S.vrr_terms[S.i,S.j,S.k-1,0,0,0, S.im] - eta / (zeta + eta) * S.vrr_terms[S.i,S.j,S.k-1,0,0,0, S.im+1])
                        S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j,S.k+1,0,0,0,S.im], tmp)
                        S.im += 1
                    S.i += 1
                S.j += 1
            S.k += 1


        # No longer need auxilliary's (i think)
        S.vrr_terms = S.vrr_terms[:,:,:,:,:,:,0]
    
        # For HS transfer relation first (xs|ys)
        La = la + lb 
        Ma = ma + mb 
        Na = na + nb 
        
        Lc = lc + ld
        Mc = mc + md
        Nc = nc + nd
        # every possible combo
        # [ps|ps], [ds|ps], [fs|ps], ... [(La + Lb + Lc + Ld - 1)s |ps]
        #          [ds|ds], [fs|ds], ... [(La + Lb + Lc + Ld - 2)s|ds]

        # X component electron transfer: this works for la,ma,na = 1,1,1; lc,mc,nc = 1,0,0, FAILS for la,ma,na = (1,2,1)
        S.q = 0
        for _ in S.while_range(lambda: S.q < Lc):
            S.k = 0
            for _ in S.while_range(lambda: S.k < Na + 1):
                S.j = 0
                for _ in S.while_range(lambda: S.j < Ma + 1):
                    S.i = 0
                    for _ in S.while_range(lambda: S.i < La + 1):
                        ai = S.i - S.q
                        ci = S.q 
                                         # This needs to fill in all possible values of S.j and S.k
                        tmp = oot_eta * ( ai * S.vrr_terms[S.i - S.q - 1, S.j, S.k, S.q, 0, 0] + ci * S.vrr_terms[S.i - S.q, S.j, S.k, S.q - 1, 0, 0] \
                                        - 2 * zeta * S.vrr_terms[S.i - S.q + 1, S.j, S.k, S.q, 0, 0] -  deltax * S.vrr_terms[S.i - S.q, S.j, S.k, S.q, 0, 0])
                        S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i - S.q,S.j,S.k,S.q + 1,0,0], tmp)
                        S.i += 1
                    S.j += 1
                S.k += 1
            S.q += 1


        #for _ in S.while_range(lambda: S.q < lc + ld):
        #    S.k = 0
        #    for _ in S.while_range(lambda: S.k < la + lb + lc + ld + 1):
        #        S.j = 0
        #        for _ in S.while_range(lambda: S.j < ma + mb + mc + md + 1):
        #            S.i = 0
        #            for _ in S.while_range(lambda: S.i < na + nb + nc + nd + 1):

        #for _ in S.while_range(lambda: S.q < lc + ld):
        #    S.k = 0
        #    for _ in S.while_range(lambda: S.k < la + lb + 1):
        #        S.j = 0
        #        for _ in S.while_range(lambda: S.j <  ma + mb + 1):
        #            S.i = 0
        #            for _ in S.while_range(lambda: S.i < na + nb + 1):
        #                #ai = S.i - S.q
        #                #ci = S.q 
        #                                  # This needs to fill in all possible values of S.j and S.k
        #                #tmp = oot_eta * ( ai * S.vrr_terms[S.i - S.q - 1, S.j, S.k, S.q, 0, 0] + ci * S.vrr_terms[S.i - S.q, S.j, S.k, S.q - 1, 0, 0] \
        #                #                - 2 * zeta * S.vrr_terms[S.i - S.q + 1, S.j, S.k, S.q, 0, 0] -  deltax * S.vrr_terms[S.i - S.q, S.j, S.k, S.q, 0, 0])
        #                #S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i - S.q,S.j,S.k,S.q + 1,0,0], tmp)

        #                tmp = oot_eta * ( S.i * S.vrr_terms[S.i - 1, S.j, S.k, S.q, 0, 0] + S.q * S.vrr_terms[S.i, S.j, S.k, S.q - 1, 0, 0] \
        #                                - 2 * zeta * S.vrr_terms[S.i + 1, S.j, S.k, S.q, 0, 0] -  deltax * S.vrr_terms[S.i, S.j, S.k, S.q, 0, 0])
        #                S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q + 1,0,0], tmp)
        #                S.i += 1
        #            S.j += 1
        #        S.k += 1
        #    S.q += 1

        # Works with (1,1,1) (1,0,0) but fails (1,2,1) (1,0,0) 
        #for _ in S.while_range(lambda: S.q < lc + ld):
        #    S.k = 0
        #    for _ in S.while_range(lambda: S.k < la + lb + 1):
        #        S.j = 0
        #        for _ in S.while_range(lambda: S.j <  ma + mb + 1):
        #            S.i = 0
        #            for _ in S.while_range(lambda: S.i < na + nb + 1):
        #                tmp = oot_eta * ( S.i * S.vrr_terms[S.i - 1, S.j, S.k, S.q, 0, 0] + S.q * S.vrr_terms[S.i, S.j, S.k, S.q - 1, 0, 0] \
        #                                - 2 * zeta * S.vrr_terms[S.i + 1, S.j, S.k, S.q, 0, 0] -  deltax * S.vrr_terms[S.i, S.j, S.k, S.q, 0, 0])
        #                S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q + 1,0,0], tmp)
        #                S.i += 1
        #            S.j += 1
        #        S.k += 1
        #    S.q += 1




    #return S.vrr_terms[la,ma,na,0,0,0,0]
    #return S.vrr_terms[la,ma,na,lc,mc,nc]
    #print(S.vrr_terms[la,ma,na,lc,mc,nc])
    print(S.vrr_terms[La, Ma, Na, Lc, Mc, Nc])
    return S.vrr_terms


xa,ya,za = 0.0,0.1,0.9
xb,yb,zb = 0.0,-0.1,-0.9
xc,yc,zc = 0.0,-0.1, 0.9
xd,yd,zd = 0.0,-0.1,-0.9
# (pp|pp) class, (px py | pz pz)
# Hamilton Schaefer: Target angular momentum vector is (1,1,2), so elements filled are
# (ss|ss), (px s | s s), (py s | s s), (pz s | ss), (dzz s| s s)
la,ma,na = 1,2,1
lb,mb,nb = 0,0,0
lc,mc,nc = 1,0,0
ld,md,nd = 0,0,0
#la,ma,na = 1,1,2
alphaa,alphab,alphac,alphad = 0.5, 0.4, 0.3, 0.2

result = vrr((la,ma,na,lb,mb,nb,lc,mb,nc,ld,md,nd,xa,ya,za,xb,yb,zb,xc,yc,zc,xd,yd,zd,alphaa,alphab,alphac,alphad))
#print(result)

for i in result.reshape(-1,1):
    if not onp.allclose(i[0],0):
        print(i[0])

