import jax 
from jax import lax
import numpy as onp
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
#https://github.com/rpmuller/pyquante2/blob/master/pyquante2/ints/two.py

#def factorial(n):
#  n = n.astype(float)
#  return jax.lax.exp(jax.lax.lgamma(n + 1))

@jax.jit
def factorial(n):
  with loops.Scope() as s:
    s.result = 1
    s.k = 1
    for _ in s.while_range(lambda: s.k < n + 1):
      s.result *= s.k
      s.k += 1
    return s.result

@jax.jit
def binom(x,y):
    C = factorial(x) / (factorial(x-y) * factorial(y))
    return C

@jax.jit
def fB(i,l1,l2,P,A,B,r,g): 
    return binomial_prefactor(i,l1,l2,P-A,P-B) * B0(i,r,g)

@jax.jit
def B0(i,r,g): 
    return fact_ratio2(i,r) * (4*g)**(r-i)

@jax.jit
def fact_ratio2(a,b):
    return factorial(a)/factorial(b)/factorial(a-2*b)

#@jax.jit
#def binomial_prefactor(s,ia,ib,xpa,xpb):
#    with loops.Scope() as L:
#        L.total = 0.0
#        L.s = s 
#        L.ia = ia
#        L.ib = ib
#        L.xpa = xpa
#        L.xpb = xpb
#
#        L.t = s + 1
#    
#        for _ in L.while_range(lambda: L.t > -1):
#            for _ in L.cond_range( (L.s - L.ia) <= L.t):
#                for _ in L.cond_range(L.t <= L.ib):
#                    L.total += binom(L.ia,L.s - L.t) * binom(L.ib,L.t) * \
#                               L.xpa**(L.ia-L.s+L.t) * L.xpb**(L.ib-L.t)
#            L.t -= 1
#        return L.total

@jax.jit
def binomial_prefactor(s,ia,ib,xpa,xpb):
    with loops.Scope() as L:
        L.total = 0.
        L.t = 0
        for _ in L.while_range(lambda: L.t < s + 1):
          for _ in L.cond_range(s-ia <= L.t):
            for _ in L.cond_range(L.t <= ib):
              L.total += binom(ia,s-L.t) * binom(ib,L.t) * xpa**(ia-s + L.t) * xpb**(ib - L.t)
          L.t += 1
        return L.total

@jax.jit
def gaussian_product(alpha1,A,alpha2,B):
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)

@jax.jit
def B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,Px,Ax,Bx,Qx,Cx,Dx,gamma1,gamma2,delta):
    val = fB(i1,l1,l2,Px,Ax,Bx,r1,gamma1) \
           * (-1)**i2 * fB(i2,l3,l4,Qx,Cx,Dx,r2,gamma2) \
           * (-1)**u * fact_ratio2(i1+i2-2*(r1+r2),u) \
           * (Qx-Px)**(i1+i2-2*(r1+r2)-2*u) \
           / delta**(i1+i2-2*(r1+r2)-u)
    return val

@jax.jit
def B_array(l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta):
    # This originally made arrays with argument-dependent shapes. Need fix size for jit compiling
    # Hard code only up to f functions (fxxx, fxxx | fxxx, fxxx) = l1 + l2 + l3 + l4 
    #Imax = l1+l2+l3+l4+1 # wait do i need to add 1 to make 13?
    #B = np.zeros(Imax)
    with loops.Scope() as s:
      s.B = np.zeros(12)
      #s.B = np.zeros(13)
      s.i2 = 0
      s.r1 = 0
      s.r2 = 0
      s.u = 0 
      s.i1 = l1 + l2  
      for _ in s.while_range(lambda: s.i1 > -1):   
        s.r1 = s.i1 // 2
        for _ in s.while_range(lambda: s.r1 > -1):
          s.i2 = l3 + l4 
          for _ in s.while_range(lambda: s.i2 > -1):
            s.r2 = s.i2 // 2
            for _ in s.while_range(lambda: s.r2 > -1):
              s.u = (s.i1 + s.i2) // 2 - s.r1 - s.r2 
              for _ in s.while_range(lambda: s.u > -1):
                I = s.i1 + s.i2 - 2 * (s.r1 + s.r2) - s.u 
                term = B_term(s.i1,s.i2,s.r1,s.r2,s.u,l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta) 
                s.B = jax.ops.index_add(s.B, I, term)
                s.u -= 1
              s.r2 -= 1
            s.i2 -= 1
          s.r1 -= 1
        s.i1 -= 1
      return s.B

@jax.jit
def boys(m,x):
    return 0.5 * (x + 1e-9)**(-(m + 0.5)) * jax.lax.igamma(m + 0.5, x + 1e-9) * np.exp(jax.lax.lgamma(m + 0.5))
    #return 0.5 * x * m

#def boys(n,x):
#    result = np.where(x < 1e-8, 1 / (2 * n + 1) - x *  (1 / (2 * n + 3)), 
#                      0.5 * (x)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5,x) * np.exp(jax.lax.lgamma(n + 0.5)))
#    return result

def contracted_tei(La,Lb,Lc,Ld, A, B, C, D, exp1, exp2, exp3, exp4, c1, c2, c3, c4): 
    """
    Computes a single contracted two electron integral given angular momentum vectors, centers, 
    and arrays of exponents and contraction coefficients for the primitives.
    """
    la, ma, na = La
    lb, mb, nb = Lb
    lc, mc, nc = Lc
    ld, md, nd = Ld
    xa,ya,za = A 
    xb,yb,zb = B 
    xc,yc,zc = C 
    xd,yd,zd = D 

    rab2 = np.dot(A-B,A-B)
    rcd2 = np.dot(C-D,C-D)

    with loops.Scope() as s:
      s.contracted_eri = 0.
      s.primitive = 0.
      s.i = 0
      s.j = 0
      s.k = 0
      s.l = 0
      s.I = 0
      s.J = 0  
      s.K = 0 
      # contraction loops
      # Problem: this loop size is locked into one size due to padding.
      # solution: add conditional to see if 'coef' is 0. this makes compilation time ridiculous
      for _ in s.while_range(lambda: s.i < c1.shape[0]):
        s.j = 0 
        for _ in s.while_range(lambda: s.j < c2.shape[0]):
          s.k = 0 
          for _ in s.while_range(lambda: s.k < c3.shape[0]):
            s.l = 0 
            for _ in s.while_range(lambda: s.l < c4.shape[0]):
              # compute this particular primitive, add to s.contracted_eri
              coef = c1[s.i] * c2[s.j] * c3[s.k] * c4[s.l]
              #TODO see if faster
              for _ in s.cond_range(coef > 0.):
                aa, bb, cc, dd = exp1[s.i], exp2[s.j], exp3[s.k], exp4[s.l]
                # all of these are dependent on different orbital exponents
                # so they must be accessed inside contraction loop
                xyzp = gaussian_product(aa,A,bb,B)
                xyzq = gaussian_product(cc,C,dd,D)
                xp,yp,zp = xyzp
                xq,yq,zq = xyzq
                rpq2 = np.dot(xyzp-xyzq,xyzp-xyzq)
                gamma1 = aa + bb
                gamma2 = cc + dd
                delta = 0.25*(1/gamma1+1/gamma2)
                Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
                By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
                Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)
                boys_arg = 0.25*rpq2/delta
                boys_eval = boys(np.arange(12), boys_arg) # f functions
                # computation loops
                s.primitive = 0.
                s.I = 0 
                for _ in s.while_range(lambda: s.I < la + lb + lc + ld + 1):
                  s.J = 0 
                  for _ in s.while_range(lambda: s.J < ma + mb + mc + md + 1):
                    s.K = 0 
                    for _ in s.while_range(lambda: s.K < na + nb + nc + nd + 1):
                      #s.primitive += Bx[s.I] * By[s.J] * Bz[s.K] * boys(s.I + s.J + s.K, boys_arg)
                      s.primitive += Bx[s.I] * By[s.J] * Bz[s.K] * boys_eval[s.I + s.J + s.K]
                      s.K += 1
                    s.J += 1
                  s.I += 1
                value = 2*jax.lax.pow(np.pi,2.5)/(gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
                        *np.exp(-aa*bb*rab2/gamma1) \
                        *np.exp(-cc*dd*rcd2/gamma2)*s.primitive*coef
                s.contracted_eri += value
              s.l += 1
            s.k += 1
          s.j += 1
        s.i += 1
      return s.contracted_eri

#
### Test single evaluation
#xyza = np.array([0.0,0.1,0.9])
#xyzb = np.array([0.0,-0.1,-0.9])
#xyzc = np.array([0.0,-0.1, 0.9])
#xyzd = np.array([0.0,-0.1,-0.9])
#coef = np.array([1.0])
#aa = np.array([0.5])
#bb = np.array([0.4])
#cc = np.array([0.3])
#dd = np.array([0.2])
#La = [0,0,0]
##
#result = contracted_tei(La,La,La,La,xyza,xyzb,xyzc,xyzd,aa,bb,cc,dd,coef,coef,coef,coef)
#print(result)


# Test vectorized evaluation
#K = 10000
#K = 100000
#A = np.asarray(onp.tile(np.array([0.0,0.1,0.9]),  (K,1)))
#B = np.asarray(onp.tile(np.array([0.0,-0.1,-0.9]),(K,1)))
#C = np.asarray(onp.tile(np.array([0.0,-0.1, 0.9]),(K,1)))
#D = np.asarray(onp.tile(np.array([0.0,-0.1,-0.9]),(K,1)))
#aa = np.asarray(onp.tile(np.array([0.5,0.4,0.3]),  (K,1)))
#coef = np.asarray(onp.tile(np.array([0.5,0.4,0.3]),  (K,1)))
#
#L = np.asarray(onp.tile(np.array([1,0,0,1,0,0,1,0,0,1,0,0]), (K,1)))
#vmapped = jax.vmap(contracted_tei, (0,0,0,0,0,0,0,0,0,0,0,0,0)) 
##result = vmapped(L,A,B,C,D,aa,aa,aa,aa,coef,coef,coef,coef)
#
#result = jax.lax.map(contracted_tei, (L,A,B,C,D,aa,aa,aa,aa,coef,coef,coef,coef))

#a    = np.repeat(0.5, K) 
#b    = np.repeat(0.4, K)
#c    = np.repeat(0.3, K)
#d    = np.repeat(0.2, K)
#coef = np.repeat(1.0, K)
#
#vmapped = jax.vmap(tei, (0,0,0,0,0,0,0,0,0,0)) 
#result = vmapped(L,A,B,C,D,a,b,c,d,coef)
#result = vmapped(la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,A,B,C,D,a,b,c,d,coef)


#for i in range(100000):
#    result = coulomb_repulsion(xyza,norma,lmna,aa,xyzb,normb,lmnb,bb,xyzc,normc,lmnc,cc,xyzd,normd,lmnd,dd)

