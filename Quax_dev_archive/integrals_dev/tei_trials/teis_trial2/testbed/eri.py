import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from integrals_utils import * 
np.set_printoptions(precision=12)


def eri_data(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Computes zeta, eta, P, Q'''
    zeta = aa + bb
    eta = cc + dd
    P = (aa * A + bb * B) * (1/zeta)
    Q = (cc * C + dd * D) * (1/eta)
    return zeta, eta, P, Q 

def eri_ssss_0(A, B, C, D, aa, bb, cc, dd, coeff):
    '''
    Helper function for higher angular momentum integrals.
    Computes ssss without the boys function, and several intermediates which appear in the Obara Saika scheme.
    '''
    zeta, eta, P, Q = eri_data(A, B, C, D, aa, bb, cc, dd, coeff)
    AmB = A-B
    CmD = C-D
    K_ab = (1/zeta) * jax.lax.exp((-aa * bb * (1/zeta)) * jax.lax.dot(AmB,AmB))
    K_cd = (1/eta)  * jax.lax.exp((-cc * dd * (1/eta)) * jax.lax.dot(CmD,CmD))
    PmQ = P - Q
    W = (zeta * P + eta * Q) / (zeta + eta)
    boys_arg = (zeta * eta / (zeta + eta)) * jax.lax.dot(PmQ,PmQ)
    ssss = 2 * np.pi**(10/4) * coeff * (zeta + eta)**(-1/2) * K_ab * K_cd
    return ssss, zeta, eta, P, Q, W, boys_arg

def eri_ssss(A, B, C, D, aa, bb, cc, dd, coeff):
    """ 
    Computes a [ss|ss] integral. Should not be used except to  
    """
    tmp_ssss, zeta, eta, P, Q, W, boys_arg = eri_ssss_0(A, B, C, D, aa, bb, cc, dd, coeff)
    #TODO turn on boys
    ssss = tmp_ssss * boys0(boys_arg)
    #ssss = tmp_ssss * general_boys_taylor(0,boys_arg)
    #ssss = tmp_ssss #* boys0(boys_arg)
    return ssss

# New ssss and psss functions
def new_eri_ssss(A, B, C, D, aa, bb, cc, dd, coeff):
    tmp_ssss, zeta, eta, P, Q, W, boys_arg = eri_ssss_0(A, B, C, D, aa, bb, cc, dd, coeff)
    ssss = tmp_ssss * new_boys0(boys_arg) 
    return ssss

def new_eri_psss(A, B, C, D, aa, bb, cc, dd, coeff):
    ssss, zeta, eta, P, Q, W, boys_arg = eri_ssss_0(A, B, C, D, aa, bb, cc, dd, coeff)
    #psss = ssss * ((P - A) * new_boys0(boys_arg) + (W - P) * new_boys1(boys_arg))
    #TODO 
    F0 = new_boys0(boys_arg)
    F1 = (F0 - np.exp(-boys_arg)) / (2*(boys_arg + 1e-12))
    psss = ssss * ((P - A) * F0 + (W - P) * F1)
    return psss

def eri_psss(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 3 integrals of psss class'''
    oot_aa = 1 / (2 * aa)
    return oot_aa * jax.jacfwd(eri_ssss, 0)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_psps(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 9 integrals (shape=(3,3)) of psps class'''
    oot_cc = 1 / (2 * cc)
    return oot_cc * jax.jacfwd(eri_psss, 2)(A, B, C, D, aa, bb, cc, dd, coeff)


#def new_eri_psps(A, B, C, D, aa, bb, cc, dd, coeff):
#    P =  



def eri_ppss(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 9 integrals (shape=(3,3)) of ppss class'''
    oot_bb = 1 / (2 * bb)
    return oot_bb * jax.jacfwd(eri_psss, 1)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_ppps(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 27 integrals (shape=(3,3,3)) of ppps class'''
    oot_cc = 1 / (2 * cc)
    return oot_cc * jax.jacfwd(eri_ppss, 2)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_pppp(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 81 integrals (shape=(3,3,3,3)) of pppp class'''
    oot_dd = 1 / (2 * dd)
    return oot_dd * jax.jacfwd(eri_ppps, 3)(A, B, C, D, aa, bb, cc, dd, coeff)


#def new_eri_psps(A, B, C, D, aa, bb, cc, dd, coeff):
#    P =  

def transfer_bra(upper, lower, args):
    '''
    Use horizontal recurrence relation to transfer angular momentum 
    't' times from left side of bra (center A) to right side of bra (center B)
    
    upper : array 
        The higher angular momentum term [(a+1i)b|cd] 
    lower : array
        The lower angular momentum term [ab|cd] 
    args: tuple
        A tuple of ERI arguments, defined elsewhere: (A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)

    '''
    AmB = args[0] - args[1]
    result = upper + np.broadcast_to(AmB, upper.shape) * np.broadcast_to(lower, upper.shape)
    return result

def transfer_ket(upper, lower, args):
    '''
    Use horizontal recurrence relation to transfer angular momentum 
    't' times from left side of ket (center C) to right side of ket (center D)
    
    upper : array 
        The higher angular momentum term [ab|(c+1i)d] 
    lower : array
        The lower angular momentum term [ab|cd] 
    args: tuple
        A tuple of ERI arguments, defined elsewhere: (A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)

    '''
    AmB = args[2] - args[3]
    result = upper + np.broadcast_to(AmB, upper.shape) * np.broadcast_to(lower, upper.shape)
    return result

# Create four distinct cartesian centers of atoms
#H       -0.4939594255     -0.2251760374      0.3240754142                 
#H        0.4211401526      1.8106751596     -0.1734137286                 
#H       -0.5304044183      1.5987236612      2.0935583523                 
#H        1.9190079941      0.0838367286      1.4064021040                 
A = np.array([-0.4939594255,-0.2251760374, 0.3240754142])
B = np.array([ 0.4211401526, 1.8106751596,-0.1734137286])
C = np.array([-0.5304044183, 1.5987236612, 2.0935583523])
D = np.array([ 1.9190079941, 0.0838367286, 1.4064021040])

alpha = 0.2
beta = 0.3
gamma = 0.4
delta = 0.5
coeff = 1.0

args = (A, B, C, D, alpha, beta, gamma, delta, coeff)

#print('psss', eri_psss(*args))
#print('psps', eri_psps(*args))


#f = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(new_eri_psss))))
#for i in range(10):
#    a = f(A,B,C,D,alpha,beta,gamma,delta,coeff)

print(new_eri_psss(A,B,C,D,alpha,beta,gamma,delta,coeff))
print(eri_psss(A,B,C,D,alpha,beta,gamma,delta,coeff))
test1 = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(new_eri_psss))))(A,B,C,D,alpha,beta,gamma,delta,coeff)
test2 = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_psss))))(A,B,C,D,alpha,beta,gamma,delta,coeff)
print(np.allclose(test1,test2))


#print(transfer_bra(eri_psss(*args), eri_ssss(*args), args))


print('OLD functions')
jaxpr = jax.make_jaxpr(eri_ssss)(*args)
print('old (ss|ss)', len(str(jaxpr).splitlines()))
jaxpr = jax.make_jaxpr(eri_psss)(*args)
print('old (ps|ss)', len(str(jaxpr).splitlines()))
jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_ssss)))))(*args)
print('old (ss|ss)', len(str(jaxpr).splitlines()))
jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_psss)))))(*args)
print('old (ps|ss)', len(str(jaxpr).splitlines()))


print('NEW functions')
jaxpr = jax.make_jaxpr(new_eri_ssss)(*args)
print('new (ss|ss)', len(str(jaxpr).splitlines()))
jaxpr = jax.make_jaxpr(new_eri_psss)(*args)
print('new (ps|ss)', len(str(jaxpr).splitlines()))

print('quartic')
jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(new_eri_ssss)))))(*args)
print('new (ss|ss)', len(str(jaxpr).splitlines()))
jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(new_eri_psss)))))(*args)
print('new (ps|ss)', len(str(jaxpr).splitlines()))

#jaxpr = jax.make_jaxpr(eri_pppp)(*args)
#print('(pp|pp)', len(str(jaxpr).splitlines()))


#jaxpr = jax.make_jaxpr(eri_psss)(*args)
#print('old (ps|ss)', len(str(jaxpr).splitlines()))

#print('quartic')
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_ssss)))))(*args)
#print('(ss|ss)', len(str(jaxpr).splitlines()))
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_psss)))))(*args)
#print('old (ps|ss)', len(str(jaxpr).splitlines()))
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(new_eri_psss)))))(*args)
#print('new (ps|ss)', len(str(jaxpr).splitlines()))
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_psps)))))(*args)
#print('old (ps|ps)', len(str(jaxpr).splitlines()))

#print('test function with just boys')
#jaxpr = jax.make_jaxpr(test)(*args)
#print('boys', len(str(jaxpr).splitlines()))
#print('quartic')
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(test)))))(*args)
#print('boys', len(str(jaxpr).splitlines()))


#jaxpr = jax.make_jaxpr(new_eri_psss)(*args)
#print(jaxpr)
#jaxpr = jax.make_jaxpr(new_eri_psss2)(*args)
#print(jaxpr)

#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(new_eri_psss2,0),0),0),0))(*args)
#print(jaxpr)

