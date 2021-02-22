import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.experimental import loops


@jax.jit
def f1(n):                                                     
  '''Note: switch to float for high values (n>20) for stability'''    
  with loops.Scope() as s:                                            
    s.result = 1
    s.k = 1                                                           
    for _ in s.while_range(lambda: s.k < n + 1):                      
      s.result *= s.k                                                 
      s.k += 1                                                        
    return s.result                                                   

@jax.jit
def f2(n):
    k = jax.lax.exp(jax.lax.lgamma(n + 1))
    return k

@jax.jit
def f3(n):
    return fact_array[n]

seed = jax.random.PRNGKey(0)
Rint = jax.random.randint(seed, (10000,), 0, 10)
Rflt = Rint.astype('float64')

def benchmark_f1_1(R):
    with loops.Scope() as s:
        s.dummy = 0.0
        s.i = 0
        for _ in s.while_range(lambda: s.i < R.shape[0]):
            s.dummy += f1(R[s.i])
            s.i += 1
        return s.dummy

def benchmark_f1_2(R):
    with loops.Scope() as s:
        s.dummy = 0.0
        for i in s.range(R.shape[0]):
            s.dummy += f1(R[i])
        return s.dummy

def benchmark_f2_1(R):
    with loops.Scope() as s:
        s.dummy = 0.0
        s.i = 0
        for _ in s.while_range(lambda: s.i < R.shape[0]):
            s.dummy += f2(R[s.i])
            s.i += 1
        return s.dummy

def benchmark_f2_2(R):
    with loops.Scope() as s:
        s.dummy = 0.0
        for i in s.range(R.shape[0]):
            s.dummy += f2(R[i])
        return s.dummy

def benchmark_f3_1(R):
    with loops.Scope() as s:
        s.dummy = 0.0
        s.i = 0
        for _ in s.while_range(lambda: s.i < R.shape[0]):
            s.dummy += f3(R[s.i])
            s.i += 1
        return s.dummy

def benchmark_f3_2(R):
    with loops.Scope() as s:
        s.dummy = 0.0
        for i in s.range(R.shape[0]):
            s.dummy += f3(R[i])
        return s.dummy


def binom(n,k):
    '''Binomial coefficient'''
    C = f1(n) // (f1(k) * f1(n-k))
    return C                                               


