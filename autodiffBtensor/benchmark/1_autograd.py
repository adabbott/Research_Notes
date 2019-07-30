import autograd.numpy as np
import autograd
from autograd import elementwise_grad
from autograd import grad
from autograd import value_and_grad
from autograd import grad_named
from autograd import multigrad_dict

import time
a = time.time()

np.set_printoptions(precision=4, linewidth=150)

# NOTES: functions may take a scalar or vector, and return a scalar or vector
# autograd.jacobian can be used to obtain 

def computation(p):
    a = p.reshape(3,3)
    inp2 = np.cos(a)
    final = np.sum(inp2,1)
    return final

test = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]]).flatten()

result = computation(test)
print(result)

gradients = autograd.jacobian(computation, 0)
hessian = autograd.jacobian(gradients, 0)
cubic = autograd.jacobian(hessian, 0)
quartic = autograd.jacobian(cubic, 0)
b = time.time()
print(np.round(gradients(test),4))
#print(np.round(hessian(test),4))
#print(np.round(cubic(test),4))
#print(np.round(quartic(test),4))
print(b-a)
