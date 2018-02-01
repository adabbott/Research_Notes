from hyperopt import hp, fmin, tpe, rand, space_eval
import numpy as np
import matplotlib.pyplot as plt

# search space

space = hp.uniform('x', -1, 1)

def polynomial(x):
    return (1 / 16) * (231 * x ** 6 - 315 * x ** 4 + 105 * x**2 - 5) 

score1 = fmin(polynomial, space, algo=rand.suggest, max_evals=100)
print('random search algorithm results:', score1)

score2 = fmin(polynomial, space, algo=tpe.suggest, max_evals=100)
print('tree of parzen estimators algorithm results:', score2)

datain = np.linspace(-1,1,200)
dataout = polynomial(datain)

p = plt.plot(datain, dataout)
plt.show(p)

