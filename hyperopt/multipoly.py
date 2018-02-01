from hyperopt import hp, fmin, tpe, rand, space_eval
import numpy as np
import matplotlib.pyplot as plt

# search space

space = hp.uniform('x', -1, 1)

def polynomial(x):
    return 3 * x ** 4 - 2 * x ** 3 + x - 6

score1 = fmin(polynomial, space, algo=rand.suggest, max_evals=100)
print('random search algorithm results:', score1)

score2 = fmin(polynomial, space, algo=tpe.suggest, max_evals=100)
print('tree of parzen estimators algorithm results:', score2)

datain = np.linspace(-1,1,200)
dataout = polynomial(datain)

p = plt.plot(datain, dataout)
plt.show(p)

