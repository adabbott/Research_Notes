from hyperopt import hp, fmin, tpe, rand, space_eval

# search space

space = hp.uniform('x', -1, 1)

# function f(x) = x^2
def polynomial(x):
    return x ** 2  

score1 = fmin(polynomial, space, algo=rand.suggest, max_evals=1000)
print('random search algorithm results:', score1)

score2 = fmin(polynomial, space, algo=tpe.suggest, max_evals=1000)
print('tree of parzen estimators algorithm results:', score2)

