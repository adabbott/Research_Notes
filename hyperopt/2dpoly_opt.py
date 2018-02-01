from hyperopt import hp, fmin, tpe, rand, space_eval

# search space

space = [hp.uniform('x', 0, 1), hp.normal('y', 0, 1)]

# objective function
def polynomial(args):
    x, y = args
    return x ** 2 + y ** 2 


# optimize the function (best result is obviously x=0 for x**2)
score1 = fmin(polynomial, space, algo=rand.suggest, max_evals=100)
print('random search algorithm results:', score1)

score2 = fmin(polynomial, space, algo=tpe.suggest, max_evals=100)
print('tree of parzen estimators algorithm results:', score2)
