
def my_func(a):
    import numpy as np
    import re
    from itertools import combinations
    import json
    import GPy

    # do some computation on vector a
    a = np.asarray(a)
    r = np.linalg.norm(a)
    return r

