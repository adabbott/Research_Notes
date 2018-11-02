
def my_func(a):
    import numpy as np
    import re
    from itertools import combinations
    import json
    import GPy
    r = np.cos(a)                 # use numpy
    junk1 = re.sub("y", "n", "ye")# use re
    c = combinations("ABCD", 2)   # use itertools
    for i in c:
        print(i)
    return r

a = my_func(5)
