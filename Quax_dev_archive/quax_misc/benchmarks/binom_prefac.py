

# Given two angular momenta values, explore 
def bf(i1, l1, l2, PAx, PBx):
    '''
    i1 starts at l1 + l2, and gets reduced to 0 
    
    '''
    total = 0.
    t = 0
    while t < i1 + 1:
        if (i1-l1) <= t <= l2:
            total += binomials[l1,i1-t] * binomials[l2,t] * PAx**(l1 - i1 + t) * PBx**(l2 - t)
            #print("PAx", l1 - i1 + t)
            print("PBx", l2 - t)
        t += 1


def bf(l1, l2, PAx, PBx):
    total = 0.
    t = 0
    while t < l1 + l2 + 1:
        if (i1-l1) <= t <= l2:
            total += binomials[l1,i1-t] * binomials[l2,t] * PAx**(l1 - i1 + t) * PBx**(l2 - t)
            #print("PAx", l1 - i1 + t)
            print("PBx", l2 - t)
        t += 1

l1,l2 = 0,0

l1,l2 = 1,0

l1,l2 = 2,0

l1,l2 = 3,0

l1,l2 = 0,1

l1,l2 = 1,1

l1,l2 = 2,1

l1,l2 = 3,1
