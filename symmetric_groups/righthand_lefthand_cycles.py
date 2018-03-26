
def find_cycles(perm):
    """
    Finds the cycle(s) required to get the permutation. For example,
    the permutation [3,1,2] is obtained by permuting [1,2,3] with the cycle [1,2,3]
    read as "1 goes to 2, 2 goes to 3, 3 goes to 1"
    Sometimes cycles are products of more than one subcycle, e.g. (12)(34)(5678)
    This function is to find them all. Ripped bits and pieces of this off from SE, 
    don't completely understand it but it works xD
    """
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break
        cycles.append(cycle[::-1])

    # only save cycles of size 2 and larger
    cycles[:] = [cyc for cyc in cycles if len(cyc) > 1]

    return cycles


def find_cycles_v2(perm):
    cycles = []
    initial = set(perm)
    while len(initial) > 0:
        n = initial.pop()
        cycle = [n]
        while True:
            n = perm[n]
            if n not in initial:
                break
            initial.remove(n)
            cycle.append(n)
        #cycles.append(cycle[::-1])
        cycles.append(cycle)
   
    cycles[:] = [cyc for cyc in cycles if len(cyc) > 1] 
            
    ## only save cycles of size 2 and larger
    #new_cycles = []
    #for cyc in cycles:
    #    if len(cyc) > 1: 
    #        new_cycles.append(cyc)
    #return new_cycles
    return cycles


print(find_cycles_v2([0,1,2,3,4,5]))
print(find_cycles([0,1,2,3,4,5]))
print(find_cycles_v2([0,1,3,2,4,5]))
print(find_cycles([0,1,3,2,4,5]))
print(find_cycles_v2([0,1,2,3,5,4]))
print(find_cycles([0,1,2,3,5,4]))
print(find_cycles_v2([2,1,0,5,4,3]))
print(find_cycles([2,1,0,5,4,3]))
print(find_cycles_v2([1,2,0,4,5,3]))
print(find_cycles([1,2,0,4,5,3]))
print(find_cycles_v2([2,0,1,5,3,4]))
print(find_cycles([2,0,1,5,3,4]))
print(find_cycles_v2([1,0,2,5,3,4]))
print(find_cycles([1,0,2,5,3,4]))

#print(find_cycles_v2([2,3,4,0,1]))
#print(find_cycles([2,3,4,0,1]))
