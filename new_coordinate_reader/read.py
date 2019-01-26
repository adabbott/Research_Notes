import peslearn
import re
import collections
import numpy as np

xyz_re = peslearn.regex.xyz_block_regex 
atom_label = peslearn.regex.xyz_block_regex 

with open('OCHCO_7800.xyz') as f:
    # read it
    data = f.read()
# extract energies
energies = re.findall("\n\s*(-?\d+\.\d+)\s*\n", data)
# extract geometries and clean up format
geoms = re.findall(xyz_re, data)
for i in range(len(geoms)):
    geoms[i] = list(filter(None, geoms[i].split('\n')))
# find atom labels
sample = geoms[0]
atom_labels = [re.findall('\w+', s)[0] for s in sample]
natoms = len(atom_labels)
# convert atom labels to standard order
sorted_atom_counts = collections.Counter(atom_labels).most_common() 
sorted_atom_counts = sorted(sorted_atom_counts, key = lambda x: (-x[1], x[0]))
sorted_atom_labels = []
for tup in sorted_atom_counts:
    for i in range(tup[1]): 
        sorted_atom_labels.append(tup[0])
# find the permutation vector which maps unsorted atom labels to standard order atom labels
p = []
for i,j in enumerate(atom_labels):
    for k,l in enumerate(sorted_atom_labels):
        if j == l:
            p.append(k)
            sorted_atom_labels[k] = 'done'
            continue

# permute all xyz geometries to standard order 
for g in range(len(geoms)):
    geoms[g] = [geoms[g][i] for i in p]

# write new xyz with standard order
with open('new_' + 'OCHCO_7800.xyz', 'w+') as f:
    for i in range(len(energies)):
        f.write(energies[i] +'\n')  
        for j in range(natoms):
            f.write(geoms[i][j] +'\n') 

# remove everything from XYZs except floats                 
#for i,g in enumerate(geoms):
#    for j,s in enumerate(g):
#        s = s.split()
#        del s[0]
#        g[j] = s
#        print(s)#g = [a.split() for a in g]
#print(np.asarray(geoms[0], dtype=np.float32))
    
    

#for g in geoms:                                             
#    for a in g:                                             
#        print(i)
        #if re.match('[A-Za-z]+', a):                        
        #    g.remove(i)                                     
#print(geoms)
#final_geoms = []                                            
## get interatom distances                                   
#for i in geoms:                                             
#    a = np.asarray(i, dtype=np.float32).reshape(natoms,3)   
#    b = get_interatom_distances(a)                          
#    # trim interatom distance matrix to just unique bonds   
#    # -1 means ignore diagonal                              
#    b = b[np.tril_indices(b.shape[0],-1)]                   
#    final_geoms.append(b)                                   
#final = np.asarray(final_geoms)                             

