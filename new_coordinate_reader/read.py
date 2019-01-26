import peslearn
import re
import collections
import numpy as np
import pandas as pd

xyz_re = peslearn.regex.xyz_block_regex 
atom_label = peslearn.regex.xyz_block_regex 

with open('OCHCO_7800.xyz') as f:
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
# convert atom labels to standard order (most common element first, alphabetical tiebreaker)
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

# write new xyz file with standard order
with open('std_' + 'OCHCO_7800.xyz', 'w+') as f:
    for i in range(len(energies)):
        f.write(energies[i] +'\n')  
        for j in range(natoms):
            f.write(geoms[i][j] +'\n') 

# remove everything from XYZs except floats and convert to numpy arrays
for i,geom in enumerate(geoms):
    for j,string in enumerate(geom):
        string = string.split()
        del string[0] # remove atom label
        geom[j] = np.asarray(string, dtype=np.float32)

# convert to interatomic distances
final_geoms = []
for i in geoms:
    idm = peslearn.geometry_transform_helper.get_interatom_distances(i)                        
    idm = idm[np.tril_indices(idm.shape[0],-1)]                 
    final_geoms.append(idm)

final_geoms = np.asarray(final_geoms) 
energies = np.asarray(energies, dtype=np.float32)

n_interatomics =  int(0.5 * (natoms * natoms - natoms))
bond_columns = []
for i in range(n_interatomics):
    bond_columns.append("r%d" % (i))
DF = pd.DataFrame(data=final_geoms, columns=bond_columns)    
DF['E'] = energies                                               
DF.to_csv('PES.dat',index=False,  float_format='%12.10f')
 
 
                                          

    
    

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

