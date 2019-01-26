import peslearn
import re
import collections

xyz_re = peslearn.regex.xyz_block_regex 
atom_label = peslearn.regex.xyz_block_regex 

with open('OCHCO_7800.xyz') as f:
    # read it
    data=f.read()
    # extract energies
    energies = re.findall("\n\s*(\d+\.\d+)\s*\n", data)
    # extract geometries
    geoms = re.findall(xyz_re, data)
    for i in range(len(geoms)):
        geoms[i] = list(filter(None, geoms[i].split('\n')))
    sample = geoms[0]
    atom_labels = [re.findall('\w+', s)[0] for s in sample]
    print(atom_labels)
    sorted_atom_counts = collections.Counter(atom_labels).most_common() 
    sorted_atom_counts = sorted(sorted_atom_counts, key = lambda x: (-x[1], x[0]))
    sorted_atom_labels = []
    for tup in sorted_atom_counts:
        for i in range(tup[1]): 
            sorted_atom_labels.append(tup[0])
    print(sorted_atom_labels)
    # find the permutation vector which maps unsorted atoms to standard order atoms
    p = []
    for i,j in enumerate(atom_labels):
        for k,l in enumerate(sorted_atom_labels):
            if j == l:
                p.append(k)
                sorted_atom_labels[k] = 'done'
                continue

    # permute all geometries to standard order 
    for g in range(len(geoms)):
        geoms[g] = [geoms[g][i] for i in p]
    print(geoms[5])
         
    

    
            
    # create list of atom labels
    #new = [mylst[i] for i in p] 

    #print(energies)
    #print(geoms[0].split('\n'))
    #mylst = geoms[0].split('\n')

    #print(list(filter(None, mylst)))
    #p = [2,3,0,1,4]
    #new = [mylst[i] for i in p] 
    #print(new)
