import matplotlib.pyplot as plt
import timeit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats


# Load in data and sort
data = pd.read_csv("h2co.dat")
#data = pd.read_csv("h2o.dat")
#data = pd.read_csv("h3o+.dat")
data = data.sort_values("E")
#data['E'] = (data['E'] - data['E'].min())
max_e = data['E'].max()
data = data.sort_values("E")

ntrain = 5000

def fast_structure_based(data=data,ntrain=ntrain):
    data = data.sort_values("E").reset_index(drop=True)
    train = []
    # accept lowest energy point as 1st training point
    train.append(data.values[0])
    # accept farthest point from first training point as 2nd training point
    tmp1 = np.tile(train[0][:-1], (data.shape[0],1))
    tmp2 = data.values[:,:-1]
    diff = tmp1 - tmp2
    norm_vector = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    idx = np.argmax(norm_vector)
    newtrain = data.values[idx]
    train.append(newtrain)

    # update norm matrix with second training point
    tmp1 = np.tile(train[1][:-1], (data.shape[0],1))
    tmp2 = data.values[:,:-1]
    diff = tmp1 - tmp2
    norm_vector_2 = np.sqrt(np.einsum('ij,ij->i', diff, diff))

    norm_matrix = np.vstack((norm_vector, norm_vector_2))
    # the minimum values along the columns of this 2xN array of norms
    min_array = np.amin(norm_matrix, axis=0)
    while len(train) < ntrain:
        # min_array contains the smallest norms into the training set, by datapoint.
        # We take the largest one.
        idx = np.argmax(min_array)
        new_geom = data.values[idx]
        train.append(new_geom)
        # update norm matrix with the norms of newly added training point
        tmp1 = np.tile(train[-1][:-1], (data.shape[0],1))
        tmp2 = data.values[:,:-1]
        diff = tmp1 - tmp2
        norm_vec = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        stack = np.vstack((min_array, norm_vec))
        min_array = np.amin(stack, axis=0)
    train = np.asarray(train).reshape(ntrain,len(data.columns))
    train = pd.DataFrame(train, columns=data.columns).sort_values("E")
    return train
        


def old_bad_fast_structure_based(data=data,ntrain=ntrain):
    """
    Will only work properly if you remove redundancies (?)
    """
    data = data.sort_values("E").reset_index(drop=True)
    print(data)
    # accept lowest energy point
    train = []
    train.append(data.values[0])

    # accept farthest point from lowest energy point
    tmp1 = np.tile(train[0][:-1], (data.shape[0],1))
    tmp2 = data.values[:,:-1]
    diff = tmp1 - tmp2
    norm_matrix = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    #print(norm_matrix)
    idx = np.argmax(norm_matrix)
    #print(idx)
    newtrain = data.values[idx]
    train.append(newtrain) 

    # fill in rest of points
    #for i in range(ntrain):
    i = 0
    while len(train) < ntrain:
        # take last element of training set, compute norm vector with whole dataset
        #start = timeit.default_timer()
        #print(i)
        tmp1 = np.tile(train[-1][:-1], (data.shape[0], 1))
        #print(train[-1][:-1])
        tmp2 = data.values[:,:-1]
        diff = tmp1 - tmp2
        norm_vec = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        # add norm vector to norm matrix
        norm_matrix = np.vstack((norm_matrix, norm_vec))
        print(norm_matrix)
#        print("One iteration of generating norm matrix takes {} seconds".format(round((timeit.default_timer() - start),2)))
        # find smallest nonzero norm in each row (corresponding to each training point) of norm matrix
        # one is guaranteed to be 0, so it is the second smallest.
        # 'idx' is the column indices of second smallest norm of each row (training point):
        # this is giving same exact smallest norm every time for older training point norms; need to not count norms already used.
        #idx = np.argpartition(norm_matrix, 1)[:,1]
        # for every training point, take smallest unused normo

        # NEW APPROACH: compare smallest nonzero norms, below we set each used norm location in norm matrix to 0.0
        # smallest nonzero value of whole norm matrix
        #mask = np.ma.masked_values(norm_matrix, 0.0, copy=False)
        # smallest nonzero norm of each row of norm matrix (training set points) 
        mins = np.amin(np.ma.masked_values(norm_matrix, 0.0, copy=False), axis=1)
        min_indices = np.argmin(np.ma.masked_values(norm_matrix, 0.0, copy=False), axis=1)
        print(min_indices)
        #if i % 2 == 0:
        #    sample = np.argmax(mins)
        #else:
        #    sample = np.argmin(mins)
        sample = np.argmax(mins)
        #new_train_point_idx = min_indices[np.argmax(mins)]
        new_train_point_idx = min_indices[sample]
        print(new_train_point_idx)
        newtrain = data.values[new_train_point_idx]
        train.append(newtrain)
        norm_matrix[sample,new_train_point_idx] = 0.0
        #norm_matrix[np.argmax(mins),new_train_point_idx] = 0.0
        i += 1
    print(np.asarray(train).shape)
    train = np.asarray(train).reshape(ntrain,len(data.columns))
    train = pd.DataFrame(train, columns=data.columns).sort_values("E")
    return train

        ##print(min_indices)
        ## now figure out which one is bigger
        #candidates = [] 
        #for i, row in enumerate(norm_matrix):
        #    #print(norm_matrix[i,min_indices[i]])
        #    candidates.append(norm_matrix[i,min_indices[i]])
        #c = np.asarray(candidates)
        #new_training_point = data.values[np.argmax(c)]
        #norm_matrix[min_indices[np.argmax(c)], np.argmax(c)] = 0.0 
        #train.append(new_training_point)
            #print(row[min_indices[i]])
        #print(norm_matrix)
        #print(min_indices)
        #print(norm_matrix[row_idx, col_idx])
        # get coordinates of the
        #np.unravel_index(min_indices, norm_matrix.shape)
        #print(min_indices)
        #print(norm_matrix)
        #print(norm_matrix[[min_indices]])
"""
        #print(min_indices)
        # the index of the training set which contains the largest, smallest nonzero norm 
        idx = np.argmax(mins)
        print(idx)
        print(mins[idx])
        # new training point index in full dataset
        new = np.argmin(mask[idx])
        new_training_point = data.values[new]
        train.append(new_training_point)
        # set norm matrix value of taken norm equal to zero so it isnt used again
        #norm_matrix[idx,new] = 0.0
        # set norm matrix value of whole row corresponding to newly drawn trainingp oint to zero so it isnt considered anymore
        # BADD norm_matrix[idx,:] = 0.0
        norm_matrix[idx,new] = 0.0 
    train = np.asarray(train).reshape(ntrain,len(data.columns))
    train = pd.DataFrame(train, columns=data.columns).sort_values("E")
    #print(train)
    return train
"""           


#def structure_based(data=data,ntrain=ntrain):
#    data = data.sort_values("E")
#
#    train = []
#    test = data.copy()
#    # take lowest energy point
#    train.append(data.values[0])
#    test.drop([0], inplace=True)
#    #while len(train) < ntrain:
#    while len(train) < 20:
#        norms = np.zeros((test.shape[0], len(train)))
#        #print(norms.shape)
#        # do not need to loop over training points. Just want norms of last training point
#        for i, trainpoint in enumerate(train):
#            for j, row in enumerate(test.itertuples()):
#                tmp1 = np.asarray(row[1:-1])
#                tmp2 = trainpoint[:-1]
#                norm = np.sqrt(np.sum(np.square(tmp1-tmp2)))
#                norms[j,i] = norm
#        # find smallest norm of dataset for each current training point
#        mins = norms.min(axis=0)
#        #print(mins)
#        minargs = np.argmin(norms,axis=0) 
#        #print(minargs)
#        idx = minargs[np.argmax(mins)]
#        #print(idx)
#        #accepted_point = n
#        #print(minargs)
#        # TEMPORARY
#        temp = test.sample(n=1)
#        train.append(temp.values[0])
#        test.drop(temp.index[0],inplace=True)
def structure_based(data=data,ntrain=ntrain):
    """
    Very slow but works. Main problem:
    You are doing the same computations over and over again.
    Should consider FIXING the size of the test set (just keep full dataset),
    and EXPANDING the size of the norm matrix so you never have to compute the same norm again and again
    """
    train = []
    test = data.sort_values("E").reset_index(drop=True)
    # accept lowest energy point, delete from test set
    train.append(test.values[0])
    test = test.drop([0]).reset_index(drop=True)

    # accept farthest point from lowest energy point, delete from test set
    tmp1 = np.tile(train[0][:-1], (test.shape[0],1))
    tmp2 = test.values[:,:-1]
    diff = tmp1 - tmp2
    norms = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    idx = np.argmax(norms)
    train.append(test.values[idx])
    test = test.drop([idx]).reset_index(drop=True)
    # to accept anothing training point,
    # 1. Compute the distance of every point in the test dataset to each training point.
    # 2. Find the test dataset point which has the smallest distance to the training point, for each training point.
    # 3. Accept the test datapoint which has the LARGEST smallest distance to a training point.
    # 4. Add accepted datapoint to training set, delete accepted training point from test set
    while len(train) < ntrain:
        minnorms = []
        indices = []
        for i, trainpoint in enumerate(train):
            tmp1 = np.tile(trainpoint[:-1], (test.shape[0],1))
            tmp2 = test.values[:,:-1]
            diff = tmp1 - tmp2
            norms = np.sqrt(np.einsum('ij,ij->i', diff, diff))
            min_norm = norms.min()
            idx = np.argmin(norms)
            minnorms.append(min_norm)
            indices.append(idx)
        largest_smallest = max(minnorms)
        final_idx = indices[np.argmax(minnorms)]
        train.append(test.values[final_idx])
        #test.drop(final_idx).reset_index(drop=True, inplace=True)
        test = test.drop(final_idx).reset_index(drop=True)
    
    train = np.asarray(train).reshape(ntrain,len(data.columns))
    train = pd.DataFrame(train, columns=data.columns).sort_values("E")
    print(train)


            
start = timeit.default_timer()
sb = fast_structure_based()
print("Training set generation finished in {} seconds".format(round((timeit.default_timer() - start),2)))


def sobol(data=data,ntrain=ntrain):
    delta = 0.002278
    denom = (1 / (max_e + delta))
    train = []
    test = data.copy()
    
    while len(train) < ntrain:
        # randomly draw a PES datapoint energy
        rand_point = test.sample(n=1)
        #idx = np.random.choice(data.shape[0])
        #rand_point = data.iloc[[idx]]
        rand_E = rand_point["E"].values
        condition = (max_e - rand_E + delta) * denom 
        rand = np.random.uniform(0.0,1.0)
        ## if the datapoint is already accepted into training set, skip it
        #if any((rand_point.values == x).all() for x in train):
        #    continue
        # add to training set if sobol condition is satisfied. 
        # also remove it from the test dataset
        if condition > rand:
            train.append(rand_point.values)
            test = test.drop(rand_point.index[0])
    
    
    # prepare for pandas
    train = np.asarray(train).reshape(ntrain,len(data.columns))
    train = pd.DataFrame(train, columns=data.columns)
    return train.sort_values("E")

def energy_ordered(data=data,ntrain=ntrain):
#df1 = df[df.index % 3 != 0]  # Excludes every 3rd row starting from 0
#df2 = df[df.index % 3 == 0]  # Selects every 3rd raw starting from 0
#This assumes, of course, that you have an index column of ordered, consecutive, integers starting at 0.
    s = round(data.shape[0] / ntrain)
    train = data[0::s]
    return train 


def random(data=data,ntrain=ntrain):
    random = data.sample(n=ntrain)
    random = random.sort_values("E")
    return random

def norm(n1,n2):
    return np.sqrt(np.sum(np.square(n1-n2)))


#eo = energy_ordered()
#sobol = sobol(data)
#random = random(data)
#
nbins = 10
hist = pd.concat([data['E'], sb['E']],axis=1)
hist.columns = ['Full', 'Structure Based']# 'Random', 'EO']
hist.plot.hist(bins = nbins, stacked=False, density=True,alpha=0.5, color=['black', 'blue'])
plt.show()
#hist = pd.concat([data['E'], eo['E']],axis=1)
#hist.columns = ['Full','EO']
#hist.plot.hist(bins = nbins, stacked=False, density=True,alpha=0.7, color=['black', 'red'])
#plt.show()


















"""
plot1 = plt.hist(hist['Full'], bins=10, density=True,alpha=0.5)

plot2 = plt.hist(hist['Sobol'], bins=10, density=True,alpha=0.5)
p12 = np.absolute(np.asarray(plot1[0])-np.asarray(plot2[0]))

plot3 = plt.hist(hist['Random'], bins=10, density=True,alpha=0.5)
p13 = np.absolute(np.asarray(plot1[0])-np.asarray(plot3[0]))
plt.clf()

#plt.subplot(121)
diff = plt.bar([0,1,2,3,4,5,6,7,8,9], height=p12, alpha=0.5, color='black')
#plt.subplot(122)
diff = plt.bar([0,1,2,3,4,5,6,7,8,9], height=p13, alpha=0.5, color='blue')
plt.show()
"""

#plot1 = hist['Full'].plot.hist(density=True,alpha=0.5, color='black')
#plot2 = hist['Sobol'].plot.hist(density=True,alpha=0.5, color='blue')
##plot3 = hist['Random'].plot.hist(density=True,alpha=0.5, color='red')
#diff = plt.bar([0,1,2,3,4,5,6,7,8,9], height=(plot1[0]-plot2[0]))

#nbins = 8
#n, bins, patches = plt.hist(hist['Full'], nbins, facecolor='black', alpha=0.5)


#diff = plt.bar(
#hist.plot.hist(cumulative=True,alpha=0.5)

## Kolmogorov Smirnov test
#result1 = stats.ks_2samp(data['E'], sobol['E'])
#result2 = stats.ks_2samp(data['E'], random['E'])
#print(result1)
#print(result2)
#if result1[1] > result2[1]:
#    wins.append(1)
#else:
#    losses.append(1)
#print("Sobol wins  ", len(wins))
#print("Random wins  ", len(losses))
    
#result3 = stats.ks_2samp(sobol['E'], random['E'])
#print(result3)


#hist = hist_data_sobol_random.hist(bins=8)
#sobol_hist = train['E'].hist(bins=bin_values)
#data_hist = data['E'].hist(bins=bin_values)
#plt.figure()
#plt.show()

## sanity check
#temp = train.drop_duplicates('E')
#if len(temp.index) != len(train.index):
#    print("Alert!!!!")
#    print("Alert!!!!")
#    print("Alert!!!!")
#
## use minmax scaling 0 to 1
#x = np.asarray(data['E']) # / (data['E'].max() - data['E'].min())   #/ data['E'].max()
#y = np.asarray(train['E']) #/ (train['E'].max() - train['E'].min())
##x = np.asarray(data['E']) / data['E'].max()
##y = np.asarray(train['E']) / train['E'].max()
#X_train, X_test, y_train, y_test  = train_test_split(data,data,train_size=ntrain)
#z = np.asarray(X_train['E'])  #/ (X_train['E'].max() - X_train['E'].min())
##z = np.asarray(X_train['E']) / X_train['E'].max()


#nbins = 8
#n, bins, patches = plt.hist(x, nbins, histtype='bar', facecolor='green', alpha=0.5, normed=True)
#bin_centers = bins[:-1] + 0.5 * np.diff(bins) 
#
#plt.plot(bin_centers, n, color='green')
#n, bins, patches = plt.hist(y, nbins, histtype='bar', facecolor='blue', alpha=0.5, normed=True)
#bin_centers = bins[:-1] + 0.5 * np.diff(bins) 
#plt.plot(bin_centers, n, color='blue')
#
#n, bins, patches = plt.hist(z, nbins, histtype='bar', facecolor='black', alpha=0.5, normed=True)
#bin_centers = bins[:-1] + 0.5 * np.diff(bins) 
#plt.plot(bin_centers, n, color='black')
#
#plt.show()

#nbins = 8
#n1, bins1, patches = plt.hist(x, nbins, histtype='bar', facecolor='green', alpha=0.5, density=True)
#bin_centers1 = bins1[:-1] + 0.5 * np.diff(bins1) 
#plt.plot(bin_centers1, n1, color='green')
#
#n2, bins2, patches = plt.hist(y, nbins, histtype='bar', facecolor='blue', alpha=0.5, density=True)
#bin_centers2 = bins2[:-1] + 0.5 * np.diff(bins2) 
#plt.plot(bin_centers1, n2, color='blue')
#
#n3, bins3, patches = plt.hist(z, nbins, histtype='bar', facecolor='black', alpha=0.5, density=True)
#bin_centers3 = bins3[:-1] + 0.5 * np.diff(bins3) 
#plt.plot(bin_centers1, n3, color='black')
#
##plt.clf()
##plt.plot(bin_centers1, n1, color='green')
##plt.plot(bin_centers2, n2, color='blue')
##plt.plot(bin_centers3, n3, color='black')
#plt.show()



#print(df)
#print("min", data['E'].min())
#print("max", data['E'].max())

