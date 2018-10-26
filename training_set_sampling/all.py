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
max_e = data['E'].max()

ntrain = 5000

def structure_based(data=data,ntrain=ntrain):
    data = data.sort_values("E").reset_index(drop=True)
    data_dim = data.shape[0]
    # accept lowest energy point as 1st training point
    train = []
    train.append(data.values[0])
    
    def norm(train_point, data=data):
        """ Computes norm between training point geometry and every point in dataset"""
        data_dim = data.shape[0]
        tmp1 = np.tile(train_point[:-1], (data_dim,1))
        tmp2 = data.values[:,:-1]
        diff = tmp1 - tmp2
        norm_vector = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        return norm_vector

    # accept farthest point from 1st training point as the 2nd training point
    norm_vector_1 = norm(train[0]) 
    idx = np.argmax(norm_vector_1)
    newtrain = data.values[idx]
    train.append(newtrain)

    # create norm matrix, whose rows are all the norms to 1st and 2nd training points 
    norm_vector_2 = norm(train[1])
    norm_matrix = np.vstack((norm_vector_1, norm_vector_2))

    # find the minimum value along the columns of this 2xN array of norms
    min_array = np.amin(norm_matrix, axis=0)

    while len(train) < ntrain:
        # min_array contains the smallest norms into the training set, by datapoint.
        # We take the largest one.
        idx = np.argmax(min_array)
        new_geom = data.values[idx]
        train.append(new_geom)
        # update norm matrix with the norms of newly added training point
        norm_vec = norm(train[-1])
        stack = np.vstack((min_array, norm_vec))
        min_array = np.amin(stack, axis=0)
    train = np.asarray(train).reshape(ntrain,len(data.columns))
    train = pd.DataFrame(train, columns=data.columns).sort_values("E")
    return train
        

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

start = timeit.default_timer()
sb = structure_based()
print("Training set generation finished in {} seconds".format(round((timeit.default_timer() - start),2)))


#eo = energy_ordered()
#sobol = sobol(data)
random = random(data)
nbins = 10
hist = pd.concat([data['E'], sb['E'], random['E']],axis=1)
hist.columns = ['Full', 'Structure Based', 'Random']
hist.plot.hist(bins = nbins, stacked=False, density=True,alpha=0.7, color=['black', 'blue', 'red'])
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

