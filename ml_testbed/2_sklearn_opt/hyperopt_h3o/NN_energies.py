from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

import pandas as pd
import numpy as np

import sys
import os
import json

data = pd.read_csv("interatomic_PES.dat") 
data = data.drop_duplicates(subset = "E")

X = data.values[:,:-1]
y = data.values[:,-1].reshape(-1,1)
#y = data.values[:,-1]
y = y - y.min()


dim = X.shape[1]
#Xscaler = MinMaxScaler(feature_range=(0,1))
#yscaler = MinMaxScaler(feature_range=(0,1))
Xscaler = StandardScaler()
yscaler = StandardScaler()

X = Xscaler.fit_transform(X)
y = yscaler.fit_transform(y)

X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, train_size = 500, random_state=42)
#X_valid, X_test, y_valid, y_test = train_test_split(X_fulltest, y_fulltest, test_size = 0.5, random_state=42)


from sklearn.neural_network import MLPRegressor

y_train = np.ravel(y_train)

# LBFGS only cares about hidden layers, activation, alpha, tol?, random_state
# does not use validation data either

#regs = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
regs = [0.0000000]
for j in regs:
    data = []
    for i in range(20):
        mlp = MLPRegressor(hidden_layer_sizes=(100,100,100),
                           activation='logistic',
                           #activation='tanh',
                           #activation='relu',
                           #alpha = 0.0050,
                           alpha = j,
                           solver='lbfgs',
                           #random_state=1,
                           random_state=i,
                           tol=1e-15, 
                           verbose=False,
                           max_iter=10000)
                           #n_iter_no_change=10)
        mlp.fit(X_train,y_train)
        
        p = mlp.predict(X_fulltest)
        predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
        actual_y = yscaler.inverse_transform(y_fulltest.reshape(-1,1))
        rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
        print(rmse)
        data.append(list([i, rmse])) 

    a = np.asarray(data)
    print('da best:')
    print(a[:,1].min(axis=0))

import GPy
kernel = GPy.kern.RBF(dim, ARD=True) #TODO add more kernels to hyperopt space
model = GPy.models.GPRegression(X_train, y_train.reshape(-1,1), kernel=kernel, normalizer=False)
model.optimize(max_iters=500, messages=False)
model.optimize_restarts(10, optimizer="bfgs", verbose=False, max_iters=500, messages=False)

p, v1 = model.predict(X_fulltest, full_cov=False) 
predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
actual_y = yscaler.inverse_transform(y_fulltest.reshape(-1,1))
rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
print('GP result',rmse)

