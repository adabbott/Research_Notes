from sklearn.preprocessing import MinMaxScaler
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
Xscaler = MinMaxScaler(feature_range=(-1,1))
yscaler = MinMaxScaler(feature_range=(0,1))

X = Xscaler.fit_transform(X)
y = yscaler.fit_transform(y)

X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, train_size = 1000, random_state=42)
#X_valid, X_test, y_valid, y_test = train_test_split(X_fulltest, y_fulltest, test_size = 0.5, random_state=42)


from sklearn.neural_network import MLPRegressor

y_train = np.ravel(y_train)

# LBFGS only cares about hidden layers, activation, alpha, tol?, random_state
# does not use validation data either
mlp = MLPRegressor(hidden_layer_sizes=(100,),
                   warm_start=True,
                   activation='tanh',
                   #activation='relu',
                   alpha = 0.0000001,
                   #solver='lbfgs',
                   solver='adam',
                   random_state=1,
                   tol=1e-12, 
                   verbose=True,
                   max_iter=1000,
                   n_iter_no_change=2000)

mlp.fit(X_train,y_train)

p = mlp.predict(X_fulltest)
predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
actual_y = yscaler.inverse_transform(y_fulltest.reshape(-1,1))
rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
print(rmse)
print(np.finfo(float).eps)


mlp.solver = 'lbfgs'
mlp.fit(X_train,y_train)
p = mlp.predict(X_fulltest)
predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
actual_y = yscaler.inverse_transform(y_fulltest.reshape(-1,1))
rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
print(rmse)
print(np.finfo(float).eps)


mlp.solver = 'adam'
mlp.fit(X_train,y_train)
p = mlp.predict(X_fulltest)
predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
actual_y = yscaler.inverse_transform(y_fulltest.reshape(-1,1))
rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
print(rmse)
print(np.finfo(float).eps)

#mlp = MLPRegressor(hidden_layer_sizes=(100,100),
#                   warm_start=True,
#                   activation='tanh',
#                   #activation='relu',
#                   alpha = 0.00051,
#                   solver='adam',
#                   random_state=1,
#                   tol=1e-5, 
#                   verbose=True,
#                   max_iter=10000,
#                   n_iter_no_change=2000)

#import GPy
#kernel = GPy.kern.RBF(dim, ARD=True) #TODO add more kernels to hyperopt space
#model = GPy.models.GPRegression(X_train, y_train.reshape(-1,1), kernel=kernel, normalizer=False)
#model.optimize(max_iters=500, messages=False)
#model.optimize_restarts(10, optimizer="bfgs", verbose=False, max_iters=500, messages=False)
#
#p, v1 = model.predict(X_fulltest, full_cov=False) 
#predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
#actual_y = yscaler.inverse_transform(y_fulltest.reshape(-1,1))
#rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
#print(rmse)

