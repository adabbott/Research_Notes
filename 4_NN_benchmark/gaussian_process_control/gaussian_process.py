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
import GPy

#datasets = ["../datasets/h2o","../datasets/h3o", "../datasets/h2co", "../datasets/ochco"]
datasets = ["../datasets/h2o_fi","../datasets/h3o_fi", "../datasets/h2co_fi", "../datasets/ochco_fi"]

def build_model(path):
    data = pd.read_csv(path) 
    X = data.values[:,:-1]
    y = data.values[:,-1].reshape(-1,1)

    inp_dim = X.shape[1]
    out_dim = y.shape[1]
    #Xscaler = StandardScaler()
    #yscaler = StandardScaler()
    Xscaler = MinMaxScaler(feature_range=(-1,1))
    yscaler = MinMaxScaler(feature_range=(-1,1))

    X = Xscaler.fit_transform(X)
    y = yscaler.fit_transform(y)

    X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, train_size = 950, random_state=42)  # for all datasets, use 800 training points

    np.random.seed(0)
    kernel = GPy.kern.RBF(inp_dim, ARD=True) 
    model = GPy.models.GPRegression(X_train, y_train, kernel=kernel, normalizer=False)
    model.optimize(max_iters=1000, messages=False)
    model.optimize_restarts(10, optimizer="bfgs", verbose=False, max_iters=1000, messages=False)
    
    p, v1 = model.predict(X_fulltest, full_cov=False) 
    predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
    actual_y = yscaler.inverse_transform(y_fulltest.reshape(-1,1))
    rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
    print(rmse*219474.63)

for path in datasets:
    print(path)
    build_model(path)

