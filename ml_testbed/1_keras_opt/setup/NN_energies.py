from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from tensorflow.contrib.opt import ScipyOptimizerInterface
#import tensorflow as tf
import keras

from custom_optimizers import ScipyOpt

import pandas as pd
import numpy as np

import sys
import os
import json

data = pd.read_csv("PES.dat") 
data = data.drop_duplicates(subset = "E")

X = data.values[:,:-1]
y = data.values[:,-1].reshape(-1,1)
y = y - y.min()


Xscaler = MinMaxScaler(feature_range=(0,1))
yscaler = MinMaxScaler(feature_range=(0,1))

X = Xscaler.fit_transform(X)
y = yscaler.fit_transform(y)

X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, test_size = 0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_fulltest, y_fulltest, test_size = 0.5, random_state=42)

in_dim = X_train.shape[1]
out_dim = y_train.shape[1]

valid_set = tuple([X_valid, y_valid])

# train a fresh model 50 times. Save the best one.
models = []
MAE = []
RMSE = []
percent_error = []

def prediction(model, X_test, y_test):
    p = model.predict(np.array(X_test))
    predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
    actual_y = yscaler.inverse_transform(y_test.reshape(-1,1))
    return actual_y - predicted_y
    #rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
    #return rmse


for i in range(1):
    model = Sequential()
    model.add(Dense(100, input_dim=in_dim, kernel_initializer='normal'))
    model.add(Activation('sigmoid'))
    model.add(Dense(out_dim, kernel_initializer='normal'))
    model.add(Activation('linear'))

    # fit the model 
    #opt = ScipyOpt(model=model, x=X_train, y=y_train, nb_epoch=1000)
    #opt = ScipyOptimizerInterface(loss=prediction, options={'maxit':100})

    #opt = tf.train.AdamOptimizer(0.01)
    #opt = keras.optimizers.Adam(lr=0.01)
    #opt = keras.optimizers.TFOptimizer(tf.train.RMSPropOptimizer(0.1))
    #opt = keras.optimizers.TFOptimizer(tfp.optimizer.bfgs_minimize())
    #loss = keras.losses.mean_squared_error(y_true, y_pred)
    opt = ScipyOpt(model=model, x=X_train, y=y_train, nb_epoch=500)
    #opt = tf.contrib.opt.ScipyOptimizerInterface(loss, method="L-BFGS-B")
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    #model.fit()
    model.fit(x=X_train,y=y_train,epochs=1000)
    #model.fit(x=X_train,y=y_train,epochs=1000,batch_size=X_train.shape[0])#validation_data=valid_set,batch_size=5,verbose=2)
    #models.append(model)
    #
    ## analyze the model performance 
    #p = model.predict(np.array(X_test))
    #predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
    #actual_y = yscaler.inverse_transform(y_test.reshape(-1,1))
    #mae = mean_absolute_error(actual_y, predicted_y) 
    #rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))

    #RMSE.append(rmse)
    #MAE.append(mae)
#    print("Done with", i)

#print(MAE)
#print(RMSE)
