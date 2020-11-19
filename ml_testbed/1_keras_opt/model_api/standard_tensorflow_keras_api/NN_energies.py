from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation
from keras.layers.core import Activation
from keras.models import load_model
from keras.optimizers import TFOptimizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from tensorflow.contrib.opt import ScipyOptimizerInterface
#from tensorflow.train import RMSPropOptimizer
#from tensorflow.losses import mean_squared_error
#import tensorflow as tf
#import keras

#from custom_optimizers import ScipyOpt
import pandas as pd
import numpy as np

data = pd.read_csv("PES.dat") 
data = data.drop_duplicates(subset = "E")
X = data.values[:,:-1]
y = data.values[:,-1].reshape(-1,1)
y = y - y.min()
Xscaler = StandardScaler()
yscaler = StandardScaler()
X = Xscaler.fit_transform(X)
y = yscaler.fit_transform(y)
X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, test_size = 0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_fulltest, y_fulltest, test_size = 0.5, random_state=42)
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
valid_set = tuple([X_valid, y_valid])

import tensorflow as tf
from keras import backend as K

# create placeholder for inputs
#inp = tf.placeholder(tf.float32, shape=(None, 3)) # 3 inputs
inp = tf.placeholder(tf.float32, shape=(None,3)) # 3 inputs
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(20, activation='tanh', input_shape=(3,)))
model.add(tf.keras.layers.Dense(1, activation='linear'))
output = model(inp)
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              #epochs=5000,
              loss='mse')

model.fit(X_train, y_train, epochs=1000, verbose=0)

p1 = model.predict(X_train)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(p1, y_train))

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    # send new data through
#    predictions = sess.run(output, {inp: X_train})
#    #prediction2 = model.predict(X_train)
#    print(mean_squared_error(predictions, y_train))

#sess = tf.Session()
#K.set_session(sess)
