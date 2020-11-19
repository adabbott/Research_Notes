from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation
from keras.layers.core import Activation
from keras.models import load_model
from keras.optimizers import TFOptimizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from tensorflow.contrib.opt import ScipyOptimizerInterface
from tensorflow.train import RMSPropOptimizer
from tensorflow.losses import mean_squared_error
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
out = tf.placeholder(tf.float32, shape=(None,1)) # 3 inputs
#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(20, activation='tanh', input_shape=(3,)))
#model.add(tf.keras.layers.Dense(1, activation='linear'))
#output = model(inp)

#a = Dense(32, activation='linear', input_shape=(3,))
#b = Dense(1, activation='linear')(a)
#model = Model(inputs=a, outputs=b)

a = Input(shape=(3,))
b = Dense(32, activation='linear')(a)
model = Model(inputs=a, outputs=b)

output = model.predict(inp)

#x = Dense(1, activation='relu')(x)
#output = model.predict(x=inp)

loss = tf.reduce_mean(tf.losses.mean_squared_error(out, output))
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method="L-BFGS-B", options={'maxiter': 100})
#
def print_loss(loss_evaled):
    print(loss_evaled)
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    optimizer.minimize(sess, feed_dict={inp:X_train, out:y_train}, loss_callback=print_loss, fetches=[loss])#, feed_dict={inp: X_train, output: y_train})
#    print('wtf',sess.run(loss, feed_dict={inp:X_train, out:y_train}))
#    print('wtf',sess.run(loss, feed_dict={inp:X_train, out:y_train}))
#    print('wtf',sess.run(loss, feed_dict={inp:X_valid, out:y_valid}))
#    print('wtf',sess.run(loss, feed_dict={inp:X_test, out:y_test}))
    #print(sess.run(tf.convert_to_tensor(X_train)) - y_train)
    #print(loss.eval(X_train, y_train))
        #sess.run(tf.convert_to_tensor(X_test))
    # send new data through
    #predictions = sess.run(output, {inp: X_train})
    #prediction2 = model.predict(X_train)
    #print(mean_squared_error(predictions, y_train))

#sess = tf.Session()
#K.set_session(sess)
