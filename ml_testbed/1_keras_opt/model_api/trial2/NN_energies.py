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
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

inp = tf.placeholder(tf.float32, shape=(None, 3)) # 3 inputs
x = Dense(64, activation='linear')(inp)
x = Dense(64, activation='linear')(x)
preds = Dense(1, activation='linear')(x)
labels = tf.placeholder(tf.float32, shape=(None, 1))
from keras.losses import mean_squared_error
loss = tf.reduce_mean(mean_squared_error(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
optimizer = ScipyOptimizerInterface(loss, method="L-BFGS-B")#.minimize(loss)
##
### Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)
#
#with sess as session:
#    for i in range(100):
#        train_step.run(feed_dict={inp : X_train,
#                                  labels: y_train})

#with sess as sess:
with sess.as_default():
#    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        train_step.run(feed_dict={inp : X_train, labels: y_train})
        #optimizer.minimize(sess, feed_dict={inp : X_train, labels: y_train})
        #mse = tf.reduce_mean(mean_squared_error(preds, y_train))
        #print('Training set error =', sess.run(mse), '\n')


#from keras.metrics import mean_absolute_error as accuracy
##acc_value = accuracy(labels, preds)
#acc_value = tf.reduce_mean(accuracy(labels, preds))
#with sess.as_default():
#    print(acc_value.eval(feed_dict={inp: X_test, labels: y_test}))
#    #print(acc_value.eval(feed_dict={inp : X_test,
#    #                                labels: y_test}))







##
##
###from tensorflow.examples.tutorials.mnist import input_data
###mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
###batch = mnist_data.train.next_batch(50)
###print(batch)
##
##
#from keras.metrics import mean_absolute_error as accuracy
#
#acc_val = accuracy(labels, preds)
#
#    print(acc_val.eval(feed_dict={inp : X_train,labels: y_train}))


#with sess.as_default():
    #for i in range(10):
        #train_step.minimize(
        #train_step.run(feed_dict={inp : inp,
#                                  labels: labels})



#model = Model(inputs=inputs, outputs=predictions)
#
##opt = TFOptimizer(RMSPropOptimizer(0.01))
#loss = mean_squared_error(y_train, predictions)
#opt = ScipyOptimizerInterface(loss, method="L-BFGS-B")
#model.compile(optimizer=opt,
#              loss='mse')
#model.fit(X_train, y_train, epochs=100)  


#for i in range(1):
#    model = Sequential()
#    model.add(Dense(100, input_dim=in_dim, kernel_initializer='normal'))
#    model.add(Activation('sigmoid'))
#    model.add(Dense(out_dim, kernel_initializer='normal'))
#    model.add(Activation('linear'))
#
#    # fit the model 
#    #opt = ScipyOpt(model=model, x=X_train, y=y_train, nb_epoch=1000)
#    #opt = ScipyOptimizerInterface(loss=prediction, options={'maxit':100})
#
#    #opt = tf.train.AdamOptimizer(0.01)
#    #opt = keras.optimizers.Adam(lr=0.01)
#    loss = keras.losses.mean_squared_error(y_true, y_pred)
#    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method="L-BFGS-B")
#
#    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
#    model.fit(x=X_train,y=y_train,epochs=1000)
#    #model.fit(x=X_train,y=y_train,epochs=1000,batch_size=X_train.shape[0])#validation_data=valid_set,batch_size=5,verbose=2)
#    #models.append(model)
#    #
#    ## analyze the model performance 
#    #p = model.predict(np.array(X_test))
#    #predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
#    #actual_y = yscaler.inverse_transform(y_test.reshape(-1,1))
#    #mae = mean_absolute_error(actual_y, predicted_y) 
#    #rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
#
#    #RMSE.append(rmse)
#    #MAE.append(mae)
#    print("Done with", i)
#
#print(MAE)
#print(RMSE)
