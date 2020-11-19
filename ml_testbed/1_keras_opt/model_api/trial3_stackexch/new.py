import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

#generate data
no = 20
data_x = np.linspace(0,1,no)
data_y = 2 * data_x #+ 2 + np.random.uniform(-0.5,0.5,no)
data_y = data_y.reshape(no,1)
data_x = data_x.reshape(no,1)

# Make model using keras layers and train
x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])
output = tf.keras.layers.Dense(1, activation='linear')(x)
loss = tf.reduce_mean(tf.losses.mean_squared_error(data_y, output))
#loss = tf.losses.mean_squared_error(y, output)
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method="L-BFGS-B", options={'maxiter': 100})


def print_loss(loss_evaled):
    print(loss_evaled)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    optimizer.minimize(sess, feed_dict={x: data_x, y: data_y}, loss_callback=print_loss, fetches=[loss])
        #print(sess.run(loss, feed_dict={x: data_x}))
        #a, b = sess.run([optimizer, loss], feed_dict={x: data_x, y: data_y})
        #print(sess.run(loss, feed_dict={x: data_x, y: data_y}))
        #print(b)

    #print(x.eval()) 

#sess = K.get_session()
#sess.run(tf.global_variables_initializer())

#with tf.Session() as session:
#    optimizer.minimize(session)

#tf_dict = {x : data_x, y : data_y}
#tf_dict = {x: data_x}
#with sess.as_default():
#    sess.run(tf.global_variables_initializer())
#    optimizer.minimize(sess, feed_dict = tf_dict, fetches=[loss], loss_callback=lambda a: print("Loss:", a))
    #optimizer.minimize(sess, feed_dict = tf_dict) #, fetches=[loss], loss_callback=lambda a: print("Loss:", a))
##
##
#from keras.metrics import mean_absolute_error as accuracy
#with sess.as_default():
#    acc_val = accuracy(data_y, output)
#    print(acc_val.eval(feed_dict={x : data_x,y: data_y}))
