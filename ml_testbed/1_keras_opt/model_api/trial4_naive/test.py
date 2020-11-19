
#import tensorflow as tf

##vector = tf.Variable([7., 7.], 'vector')
#vector = tf.constant([[1.0], [2.0], [3.0]])
#
## Make vector norm as small as possible.
#loss = tf.reduce_sum(tf.square(vector))
#optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': 100})
#with tf.Session() as session:
#    session.run(tf.global_variables_initializer())
#    optimizer.minimize(session, fetches=[loss])
### The value of vector should now be [0., 0.].
#print(vector)
#
#
#x = tf.placeholder(tf.float32, shape=(10, 10))
#y = tf.matmul(x, x)
#
#with tf.Session() as sess:
#    rand_array = np.random.rand(10, 10)
#    optimizer.minimize(session, feed_dict={x: rand_array} fetches=[loss])
#    print(sess.run(y, ))


import tensorflow as tf
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops 

ScipyOptimizerInterface = tf.contrib.opt.ScipyOptimizerInterface
#vector = tf.Variable([7., 7.], 'vector')
x = variables.Variable(array_ops.ones(5))
loss = tf.reduce_sum(tf.square(x))
optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 5})

sess = tf.Session()
with sess as s:
    s.run(variables.global_variables_initializer())
    print(optimizer._loss.eval())
    optimizer.minimize(s)
    print(optimizer._loss.eval())
    stuff = sess.run(x)
    print(stuff)
    

#sess.run(tf.global_variables_initializer())
#
#with sess as session:
#    optimizer.minimize(session)
    #print(optimizer._loss.eval())
    #print(optimizer._var_updates)
    #print(optimizer._var_updates[0].eval())
