

import tensorflow as tf
x = tf.Variable(2, name='x', dtype=tf.float32)           # some input data structure
log_x_squared = tf.square(tf.log(x))                     # the objective function to be minimized which is a function of the input data structure

optimizer = tf.train.GradientDescentOptimizer(0.5)       # define the optimizer
train = optimizer.minimize(log_x_squared)                # define a training step as minimizing the objective function

init = tf.initialize_all_variables()

def optimize():
  with tf.Session() as session:
    session.run(init)
    print("starting at", "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))
    for step in range(10):  
      session.run(train)
      print("step", step, "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))
        

optimize()
