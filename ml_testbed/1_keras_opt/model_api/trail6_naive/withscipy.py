import tensorflow as tf

#x = tf.Variable(2, name='x', dtype=tf.float32)           # some input data structure
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
log_x_squared = tf.square(tf.log(x))                     # the objective function to be minimized which is a function of the input data structure

#optimizer = tf.train.GradientDescentOptimizer(0.5)       # define the optimizer
#train = optimizer.minimize(log_x_squared)                # define a training step as minimizing the objective function
optimizer = tf.contrib.opt.ScipyOptimizerInterface(log_x_squared)
init = tf.global_variables_initializer()

def optimize():
    with tf.Session() as session:
        session.run(init)
        #print("starting at", "x:", session.run(x), "f(x):", session.run(log_x_squared))
        for step in range(10):  
            #session.run(train)
            optimizer.minimize(session, feed_dict={x:2}) #, Y:log_x_squared})
            print("step", step, "x:", session.run(x), "f(x):", session.run(log_x_squared))
        

optimize()
