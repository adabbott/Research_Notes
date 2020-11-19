#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
np.set_printoptions(linewidth=200)
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.enable_eager_execution()

## Looks like batch_jacobian is your friend, but just use regular jacobain first
#p = tf.constant([[1,2,3],
#                 [4,5,6],
#                 [7,8,9]])
p = tf.constant([[1,2,3],
               [4,5,6],
               [7,8,9]], dtype=tf.float64)

def computation(p):
    a = tf.reshape(p, [3,3]) 
    inp2 = tf.cos(a)
    final = tf.reduce_sum(inp2, 1)
    return final

#result = computation(p)
#print(result)

#with tf.GradientTape() as t:
#    with tf.GradientTape() as t2:
#        y = computation(p)


#p = tf.constant([[1,2,3],
#                 [4,5,6],
#                 [7,8,9]], dtype=tf.float64)
#
#p = tf.constant([1,2,3,4,5,6,7,8,9], dtype=tf.float64)
#
#with tf.GradientTape(persistent=True) as t:
#    t.watch(p)
#    with tf.GradientTape(persistent=True) as t2:
#        t2.watch(p)
#        result = computation(p)
#    gradient = t2.jacobian(result,p, experimental_use_pfor=False)
#hessian = t.jacobian(gradient, p, experimental_use_pfor=False)
##gradient = t2.jacobian(result, p)
##hessian = t.batch_jacobian(gradient, p)
#
#print("\n\n\n")
#print(result)
#print(gradient)
#print(hessian)
    
#p = tf.constant([1,2,3,4,5,6,7,8,9], dtype=tf.float64)
#with tf.GradientTape(persistent=True) as t:
#    t.watch(p)
#    with tf.GradientTape(persistent=True) as t2:
#        t2.watch(p)
#        with tf.GradientTape(persistent=True) as t3:
#            t3.watch(p)
#            result = computation(p)
#        gradient = t3.jacobian(result,p, experimental_use_pfor=False)
#    hessian = t2.jacobian(gradient, p, experimental_use_pfor=False)
#cubic = t.jacobian(hessian, p, experimental_use_pfor=False) 
#print(result)
#print(cubic)

# Print up to quartic
p = tf.constant([1,2,3,4,5,6,7,8,9], dtype=tf.float64)
with tf.GradientTape(persistent=True) as t:
    t.watch(p)
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(p)
        with tf.GradientTape(persistent=True) as t3:
            t3.watch(p)
            with tf.GradientTape(persistent=True) as t4:
                t4.watch(p)
                result = computation(p)
            gradient = t4.jacobian(result,p, experimental_use_pfor=False)
        hessian = t3.jacobian(gradient, p, experimental_use_pfor=False)
    cubic = t2.jacobian(hessian, p, experimental_use_pfor=False) 
quartic = t.jacobian(cubic, p, experimental_use_pfor=False) 
#print(result)
#print(cubic)



#p = tf.constant([[1,2,3],
#                 [4,5,6],
#                 [7,8,9]], dtype=tf.float64)
#
#with tf.GradientTape(persistent=True) as t:
#    t.watch(p)
#    with tf.GradientTape(persistent=True) as t2:
#        t2.watch(p)
#        with tf.GradientTape(persistent=True) as t3:
#            t3.watch(p)
#            result = computation(p)
#        gradient = t3.jacobian(result,p, experimental_use_pfor=False)
#    hessian = t2.batch_jacobian(gradient, p, experimental_use_pfor=False)
#cubic = t.batch_jacobian(hessian, p, experimental_use_pfor=False) 
#print(hessian)
#hessian = t.jacobian(gradient, p, experimental_use_pfor=False)
#gradient = t2.jacobian(result, p)
#hessian = t.batch_jacobian(gradient, p)
    
        
#sess = tf.Session()
#sess.run(tf.initialize_all_variables())
#print(sess.run(result))


#x = tf.Variable(1.0)
#
#with tf.GradientTape() as t:
#    with tf.GradientTape() as t2:
#        y = x * x * x
#    dy_dx = t2.gradient(y,x)
#d2y_dx2 = t.gradient(dy_dx, x)
#
#print(d2y_dx2)
