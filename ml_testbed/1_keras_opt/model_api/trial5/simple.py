
import tensorflow as tf

def print_loss(loss_evaled, vector_evaled):
  print(loss_evaled, vector_evaled)

vector = tf.Variable([7., 7.], 'vector')
loss = tf.reduce_sum(tf.square(vector))

optimizer = tf.contrib.opt.ScipyOptimizerInterface(
    loss, method='L-BFGS-B',
    options={'maxiter': 100})

with tf.Session() as session:
  tf.global_variables_initializer().run()
  optimizer.minimize(session,
                     loss_callback=print_loss,
                     fetches=[loss, vector])
  #print(vector.eval())
