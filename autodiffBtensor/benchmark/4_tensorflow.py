import tensorflow as tf

@tf.function
def computation(p):
    #a = tf.reshape(p, [3,3]) 
    #inp2 = tf.cos(a)
    #final = tf.reduce_sum(inp2, 1)
    inp2 = tf.cos(p)
    final = tf.reduce_sum(inp2, 0)
    return final

@tf.function
def gradient(p):
    with tf.GradientTape(persistent=True) as t:
        t.watch(p)
        result = computation(p)
    gradient = t.jacobian(result,p, experimental_use_pfor=False)
    return gradient

@tf.function
def hessian(p):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(p)
        g = gradient(p)
    hessian = t2.jacobian(g,p, experimental_use_pfor=False)
    return hessian
    

#@tf.function
def test(p):
    with tf.GradientTape(persistent=False) as t:
        t.watch(p)
        with tf.GradientTape(persistent=False) as tt:
            tt.watch(p)
            y = computation(p)
        g = tt.jacobian(y, p, experimental_use_pfor=False)
    h = t.jacobian(g, p, experimental_use_pfor=False)
    return h
#@tf.function
#def hessian(p):
#    with tf.GradientTape(persistent=True) as t1:
#        t1.watch(p)
#        with tf.GradientTape(persistent=True) as t2:
#            t2.watch(p)
#            result = computation(p)
#        gradient = t2.jacobian(result,p, experimental_use_pfor=False)
#    hessian = t1.jacobian(gradient,p,experimental_use_pfor=False)
#    return hessian

p = tf.constant([1,2,3,4,5,6,7,8,9], dtype=tf.float64)
g = test(p)
print(g)
#r = computation(p)
#g = gradient(p)
#h = hessian(p)
#print(r)
#print(g)
#r = hessian(p)
#print(r)

#result = computation(p)
#with tf.GradientTape(persistent=True) as t1:
#    t1.watch(p)
#    with tf.GradientTape(persistent=True) as t2:
#        t2.watch(p)
#        result = computation(p)
#    gradient = t2.jacobian(result,p, experimental_use_pfor=False)
#hessian = t1.jacobian(gradient,p,experimental_use_pfor=False)
#print(gradient)

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
#print(gradient)



#with tf.GradientTape(persistent=True) as t1:
#    t1.watch(p)
#    with tf.GradientTape(persistent=True) as t2:
#        t2.watch(p)
#        result = computation(p)
#    gradient = t2.jacobian(result,p, experimental_use_pfor=False)
#    print(gradient)
#hessian = t1.jacobian(gradient,p,experimental_use_pfor=False)
#print(hessian)


#g = gradient(p)
#print(g)
#h = hessian(p)



