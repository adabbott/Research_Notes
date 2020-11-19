#https://stackoverflow.com/questions/38450729/keras-bfgs-training-using-scipy-minimize/41004079
import numpy as np
from scipy.optimize import minimize
from keras.models import Sequential
from keras.layers.core import Dense

# Dummy training examples
X = np.array([[-1,2,-3,-1],[3,2,-1,-4]]).astype('float')
Y = np.array([[2],[-1]]).astype('float')

model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=4))

def loss(W):
    weightsList = [np.zeros((4,1)), np.zeros(1)]
    for i in range(4):
        weightsList[0][i,0] = W[i]
    weightsList[1][0] = W[4]
    model.set_weights(weightsList)
    preds = model.predict(X)
    mse = np.sum(np.square(np.subtract(preds,Y)))/len(X[:,0])
    return mse

V = [1.0, 2.0, 3.0, 4.0, 1.0]
print('Starting loss = {}'.format(loss(V)))
# set the eps option to increase the epsilon used in numerical diff
res = minimize(loss, x0=V, method = 'BFGS', options={'eps':1e-6,'disp':True})
print('Ending loss = {}'.format(loss(res.x)))
