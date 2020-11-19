from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

datasets = ["../../datasets/h2o","../../datasets/h3o", "../../datasets/h2co", "../../datasets/ochco"]
#datasets = ["../datasets/h2o_fi","../datasets/h3o_fi", "../datasets/h2co_fi", "../datasets/ochco_fi"]

# LBFGS only cares about hidden layers, activation, alpha, tol?, random_state
def build_model(path):
    data = pd.read_csv(path) 
    X = data.values[:,:-1]
    y = data.values[:,-1].reshape(-1,1)
    # convert to wavenumber domain?
    #y = y*219474.63
    #y = y - y.min()

    inp_dim = X.shape[1]
    out_dim = y.shape[1]
    Xscaler = StandardScaler()
    yscaler = StandardScaler()

    X = Xscaler.fit_transform(X)
    y = yscaler.fit_transform(y)

    X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, train_size = 950, random_state=42)
    
    y_train = np.ravel(y_train)
    actual_y = yscaler.inverse_transform(y_fulltest.reshape(-1,1))
    # hyperparameters
    #layers = [(50,), (100,), (200,), (50,50), (100,100), (200,200), (50,50,50,50), (100,100,100,100)]
    layers = [(300,), (300,300)]
    regularization = [0.0, 0.0001]
    # perform straightforward grid search
    errors = []
    for l in layers:
        for r in regularization:
            for rstate in range(5):
                mlp = MLPRegressor(hidden_layer_sizes=l,
                                   activation='tanh',
                                   alpha = r,
                                   solver='lbfgs',
                                   random_state=rstate,
                                   tol=1e-15,      # never reached, see below
                                   verbose=False,
                                   max_iter=10000) # never reached because sklearn does not expose `factr` parameter, default accuracy is standard, about 1e-8 in scaled energy domain
                mlp.fit(X_train,y_train)
                p = mlp.predict(X_fulltest)
                predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
                rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
                #print(rmse)
                print(rmse*219474.63)
                errors.append(rmse)
    print("Best RMSE", min(errors)*219474.63)

for path in datasets:
    print(path)
    build_model(path)
