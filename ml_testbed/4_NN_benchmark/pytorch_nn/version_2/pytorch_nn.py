import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

datasets = ["../../datasets/h2o","../../datasets/h3o", "../../datasets/h2co", "../../datasets/ochco"]

def build_model(path):
    data = pd.read_csv(path) 
    X = data.values[:,:-1]
    y = data.values[:,-1].reshape(-1,1)

    inp_dim = X.shape[1]
    out_dim = y.shape[1]
    Xscaler = StandardScaler()
    yscaler = StandardScaler()
    
    X = Xscaler.fit_transform(X)
    y = yscaler.fit_transform(y)

    X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, train_size = 500, random_state=42)

    xtrain = torch.Tensor(data=X_train)
    ytrain = torch.Tensor(data=y_train)
    xtest = torch.Tensor(data=X_fulltest)
    ytest = torch.Tensor(data=y_fulltest)

    #layers = [(50,), (100,), (200,), (50,50), (100,100), (200,200), (50,50,50,50), (100,100,100,100)]
    #layers = [(50,), (50,50), (50,50,50), (50,50,50,50)]
    layers = [(20,), (20,20), (20,20,20), (20,20,20,20), (20,20,20,20,20),
    (40,), (40,40), (40,40,40), (40,40,40,40), (40,40,40,40,40),
    (60,), (60,60), (60,60,60), (60,60,60,60), (60,60,60,60,60),
    (80,), (80,80), (80,80,80), (80,80,80,80), (80,80,80,80,80)]
    #layers = [(100,), (100,100), (100,100,100), (100,100,100,100)]
    errors = []
    
    factor = yscaler.var_[0]
    for l in layers:
        for rstate in range(1):
            torch.manual_seed(rstate)
            depth = len(l)
            # define input structure 
            structure = OrderedDict([('input', nn.Linear(inp_dim, l[0])),
                                     ('activ_in' , nn.Tanh())])
            model = nn.Sequential(structure)
            # add whatever else
            for i in range(depth-1):
                model.add_module('layer' + str(i), nn.Linear(l[i], l[i+1]))
                model.add_module('activ' + str(i), nn.Tanh())
            # add final output layer
            model.add_module('output', nn.Linear(l[depth-1], 1))

            metric = torch.nn.MSELoss()
            optimizer = torch.optim.LBFGS(model.parameters(), tolerance_grad=1e-7, tolerance_change=1e-12)   #, lr=0.01)
            #optimizer = torch.optim.Adam(model.parameters())   #, lr=0.01)
            prev_loss = 1.0
            bad_iter_count = 0
            for epoch in range(1,1001):
                def closure():
                    optimizer.zero_grad()
                    y_pred = model(xtrain)
                    loss = metric(y_pred, ytrain)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                if epoch % 5 == 0: 
                    with torch.no_grad():
                        tmp_pred = model(xtest)
                        loss = metric(tmp_pred, ytest)
                        test_error_rmse = np.sqrt(loss.item() * factor) * 219474.63
                        #print('epoch: ', epoch,'Test set RMSE (cm-1): ', test_error_rmse)
                        # very simple early stopping implementation
                        if epoch > 1:
                            # if test set error is not improved by more than 0.1% three iters in a row, stop training this model
                            if ((prev_loss - test_error_rmse) / prev_loss) < 1e-3:
                                bad_iter_count += 1
                                if bad_iter_count > 3:
                                    prev_loss = test_error_rmse * 1.0
                                    break
                            else:
                                bad_iter_count = 0
                        prev_loss = test_error_rmse * 1.0  # save previous loss to track improvement

            print(l,prev_loss)
            errors.append(prev_loss)
            #with torch.no_grad():
            #    p = model(xtest)
            #    tmploss = metric(p, ytest)
            #    p = p.detach().numpy()
            #    predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
            #    actual_y = yscaler.inverse_transform(y_fulltest)
            #    # compute error with unscaled data
            #    rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
            #    print('test set error traditional',rmse*219474.63)
            #    # compute error with scaled data
            #    hmm = np.sqrt(factor * mean_squared_error(y_fulltest, p.reshape(-1,1)))
            #    print('test set error direct with factor',hmm*219474.63)
            #    #errors.append(rmse*219474.63)
    #print("Best RMSE", min(errors))

build_model(datasets[1])

#for path in datasets:
#    print(path)
#    build_model(path)
