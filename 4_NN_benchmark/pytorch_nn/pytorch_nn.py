import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#datasets = ["../datasets/h2o","../datasets/h3o", "../datasets/h2co", "../datasets/ochco"]
datasets = ["../datasets/ochco"]

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

    X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, train_size = 5000, random_state=42)

    xtrain = torch.Tensor(data=X_train)
    ytrain = torch.Tensor(data=y_train)
    xtest = torch.Tensor(data=X_fulltest)
    ytest = torch.Tensor(data=y_fulltest)

    layers = [(50,), (100,), (200,), (50,50), (100,100), (200,200), (50,50,50,50), (100,100,100,100)]

    errors = []
    for l in layers:
        for rstate in range(5):
            torch.manual_seed(rstate)
            depth = len(l)
            if depth == 1:
                model = nn.Sequential(nn.Linear(inp_dim, l[0]),  
                                                     nn.Tanh(),  
                                            nn.Linear(l[0], 1))  
            if depth == 2:
                model = nn.Sequential(nn.Linear(inp_dim, l[0]),  
                                                     nn.Tanh(),  
                                         nn.Linear(l[0], l[1]),
                                                     nn.Tanh(),  
                                         nn.Linear(l[1], 1))
            if depth == 4:
                model = nn.Sequential(nn.Linear(inp_dim, l[0]),  
                                                     nn.Tanh(),  
                                         nn.Linear(l[0], l[1]),
                                                     nn.Tanh(),  
                                         nn.Linear(l[1], l[2]),
                                                     nn.Tanh(),  
                                         nn.Linear(l[2], l[3]),
                                                     nn.Tanh(),  
                                            nn.Linear(l[3], 1))


            metric = torch.nn.MSELoss()
            optimizer = torch.optim.LBFGS(model.parameters(), tolerance_grad=1e-7, tolerance_change=1e-12)   #, lr=0.01)
            #optimizer = torch.optim.Adam(model.parameters())   #, lr=0.01)
            for epoch in range(1,100001):
                def closure():
                    optimizer.zero_grad()
                    y_pred = model(xtrain)
                    loss = metric(y_pred, ytrain)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                if epoch % 10 == 0: 
                    with torch.no_grad():
                        tmp_pred = model(xtest)
                        loss = metric(tmp_pred, ytest)
                        print('epoch: ', epoch,'Test set RMSE: ', loss.item())
            
            
            
            with torch.no_grad():
                p = model(xtest)
                p = p.detach().numpy()
                predicted_y = yscaler.inverse_transform(p.reshape(-1,1))
                actual_y = yscaler.inverse_transform(y_fulltest)
                rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
                print(rmse*219474.63)
                errors.append(rmse*219474.63)
    print("Best RMSE", min(errors))

for path in datasets:
    print(path)
    build_model(path)
