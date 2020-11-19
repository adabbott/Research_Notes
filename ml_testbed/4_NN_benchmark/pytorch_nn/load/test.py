import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

layers = [(40,)]
inp_dim = 3
for l in layers:
    torch.manual_seed(0)
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


print(model.modules)
