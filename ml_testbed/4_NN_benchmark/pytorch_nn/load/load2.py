import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import OrderedDict


structure = torch.load('model_structure')
model = nn.Sequential(structure)
model.load_state_dict(torch.load('model.pt'))

#model = nn.Module()
#structure = OrderedDict([('input', nn.Linear(3, 40)),
#                         ('activ_in' , nn.Tanh())])
#model.add_module('output', nn.Linear(40, 1))
#structure = json.loads( , object_pairs_hook=OrderedDict)
#model = nn.Sequential(structure)
#model.load_state_dict(torch.load('model.pt'))


#model.add_module('output', nn.Linear(40, 1))
#state = model.load_state_dict(torch.load('model.pt'))

#model.load_state_dict(torch.load('model.pt'), strict=False)
#model = torch.load('model.pt')
#model.load_state_dict(torch.load('model.pt'))
#model.eval()


data = pd.read_csv("../../datasets/h2o_fi") 
X = data.values[:,:-1]
y = data.values[:,-1].reshape(-1,1)
Xscaler = StandardScaler()
yscaler = StandardScaler()

X = Xscaler.fit_transform(X)
y = yscaler.fit_transform(y)

x = torch.Tensor(data=X)
y = torch.Tensor(data=y)

with torch.no_grad():
    pred_y = model(x)

print(y[:10])
print(pred_y[:10])
#
