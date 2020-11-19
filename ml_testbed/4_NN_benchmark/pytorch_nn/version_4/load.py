import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

model = nn.Module()
#model.load_state_dict(torch.load('model.pt'), strict=False)
model = torch.load('model.pt')
#model.load_state_dict(torch.load('model.pt'))
model.eval()


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

