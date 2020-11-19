import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import OrderedDict


#structure = torch.load('model_structure')
model = nn.Sequential()
model.load_state_dict(torch.load('model.pt'))
model.eval()

