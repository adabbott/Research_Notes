import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

X = np.linspace(-10,10, 101)
y = X**2

Xtrain, Xtmp, ytrain, ytmp = train_test_split(X,y, train_size=60, random_state=1)
Xvalid, Xtmp, ytrain, ytmp = train_test_split(X,y, train_size=60, random_state=1)

