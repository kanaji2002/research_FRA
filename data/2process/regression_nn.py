import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import torch.nn as nn
 
# CSVの読み込み
df = pd.read_csv('power.csv')

print(df.head())


y = df['Power']
X = df.drop(['Power', 'model1', 'model2', 'Tem', 'model1+2_P', 'CommonP', 'model1_P_2pir', 'model2_P_2pair', 'model_py', 'P_Usage', 'model1_FLOPS'], axis=1)
print(X.shape)
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

# Define the model
# model = nn.Sequential(
#     nn.Linear(8, 24),
#     nn.ReLU(),
#     nn.Linear(24, 12),
#     nn.ReLU(),
#     nn.Linear(12, 6),
#     nn.ReLU(),
#     nn.Linear(6, 1)
# )

# model = nn.Sequential(
#     nn.Linear(8, 24),
#     nn.ReLU(),
#     nn.Dropout(0.3),  # ドロップアウトを追加
#     nn.Linear(24, 12),
#     nn.ReLU(),
#     nn.Dropout(0.3),
#     nn.Linear(12, 6),
#     nn.ReLU(),
#     nn.Dropout(0.3),
#     nn.Linear(6, 1)
# )

# model = nn.Sequential(
#     nn.Linear(8, 24),
#     nn.BatchNorm1d(24),  # バッチ正規化
#     nn.ReLU(),
#     nn.Linear(24, 12),
#     nn.BatchNorm1d(12),
#     nn.ReLU(),
#     nn.Linear(12, 6),
#     nn.BatchNorm1d(6),
#     nn.ReLU(),
#     nn.Linear(6, 1)
# )

# most good model with epoch 2000
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# model = nn.Sequential(
#     nn.Linear(8, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 8),
#     nn.ReLU(),
#     nn.Linear(8, 1)
# )






import torch.nn as nn
import torch.optim as optim
 
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)


import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
 
# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
 
# training parameters
n_epochs = 5000   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
 
# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
 
# training loop
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
 
# restore model and return best accuracy
model.load_state_dict(best_weights)

import copy
 
import matplotlib.pyplot as plt
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()