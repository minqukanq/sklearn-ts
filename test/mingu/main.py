import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from dataset import SlidingWindowTransformer
from models.lstm import LSTMForecaster
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from training_arguments import TrainingArguments

AMZN = yf.download('AMZN', start='2013-01-01', end='2019-12-31', progress=False)
all_data = AMZN[['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]].round(2)

train_df, test_df = train_test_split(all_data, test_size=0.2, shuffle=False)

#input_cols = ['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]
input_cols = ['Adj Close']
output_cols = ['Adj Close']

X_train, y_train = train_df[input_cols], train_df[output_cols]
X_test, y_test = test_df[input_cols], test_df[output_cols]

window_size = 5
forecast_size = 2
step_size = 1

lstm = LSTMForecaster(
    window_size=window_size,
    forecast_size=forecast_size,
    hidden_size=64,
    num_layers=1,
    in_features=len(input_cols),
    out_features=len(output_cols),
    training_args=TrainingArguments(
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.003,
        max_epochs=10,
        batch_size=1,
        device='cpu',
    ),
)

model = Pipeline(
    steps=[
        ('scaler', StandardScaler()),
        ('slding', SlidingWindowTransformer(window_size=window_size, forecast_size=forecast_size, step_size=step_size)),
        ('lstm', lstm),
    ]
)
model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)
_, y_true = model['slding'].transform(y=y_test)

mse_list = [mean_squared_error(true, pred) for true, pred in zip(y_true, y_pred)]
average_mse = np.mean(mse_list)

print(average_mse)
## loss = F.mse_loss(torch.from_numpy(y_pred), torch.from_numpy(y_true)).item()
## print(loss)
