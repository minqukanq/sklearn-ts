import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from models.lstm import LSTMForecaster
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from training_arguments import TrainingArguments

AMZN = yf.download('AMZN', start='2013-01-01', end='2019-12-31', progress=False)
col = ['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]
all_data = AMZN[['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]].round(2)

train_df, test_df = train_test_split(all_data, test_size=0.2, shuffle=False)

input_cols = ['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]
output_cols = ['Adj Close']

X_train, X_test = train_df[input_cols], test_df[input_cols]
y_train, y_test = train_df[output_cols], test_df[output_cols]

lstm = LSTMForecaster(
    window_size=24,
    forecast_size=3,
    hidden_size=128,
    num_layers=1,
    in_features=len(input_cols),
    out_features=len(output_cols),
    training_args=TrainingArguments(
        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
        lr=0.003,
        max_epochs=20,
        batch_size=32,
        device='cuda',
    ),
)

# lstm.fit(X=X, y=y)
# out = lstm.forward(torch.randn(1, 24, 6).to('cuda'))
# print(out)
model = Pipeline(steps=[('scaler', StandardScaler()), ('lstm', lstm)])
model.fit(X=X_train, y=y_train)

out = model.predict(X_test)
print(out)
