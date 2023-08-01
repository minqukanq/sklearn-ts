import numpy as np
import torch
import torch.nn as nn
from dataset import split_sequences
from skorch import NeuralNetRegressor
from skorch.utils import is_dataset
from training_arguments import TrainingArguments


class LSTMForecaster(NeuralNetRegressor):
    def __init__(
        self, window_size, forecast_size, in_features, hidden_size, num_layers, out_features, training_args: TrainingArguments = None
    ):
        super().__init__(
            module=LSTM(
                seq_len=window_size,
                pred_len=forecast_size,
                n_features_in=in_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                n_features_out=out_features,
            ),
            **training_args.__dict__
        )
        self.seq_len = window_size
        self.pred_len = forecast_size

    def fit(self, X, y=None, sliding_steps=1, **fit_params):
        self.sliding_steps = sliding_steps

        if not is_dataset(X):
            X, y = split_sequences(
                features=X.values if not isinstance(X, np.ndarray) else X,
                targets=y.values if not isinstance(y, np.ndarray) else y,
                window_size=self.seq_len,
                forecast_size=self.pred_len,
                sliding_steps=sliding_steps,
            )
            X = self.get_dataset(X, y)

        super().fit(X=X, y=None, **fit_params)

    def predict(self, X):
        X, _ = split_sequences(
            features=X.values if not isinstance(X, np.ndarray) else X,
            targets=None,
            window_size=self.seq_len,
            forecast_size=self.pred_len,
            sliding_steps=self.sliding_steps,
        )
        return super().predict(X)


class LSTM(nn.Module):
    def __init__(self, seq_len, pred_len, n_features_in, hidden_size, num_layers, n_features_out):
        super(LSTM, self).__init__()
        self.n_features_out = n_features_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.lstm = nn.LSTM(n_features_in, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * self.seq_len, self.pred_len * self.n_features_out)

    def forward(self, x):
        B = x.shape[0]

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(B, -1)
        out = self.fc(out)
        out = out.reshape(B, self.pred_len, self.n_features_out)

        return out
