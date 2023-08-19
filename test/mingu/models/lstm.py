import inspect

import torch
import torch.nn as nn
from dataset import SlidingWindowTransformer
from sklearn.pipeline import Pipeline
from skorch import NeuralNetRegressor
from skorch.utils import is_dataset
from training_arguments import TrainingArguments


class LSTMForecaster(NeuralNetRegressor):
    def __init__(
        self, window_size, forecast_size, in_features, hidden_size, num_layers, out_features, training_args: TrainingArguments = None
    ):
        super().__init__(
            module=AutoregressiveLSTM(
                seq_len=window_size,
                pred_len=forecast_size,
                n_features_in=in_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                n_features_out=out_features,
            ),
            **training_args.__dict__
        )

    def fit(self, X, y=None, **fit_params):
        if not is_dataset(X):
            if y is None:
                raise ValueError('X가 Dataset이 아닐 경우, y는 None이 될 수 없습니다.')

            caller_frame = inspect.stack()[1][0]
            caller_locals = caller_frame.f_locals
            if 'self' in caller_locals and isinstance(caller_locals['self'], Pipeline):
                for name, transformer in caller_locals['self'].steps:
                    if isinstance(transformer, SlidingWindowTransformer):
                        X, y = X[0], X[1]
                        break

            X = self.get_dataset(X, y)

        return super().fit(X=X, y=None, **fit_params)

    def predict(self, X):
        if not is_dataset(X):
            caller_frame = inspect.stack()[1][0]
            caller_locals = caller_frame.f_locals
            if 'self' in caller_locals and isinstance(caller_locals['self'], Pipeline):
                for name, transformer in caller_locals['self'].steps:
                    if isinstance(transformer, SlidingWindowTransformer):
                        X, y = X[0], X[1]
                        print(X, y)
                        break

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

class AutoregressiveLSTM(nn.Module):
    def __init__(self, seq_len, pred_len, n_features_in, hidden_size, num_layers, n_features_out):
        super(AutoregressiveLSTM, self).__init__()
        self.n_features_out = n_features_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.lstm = nn.LSTM(n_features_in, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_features_out)

    def forward(self, x):

        input_x = x
        output_x = x

        means = x.mean(1, keepdim=True).detach()
        var = torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        std = torch.sqrt(var)

        B = x.shape[0]

        h0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)

        for i in range(self.pred_len):
            
            x_data = input_x[:, i:i + self.seq_len, :]

            out, (h0, c0) = self.lstm(x_data, (h0, c0))
            new_x_data = self.fc(out[:, -1, :])

            new_x_data = new_x_data.unsqueeze(0)
            x_append = (new_x_data - means) / std
            
            input_x = torch.cat((input_x, x_append), dim=1)
            output_x = torch.cat((output_x, new_x_data), dim=1)

        return output_x[:, self.seq_len:, :]