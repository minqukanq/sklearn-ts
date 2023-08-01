import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]).float().squeeze(1)


class TimeSeriesValidSplit:
    def __init__(self, train_size=None, valid_size=None):
        self.train_size = train_size
        self.valid_size = valid_size

    def __call__(self, dataset):
        dataset_size = len(dataset)
        default_valid_size = 0.25

        if self.train_size is None and self.valid_size is None:
            self.valid_size = default_valid_size
            self.train_size = 1.0 - self.valid_size

        if self.train_size is None:
            self.train_size = 1.0 - self.valid_size
        elif self.valid_size is None:
            self.valid_size = 1.0 - self.train_size

        if isinstance(self.train_size, float):
            self.train_size = int(self.train_size * dataset_size)
        if isinstance(self.valid_size, float):
            self.valid_size = int(self.valid_size * dataset_size)

        train_indices = list(range(self.train_size))
        valid_indices = list(range(self.train_size, self.train_size + self.valid_size))

        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)

        return train_dataset, valid_dataset


def split_sequences(window_size, forecast_size, features=None, targets=None, sliding_steps=1):
    X, y = list(), list()
    for i in range(0, len(features), sliding_steps):
        end_ix = i + window_size
        out_end_ix = end_ix + forecast_size

        if out_end_ix > len(features):
            break

        if features is not None:
            seq_x = features[i:end_ix, :]
            X.append(seq_x)

        if targets is not None:
            seq_y = targets[end_ix:out_end_ix, :]
            y.append(seq_y)

    X_array = np.array(X).astype(np.float32) if features is not None else None
    y_array = np.array(y).astype(np.float32) if targets is not None else None

    return X_array, y_array


# def split_sequences(features, targets, window_size, forecast_size, sliding_steps):
#     X, y = list(), list()
#     for i in range(0, len(features), sliding_steps):
#         end_ix = i + window_size
#         out_end_ix = end_ix + forecast_size

#         if out_end_ix > len(features):
#             break

#         seq_x = features[i:end_ix, :]
#         if targets is not None:
#             seq_y = targets[end_ix:out_end_ix, :]
#             y.append(seq_y)

#         X.append(seq_x)

#     return np.array(X).astype(np.float32), np.array(y).astype(np.float32) if targets is not None else None

# def split_sequences(features, targets, window_size, forecast_size, sliding_steps):
#     X, y = list(), list()
#     for i in range(0, len(features), sliding_steps):
#         end_ix = i + window_size
#         out_end_ix = end_ix + forecast_size

#         if out_end_ix > len(features):
#             break

#         seq_x, seq_y = features[i:end_ix, :], targets[end_ix:out_end_ix, :]

#         X.append(seq_x)
#         y.append(seq_y)

#     return np.array(X).astype(np.float32), np.array(y).astype(np.float32)
