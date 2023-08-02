import numpy as np
from torch.utils.data import Subset


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


from sklearn.base import BaseEstimator, TransformerMixin


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, forecast_size=1, step_size=1):
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.step_size = step_size

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        if X is None and y is None:
            raise ValueError("Either X or y must be provided.")

        X_result, y_result = None, None

        if y is not None:
            y_result = [
                y[i + self.window_size : i + self.window_size + self.forecast_size]
                for i in range(0, len(y) - self.window_size - self.forecast_size + 1, self.step_size)
            ]
            y_result = np.array(y_result).astype(np.float32)

        if X is not None:
            X_result = [
                X[i : i + self.window_size] for i in range(0, len(X) - self.window_size - self.forecast_size + 1, self.step_size)
            ]
            X_result = np.array(X_result).astype(np.float32)

        return X_result, y_result

    def fit_transform(self, X=None, y=None):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        if len(X.shape) != 3:
            raise ValueError("Input should be a 3D array.")
        if X.shape[1] != self.window_size:
            raise ValueError("Invalid window_size in the input.")

        X_original_shape = (X.shape[0] - 1) * self.step_size + self.window_size

        if y is not None:
            if len(y.shape) != 3:
                raise ValueError("Input should be a 3D array.")
            if y.shape[1] != self.forecast_size:
                raise ValueError("Invalid window_size in the input.")
            y_original_shape = (y.shape[0] - 1) * self.step_size + self.forecast_size
            y_result = np.empty((y_original_shape, y.shape[2]))

        X_result = np.empty((X_original_shape, X.shape[2]))

        for i in range(len(X)):
            X_result[i * self.step_size : i * self.step_size + self.window_size] = X[i]
            if y is not None:
                y_result[i * self.step_size : i * self.step_size + self.forecast_size] = y[i]
        if y is not None:
            return X_result, y_result
        else:
            return X_result
