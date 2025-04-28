import os

import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import TimeSeriesSplit

from rampwf.score_types.base import BaseScoreType
from sklearn.metrics import r2_score

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.EstimatorExternalData()


class R2Score(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0  # sklearn behavior
    maximum = 1.0

    def __init__(self, name='r2', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
    

score_types = [
    rw.score_types.RMSE(name="rmse", precision=3),
    R2Score(name='r2', precision=3),
]


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def _read_data(path, f_name):
    data = pd.read_parquet(os.path.join(path, "data", f_name))
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_train_data(path="."):
    f_name = "train.parquet"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.parquet"
    return _read_data(path, f_name)
