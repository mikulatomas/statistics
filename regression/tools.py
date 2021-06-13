from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import pandas as pd
import numpy as np


def model_performance(model, train_X, train_y, test_X, test_y):
    print("R-squared:")
    print("Train score")
    print(model.score(train_X, train_y))
    print("Test score")
    print(model.score(test_X, test_y))
    print("MSE:")
    print("Train score")
    print(mean_squared_error(train_y, model.predict(train_X)))
    print("Test score")
    print(mean_squared_error(test_y, model.predict(test_X)))


def coef_table(coef, features):
    return pd.DataFrame(zip(features, coef), columns=["feature", "coef"]).sort_values(
        "coef", ascending=False
    )


def polynomial_features(data, p):
    new_columns = []
    for i in range(p):
        degree = i + 1
        if degree > 1:
            new_columns.extend([f"{column}^{degree}" for column in data.columns])
        else:
            new_columns.extend(data.columns)
    return pd.DataFrame(
        np.hstack(tuple((data ** (i + 1) for i in range(p)))), columns=new_columns
    )
