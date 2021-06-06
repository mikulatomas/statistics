from sklearn.metrics import confusion_matrix as confusion
from sklearn.utils import resample
import pandas as pd
import numpy as np


def model_performance(model, train_X, train_y, test_X, test_y):
    print("Train score")
    print(model.score(train_X, train_y))
    print("Test score")
    print(model.score(test_X, test_y))


def confusion_matrix(train_X, train_y, test_X, test_y):
    tn, fp, fn, tp = confusion(train_y, train_X.astype(float)).ravel()

    print("Train")
    print(f"TN: {tn}, TP: {tp}, FN: {fn}, FP: {fp}")

    tn, fp, fn, tp = confusion(test_y, test_X.astype(float)).ravel()

    print("Test")
    print(f"TN: {tn}, TP: {tp}, FN: {fn}, FP: {fp}")


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
