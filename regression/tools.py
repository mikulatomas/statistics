from sklearn.metrics import confusion_matrix as confusion
from sklearn.utils import resample
import pandas as pd


def model_performance(model, train_X, train_y, test_X, test_y):
    print("Train score")
    print(model.score(train_X, train_y))
    print("Test score")
    print(model.score(test_X, test_y))


def coef_table(coef, features):
    return pd.DataFrame(zip(features, coef), columns=["feature", "coef"]).sort_values(
        "coef", ascending=False
    )
