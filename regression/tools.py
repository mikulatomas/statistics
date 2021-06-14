from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin


class SMWrapper(BaseEstimator, RegressorMixin):
    """A universal sklearn-style wrapper for statsmodels regressors"""

    def __init__(
        self, model_class, fit_intercept=True, ridge=False, lasso=False, alpha=0.0
    ):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.ridge = ridge
        self.lasso = lasso
        self.alpha = alpha

    def fit(self, X, y):
        # features = X.columns
        # X = X.values
        # y = y.values
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)

        if not (self.ridge or self.lasso):
            self.results_ = self.model_.fit()
        elif self.ridge:
            self.results_ = self.model_.fit(L1_wt=0, alpha=self.alpha)
        elif self.lasso:
            self.results_ = self.model_.fit(L1_wt=1, alpha=self.alpha)

        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)

    def summary(self):
        return self.results_.summary()


class SMWrapperRegularized(BaseEstimator, RegressorMixin):
    """A universal sklearn-style wrapper for statsmodels regressors"""

    def __init__(self, model_class, fit_intercept=True, alpha=0.0):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.alpha = 0.0

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)

        self.results_ = self.model_.fit_regularized(alpha=self.alpha)

        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)

    def summary(self):
        return self.results_.summary()


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
