import pandas as pd
import os
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import numpy as np


def load_dutch():
    df = pd.read_csv("datasets/artifacts.csv", index_col=0)
    df.columns.name = "features"

    return df


def load_titanic():
    # https://www.kaggle.com/c/titanic/data

    def load(filename):
        df = pd.read_csv(os.path.join("datasets", "titanic", filename), index_col=0)

        df.columns = df.columns.str.lower()
        df.index.name = None

        if "survived" in df.columns:
            df["target"] = df["survived"]
            df = df.drop("survived", axis=1)

        return df

    drop_features = ["name", "ticket", "cabin"]

    df = replace_nan_titanic(load("train.csv"))

    df.drop(drop_features, axis=1, inplace=True)

    # train_X, train_y, test_X, test_y
    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=22, stratify=df["target"]
    )

    df_train = pd.DataFrame(df_train, columns=df.columns)
    df_test = pd.DataFrame(df_test, columns=df.columns)

    df_train = df_train.astype(df.dtypes.to_dict())
    df_test = df_test.astype(df.dtypes.to_dict())
    # df_train = pd.DataFrame(np.stack((train_X, train_y), axis=1), columns=df.columns)
    # df_test = pd.DataFrame(np.stack((test_X, test_y), axis=1), columns=df.columns)

    return df, df_train, df_test


def replace_nan_titanic(df):
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())

    df["embarked"] = df["embarked"].fillna(df["embarked"].mode().values[0])

    return df


HAPPINESS_DIR = os.path.join("datasets", "happines")


def load_happines():
    # https://www.kaggle.com/unsdsn/world-happiness

    df = pd.read_csv(os.path.join(HAPPINESS_DIR, "2015.csv"), index_col=0)

    # to lowercase
    df.columns = df.columns.str.lower()

    target = df["happiness score"]
    df.drop(
        ["happiness rank", "happiness score", "standard error", "dystopia residual"],
        inplace=True,
        axis=1,
    )

    df["target"] = target

    df.rename(
        columns={
            "economy (gdp per capita)": "economy",
            "health (life expectancy)": "health",
            "trust (government corruption)": "trust",
        },
        inplace=True,
    )

    df.index.name = "country"

    # df.sort_values("target", inplace=True)
    df_train = df.sample(frac=1, random_state=22)

    df = pd.read_csv(os.path.join(HAPPINESS_DIR, "2016.csv"), index_col=0)

    # to lowercase
    df.columns = df.columns.str.lower()

    target = df["happiness score"]
    df.drop(
        [
            "happiness rank",
            "happiness score",
            "upper confidence interval",
            "lower confidence interval",
            "dystopia residual",
        ],
        inplace=True,
        axis=1,
    )

    df["target"] = target

    df.rename(
        columns={
            "economy (gdp per capita)": "economy",
            "health (life expectancy)": "health",
            "trust (government corruption)": "trust",
        },
        inplace=True,
    )

    df.index.name = "country"

    df_test = df.sample(frac=1, random_state=22)

    return df_train, df_test
