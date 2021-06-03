import os
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


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

    df_train, df_test = train_test_split(df, train_size=0.2, random_state=22)

    return df, df_train, df_test


def replace_nan_titanic(df):
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())

    df["embarked"] = df["embarked"].fillna(df["embarked"].mode().values[0])

    return df


def balance_classes(df):
    sample_size = min(
        df[df["target"] == 1].count()[0],
        df[df["target"] == 0].count()[0],
    )

    return pd.concat(
        [
            resample(df[df["target"] == 0], replace=False, n_samples=sample_size),
            resample(df[df["target"] == 1], replace=False, n_samples=sample_size),
        ]
    )
