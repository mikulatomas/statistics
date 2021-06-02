import os
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

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

    df.sort_values("target", inplace=True)

    df_train = df

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

    df.sort_values("target", inplace=True)

    df_test = df

    return df_train, df_test


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
