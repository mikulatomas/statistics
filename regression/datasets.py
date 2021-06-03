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
