import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_wine():
    # https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

    df = pd.read_csv(os.path.join(os.path.join("datasets", "wine"), "wine.csv"))

    target = df["quality"]
    df.drop("quality", axis=1, inplace=True)

    df["target"] = target
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=22)

    return df, df_train, df_test


def load_tips():
    # https://www.kaggle.com/jsphyg/tipping?select=tips.csv

    df = pd.read_csv(os.path.join(os.path.join("datasets", "tips"), "tips.csv"))

    df["sex"] = df["sex"].replace({"Female": 1, "Male": 0})
    df["smoker"] = df["smoker"].replace({"Yes": 1, "No": 0})

    df = pd.get_dummies(df)

    target = df["tip"]
    df.drop("tip", axis=1, inplace=True)

    df["target"] = target
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=22)

    return df, df_train, df_test


def load_home():
    # https://www.kaggle.com/shree1992/housedata

    df = pd.read_csv(os.path.join(os.path.join("datasets", "home"), "data.csv"))

    df.drop("date", axis=1, inplace=True)
    df.drop("street", axis=1, inplace=True)
    # df.drop("city", axis=1, inplace=True)
    df.drop("statezip", axis=1, inplace=True)
    df.drop("country", axis=1, inplace=True)
    df.drop("yr_renovated", axis=1, inplace=True)

    df = pd.get_dummies(df)

    target = df["price"]
    df.drop("price", axis=1, inplace=True)

    df["target"] = target

    df_train, df_test = train_test_split(df, test_size=0.5, random_state=22)

    return df, df_train, df_test


# HAPPINESS_DIR = os.path.join("datasets", "happines")


# def load_happines():
#     # https://www.kaggle.com/unsdsn/world-happiness

#     df = pd.read_csv(os.path.join(HAPPINESS_DIR, "2015.csv"), index_col=0)

#     # to lowercase
#     df.columns = df.columns.str.lower()

#     target = df["happiness score"]
#     df.drop(
#         ["happiness rank", "happiness score", "standard error", "dystopia residual"],
#         inplace=True,
#         axis=1,
#     )

#     df["target"] = target

#     df.rename(
#         columns={
#             "economy (gdp per capita)": "economy",
#             "health (life expectancy)": "health",
#             "trust (government corruption)": "trust",
#         },
#         inplace=True,
#     )

#     df.index.name = "country"

#     # df.sort_values("target", inplace=True)
#     df_train = df.sample(frac=1, random_state=22)

#     df = pd.read_csv(os.path.join(HAPPINESS_DIR, "2016.csv"), index_col=0)

#     # to lowercase
#     df.columns = df.columns.str.lower()

#     target = df["happiness score"]
#     df.drop(
#         [
#             "happiness rank",
#             "happiness score",
#             "upper confidence interval",
#             "lower confidence interval",
#             "dystopia residual",
#         ],
#         inplace=True,
#         axis=1,
#     )

#     df["target"] = target

#     df.rename(
#         columns={
#             "economy (gdp per capita)": "economy",
#             "health (life expectancy)": "health",
#             "trust (government corruption)": "trust",
#         },
#         inplace=True,
#     )

#     df.index.name = "country"

#     df_test = df.sample(frac=1, random_state=22)

#     return df_train, df_test
