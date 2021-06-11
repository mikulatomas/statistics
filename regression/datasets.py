import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_tips():
    # https://www.kaggle.com/jsphyg/tipping?select=tips.csv

    df = pd.read_csv(os.path.join(os.path.join("datasets", "tips"), "tips.csv"))

    df["sex"] = df["sex"].replace({"Female": 1, "Male": 0})
    df["smoker"] = df["smoker"].replace({"Yes": 1, "No": 0})

    df = pd.get_dummies(df)

    target = df["tip"]
    df.drop("tip", axis=1, inplace=True)

    df["target"] = target

    return train_test_split(df, test_size=0.3, random_state=22)


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
