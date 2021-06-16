import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def titanic_data():
    """Prepares and loads diabetes data

    More information about data: https://www.kaggle.com/c/titanic/data

    """

    drop_features = ["name", "ticket", "cabin"]

    df = pd.read_csv(os.path.join("datasets", "titanic", "train.csv"), index_col=0)

    # lowercase feature names, remove name of index
    df.columns = df.columns.str.lower()
    df.index.name = None

    # extract target feature
    df["target"] = df["survived"]
    df = df.drop("survived", axis=1)

    # drop features
    df.drop(drop_features, axis=1, inplace=True)

    # fill missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode().values[0])

    # scale sex feature to 0/1 instead of male/female
    df["sex"].replace({"male": 0, "female": 1}, inplace=True)

    # create dummy variables for rest qualitative features
    df = pd.get_dummies(df)

    # split data into train and test, use stratification if possible
    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=22, stratify=df["target"]
    )

    df_train = pd.DataFrame(df_train, columns=df.columns)
    df_test = pd.DataFrame(df_test, columns=df.columns)

    df_train = df_train.astype(df.dtypes.to_dict())
    df_test = df_test.astype(df.dtypes.to_dict())

    # separate input and output variables
    train_X = df_train.drop(["target"], axis=1).reset_index(drop=True)
    train_y = df_train["target"].reset_index(drop=True)
    test_X = df_test.drop(["target"], axis=1).reset_index(drop=True)
    test_y = df_test["target"].reset_index(drop=True)
    original_X = df.drop(["target"], axis=1).reset_index(drop=True)
    original_y = df["target"].reset_index(drop=True)

    # standartize data
    scaler = StandardScaler().fit(original_X)
    original_X = pd.DataFrame(scaler.transform(original_X), columns=original_X.columns)
    train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
    test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    return original_X, original_y, train_X, train_y, test_X, test_y
