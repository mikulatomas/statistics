import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def diabetes_data():
    """Prepares and loads diabetes data

    More information about data: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

    """
    dataset = load_diabetes(as_frame=True)

    df = dataset["data"]
    df["target"] = dataset["target"]

    # lowercase feature names
    df.columns = df.columns.str.lower()

    # split data into train and test
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=22)

    # separate input and output variables
    train_X = df_train.drop(["target"], axis=1).reset_index(drop=True)
    train_y = df_train["target"].reset_index(drop=True)
    test_X = df_test.drop(["target"], axis=1).reset_index(drop=True)
    test_y = df_test["target"].reset_index(drop=True)
    original_X = df.drop(["target"], axis=1).reset_index(drop=True)
    original_y = df["target"].reset_index(drop=True)

    return original_X, original_y, train_X, train_y, test_X, test_y
