import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes as diabetes


def load_diabetes():
    dataset = diabetes(as_frame=True)

    df = dataset["data"]

    df["target"] = dataset["target"]

    df.columns = df.columns.str.lower()

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=22)

    return df, df_train, df_test
