import pandas as pd


def load_dutch():
    df = pd.read_csv("datasets/artifacts.csv", index_col=0)
    df.columns.name = "features"

    return df
