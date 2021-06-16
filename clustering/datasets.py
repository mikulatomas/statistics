import pandas as pd


def dutch_data():
    """Prepares and loads diabetes data

    More information about data: De Deyne, Simon, et al. "Exemplar by feature applicability matrices and other Dutch normative data for semantic concepts." Behavior research methods 40.4 (2008): 1030-1048.

    """

    df = pd.read_csv("datasets/artifacts.csv", index_col=0)
    df.columns.name = "features"

    return df
