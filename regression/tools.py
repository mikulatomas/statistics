import pandas as pd
import numpy as np


def polynomial_features(data, p):
    """Create polynomial features from data and degree p."""
    new_columns = []
    for i in range(p):
        degree = i + 1
        if degree > 1:
            new_columns.extend([f"{column}^{degree}" for column in data.columns])
        else:
            new_columns.extend(data.columns)

    df_poly = pd.DataFrame(
        np.hstack(tuple((data ** (i + 1) for i in range(p)))), columns=new_columns
    )

    return df_poly.reset_index(drop=True)
