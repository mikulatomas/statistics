from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.express as px
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)


def roc(model, data_X, data_y):
    predicted_y = model.predict_proba(data_X)

    try:
        predicted_y = predicted_y[:, 1]
    except:
        pass

    fpr, tpr, thresholds = roc_curve(data_y, predicted_y)

    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC Curve (AUC={auc(fpr, tpr):.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        width=700,
        height=500,
    )

    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    return fig


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
