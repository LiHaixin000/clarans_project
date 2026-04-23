import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def load_data(
    filepath: str,
    columns: Optional[List[str]] = None,
    normalize: bool = False,
    standardize: bool = False,
    dropna: bool = True,
    header: Optional[int] = "infer",
) -> Tuple[np.ndarray, pd.DataFrame]:
    if normalize and standardize:
        raise ValueError("Use either normalize or standardize, not both.")

    if filepath.lower().endswith(".csv"):
        if header == "infer":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_csv(filepath, header=header)
    else:
        if header == "infer":
            df = pd.read_table(filepath)
        else:
            df = pd.read_table(filepath, header=header)

    if dropna:
        df = df.dropna()

    # Special handling for Iris-like no-header dataset:
    # 5 columns = 4 numeric features + 1 species label
    if header is None and df.shape[1] == 5:
        df.columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ]

    if columns is not None:
        df_selected = df[columns].copy()
    else:
        df_selected = df.select_dtypes(include=["number"]).copy()

    if df_selected.shape[1] == 0:
        raise ValueError("No numerical columns found or selected.")

    X = df_selected.to_numpy(dtype=float)

    if normalize:
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        X = (X - min_vals) / denom

    if standardize:
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds == 0, 1, stds)
        X = (X - means) / stds

    return X, df_selected
