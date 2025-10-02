import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


# --------------------------
# Custom Frequency Encoder
# --------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features with their frequency counts."""

    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps[col] = freq
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            freq = self.freq_maps.get(col, {})
            X[col] = X[col].map(freq).fillna(0)
        return X


# --------------------------
# Preprocessor Builder
# --------------------------
def build_preprocessor(num_cols, cat_cols, model_type="linear"):
    """
    Build preprocessing pipeline.
    - Linear models → OneHotEncoder
    - Tree-based models → FrequencyEncoder
    """
    # numeric pipeline
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # categorical pipeline
    if model_type in ["tree", "forest", "boost"]:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("freqenc", FrequencyEncoder())
        ])
    else:  # linear models
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
