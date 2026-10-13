"""Task detection, column selection and the shared preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASSIFICATION = "classification"
REGRESSION = "regression"

# A numeric target with at most this many distinct values is treated as classes.
MAX_CLASSES_FOR_NUMERIC_TARGET = 20
# Categorical columns with more distinct values than this are likely free text / IDs.
MAX_CATEGORIES = 50


@dataclass
class FeatureInfo:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # column -> reason

    @property
    def used(self) -> list[str]:
        return self.numeric + self.categorical


def detect_task(y: pd.Series) -> str:
    """Guess whether predicting ``y`` is a classification or regression problem."""
    y = y.dropna()
    if y.dtype == bool or not pd.api.types.is_numeric_dtype(y):
        return CLASSIFICATION
    n_unique = y.nunique()
    is_integer = np.all(np.mod(y.to_numpy(dtype=float), 1) == 0)
    if is_integer and n_unique <= MAX_CLASSES_FOR_NUMERIC_TARGET:
        return CLASSIFICATION
    return REGRESSION


def select_features(df: pd.DataFrame, target: str) -> FeatureInfo:
    """Split feature columns into numeric/categorical and drop unusable ones."""
    info = FeatureInfo()
    n_rows = len(df)
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if s.isna().all():
            info.dropped[col] = "column is empty"
        elif s.nunique(dropna=True) <= 1:
            info.dropped[col] = "column has a single value"
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.dropped[col] = "date/time columns are not supported yet"
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            info.numeric.append(col)
        else:
            n_unique = s.nunique(dropna=True)
            if n_unique > MAX_CATEGORIES and n_unique > 0.5 * n_rows:
                info.dropped[col] = "too many distinct text values (looks like an ID or free text)"
            else:
                info.categorical.append(col)
    if not info.used:
        raise ValueError("No usable feature columns were found besides the target.")
    return info


def build_preprocessor(info: FeatureInfo) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical ones."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=MAX_CATEGORIES,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, info.numeric), ("cat", categorical, info.categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def clean_features(df: pd.DataFrame, info: FeatureInfo) -> pd.DataFrame:
    """Select the feature columns and normalise their types for the pipeline."""
    X = df[info.used].copy()
    for col in info.categorical:
        # Mixed types (e.g. 1 and "1") would break the encoder; keep NaN as NaN.
        X[col] = X[col].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    for col in info.numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    return X


def prepare_xy(df: pd.DataFrame, target: str, info: FeatureInfo) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target, dropping rows where the target is missing."""
    data = df.dropna(subset=[target])
    if len(data) < 10:
        raise ValueError("At least 10 rows with a non-empty target are required.")
    return clean_features(data, info), data[target]
