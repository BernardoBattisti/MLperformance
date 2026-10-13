import numpy as np
import pandas as pd

from mlperformance.preprocessing import CLASSIFICATION, REGRESSION, detect_task, select_features


def test_detect_task():
    assert detect_task(pd.Series(["a", "b", "a"])) == CLASSIFICATION
    assert detect_task(pd.Series([0, 1, 1, 0])) == CLASSIFICATION
    assert detect_task(pd.Series([True, False])) == CLASSIFICATION
    assert detect_task(pd.Series(np.linspace(0, 1, 100))) == REGRESSION
    assert detect_task(pd.Series(range(100))) == REGRESSION


def test_select_features_drops_useless_columns():
    n = 200
    df = pd.DataFrame({
        "id": [f"user-{i}" for i in range(n)],
        "constant": 1,
        "empty": np.nan,
        "age": np.arange(n),
        "city": ["A", "B"] * (n // 2),
        "target": [0, 1] * (n // 2),
    })
    info = select_features(df, "target")
    assert info.numeric == ["age"]
    assert info.categorical == ["city"]
    assert set(info.dropped) == {"id", "constant", "empty"}
