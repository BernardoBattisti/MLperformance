import numpy as np
import pandas as pd
import pytest

from conftest import EXAMPLES
from mlperformance.data import load_dataset
from mlperformance.models import get_algorithms
from mlperformance.storage import (
    delete_experiment,
    list_experiments,
    load_experiment,
    load_model,
    model_path,
    save_experiment,
)
from mlperformance.trainer import Settings, run_experiment

FAST = Settings(cv_folds=3)


def test_classification_all_algorithms():
    df = load_dataset(EXAMPLES / "iris.csv")
    exp, bundles = run_experiment(df, "species", settings=FAST)
    assert exp.task == "classification"
    assert len(exp.results) == len(get_algorithms("classification"))
    failed = [r for r in exp.results if r["status"] != "ok"]
    assert not failed, failed
    best = exp.best
    assert best["metrics"]["accuracy"] > 0.85
    assert len(best["cv_scores"]) == 3
    assert np.asarray(best["confusion_matrix"]).sum() == exp.settings["n_test"]
    assert len(best["roc"]) == 3  # one-vs-rest curve per class
    assert set(bundles[best["name"]].predict(df.head(5))) <= set(exp.classes)


def test_regression_selected_algorithms():
    df = load_dataset(EXAMPLES / "diabetes.csv")
    exp, _ = run_experiment(df, "disease_progression", ["Ridge Regression", "Random Forest"],
                            settings=FAST)
    assert exp.task == "regression"
    assert [r["name"] for r in exp.results] == ["Ridge Regression", "Random Forest"]
    assert exp.best["metrics"]["r2"] > 0.2
    assert len(exp.best["y_true"]) == exp.settings["n_test"]
    assert exp.best["feature_importance"]


def test_mixed_types_and_missing_values():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "income": rng.normal(5000, 1000, n),
        "city": rng.choice(["Porto Alegre", "São Paulo", None], n),
        "age": rng.integers(18, 80, n).astype(float),
    })
    df.loc[::7, "age"] = np.nan
    df["buys"] = np.where(df["income"] > 5000, "yes", "no")
    exp, _ = run_experiment(df, "buys", ["Logistic Regression", "Decision Tree"], settings=FAST)
    assert all(r["status"] == "ok" for r in exp.results)
    assert exp.classes == ["no", "yes"]


def test_unknown_algorithm_rejected():
    df = load_dataset(EXAMPLES / "iris.csv")
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_experiment(df, "species", ["Linear Regression"])


def test_storage_roundtrip(tmp_path):
    df = load_dataset(EXAMPLES / "iris.csv")
    exp, bundles = run_experiment(df, "species", ["Naive Bayes"], settings=FAST)
    save_experiment(exp, bundles, tmp_path)
    assert [e.id for e in list_experiments(tmp_path)] == [exp.id]
    assert load_experiment(exp.id, tmp_path).best["name"] == "Naive Bayes"
    model = load_model(model_path(exp.id, "Naive Bayes", tmp_path))
    assert len(model.predict(df)) == len(df)
    delete_experiment(exp.id, tmp_path)
    assert list_experiments(tmp_path) == []
