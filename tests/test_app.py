"""Headless UI tests using Streamlit's AppTest."""

import pytest
from streamlit.testing.v1 import AppTest

from conftest import EXAMPLES, ROOT
from mlperformance.data import load_dataset
from mlperformance.storage import list_experiments, save_experiment
from mlperformance.trainer import Settings, run_experiment


def _app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def test_train_page_runs_an_experiment(runs_dir):
    at = _app().run()
    assert not at.exception
    at.radio[0].set_value("Use an example dataset").run()
    at.selectbox[0].set_value("iris.csv").run()
    assert at.selectbox[1].value == "species"
    at.radio[2].set_value("Choose specific algorithms").run()
    at.multiselect[0].set_value(["Naive Bayes", "Decision Tree"]).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("Best algorithm" in s.value for s in at.success)
    assert len(list_experiments()) == 1


@pytest.mark.parametrize("dataset,target", [("iris.csv", "species"),
                                            ("diabetes.csv", "disease_progression")])
def test_results_page_renders(runs_dir, dataset, target):
    df = load_dataset(EXAMPLES / dataset)
    exp, bundles = run_experiment(df, target, settings=Settings(cv_folds=2))
    save_experiment(exp, bundles)

    at = _app().run()
    at.switch_page("app_pages/results.py").run()
    assert not at.exception, at.exception
    assert any(m.value == exp.best["name"] for m in at.metric)
