"""Drives the desktop app end to end with an offscreen Qt platform."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mlperformance.storage import list_experiments  # noqa: E402


@pytest.fixture(scope="module")
def app():
    import matplotlib

    matplotlib.use("QtAgg")
    return QApplication.instance() or QApplication([])


def pump(ms: int = 100) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def train(window, dataset: str, algorithms: list[str]) -> None:
    view = window.train_view
    view.example_combo.setCurrentText(dataset)
    view._load_example()
    view.pick_radio.setChecked(True)
    for i in range(view.algo_list.count()):
        item = view.algo_list.item(i)
        item.setCheckState(Qt.Checked if item.text() in algorithms else Qt.Unchecked)
    assert view._selected_algorithms() == algorithms
    view.train_btn.click()
    for _ in range(600):
        if not view.is_training():
            break
        pump(100)
    assert not view.is_training()


@pytest.mark.parametrize("dataset,task,algorithms", [
    ("iris.csv", "classification", ["Decision Tree", "Naive Bayes"]),
    ("diabetes.csv", "regression", ["Ridge Regression", "Random Forest"]),
])
def test_train_and_open_results(app, runs_dir, dataset, task, algorithms):
    from mlperformance.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    train(window, dataset, algorithms)
    assert window.train_view.result_banner.isVisible()

    [experiment] = list_experiments()
    assert experiment.task == task
    window.train_view.open_btn.click()
    pump()
    results = window.results_view
    assert window.stack.currentIndex() == 1
    assert results.exp.id == experiment.id
    assert results.best_tile.value.text() == experiment.best["name"]
    assert results.board_model.rowCount() == len(algorithms)
    for i in range(results.tabs.count()):  # every tab and chart renders
        results.tabs.setCurrentIndex(i)
        pump()
    for choice in ("Dark", "Light"):  # theme switch redraws every chart
        window.theme_combo.setCurrentText(choice)
        pump()
    window.close()


def test_results_empty_state(app, runs_dir):
    from mlperformance.gui.main_window import MainWindow

    window = MainWindow()
    window._go(1)
    assert window.results_view.stack.currentWidget() is window.results_view.empty
    window.close()
