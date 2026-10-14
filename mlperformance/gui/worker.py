"""Runs training in a background thread so the window stays responsive."""

from __future__ import annotations

import traceback

import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal

from ..storage import save_experiment
from ..trainer import Settings, run_experiment


class TrainingWorker(QObject):
    progress = Signal(int, int, str)  # index, total, algorithm name
    finished = Signal(object)  # Experiment
    failed = Signal(str)

    def __init__(self, df: pd.DataFrame, target: str, algorithms: list[str], task: str,
                 settings: Settings, dataset_name: str, experiment_name: str):
        super().__init__()
        self.args = dict(df=df, target=target, algorithms=algorithms, task=task, settings=settings,
                         dataset_name=dataset_name, experiment_name=experiment_name)

    def run(self) -> None:
        try:
            experiment, bundles = run_experiment(**self.args, progress=self.progress.emit)
            save_experiment(experiment, bundles)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))
            return
        self.finished.emit(experiment)


def start_in_thread(worker: TrainingWorker) -> QThread:
    """Move ``worker`` to a new thread, start it and clean both up when done."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
