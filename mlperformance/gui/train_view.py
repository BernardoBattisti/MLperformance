"""'New experiment' screen: load data, choose the target and algorithms, train."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data import SUPPORTED_EXTENSIONS, load_dataset
from ..models import get_algorithms
from ..preprocessing import CLASSIFICATION, REGRESSION, FeatureInfo, detect_task, select_features
from ..trainer import METRICS, Experiment, Settings
from .widgets import Banner, Card, DataFrameModel, StatTile, button, label, resource_path, table_view
from .worker import TrainingWorker, start_in_thread

PREVIEW_ROWS = 200


class TrainView(QWidget):
    open_results = Signal(str)  # experiment id
    experiment_saved = Signal(str)

    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame | None = None
        self.dataset_name = ""
        self.info: FeatureInfo | None = None
        self._thread = None
        self._worker = None
        self._last_experiment: Experiment | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 36)
        root.setSpacing(18)
        root.addWidget(label("New experiment", "title"))
        root.addWidget(label("Load your data, choose what to predict and let MLperformance "
                             "compare machine learning algorithms for you.", "subtitle", wrap=True))

        root.addWidget(self._build_data_card())
        self.target_card = self._build_target_card()
        self.algo_card = self._build_algorithms_card()
        self.settings_card = self._build_settings_card()
        self.run_card = self._build_run_card()
        for card in (self.target_card, self.algo_card, self.settings_card, self.run_card):
            root.addWidget(card)
            card.setEnabled(False)
        root.addStretch()

    # ----------------------------------------------------------------- data
    def _build_data_card(self) -> Card:
        card = Card("Data", 1, "CSV, TSV, Excel, JSON or Parquet. One row per sample, one column "
                               "per variable, column names in the first row.")
        row = QHBoxLayout()
        open_btn = button("Open file…", primary=True)
        open_btn.clicked.connect(self._open_file)
        row.addWidget(open_btn)
        row.addSpacing(12)
        row.addWidget(label("or try an example:", "hint"))
        self.example_combo = QComboBox()
        examples = sorted(p.name for p in resource_path("examples").glob("*.csv"))
        self.example_combo.addItems(examples)
        self.example_combo.setMinimumWidth(180)
        row.addWidget(self.example_combo)
        example_btn = button("Load example")
        example_btn.clicked.connect(self._load_example)
        example_btn.setEnabled(bool(examples))
        row.addWidget(example_btn)
        row.addStretch()
        card.body.addLayout(row)

        self.file_label = label("No file loaded yet.", "muted")
        card.body.addWidget(self.file_label)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.rows_tile, self.cols_tile, self.missing_tile = (
            StatTile("Rows"), StatTile("Columns"), StatTile("Missing values"))
        for tile in (self.rows_tile, self.cols_tile, self.missing_tile):
            tiles.addWidget(tile)
        self.tiles_widget = QWidget()
        self.tiles_widget.setLayout(tiles)
        tiles.setContentsMargins(0, 0, 0, 0)
        self.tiles_widget.hide()
        card.body.addWidget(self.tiles_widget)

        self.preview_model = DataFrameModel(float_format="{:g}")
        self.preview = table_view(self.preview_model, 230)
        self.preview.hide()
        card.body.addWidget(self.preview)
        return card

    def _open_file(self) -> None:
        patterns = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", "", f"Data files ({patterns})")
        if path:
            self._load(Path(path))

    def _load_example(self) -> None:
        name = self.example_combo.currentText()
        if name:
            self._load(resource_path("examples", name))

    def _load(self, path: Path) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            df = load_dataset(path)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Could not read the file", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.df, self.dataset_name = df, path.name
        self.file_label.setText(f"Loaded  {path.name}")
        self.rows_tile.set_value(f"{len(df):,}")
        self.cols_tile.set_value(str(len(df.columns)))
        self.missing_tile.set_value(f"{int(df.isna().sum().sum()):,}")
        self.tiles_widget.show()
        self.preview_model.set_frame(df.head(PREVIEW_ROWS))
        self.preview.resizeColumnsToContents()
        self.preview.show()

        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems([str(c) for c in df.columns])
        self.target_combo.setCurrentIndex(len(df.columns) - 1)
        self.target_combo.blockSignals(False)
        for card in (self.target_card, self.algo_card, self.settings_card, self.run_card):
            card.setEnabled(True)
        self.result_banner.hide()
        self._on_target_changed()

    # --------------------------------------------------------------- target
    def _build_target_card(self) -> Card:
        card = Card("Target", 2, "The column you want the model to predict.")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(260)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        form.addRow(label("Column to predict"), self.target_combo)

        task_row = QHBoxLayout()
        self.task_group = QButtonGroup(self)
        self.classification_radio = QRadioButton("Classification")
        self.regression_radio = QRadioButton("Regression")
        self.classification_radio.setToolTip("Predicts a category, e.g. 'spam' or 'not spam'.")
        self.regression_radio.setToolTip("Predicts a number, e.g. a price.")
        for radio in (self.classification_radio, self.regression_radio):
            self.task_group.addButton(radio)
            task_row.addWidget(radio)
            radio.toggled.connect(self._refresh_algorithms)
        task_row.addStretch()
        form.addRow(label("Problem type"), task_row)
        card.body.addLayout(form)

        self.features_label = label("", "hint", wrap=True)
        card.body.addWidget(self.features_label)
        self.target_warning = Banner("warning")
        card.body.addWidget(self.target_warning)
        return card

    def _task(self) -> str:
        return CLASSIFICATION if self.classification_radio.isChecked() else REGRESSION

    def _on_target_changed(self) -> None:
        if self.df is None or not self.target_combo.currentText():
            return
        target = self.df.columns[self.target_combo.currentIndex()]
        detected = detect_task(self.df[target])
        self.classification_radio.setText(
            "Classification" + ("  (detected)" if detected == CLASSIFICATION else ""))
        self.regression_radio.setText(
            "Regression" + ("  (detected)" if detected == REGRESSION else ""))
        (self.classification_radio if detected == CLASSIFICATION else self.regression_radio).setChecked(True)

        self.target_warning.hide()
        try:
            self.info = select_features(self.df, target)
        except ValueError as exc:
            self.info = None
            self.features_label.setText("")
            self.target_warning.show_message(str(exc))
            self._update_run_state()
            return
        text = (f"Using {len(self.info.numeric)} numeric and {len(self.info.categorical)} "
                f"text column(s) as input.")
        if self.info.dropped:
            text += "\nIgnored: " + "; ".join(f"{c} ({why})" for c, why in self.info.dropped.items())
        self.features_label.setText(text)
        if detected == CLASSIFICATION and self.df[target].nunique() > 50:
            self.target_warning.show_message(
                f"'{target}' has {self.df[target].nunique()} different values. "
                "Regression may be a better fit for this column.")
        self.name_edit.setText(f"{self.dataset_name} → {target}")
        self._refresh_algorithms()

    # ----------------------------------------------------------- algorithms
    def _build_algorithms_card(self) -> Card:
        card = Card("Algorithms", 3, "Not sure? Test them all: the comparison is the point.")
        mode_row = QHBoxLayout()
        self.all_radio = QRadioButton("Test all algorithms")
        self.pick_radio = QRadioButton("Choose specific algorithms")
        self.all_radio.setChecked(True)
        self.mode_group = QButtonGroup(self)
        for radio in (self.all_radio, self.pick_radio):
            self.mode_group.addButton(radio)
            mode_row.addWidget(radio)
            radio.toggled.connect(self._on_mode_changed)
        mode_row.addStretch()
        card.body.addLayout(mode_row)

        self.algo_list = QListWidget()
        self.algo_list.setFixedHeight(250)
        self.algo_list.itemChanged.connect(self._update_run_state)
        card.body.addWidget(self.algo_list)
        card.body.addWidget(label("Hover over an algorithm to see what it does.", "muted"))
        self.slow_warning = Banner("warning")
        card.body.addWidget(self.slow_warning)
        return card

    def _refresh_algorithms(self) -> None:
        if not self.classification_radio.isChecked() and not self.regression_radio.isChecked():
            return
        self.algo_list.blockSignals(True)
        self.algo_list.clear()
        for name, algo in get_algorithms(self._task()).items():
            item = QListWidgetItem(name)
            item.setToolTip(algo.description)
            item.setCheckState(Qt.Checked)
            self.algo_list.addItem(item)
        self.algo_list.blockSignals(False)
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        pick = self.pick_radio.isChecked()
        for i in range(self.algo_list.count()):
            item = self.algo_list.item(i)
            flags = Qt.ItemIsEnabled | Qt.ItemIsUserCheckable if pick else Qt.ItemIsEnabled
            item.setFlags(flags)
            if not pick:
                item.setCheckState(Qt.Checked)
        self._update_run_state()

    def _selected_algorithms(self) -> list[str]:
        return [self.algo_list.item(i).text() for i in range(self.algo_list.count())
                if self.algo_list.item(i).checkState() == Qt.Checked]

    # ------------------------------------------------------------- settings
    def _build_settings_card(self) -> Card:
        card = Card("Settings", 4, "The defaults work well for most datasets.")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        self.test_size = QSpinBox()
        self.test_size.setRange(10, 50)
        self.test_size.setSingleStep(5)
        self.test_size.setValue(20)
        self.test_size.setSuffix(" %")
        self.test_size.setToolTip("Share of rows held out to evaluate the final models.")
        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(0, 20)
        self.cv_folds.setValue(5)
        self.cv_folds.setToolTip("Number of cross-validation folds. 0 disables it (faster).")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(42)
        self.seed.setToolTip("Fix it to get reproducible results.")
        self.name_edit = QLineEdit()
        for col, (caption, widget) in enumerate([("Test set size", self.test_size),
                                                 ("Cross-validation folds", self.cv_folds),
                                                 ("Random seed", self.seed)]):
            grid.addWidget(label(caption, "tileLabel"), 0, col)
            grid.addWidget(widget, 1, col)
        grid.addWidget(label("Experiment name", "tileLabel"), 2, 0)
        grid.addWidget(self.name_edit, 3, 0, 1, 3)
        card.body.addLayout(grid)
        return card

    # ------------------------------------------------------------------ run
    def _build_run_card(self) -> Card:
        card = Card()
        row = QHBoxLayout()
        self.train_btn = button("Train models", primary=True)
        self.train_btn.setMinimumWidth(170)
        self.train_btn.clicked.connect(self._train)
        row.addWidget(self.train_btn)
        row.addSpacing(12)
        status = QVBoxLayout()
        self.status_label = label("", "hint")
        self.progress = QProgressBar()
        self.progress.setFixedHeight(8)
        self.progress.hide()
        status.addWidget(self.status_label)
        status.addWidget(self.progress)
        row.addLayout(status, 1)
        card.body.addLayout(row)

        self.result_banner = Banner("success")
        self.open_btn = button("Open results  →", primary=True)
        self.open_btn.clicked.connect(
            lambda: self._last_experiment and self.open_results.emit(self._last_experiment.id))
        self.result_banner.layout_.addWidget(self.open_btn)
        card.body.addWidget(self.result_banner)
        self.fail_banner = Banner("warning")
        card.body.addWidget(self.fail_banner)
        return card

    def _update_run_state(self) -> None:
        selected = self._selected_algorithms()
        training = self._thread is not None
        self.train_btn.setEnabled(bool(selected) and self.info is not None and not training)
        if not training:
            self.status_label.setText(f"{len(selected)} algorithm(s) selected." if selected
                                      else "Select at least one algorithm.")
        if self.df is not None:
            available = get_algorithms(self._task())
            slow = [n for n in selected if n in available and available[n].slow_above_rows
                    and len(self.df) > available[n].slow_above_rows]
            if slow:
                self.slow_warning.show_message(
                    f"These can be very slow on {len(self.df):,} rows: {', '.join(slow)}.")
            else:
                self.slow_warning.hide()

    def _train(self) -> None:
        target = self.df.columns[self.target_combo.currentIndex()]
        algorithms = self._selected_algorithms()
        settings = Settings(self.test_size.value() / 100, self.cv_folds.value(), self.seed.value())
        self.result_banner.hide()
        self.fail_banner.hide()
        self.progress.setRange(0, len(algorithms))
        self.progress.setValue(0)
        self.progress.show()
        for card in (self.target_card, self.algo_card, self.settings_card):
            card.setEnabled(False)

        self._worker = TrainingWorker(self.df, target, algorithms, self._task(), settings,
                                      self.dataset_name, self.name_edit.text().strip() or None)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread = start_in_thread(self._worker)
        self._thread.finished.connect(self._on_thread_done)
        self._update_run_state()

    def _on_progress(self, i: int, total: int, name: str) -> None:
        self.progress.setValue(i)
        self.status_label.setText("Finishing…" if i == total else f"Training {name}  ({i + 1} of {total})")

    def _on_finished(self, experiment: Experiment) -> None:
        self._last_experiment = experiment
        self.experiment_saved.emit(experiment.id)
        best = experiment.best
        if best is None:
            self.fail_banner.show_message("No algorithm could be trained on this data.")
        else:
            metric = METRICS[experiment.task][experiment.primary_metric][0]
            self.result_banner.show_message(
                f"<b>Best algorithm: {best['name']}</b><br>"
                f"{metric} = {best['metrics'][experiment.primary_metric]:.4f} on the test set")
        failed = [r for r in experiment.results if r["status"] == "error"]
        if failed:
            self.fail_banner.show_message(
                "Some algorithms failed: " + "; ".join(f"{r['name']}: {r['error']}" for r in failed))

    def _on_failed(self, message: str) -> None:
        self.fail_banner.show_message(f"Training failed: {message}")

    def _on_thread_done(self) -> None:
        self._thread = None
        self._worker = None
        self.progress.hide()
        for card in (self.target_card, self.algo_card, self.settings_card):
            card.setEnabled(True)
        self._update_run_state()

    def is_training(self) -> bool:
        return self._thread is not None
