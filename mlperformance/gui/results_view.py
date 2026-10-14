"""'Results' screen: reopen any saved experiment and compare its algorithms."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..preprocessing import CLASSIFICATION
from ..storage import delete_experiment, list_experiments, model_path, runs_root
from ..trainer import METRICS, Experiment
from . import charts
from .theme import current
from .widgets import (
    Banner,
    Card,
    ChartView,
    DataFrameModel,
    StatTile,
    fit_table_height,
    button,
    label,
    scroll_page,
    table_view,
)


def leaderboard(exp: Experiment) -> pd.DataFrame:
    metrics = METRICS[exp.task]
    higher = metrics[exp.primary_metric][1]
    rows = []
    for rank, r in enumerate(sorted(exp.successful, key=lambda r: r["metrics"][exp.primary_metric],
                                    reverse=higher), 1):
        row = {"#": rank, "Algorithm": r["name"]}
        row.update({lbl: r["metrics"].get(key) for key, (lbl, _) in metrics.items()})
        if r.get("cv_scores"):
            row["CV mean"] = float(np.mean(r["cv_scores"]))
            row["CV std"] = float(np.std(r["cv_scores"]))
        row["Train time (s)"] = r["train_time"]
        rows.append(row)
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def _tab(*widgets) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 16, 6, 16)
    layout.setSpacing(16)
    for w in widgets:
        layout.addWidget(w)
    layout.addStretch()
    return scroll_page(page)


class ResultsView(QWidget):
    def __init__(self):
        super().__init__()
        self.experiments: list[Experiment] = []
        self.exp: Experiment | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 20)
        root.setSpacing(14)
        root.addWidget(label("Results", "title"))

        picker = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(420)
        self.combo.currentIndexChanged.connect(self._on_select)
        picker.addWidget(self.combo, 1)
        refresh = button("Refresh")
        refresh.clicked.connect(lambda: self.reload())
        picker.addWidget(refresh)
        self.delete_btn = button("Delete", danger=True)
        self.delete_btn.clicked.connect(self._delete)
        picker.addWidget(self.delete_btn)
        root.addLayout(picker)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.empty = QWidget()
        empty_layout = QVBoxLayout(self.empty)
        empty_layout.addStretch()
        msg = label("No experiments yet.\nTrain your first models on the New experiment screen.",
                    "subtitle")
        msg.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(msg)
        empty_layout.addStretch()
        self.stack.addWidget(self.empty)
        self.content = self._build_content()
        self.stack.addWidget(self.content)

    # -------------------------------------------------------------- layout
    def _build_content(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        self.caption = label("", "hint", wrap=True)
        layout.addWidget(self.caption)
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.best_tile, self.score_tile, self.count_tile = (
            StatTile("Best algorithm"), StatTile("Score"), StatTile("Algorithms compared"))
        for tile in (self.best_tile, self.score_tile, self.count_tile):
            tiles.addWidget(tile)
        layout.addLayout(tiles)
        self.error_banner = Banner("warning")
        layout.addWidget(self.error_banner)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_overview(), "Leaderboard")
        self.tabs.addTab(self._build_details(), "Algorithm details")
        self.tabs.addTab(self._build_curves(), "ROC curves")
        self.tabs.addTab(self._build_info(), "Experiment info")
        layout.addWidget(self.tabs, 1)
        return w

    def _build_overview(self) -> QWidget:
        board = Card("Leaderboard", subtitle="All scores are measured on the test set. "
                                             "CV mean/std summarize the cross-validation folds.")
        self.board_model = DataFrameModel()
        self.board_view = table_view(self.board_model, 260)
        board.body.addWidget(self.board_view)
        export = button("Export CSV…")
        export.clicked.connect(self._export_csv)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(export)
        board.body.addLayout(row)

        compare = Card("Comparison by metric")
        self.metric_combo = QComboBox()
        self.metric_combo.currentIndexChanged.connect(self._draw_comparison)
        row = QHBoxLayout()
        row.addWidget(label("Metric"))
        row.addWidget(self.metric_combo)
        row.addStretch()
        compare.body.addLayout(row)
        self.compare_chart = ChartView()
        compare.body.addWidget(self.compare_chart)

        self.cv_card = Card("Cross-validation stability",
                            subtitle="Each dot is one fold. A narrow box means the algorithm "
                                     "performs consistently on different parts of the data.")
        self.cv_chart = ChartView()
        self.cv_card.body.addWidget(self.cv_chart)

        time_card = Card("Training time")
        self.time_chart = ChartView()
        time_card.body.addWidget(self.time_chart)
        return _tab(board, compare, self.cv_card, time_card)

    def _build_details(self) -> QWidget:
        top = Card()
        row = QHBoxLayout()
        row.addWidget(label("Algorithm"))
        self.algo_combo = QComboBox()
        self.algo_combo.setMinimumWidth(260)
        self.algo_combo.currentIndexChanged.connect(self._draw_details)
        row.addWidget(self.algo_combo)
        row.addStretch()
        self.export_model_btn = button("Export trained model…")
        self.export_model_btn.clicked.connect(self._export_model)
        row.addWidget(self.export_model_btn)
        top.body.addLayout(row)
        self.metric_tiles_row = QHBoxLayout()
        self.metric_tiles_row.setSpacing(10)
        top.body.addLayout(self.metric_tiles_row)

        self.cm_card = Card("Confusion matrix",
                            subtitle="Rows are the real classes, columns the predicted ones. "
                                     "The diagonal holds the correct predictions.")
        self.normalize_box = QCheckBox("Show percentages per actual class")
        self.normalize_box.setChecked(True)
        self.normalize_box.toggled.connect(self._draw_details)
        self.cm_card.body.addWidget(self.normalize_box)
        self.cm_chart = ChartView(420)
        self.cm_card.body.addWidget(self.cm_chart)

        self.pva_card = Card("Predicted vs. actual",
                             subtitle="Each dot is a test sample; dots on the dashed line are "
                                      "perfect predictions.")
        self.pva_chart = ChartView(380)
        self.pva_card.body.addWidget(self.pva_chart)
        self.resid_card = Card("Error distribution",
                               subtitle="A good model is centered on 0 with a narrow spread.")
        self.resid_chart = ChartView(320)
        self.resid_card.body.addWidget(self.resid_chart)

        self.fi_card = Card("Most influential features",
                            subtitle="What the model relies on most. This shows correlation, "
                                     "not causation.")
        self.fi_chart = ChartView()
        self.fi_card.body.addWidget(self.fi_chart)
        self.fi_empty = label("This algorithm does not report feature importances.", "muted")
        self.fi_card.body.addWidget(self.fi_empty)
        return _tab(top, self.cm_card, self.pva_card, self.resid_card, self.fi_card)

    def _build_curves(self) -> QWidget:
        self.roc_card = Card("ROC curves",
                             subtitle="The closer a curve is to the top-left corner, the better. "
                                      "AUC 1.0 is perfect, 0.5 is random guessing.")
        row = QHBoxLayout()
        self.roc_list = QListWidget()
        self.roc_list.setFixedSize(260, 200)
        self.roc_list.itemChanged.connect(self._on_roc_pick)
        side = QVBoxLayout()
        side.addWidget(label(f"Algorithms (up to {charts.MAX_CURVES})", "tileLabel"))
        side.addWidget(self.roc_list)
        side.addWidget(label("Class (one-vs-rest)", "tileLabel"))
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._draw_roc)
        side.addWidget(self.class_combo)
        side.addStretch()
        row.addLayout(side)
        self.roc_chart = ChartView(440)
        row.addWidget(self.roc_chart, 1)
        self.roc_card.body.addLayout(row)

        self.errors_card = Card("Error comparison",
                                subtitle="Absolute error of every test sample, per algorithm. "
                                         "The line inside each box is the median error.")
        self.errors_chart = ChartView()
        self.errors_card.body.addWidget(self.errors_chart)
        return _tab(self.roc_card, self.errors_card)

    def _build_info(self) -> QWidget:
        card = Card("Experiment info")
        self.info_label = label("", wrap=True)
        self.info_label.setTextFormat(Qt.RichText)
        card.body.addWidget(self.info_label)
        return _tab(card)

    # ---------------------------------------------------------------- data
    def reload(self, select_id: str | None = None) -> None:
        current_id = select_id or (self.exp.id if self.exp else None)
        self.experiments = list_experiments()
        self.combo.blockSignals(True)
        self.combo.clear()
        for e in self.experiments:
            self.combo.addItem(f"{e.name}   ·   {e.created_at.replace('T', ' ')}", e.id)
        ids = [e.id for e in self.experiments]
        self.combo.setCurrentIndex(ids.index(current_id) if current_id in ids else 0)
        self.combo.blockSignals(False)
        self.delete_btn.setEnabled(bool(self.experiments))
        self._on_select()

    def _on_select(self) -> None:
        idx = self.combo.currentIndex()
        if idx < 0 or not self.experiments:
            self.exp = None
            self.stack.setCurrentWidget(self.empty)
            return
        self.exp = self.experiments[idx]
        self.stack.setCurrentWidget(self.content)
        self._render()

    def _render(self) -> None:
        exp = self.exp
        s = exp.settings
        self.caption.setText(
            f"{exp.dataset}  ·  target: {exp.target}  ·  {exp.task}  ·  {exp.n_rows:,} rows "
            f"({s['n_train']:,} train / {s['n_test']:,} test)  ·  {s['cv_folds']}-fold CV  ·  "
            f"seed {s['random_state']}")
        failed = [r for r in exp.results if r["status"] == "error"]
        if failed:
            self.error_banner.show_message(
                f"{len(failed)} algorithm(s) failed: " +
                "; ".join(f"{r['name']}: {r['error']}" for r in failed))
        else:
            self.error_banner.hide()

        best = exp.best
        metric_label = METRICS[exp.task][exp.primary_metric][0]
        self.best_tile.set_value(best["name"] if best else "—")
        self.score_tile.set_caption(f"{metric_label} (test set)")
        self.score_tile.set_value(f"{best['metrics'][exp.primary_metric]:.4f}" if best else "—")
        self.count_tile.set_value(str(len(exp.successful)))
        self.tabs.setEnabled(best is not None)
        if best is None:
            return

        board = leaderboard(exp)
        self.board_model.set_frame(board)
        self.board_view.resizeColumnsToContents()
        header = self.board_view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # "Algorithm" takes the free space
        fit_table_height(self.board_view, max_rows=len(board))

        metrics = METRICS[exp.task]
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for key, (lbl, _) in metrics.items():
            if any(key in r["metrics"] for r in exp.successful):
                self.metric_combo.addItem(lbl, key)
        self.metric_combo.setCurrentIndex(self.metric_combo.findData(exp.primary_metric))
        self.metric_combo.blockSignals(False)

        names = [r["name"] for r in sorted(exp.successful, key=lambda r: r["metrics"][exp.primary_metric],
                                           reverse=metrics[exp.primary_metric][1])]
        self.algo_combo.blockSignals(True)
        self.algo_combo.clear()
        self.algo_combo.addItems(names)
        self.algo_combo.blockSignals(False)

        is_clf = exp.task == CLASSIFICATION
        self.tabs.setTabText(2, "ROC curves" if is_clf else "Error comparison")
        self.roc_card.setVisible(is_clf)
        self.errors_card.setVisible(not is_clf)
        if is_clf:
            self._fill_roc_controls()
        self._fill_info()
        self.redraw()

    def redraw(self) -> None:
        """Draw every chart (also called when the theme changes)."""
        if self.exp is None or self.exp.best is None:
            return
        self._draw_comparison()
        exp, n = self.exp, len(self.exp.successful)
        has_cv = any(r.get("cv_scores") for r in exp.successful)
        self.cv_card.setVisible(has_cv)
        if has_cv:
            self.cv_chart.plot(lambda f: charts.cv_distribution(f, exp, current()),
                               charts.bar_chart_height(n))
        self.time_chart.plot(lambda f: charts.training_time(f, exp, current()),
                             charts.bar_chart_height(n))
        self._draw_details()
        if exp.task == CLASSIFICATION:
            self._draw_roc()
        else:
            self.errors_chart.plot(lambda f: charts.error_comparison(f, exp, current()),
                                   charts.bar_chart_height(n))

    def _draw_comparison(self) -> None:
        metric = self.metric_combo.currentData()
        if self.exp is None or metric is None:
            return
        exp = self.exp
        self.compare_chart.plot(lambda f: charts.metric_comparison(f, exp, metric, current()),
                                charts.bar_chart_height(len(exp.successful)))

    def _result(self) -> dict | None:
        name = self.algo_combo.currentText()
        return next((r for r in self.exp.successful if r["name"] == name), None) if self.exp else None

    def _draw_details(self) -> None:
        result = self._result()
        if result is None:
            return
        exp = self.exp
        while self.metric_tiles_row.count():
            self.metric_tiles_row.takeAt(0).widget().deleteLater()
        for key, value in result["metrics"].items():
            self.metric_tiles_row.addWidget(StatTile(METRICS[exp.task][key][0], f"{value:.4f}"))
        self.metric_tiles_row.addWidget(StatTile("Train time", charts.format_seconds(result["train_time"])))

        is_clf = exp.task == CLASSIFICATION
        self.cm_card.setVisible(is_clf)
        self.pva_card.setVisible(not is_clf)
        self.resid_card.setVisible(not is_clf)
        if is_clf:
            n = len(exp.classes)
            normalize = self.normalize_box.isChecked()
            self.cm_chart.plot(
                lambda f: charts.confusion_matrix(f, result, exp.classes, normalize, current()),
                min(640, max(300, 52 * n + 140)))
        else:
            self.pva_chart.plot(lambda f: charts.predicted_vs_actual(f, result, current()))
            self.resid_chart.plot(lambda f: charts.residuals(f, result, current()))

        importances = result.get("feature_importance") or {}
        self.fi_chart.setVisible(bool(importances))
        self.fi_empty.setVisible(not importances)
        if importances:
            self.fi_chart.plot(lambda f: charts.feature_importance(f, result, current()),
                               charts.bar_chart_height(min(15, len(importances))))
        self.export_model_btn.setEnabled(model_path(exp.id, result["name"]).exists())

    def _fill_roc_controls(self) -> None:
        exp = self.exp
        with_roc = [r for r in exp.successful if r.get("roc")]
        with_roc.sort(key=lambda r: r["metrics"].get("roc_auc", 0), reverse=True)
        self.roc_list.blockSignals(True)
        self.roc_list.clear()
        for i, r in enumerate(with_roc):
            item = QListWidgetItem(r["name"])
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i < charts.MAX_CURVES else Qt.Unchecked)
            self.roc_list.addItem(item)
        self.roc_list.blockSignals(False)
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        if len(exp.classes) == 2:
            self.class_combo.addItem(f"{exp.classes[1]} (positive class)", 1)
        else:
            for i, c in enumerate(exp.classes):
                self.class_combo.addItem(c, i)
        self.class_combo.blockSignals(False)

    def _checked_roc(self) -> list[str]:
        return [self.roc_list.item(i).text() for i in range(self.roc_list.count())
                if self.roc_list.item(i).checkState() == Qt.Checked]

    def _on_roc_pick(self, item: QListWidgetItem) -> None:
        if item.checkState() == Qt.Checked and len(self._checked_roc()) > charts.MAX_CURVES:
            self.roc_list.blockSignals(True)
            item.setCheckState(Qt.Unchecked)
            self.roc_list.blockSignals(False)
            return
        self._draw_roc()

    def _draw_roc(self) -> None:
        if self.exp is None or self.exp.task != CLASSIFICATION:
            return
        exp, chosen, cls = self.exp, self._checked_roc(), self.class_combo.currentData()
        if cls is None:
            return
        self.roc_chart.plot(lambda f: charts.roc_curves(f, exp, chosen, cls, current()))

    def _fill_info(self) -> None:
        exp, f = self.exp, self.exp.features
        parts = [
            f"<b>Dataset:</b> {exp.dataset}",
            f"<b>Created:</b> {exp.created_at.replace('T', ' ')}",
            f"<b>Numeric features ({len(f['numeric'])}):</b> {', '.join(f['numeric']) or '—'}",
            f"<b>Text features ({len(f['categorical'])}):</b> {', '.join(f['categorical']) or '—'}",
        ]
        if f["dropped"]:
            parts.append("<b>Ignored columns:</b> " +
                         "; ".join(f"{c} ({why})" for c, why in f["dropped"].items()))
        if exp.classes:
            parts.append(f"<b>Classes:</b> {', '.join(exp.classes)}")
        s = exp.settings
        parts.append(f"<b>Settings:</b> test size {s['test_size']:.0%}, {s['cv_folds']} CV folds, "
                     f"seed {s['random_state']}")
        parts.append(f"<b>Stored in:</b> {runs_root() / exp.id}")
        self.info_label.setText("<br><br>".join(parts))

    # ------------------------------------------------------------- actions
    def _export_csv(self) -> None:
        if self.exp is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export leaderboard",
                                              f"{self.exp.id}-leaderboard.csv", "CSV (*.csv)")
        if path:
            leaderboard(self.exp).to_csv(path, index=False)

    def _export_model(self) -> None:
        result = self._result()
        if result is None:
            return
        src = model_path(self.exp.id, result["name"])
        path, _ = QFileDialog.getSaveFileName(self, "Export trained model", src.name,
                                              "Joblib model (*.joblib)")
        if path:
            shutil.copyfile(src, path)

    def _delete(self) -> None:
        if self.exp is None:
            return
        answer = QMessageBox.question(
            self, "Delete experiment",
            f"Permanently delete “{self.exp.name}” and its trained models?")
        if answer == QMessageBox.Yes:
            delete_experiment(self.exp.id)
            self.exp = None
            self.reload()
