"""Main window: sidebar navigation on the left, screens on the right."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from . import theme
from .results_view import ResultsView
from .train_view import TrainView
from .widgets import label, resource_path, scroll_page

THEME_CHOICES = ["System", "Light", "Dark"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLperformance")
        self.setWindowIcon(QIcon(str(resource_path("assets", "icon.png"))))
        self.resize(1280, 860)
        self.setMinimumSize(980, 640)
        self.settings = QSettings("MLperformance", "MLperformance")

        self.train_view = TrainView()
        self.results_view = ResultsView()
        self.stack = QStackedWidget()
        self.stack.addWidget(scroll_page(self.train_view))
        self.stack.addWidget(self.results_view)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.train_view.open_results.connect(self._show_results)
        self.train_view.experiment_saved.connect(lambda eid: self.results_view.reload(eid))
        QApplication.styleHints().colorSchemeChanged.connect(lambda _: self.apply_theme())

        saved = self.settings.value("theme", "System")
        self.theme_combo.setCurrentText(saved if saved in THEME_CHOICES else "System")
        self.apply_theme()

    def _build_sidebar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(232)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(16, 22, 16, 16)
        layout.setSpacing(4)

        brand = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(QPixmap(str(resource_path("assets", "icon.png"))).scaled(
            30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        brand.addWidget(logo)
        name = QLabel("MLperformance")
        name.setObjectName("Brand")
        brand.addWidget(name)
        brand.addStretch()
        layout.addLayout(brand)
        layout.addWidget(label("Find the best algorithm\nfor your data", "muted"))
        layout.addSpacing(22)

        self.nav = QButtonGroup(self)
        for index, text in enumerate(["New experiment", "Results"]):
            b = QPushButton(text)
            b.setProperty("nav", True)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=index: self._go(i))
            self.nav.addButton(b, index)
            layout.addWidget(b)
        self.nav.button(0).setChecked(True)
        layout.addStretch()

        layout.addWidget(label("Appearance", "tileLabel"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(THEME_CHOICES)
        self.theme_combo.currentTextChanged.connect(self._on_theme_choice)
        layout.addWidget(self.theme_combo)
        layout.addSpacing(10)
        layout.addWidget(label(f"Version {__version__}  ·  MIT License", "muted"))
        return bar

    def _go(self, index: int) -> None:
        self.nav.button(index).setChecked(True)
        if index == 1:
            self.results_view.reload()
        self.stack.setCurrentIndex(index)

    def _show_results(self, experiment_id: str) -> None:
        self._go(1)
        self.results_view.reload(experiment_id)

    def _on_theme_choice(self, choice: str) -> None:
        self.settings.setValue("theme", choice)
        self.apply_theme()

    def apply_theme(self) -> None:
        choice = self.theme_combo.currentText()
        t = {"Light": theme.LIGHT, "Dark": theme.DARK}.get(choice) or theme.system_theme()
        theme.set_current(t)
        app = QApplication.instance()
        app.setPalette(theme.palette(t))
        app.setStyleSheet(theme.stylesheet(t))
        self.results_view.redraw()

    def closeEvent(self, event) -> None:
        if self.train_view.is_training():
            answer = QMessageBox.question(
                self, "Training in progress",
                "Models are still being trained. Quit anyway? The current experiment will be lost.")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()
