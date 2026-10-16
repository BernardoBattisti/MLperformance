"""Reusable building blocks for the desktop interface."""

from __future__ import annotations

import warnings
from typing import Callable

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableView,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .resources import resource_path  # noqa: F401 (re-exported for the views)



def label(text: str = "", role: str | None = None, wrap: bool = False) -> QLabel:
    w = QLabel(text)
    if role:
        w.setProperty("role", role)
    w.setWordWrap(wrap)
    w.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return w


def button(text: str, primary: bool = False, danger: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    if primary:
        b.setProperty("primary", True)
    if danger:
        b.setProperty("danger", True)
    return b


class Card(QFrame):
    """White rounded panel with an optional numbered step and title."""

    def __init__(self, title: str | None = None, step: int | None = None, subtitle: str | None = None):
        super().__init__()
        self.setProperty("card", True)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(22, 18, 22, 20)
        self.body.setSpacing(12)
        if title:
            header = QHBoxLayout()
            header.setSpacing(10)
            if step is not None:
                header.addWidget(label(str(step), "step"))
            header.addWidget(label(title, "section"))
            header.addStretch()
            self.body.addLayout(header)
        if subtitle:
            self.body.addWidget(label(subtitle, "hint", wrap=True))


class StatTile(QFrame):
    def __init__(self, caption: str, value: str = "—"):
        super().__init__()
        self.setProperty("tile", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)
        self.caption = label(caption, "tileLabel")
        self.value = label(value, "tileValue")
        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set_caption(self, caption: str) -> None:
        self.caption.setText(caption)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


class Banner(QFrame):
    """Colored message strip (``kind`` is "success" or "warning")."""

    def __init__(self, kind: str = "success"):
        super().__init__()
        self.setProperty("banner", kind)
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(16, 12, 16, 12)
        self.text = label("", wrap=True)
        self.layout_.addWidget(self.text, 1)
        self.hide()

    def show_message(self, text: str) -> None:
        self.text.setText(text)
        self.show()


def scroll_page(content: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    content.setObjectName("Page")
    area.setWidget(content)
    return area


class ChartView(FigureCanvasQTAgg):
    """A matplotlib figure inside the window, with hover tooltips.

    ``draw_fn(figure) -> hover targets`` is called on every (re)draw, so the
    chart follows theme changes.
    """

    def __init__(self, height: int = 320):
        self.figure = Figure(dpi=100)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(height)
        self._targets: list = []
        self._draw_fn: Callable[[Figure], list] | None = None
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("figure_leave_event", lambda _: QToolTip.hideText())

    def plot(self, draw_fn: Callable[[Figure], list], height: int | None = None) -> None:
        if height:
            self.setFixedHeight(height)
        self._draw_fn = draw_fn
        self.redraw()

    def redraw(self) -> None:
        if self._draw_fn is None:
            return
        self._targets = self._draw_fn(self.figure) or []
        self._fit()
        self.draw_idle()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._draw_fn is not None:
            self._fit()

    def _fit(self) -> None:
        # Hidden tabs report a tiny size, which makes tight_layout warn; it is
        # re-applied with the real size on the next resize.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.figure.tight_layout()

    def _on_move(self, event) -> None:
        for artist, text in self._targets:
            hit, info = artist.contains(event)
            if not hit:
                continue
            if isinstance(text, list):
                idx = info.get("ind", [])
                if len(idx) == 0:
                    continue
                text = text[int(idx[0])]
            QToolTip.showText(QCursor.pos(), text, self)
            return
        QToolTip.hideText()


class DataFrameModel(QAbstractTableModel):
    """Read-only Qt model over a DataFrame.

    ``float_format`` formats float cells: 4 fixed decimals for scores, or
    ``"{:g}"`` to show raw data as it appears in the file.
    """

    def __init__(self, float_format: str = "{:.4f}"):
        super().__init__()
        self._df = pd.DataFrame()
        self._float_format = float_format

    def set_frame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return self._float_format.format(value)
            return str(value)
        if role == Qt.TextAlignmentRole and isinstance(value, (int, float)):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)


def fit_table_height(view: QTableView, max_rows: int = 12) -> None:
    """Size the table to show all rows (up to ``max_rows``) without inner scrolling."""
    rows = min(view.model().rowCount(), max_rows)
    header = view.horizontalHeader().sizeHint().height()
    scrollbar = view.horizontalScrollBar().sizeHint().height()
    view.setFixedHeight(header + rows * view.verticalHeader().defaultSectionSize() + scrollbar + 6)


def table_view(model: DataFrameModel, height: int = 240) -> QTableView:
    view = QTableView()
    view.setModel(model)
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QTableView.SelectRows)
    view.setEditTriggers(QTableView.NoEditTriggers)
    view.verticalHeader().setVisible(False)
    view.horizontalHeader().setStretchLastSection(True)
    view.setShowGrid(False)
    view.setFixedHeight(height)
    view.setHorizontalScrollMode(QTableView.ScrollPerPixel)
    view.setVerticalScrollMode(QTableView.ScrollPerPixel)
    return view
