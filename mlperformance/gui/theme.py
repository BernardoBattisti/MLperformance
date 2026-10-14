"""Color tokens and the Qt style sheet for the light and dark themes.

Chart colors come from a validated colorblind-safe palette; the light and
dark variants are separate steps of the same hues, not an automatic invert.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    surface: str
    surface_2: str
    border: str
    text: str
    text_2: str
    muted: str
    accent: str
    accent_hover: str
    accent_soft: str
    on_accent: str
    success_bg: str
    success_text: str
    warning_bg: str
    warning_text: str
    danger: str
    # charts
    series: tuple[str, str, str]
    bar_muted: str
    reference: str
    grid: str
    sequential: tuple[str, ...]


LIGHT = Theme(
    name="light",
    bg="#f4f5f7", surface="#ffffff", surface_2="#f0f2f5", border="#e2e5ea",
    text="#14161a", text_2="#545b67", muted="#8a909c",
    accent="#2a78d6", accent_hover="#256abf", accent_soft="#e7f0fc", on_accent="#ffffff",
    success_bg="#e6f4ec", success_text="#17663a", warning_bg="#fdf3e1", warning_text="#7a4f00",
    danger="#c93b3b",
    series=("#2a78d6", "#eb6834", "#1baf7a"), bar_muted="#9ec5f4", reference="#8a8984",
    grid="#e8eaee",
    sequential=("#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"),
)

DARK = Theme(
    name="dark",
    bg="#111318", surface="#1a1d23", surface_2="#22262e", border="#2d323b",
    text="#eceef2", text_2="#a9afba", muted="#7b818d",
    accent="#3987e5", accent_hover="#5598e7", accent_soft="#1b2a3f", on_accent="#ffffff",
    success_bg="#15301f", success_text="#8fd8a8", warning_bg="#33280f", warning_text="#f0c46a",
    danger="#e66767",
    series=("#3987e5", "#d95926", "#199e70"), bar_muted="#1c5cab", reference="#8a8984",
    grid="#2a2e36",
    sequential=("#1a1d23", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"),
)


def system_theme() -> Theme:
    scheme = QGuiApplication.styleHints().colorScheme()
    return DARK if scheme == Qt.ColorScheme.Dark else LIGHT


def palette(t: Theme) -> QPalette:
    p = QPalette()
    roles = {
        QPalette.Window: t.bg, QPalette.WindowText: t.text, QPalette.Base: t.surface,
        QPalette.AlternateBase: t.surface_2, QPalette.Text: t.text, QPalette.Button: t.surface,
        QPalette.ButtonText: t.text, QPalette.Highlight: t.accent,
        QPalette.HighlightedText: t.on_accent, QPalette.ToolTipBase: t.surface,
        QPalette.ToolTipText: t.text, QPalette.PlaceholderText: t.muted,
    }
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(t.muted))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(t.muted))
    return p


_ICONS = {
    "chevron": '<path d="M4 6l4 4 4-4" fill="none" stroke="{c}" stroke-width="1.8" '
               'stroke-linecap="round" stroke-linejoin="round"/>',
    "up": '<path d="M5 10l3-3 3 3" fill="none" stroke="{c}" stroke-width="1.6" '
          'stroke-linecap="round" stroke-linejoin="round"/>',
    "down": '<path d="M5 6l3 3 3-3" fill="none" stroke="{c}" stroke-width="1.6" '
            'stroke-linecap="round" stroke-linejoin="round"/>',
    "check": '<path d="M3.5 8.5l3 3 6-6.5" fill="none" stroke="{c}" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round"/>',
}


def _icon(name: str, color: str) -> str:
    """Write a small SVG icon in ``color`` to a cache folder; return a QSS-ready path."""
    folder = Path(tempfile.gettempdir()) / "mlperformance-ui"
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}-{color.lstrip('#')}.svg"
    if not path.exists():
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
                        f'viewBox="0 0 16 16">{_ICONS[name].format(c=color)}</svg>')
    return path.as_posix()


def stylesheet(t: Theme) -> str:
    chevron, up, down = _icon("chevron", t.text_2), _icon("up", t.text_2), _icon("down", t.text_2)
    check = _icon("check", t.on_accent)
    return f"""
    * {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; }}
    QMainWindow, QWidget#Page, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {t.bg}; }}
    QLabel {{ color: {t.text}; background: transparent; }}
    QLabel[role="title"] {{ font-size: 20pt; font-weight: 600; }}
    QLabel[role="subtitle"] {{ color: {t.text_2}; font-size: 10.5pt; }}
    QLabel[role="section"] {{ font-size: 12pt; font-weight: 600; }}
    QLabel[role="hint"] {{ color: {t.text_2}; }}
    QLabel[role="muted"] {{ color: {t.muted}; font-size: 9pt; }}
    QLabel[role="tileLabel"] {{ color: {t.text_2}; font-size: 9pt; }}
    QLabel[role="tileValue"] {{ font-size: 17pt; font-weight: 600; }}
    QLabel[role="step"] {{
        background: {t.accent_soft}; color: {t.accent}; border-radius: 12px;
        min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px;
        font-weight: 700; qproperty-alignment: AlignCenter;
    }}

    QFrame#Sidebar {{ background: {t.surface}; border-right: 1px solid {t.border}; }}
    QLabel#Brand {{ font-size: 14pt; font-weight: 700; }}
    QPushButton[nav="true"] {{
        text-align: left; padding: 10px 14px; border: none; border-radius: 8px;
        color: {t.text_2}; background: transparent; font-size: 10.5pt;
    }}
    QPushButton[nav="true"]:hover {{ background: {t.surface_2}; color: {t.text}; }}
    QPushButton[nav="true"]:checked {{ background: {t.accent_soft}; color: {t.accent}; font-weight: 600; }}

    QFrame[card="true"] {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 12px; }}
    QFrame[tile="true"] {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 10px; }}
    QFrame[banner="success"] {{ background: {t.success_bg}; border-radius: 10px; }}
    QFrame[banner="success"] QLabel {{ color: {t.success_text}; }}
    QFrame[banner="warning"] {{ background: {t.warning_bg}; border-radius: 10px; }}
    QFrame[banner="warning"] QLabel {{ color: {t.warning_text}; }}

    QPushButton {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
        border-radius: 8px; padding: 7px 14px;
    }}
    QPushButton:hover {{ background: {t.surface_2}; }}
    QPushButton:disabled {{ color: {t.muted}; }}
    QPushButton[primary="true"] {{
        background: {t.accent}; color: {t.on_accent}; border: none; font-weight: 600; padding: 9px 20px;
    }}
    QPushButton[primary="true"]:hover {{ background: {t.accent_hover}; }}
    QPushButton[primary="true"]:disabled {{ background: {t.border}; color: {t.muted}; }}
    QPushButton[danger="true"] {{ color: {t.danger}; }}

    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
        border-radius: 8px; padding: 6px 10px; min-height: 20px;
        selection-background-color: {t.accent};
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border: 1px solid {t.accent}; }}
    QComboBox {{ padding-right: 28px; }}
    QComboBox::drop-down {{ border: none; width: 28px; subcontrol-position: center right; }}
    QComboBox::down-arrow {{ image: url("{chevron}"); width: 14px; height: 14px; }}
    QSpinBox {{ padding-right: 26px; }}
    QSpinBox::up-button, QSpinBox::down-button {{
        subcontrol-origin: border; width: 24px; border: none; background: transparent;
    }}
    QSpinBox::up-button {{ subcontrol-position: top right; margin-top: 3px; }}
    QSpinBox::down-button {{ subcontrol-position: bottom right; margin-bottom: 3px; }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {t.surface_2}; border-radius: 4px; }}
    QSpinBox::up-arrow {{ image: url("{up}"); width: 12px; height: 12px; }}
    QSpinBox::down-arrow {{ image: url("{down}"); width: 12px; height: 12px; }}
    QComboBox QAbstractItemView {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
        selection-background-color: {t.accent_soft}; selection-color: {t.text}; outline: none;
    }}

    QRadioButton, QCheckBox {{ color: {t.text}; spacing: 8px; background: transparent; }}
    QRadioButton::indicator {{
        width: 16px; height: 16px; border-radius: 9px; border: 1.5px solid {t.muted};
        background: {t.surface};
    }}
    QRadioButton::indicator:hover {{ border-color: {t.accent}; }}
    QRadioButton::indicator:checked {{ border: 5px solid {t.accent}; width: 8px; height: 8px; }}
    QCheckBox::indicator, QListWidget::indicator {{
        width: 16px; height: 16px; border-radius: 4px; border: 1.5px solid {t.muted};
        background: {t.surface};
    }}
    QCheckBox::indicator:hover, QListWidget::indicator:hover {{ border-color: {t.accent}; }}
    QCheckBox::indicator:checked, QListWidget::indicator:checked {{
        background: {t.accent}; border-color: {t.accent}; image: url("{check}");
    }}
    QListWidget::indicator:checked:disabled, QListWidget::indicator:disabled {{ border-color: {t.border}; }}

    QListWidget {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
        border-radius: 8px; padding: 4px; outline: none;
    }}
    QListWidget::item {{ padding: 5px 6px; border-radius: 6px; }}
    QListWidget::item:hover {{ background: {t.surface_2}; }}
    QListWidget::item:selected {{ background: {t.accent_soft}; color: {t.text}; }}

    QTableView {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; border-radius: 8px;
        gridline-color: {t.border}; selection-background-color: {t.accent_soft};
        selection-color: {t.text}; alternate-background-color: {t.surface_2};
    }}
    QHeaderView::section {{
        background: {t.surface_2}; color: {t.text_2}; border: none;
        border-bottom: 1px solid {t.border}; padding: 6px 8px; font-weight: 600;
    }}
    QTableCornerButton::section {{ background: {t.surface_2}; border: none; }}

    QTabWidget::pane {{ border: none; }}
    QTabBar::tab {{
        background: transparent; color: {t.text_2}; padding: 8px 16px; margin-right: 4px;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:hover {{ color: {t.text}; }}
    QTabBar::tab:selected {{ color: {t.accent}; border-bottom: 2px solid {t.accent}; font-weight: 600; }}

    QProgressBar {{
        background: {t.surface_2}; border: none; border-radius: 4px; height: 8px;
        text-align: center; color: transparent;
    }}
    QProgressBar::chunk {{ background: {t.accent}; border-radius: 4px; }}

    QScrollArea {{ border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {t.muted}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {t.border}; border-radius: 4px; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QToolTip {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
        border-radius: 6px; padding: 6px 8px;
    }}
    QMessageBox {{ background: {t.surface}; }}
    """


_current: Theme = LIGHT


def current() -> Theme:
    """The active theme; charts read it at draw time so they follow theme changes."""
    return _current


def set_current(t: Theme) -> None:
    global _current
    _current = t
