"""Matplotlib charts for the results screen.

Every function draws onto an empty ``Figure`` and returns hover targets:
a list of ``(artist, text)`` for single shapes (bars) or
``(collection, [text per point])`` for scatter points. ``ChartView`` turns
them into tooltips.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from ..trainer import METRICS, Experiment
from .theme import Theme

MAX_CURVES = 3  # more overlapping hues stop being distinguishable
BAR_ROW_PX = 30


def bar_chart_height(n_rows: int) -> int:
    return max(220, BAR_ROW_PX * n_rows + 90)


def _style(ax: Axes, t: Theme, grid_axis: str = "x") -> None:
    ax.set_facecolor(t.surface)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t.border)
    ax.tick_params(colors=t.text_2, labelsize=9, length=0, pad=6)
    ax.xaxis.label.set_color(t.text_2)
    ax.yaxis.label.set_color(t.text_2)
    ax.grid(axis=grid_axis, color=t.grid, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_ax(fig: Figure, t: Theme) -> Axes:
    fig.clear()
    fig.set_facecolor(t.surface)
    ax = fig.add_subplot(111)
    _style(ax, t)
    return ax


def _hbars(ax: Axes, t: Theme, names, values, colors, labels, hover_texts) -> list:
    y = np.arange(len(names))
    bars = ax.barh(y, values, height=0.62, color=colors, linewidth=0)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    span = max(values) if len(values) and max(values) > 0 else 1
    for bar, label in zip(bars, labels):
        ax.text(bar.get_width() + span * 0.01, bar.get_y() + bar.get_height() / 2, label,
                va="center", ha="left", fontsize=8.5, color=t.text_2)
    ax.set_xlim(min(0, min(values, default=0)), span * 1.12)
    return list(zip(bars, hover_texts))


def metric_comparison(fig: Figure, exp: Experiment, metric: str, t: Theme) -> list:
    """Horizontal bars, best model on top and highlighted."""
    label, higher = METRICS[exp.task][metric]
    rows = [r for r in exp.successful if metric in r["metrics"]]
    rows.sort(key=lambda r: r["metrics"][metric], reverse=higher)
    values = [r["metrics"][metric] for r in rows]
    ax = _new_ax(fig, t)
    targets = _hbars(
        ax, t, [r["name"] for r in rows], values,
        [t.accent if i == 0 else t.bar_muted for i in range(len(rows))],
        [f"{v:.4f}" for v in values],
        [f"{r['name']}\n{label}: {v:.4f}" for r, v in zip(rows, values)],
    )
    ax.set_xlabel(f"{label} on the test set ({'higher' if higher else 'lower'} is better)")
    return targets


def cv_distribution(fig: Figure, exp: Experiment, t: Theme) -> list:
    """Score per fold: a narrow spread means a stable model."""
    label, higher = METRICS[exp.task][exp.primary_metric]
    rows = [r for r in exp.successful if r.get("cv_scores")]
    rows.sort(key=lambda r: np.mean(r["cv_scores"]), reverse=higher)
    ax = _new_ax(fig, t)
    data = [r["cv_scores"] for r in rows]
    positions = np.arange(len(rows))
    ax.boxplot(
        data, positions=positions, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=t.accent_soft, edgecolor=t.accent, linewidth=1.2),
        whiskerprops=dict(color=t.accent, linewidth=1.2), capprops=dict(color=t.accent, linewidth=1.2),
        medianprops=dict(color=t.text, linewidth=1.5),
    )
    targets = []
    for pos, r in zip(positions, rows):
        scores = r["cv_scores"]
        pts = ax.scatter(scores, np.full(len(scores), pos), s=30, color=t.accent,
                         edgecolors=t.surface, linewidths=1, zorder=3)
        targets.append((pts, [f"{r['name']}\nFold {i + 1}: {s:.4f}" for i, s in enumerate(scores)]))
    ax.set_yticks(positions, [r["name"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"{label} per cross-validation fold")
    return targets


def training_time(fig: Figure, exp: Experiment, t: Theme) -> list:
    rows = sorted(exp.successful, key=lambda r: r["train_time"])
    values = [r["train_time"] for r in rows]
    ax = _new_ax(fig, t)
    targets = _hbars(
        ax, t, [r["name"] for r in rows], values, [t.accent] * len(rows),
        [format_seconds(v) for v in values],
        [f"{r['name']}\nTraining time: {v:.3f}s" for r, v in zip(rows, values)],
    )
    ax.set_xlabel("Training time in seconds (lower is faster)")
    return targets


def format_seconds(s: float) -> str:
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s"


def confusion_matrix(fig: Figure, result: dict, classes: list[str], normalize: bool,
                     t: Theme) -> list:
    cm = np.asarray(result["confusion_matrix"], dtype=float)
    shown = cm
    if normalize:
        totals = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)
    fig.clear()
    fig.set_facecolor(t.surface)
    ax = fig.add_subplot(111)
    ax.set_facecolor(t.surface)
    cmap = LinearSegmentedColormap.from_list("seq", t.sequential)
    n = len(classes)
    mesh = ax.pcolormesh(shown, cmap=cmap, edgecolors=t.surface, linewidth=2,
                         vmin=0, vmax=shown.max() or 1)
    ax.set_xticks(np.arange(n) + 0.5, classes, rotation=30 if n > 4 else 0,
                  ha="right" if n > 4 else "center")
    ax.set_yticks(np.arange(n) + 0.5, classes)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted class", color=t.text_2)
    ax.set_ylabel("Actual class", color=t.text_2)
    ax.tick_params(colors=t.text_2, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_aspect("equal")
    top = shown.max() or 1
    # Strong cells are dark in light mode and light in dark mode: flip the ink to match.
    strong, weak = (t.surface, t.text) if t.name == "light" else ("#0b0b0b", t.text)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                text = f"{shown[i, j]:.0%}" if normalize else f"{int(cm[i, j])}"
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=9,
                        color=strong if shown[i, j] / top > 0.5 else weak)
    texts = [f"Actual: {classes[i]}\nPredicted: {classes[j]}\nCount: {int(cm[i, j])}"
             for i in range(n) for j in range(n)]
    return [(mesh, texts)]


def roc_curves(fig: Figure, exp: Experiment, algorithms: list[str], class_index: int,
               t: Theme) -> list:
    ax = _new_ax(fig, t)
    ax.grid(axis="both", color=t.grid, linewidth=0.8)
    ax.plot([0, 1], [0, 1], linestyle="--", color=t.reference, linewidth=1.2, label="Random guess")
    by_name = {r["name"]: r for r in exp.successful}
    for color, name in zip(t.series, algorithms[:MAX_CURVES]):
        curve = next((c for c in by_name[name].get("roc", []) if c["class"] == class_index), None)
        if curve:
            ax.plot(curve["fpr"], curve["tpr"], color=color, linewidth=2,
                    label=f"{name} (AUC {curve['auc']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    legend = ax.legend(loc="lower right", frameon=True, fontsize=9)
    legend.get_frame().set_facecolor(t.surface)
    legend.get_frame().set_edgecolor(t.border)
    for text in legend.get_texts():
        text.set_color(t.text)
    return []


def predicted_vs_actual(fig: Figure, result: dict, t: Theme) -> list:
    y_true, y_pred = np.asarray(result["y_true"]), np.asarray(result["y_pred"])
    ax = _new_ax(fig, t)
    ax.grid(axis="both", color=t.grid, linewidth=0.8)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=t.reference, linewidth=1.2)
    pts = ax.scatter(y_true, y_pred, s=36, color=t.accent, alpha=0.65,
                     edgecolors=t.surface, linewidths=0.8)
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    return [(pts, [f"Actual: {a:.4g}\nPredicted: {p:.4g}" for a, p in zip(y_true, y_pred)])]


def residuals(fig: Figure, result: dict, t: Theme) -> list:
    err = np.asarray(result["y_pred"]) - np.asarray(result["y_true"])
    ax = _new_ax(fig, t)
    ax.grid(axis="y", color=t.grid, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    counts, edges, patches = ax.hist(err, bins="auto", color=t.accent, edgecolor=t.surface,
                                     linewidth=1.5)
    ax.axvline(0, linestyle="--", color=t.reference, linewidth=1.2)
    ax.set_xlabel("Prediction error (predicted − actual)")
    ax.set_ylabel("Samples")
    return [(p, f"Error {lo:.4g} to {hi:.4g}\n{int(c)} samples")
            for p, c, lo, hi in zip(patches, counts, edges[:-1], edges[1:])]


def error_comparison(fig: Figure, exp: Experiment, t: Theme) -> list:
    """Absolute test errors per regression model, most accurate on top."""
    rows = [(r["name"], np.abs(np.asarray(r["y_pred"]) - np.asarray(r["y_true"])))
            for r in exp.successful]
    rows.sort(key=lambda item: np.median(item[1]))
    ax = _new_ax(fig, t)
    positions = np.arange(len(rows))
    ax.boxplot(
        [e for _, e in rows], positions=positions, orientation="horizontal", widths=0.5, patch_artist=True,
        boxprops=dict(facecolor=t.accent_soft, edgecolor=t.accent, linewidth=1.2),
        whiskerprops=dict(color=t.accent, linewidth=1.2), capprops=dict(color=t.accent, linewidth=1.2),
        medianprops=dict(color=t.text, linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=t.bar_muted, markeredgecolor="none"),
    )
    ax.set_yticks(positions, [name for name, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute error per test sample (lower is better)")
    return []


def feature_importance(fig: Figure, result: dict, t: Theme, top: int = 15) -> list:
    items = list((result.get("feature_importance") or {}).items())[:top]
    ax = _new_ax(fig, t)
    values = [v for _, v in items]
    targets = _hbars(
        ax, t, [k for k, _ in items], values, [t.accent] * len(items),
        [f"{v:.1%}" for v in values], [f"{k}\nRelative importance: {v:.2%}" for k, v in items],
    )
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Relative importance")
    return targets
