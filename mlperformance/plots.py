"""Plotly figures for the results page.

Colors follow a validated colorblind-safe palette: one hue (blue) for
single-series charts, with the best model emphasised; at most three
categorical hues when curves of several models share one plot.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .trainer import METRICS, Experiment

PALETTE = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "accent": "#2a78d6",
        "muted": "#86b6ef",
        "reference": "#8a8984",
        "sequential": ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"],
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70"],
        "accent": "#3987e5",
        "muted": "#184f95",
        "reference": "#8a8984",
        "sequential": ["#1a1a19", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"],
    },
}
MAX_CURVES = 3  # more than three overlapping hues are no longer distinguishable


def _colors(dark: bool) -> dict:
    return PALETTE["dark" if dark else "light"]


def _layout(fig: go.Figure, title: str | None = None, height: int | None = None, **kwargs) -> go.Figure:
    layout = dict(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barcornerradius=4,
        bargap=0.35,
        hoverlabel=dict(namelength=-1),
    )
    fig.update_layout(**{**layout, **kwargs})
    return fig


def metric_label(task: str, metric: str) -> str:
    return METRICS[task][metric][0]


def metric_comparison(exp: Experiment, metric: str, dark: bool = False) -> go.Figure:
    """Horizontal bars, best model first and highlighted."""
    c = _colors(dark)
    label, higher = METRICS[exp.task][metric]
    rows = [r for r in exp.successful if metric in r["metrics"]]
    rows.sort(key=lambda r: r["metrics"][metric], reverse=higher)
    names = [r["name"] for r in rows]
    values = [r["metrics"][metric] for r in rows]
    colors = [c["accent"] if i == 0 else c["muted"] for i in range(len(rows))]
    fig = go.Figure(
        go.Bar(
            x=values, y=names, orientation="h", marker_color=colors,
            text=[f"{v:.4f}" for v in values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>" + label + ": %{x:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=f"{label} ({'higher' if higher else 'lower'} is better)")
    return _layout(fig, height=max(250, 36 * len(rows) + 80))


def cv_distribution(exp: Experiment, dark: bool = False) -> go.Figure:
    """Spread of the cross-validation scores: a narrow box means a stable model."""
    c = _colors(dark)
    label, higher = METRICS[exp.task][exp.primary_metric]
    rows = [r for r in exp.successful if r.get("cv_scores")]
    rows.sort(key=lambda r: np.mean(r["cv_scores"]), reverse=higher)
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Box(
            x=r["cv_scores"], name=r["name"], marker_color=c["accent"], line_width=1.5,
            boxpoints="all", jitter=0.3, pointpos=0, marker_size=8, showlegend=False,
            hovertemplate=r["name"] + "<br>" + label + ": %{x:.4f}<extra></extra>",
        ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=f"{label} per cross-validation fold")
    return _layout(fig, height=max(250, 36 * len(rows) + 80))


def training_time(exp: Experiment, dark: bool = False) -> go.Figure:
    c = _colors(dark)
    rows = sorted(exp.successful, key=lambda r: r["train_time"])
    fig = go.Figure(go.Bar(
        x=[r["train_time"] for r in rows], y=[r["name"] for r in rows], orientation="h",
        marker_color=c["accent"], text=[f"{r['train_time']:.2f}s" for r in rows],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Training time: %{x:.3f}s<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Training time on the train split (seconds, lower is faster)")
    return _layout(fig, height=max(250, 36 * len(rows) + 80))


def confusion_matrix(result: dict, classes: list[str], normalize: bool = True,
                     dark: bool = False) -> go.Figure:
    c = _colors(dark)
    cm = np.asarray(result["confusion_matrix"], dtype=float)
    shown = cm
    if normalize:
        totals = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)
    text = [[f"{shown[i, j]:.0%}" if normalize else f"{int(cm[i, j])}" for j in range(len(classes))]
            for i in range(len(classes))]
    steps = c["sequential"]
    fig = go.Figure(go.Heatmap(
        z=shown, x=classes, y=classes,
        colorscale=[[i / (len(steps) - 1), s] for i, s in enumerate(steps)],
        customdata=cm, xgap=2, ygap=2, showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{customdata:.0f}<extra></extra>",
    ))
    if len(classes) <= 20:
        # Strong cells are dark in light mode and light in dark mode; flip the ink to match.
        top = shown.max() or 1
        strong_ink, weak_ink = ("#0b0b0b", "#f0efec") if dark else ("#ffffff", "#0b0b0b")
        for i in range(len(classes)):
            for j in range(len(classes)):
                fig.add_annotation(x=classes[j], y=classes[i], text=text[i][j], showarrow=False,
                                   font_color=strong_ink if shown[i, j] / top > 0.5 else weak_ink)
    fig.update_xaxes(title="Predicted class", type="category", side="bottom")
    fig.update_yaxes(title="Actual class", type="category", autorange="reversed")
    size = min(700, max(320, 60 * len(classes) + 120))
    return _layout(fig, height=size)


def roc_curves(exp: Experiment, algorithms: list[str], class_index: int,
               dark: bool = False) -> go.Figure:
    c = _colors(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
        line=dict(color=c["reference"], dash="dash", width=1.5), hoverinfo="skip",
    ))
    by_name = {r["name"]: r for r in exp.successful}
    for color, name in zip(c["series"], algorithms[:MAX_CURVES]):
        curve = next((cv for cv in by_name[name].get("roc", []) if cv["class"] == class_index), None)
        if curve is None:
            continue
        fig.add_trace(go.Scatter(
            x=curve["fpr"], y=curve["tpr"], mode="lines", line=dict(color=color, width=2),
            name=f"{name} (AUC {curve['auc']:.3f})",
            hovertemplate=name + "<br>FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
        ))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return _layout(fig, height=450, legend=dict(orientation="h", y=-0.2))


def predicted_vs_actual(result: dict, dark: bool = False) -> go.Figure:
    c = _colors(dark)
    y_true, y_pred = np.asarray(result["y_true"]), np.asarray(result["y_pred"])
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
        line=dict(color=c["reference"], dash="dash", width=1.5), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers", name="Test samples",
        marker=dict(color=c["accent"], size=8, opacity=0.6,
                    line=dict(width=1, color="rgba(255,255,255,0.6)")),
        hovertemplate="Actual %{x:.4g}<br>Predicted %{y:.4g}<extra></extra>",
    ))
    fig.update_xaxes(title="Actual value")
    fig.update_yaxes(title="Predicted value")
    return _layout(fig, height=450, legend=dict(orientation="h", y=-0.2))


def residuals(result: dict, dark: bool = False) -> go.Figure:
    c = _colors(dark)
    err = np.asarray(result["y_pred"]) - np.asarray(result["y_true"])
    fig = go.Figure(go.Histogram(
        x=err, marker_color=c["accent"], marker_line=dict(width=1, color="rgba(0,0,0,0)"),
        hovertemplate="Error %{x}<br>Samples: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=c["reference"])
    fig.update_xaxes(title="Prediction error (predicted − actual)")
    fig.update_yaxes(title="Number of samples")
    return _layout(fig, height=400, bargap=0.05)


def error_comparison(exp: Experiment, dark: bool = False) -> go.Figure:
    """Absolute test-set errors of every regression model, most accurate first."""
    c = _colors(dark)
    rows = [(r["name"], np.abs(np.asarray(r["y_pred"]) - np.asarray(r["y_true"])))
            for r in exp.successful]
    rows.sort(key=lambda item: np.median(item[1]))
    fig = go.Figure()
    for name, err in rows:
        fig.add_trace(go.Box(
            x=err, name=name, marker_color=c["accent"], line_width=1.5, boxpoints="outliers",
            marker_size=6, showlegend=False,
            hovertemplate=name + "<br>Absolute error: %{x:.4g}<extra></extra>",
        ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Absolute error per test sample (lower is better)")
    return _layout(fig, height=max(250, 36 * len(rows) + 80))


def feature_importance(result: dict, top: int = 15, dark: bool = False) -> go.Figure | None:
    imp = result.get("feature_importance") or {}
    if not imp:
        return None
    c = _colors(dark)
    items = list(imp.items())[:top]
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        marker_color=c["accent"], text=[f"{v:.1%}" for _, v in items],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Relative importance: %{x:.2%}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Relative importance", tickformat=".0%")
    return _layout(fig, height=max(250, 30 * len(items) + 80))
