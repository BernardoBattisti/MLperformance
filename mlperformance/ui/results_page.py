"""'Results' page: reopen any saved experiment and compare its algorithms."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from .. import plots
from ..preprocessing import CLASSIFICATION
from ..storage import delete_experiment, list_experiments, model_path
from ..trainer import METRICS, Experiment


def _is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except AttributeError:
        return False


def _leaderboard(exp: Experiment) -> pd.DataFrame:
    metrics = METRICS[exp.task]
    higher = metrics[exp.primary_metric][1]
    rows = []
    for r in sorted(exp.successful, key=lambda r: r["metrics"][exp.primary_metric], reverse=higher):
        row = {"Algorithm": r["name"]}
        row.update({label: r["metrics"].get(key) for key, (label, _) in metrics.items()})
        if r.get("cv_scores"):
            row["CV mean"] = float(np.mean(r["cv_scores"]))
            row["CV std"] = float(np.std(r["cv_scores"]))
        row["Train time (s)"] = r["train_time"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.dropna(axis=1, how="all")


def render() -> None:
    st.title("Results")
    experiments = list_experiments()
    if not experiments:
        st.info("No experiments yet. Train your first models on the **New experiment** page.")
        return

    ids = [e.id for e in experiments]
    by_id = {e.id: e for e in experiments}
    default = st.session_state.get("selected_experiment")
    exp_id = st.selectbox(
        "Experiment", ids, index=ids.index(default) if default in ids else 0,
        format_func=lambda i: f"{by_id[i].name}  ·  {by_id[i].created_at.replace('T', ' ')}",
    )
    st.session_state["selected_experiment"] = exp_id
    exp = by_id[exp_id]
    dark = _is_dark()

    s = exp.settings
    st.caption(
        f"Dataset **{exp.dataset}** · target **{exp.target}** · {exp.task} · "
        f"{exp.n_rows:,} rows ({s['n_train']:,} train / {s['n_test']:,} test) · "
        f"{s['cv_folds']}-fold CV · seed {s['random_state']}"
    )

    best = exp.best
    if best is None:
        st.error("Every algorithm failed in this experiment.")
        _show_errors(exp)
        return

    primary_label = METRICS[exp.task][exp.primary_metric][0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best algorithm", best["name"])
    c2.metric(f"{primary_label} (test set)", f"{best['metrics'][exp.primary_metric]:.4f}")
    c3.metric("Algorithms compared", len(exp.successful))

    tab_overview, tab_algo, tab_curves, tab_details = st.tabs(
        ["Leaderboard", "Algorithm details",
         "ROC curves" if exp.task == CLASSIFICATION else "Error comparison", "Experiment info"]
    )
    with tab_overview:
        _overview(exp, dark)
    with tab_algo:
        _algorithm_details(exp, dark)
    with tab_curves:
        if exp.task == CLASSIFICATION:
            _roc(exp, dark)
        else:
            st.markdown("**How large are the errors of each algorithm?** Each box shows the "
                        "absolute errors on the test set; the line inside is the median.")
            st.plotly_chart(plots.error_comparison(exp, dark))
    with tab_details:
        _details(exp)


def _overview(exp: Experiment, dark: bool) -> None:
    board = _leaderboard(exp)
    st.dataframe(
        board, hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.4f")
                       for c in board.columns if c not in ("Rank", "Algorithm")},
    )
    st.download_button("Download leaderboard (CSV)", board.to_csv(index=False),
                       file_name=f"{exp.id}-leaderboard.csv", mime="text/csv")
    _show_errors(exp)

    metrics = METRICS[exp.task]
    metric = st.selectbox(
        "Compare by metric", list(metrics),
        format_func=lambda m: metrics[m][0],
        index=list(metrics).index(exp.primary_metric),
    )
    st.plotly_chart(plots.metric_comparison(exp, metric, dark))

    if any(r.get("cv_scores") for r in exp.successful):
        st.markdown("**Cross-validation stability** — each dot is one fold. "
                    "A narrow spread means the model performs consistently.")
        st.plotly_chart(plots.cv_distribution(exp, dark))

    st.markdown("**Training time**")
    st.plotly_chart(plots.training_time(exp, dark))


def _algorithm_details(exp: Experiment, dark: bool) -> None:
    names = [r["name"] for r in exp.successful]
    name = st.selectbox("Algorithm", names, index=names.index(exp.best["name"]))
    result = next(r for r in exp.successful if r["name"] == name)

    cols = st.columns(len(result["metrics"]))
    for col, (key, value) in zip(cols, result["metrics"].items()):
        col.metric(METRICS[exp.task][key][0], f"{value:.4f}")

    if exp.task == CLASSIFICATION:
        normalize = st.toggle("Show percentages per actual class", value=True)
        st.markdown("**Confusion matrix** (test set)")
        st.plotly_chart(plots.confusion_matrix(result, exp.classes, normalize, dark))
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Predicted vs. actual** (test set)")
            st.plotly_chart(plots.predicted_vs_actual(result, dark))
        with right:
            st.markdown("**Error distribution**")
            st.plotly_chart(plots.residuals(result, dark))

    fig = plots.feature_importance(result, dark=dark)
    if fig is not None:
        st.markdown("**Most influential features**")
        st.plotly_chart(fig)
    else:
        st.caption("This algorithm does not expose feature importances.")

    path = model_path(exp.id, name)
    if path.exists():
        st.download_button(
            f"Download trained {name} model (.joblib)", path.read_bytes(),
            file_name=path.name, mime="application/octet-stream",
            help="Load it in Python with mlperformance.storage.load_model(path).predict(df)",
        )


def _roc(exp: Experiment, dark: bool) -> None:
    names = [r["name"] for r in exp.successful if r.get("roc")]
    if not names:
        st.info("None of the trained algorithms produce probabilities, so no ROC curves are available.")
        return
    higher_first = sorted(names, key=lambda n: -next(
        r for r in exp.successful if r["name"] == n)["metrics"].get("roc_auc", 0))
    chosen = st.multiselect(
        f"Algorithms (up to {plots.MAX_CURVES})", names, default=higher_first[:plots.MAX_CURVES],
        max_selections=plots.MAX_CURVES,
    )
    if len(exp.classes) == 2:
        class_index = 1
        st.caption(f"Positive class: **{exp.classes[1]}**")
    else:
        class_index = st.selectbox("Class (one-vs-rest)", range(len(exp.classes)),
                                   format_func=lambda i: exp.classes[i])
    st.plotly_chart(plots.roc_curves(exp, chosen, class_index, dark))
    st.caption("The closer a curve gets to the top-left corner, the better the model separates "
               "this class from the others. AUC = 1 is perfect, 0.5 is random guessing.")


def _details(exp: Experiment) -> None:
    f = exp.features
    st.markdown(f"**Numeric features ({len(f['numeric'])}):** {', '.join(f['numeric']) or '—'}")
    st.markdown(f"**Categorical features ({len(f['categorical'])}):** "
                f"{', '.join(f['categorical']) or '—'}")
    if f["dropped"]:
        st.markdown("**Ignored columns:**")
        st.table(pd.DataFrame({"Column": list(f["dropped"]), "Reason": list(f["dropped"].values())}))
    if exp.classes:
        st.markdown(f"**Classes:** {', '.join(exp.classes)}")
    st.markdown("**Settings**")
    st.code(json.dumps(exp.settings, indent=2), language="json")

    with st.popover("Delete this experiment"):
        st.write("This permanently removes the experiment and its saved models.")
        if st.button("Delete", type="primary"):
            delete_experiment(exp.id)
            st.session_state.pop("selected_experiment", None)
            st.rerun()


def _show_errors(exp: Experiment) -> None:
    failed = [r for r in exp.results if r["status"] == "error"]
    if failed:
        with st.expander(f"{len(failed)} algorithm(s) failed"):
            for r in failed:
                st.markdown(f"- **{r['name']}**: {r['error']}")
