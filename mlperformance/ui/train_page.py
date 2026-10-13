"""'New experiment' page: upload data, pick the target and algorithms, train."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from ..data import SUPPORTED_EXTENSIONS, load_dataset
from ..models import get_algorithms
from ..preprocessing import CLASSIFICATION, REGRESSION, detect_task, select_features
from ..storage import save_experiment
from ..trainer import METRICS, Settings, run_experiment

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@st.cache_data(show_spinner="Reading file…")
def _read(data: bytes, filename: str) -> pd.DataFrame:
    return load_dataset(io.BytesIO(data), filename=filename)


def _choose_dataset() -> tuple[pd.DataFrame, str] | None:
    source = st.radio("Data source", ["Upload a file", "Use an example dataset"],
                      horizontal=True, label_visibility="collapsed")
    if source == "Upload a file":
        upload = st.file_uploader(
            "Upload your dataset",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            help="CSV/TSV (any common separator), Excel, JSON or Parquet. "
                 "One row per sample, one column per variable.",
        )
        if upload is None:
            return None
        return _read(upload.getvalue(), upload.name), upload.name

    examples = sorted(p.name for p in EXAMPLES_DIR.glob("*.csv"))
    if not examples:
        st.info("No example datasets found in the examples/ folder.")
        return None
    name = st.selectbox("Example dataset", examples)
    return _read((EXAMPLES_DIR / name).read_bytes(), name), name


def render() -> None:
    st.title("New experiment")
    st.caption("Upload your data, choose what to predict, and let MLperformance "
               "compare machine learning algorithms for you.")

    st.subheader("1 · Data")
    try:
        chosen = _choose_dataset()
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return
    if chosen is None:
        return
    df, dataset_name = chosen

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(100), hide_index=True)

    st.subheader("2 · Target")
    target = st.selectbox("Which column do you want to predict?", df.columns,
                          index=len(df.columns) - 1)
    detected = detect_task(df[target])
    task = st.radio(
        "Problem type",
        [CLASSIFICATION, REGRESSION],
        index=0 if detected == CLASSIFICATION else 1,
        format_func=lambda t: f"{t.capitalize()}{' (detected)' if t == detected else ''}",
        horizontal=True,
        help="Classification predicts a category (e.g. 'spam' / 'not spam'). "
             "Regression predicts a number (e.g. a price).",
    )
    if task == CLASSIFICATION and df[target].nunique() > 50:
        st.warning(f"'{target}' has {df[target].nunique()} distinct values. "
                   "Regression may be a better fit for this column.")

    try:
        info = select_features(df, target)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(f"Using **{len(info.numeric)}** numeric and **{len(info.categorical)}** "
               "categorical columns as input.")
    if info.dropped:
        with st.expander(f"{len(info.dropped)} column(s) will be ignored"):
            st.table(pd.DataFrame({"Column": list(info.dropped),
                                   "Reason": list(info.dropped.values())}))

    st.subheader("3 · Algorithms")
    available = get_algorithms(task)
    mode = st.radio("Mode", ["Test all algorithms", "Choose specific algorithms"], horizontal=True)
    if mode == "Test all algorithms":
        selected = list(available)
    else:
        selected = st.multiselect("Algorithms", list(available), default=list(available)[:1])
    with st.expander("What does each algorithm do?"):
        st.table(pd.DataFrame({"Algorithm": list(available),
                               "Description": [a.description for a in available.values()]}))
    slow = [n for n in selected
            if available[n].slow_above_rows and len(df) > available[n].slow_above_rows]
    if slow:
        st.warning(f"These algorithms can be very slow on {len(df):,} rows: {', '.join(slow)}.")

    with st.expander("Advanced settings"):
        a1, a2, a3 = st.columns(3)
        test_size = a1.slider("Test set size", 0.1, 0.5, 0.2, 0.05,
                              help="Fraction of rows held out to evaluate the final models.")
        cv_folds = a2.number_input("Cross-validation folds", 0, 20, 5,
                                   help="0 disables cross-validation (faster).")
        seed = a3.number_input("Random seed", 0, 2**31 - 1, 42,
                               help="Fix it to get reproducible results.")
        name = st.text_input("Experiment name", f"{dataset_name} → {target}")

    if st.button("Train models", type="primary", disabled=not selected):
        _train(df, target, selected, task, dataset_name, name,
               Settings(test_size=test_size, cv_folds=int(cv_folds), random_state=int(seed)))
    _show_last_run()


def _train(df, target, selected, task, dataset_name, name, settings: Settings) -> None:
    st.session_state.pop("last_run", None)
    bar = st.progress(0.0, text="Starting…")

    def on_progress(i: int, total: int, algo: str) -> None:
        text = "Done!" if i == total else f"Training {algo} ({i + 1}/{total})…"
        bar.progress(i / total, text=text)

    try:
        experiment, bundles = run_experiment(
            df, target, selected, task=task, settings=settings,
            dataset_name=dataset_name, experiment_name=name, progress=on_progress,
        )
    except Exception as exc:
        bar.empty()
        st.error(f"Training failed: {exc}")
        return
    save_experiment(experiment, bundles)
    st.session_state["last_run"] = experiment
    st.session_state["selected_experiment"] = experiment.id


def _show_last_run() -> None:
    """Shown after training and kept across reruns, so the link stays clickable."""
    experiment = st.session_state.get("last_run")
    if experiment is None:
        return
    failed = [r for r in experiment.results if r["status"] == "error"]
    if failed:
        st.warning("Some algorithms failed: " + "; ".join(f"{r['name']}: {r['error']}" for r in failed))
    best = experiment.best
    if best is None:
        st.error("No algorithm could be trained on this data.")
        return
    label = METRICS[experiment.task][experiment.primary_metric][0]
    st.success(f"Best algorithm: **{best['name']}** — {label} = "
               f"{best['metrics'][experiment.primary_metric]:.4f}  \n"
               f"Experiment saved as `{experiment.id}`.")
    st.page_link("app_pages/results.py", label="Open results and charts", icon="📊")
