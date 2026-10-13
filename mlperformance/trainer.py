"""Train, cross-validate and evaluate algorithms on a dataset."""

from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .models import get_algorithms
from .preprocessing import (
    CLASSIFICATION,
    FeatureInfo,
    build_preprocessor,
    clean_features,
    detect_task,
    prepare_xy,
    select_features,
)

# metric key -> (label, higher_is_better)
METRICS = {
    CLASSIFICATION: {
        "f1_weighted": ("F1 (weighted)", True),
        "accuracy": ("Accuracy", True),
        "balanced_accuracy": ("Balanced accuracy", True),
        "precision_weighted": ("Precision (weighted)", True),
        "recall_weighted": ("Recall (weighted)", True),
        "roc_auc": ("ROC AUC", True),
    },
    "regression": {
        "r2": ("R²", True),
        "rmse": ("RMSE", False),
        "mae": ("MAE", False),
        "median_ae": ("Median absolute error", False),
    },
}
PRIMARY_METRIC = {CLASSIFICATION: "f1_weighted", "regression": "r2"}

MAX_STORED_POINTS = 5000  # regression scatter points kept per model
MAX_IMPORTANCES = 25


@dataclass
class Settings:
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class ModelBundle:
    """A fitted model plus everything needed to predict on new raw data."""

    algorithm: str
    pipeline: Pipeline
    features: list[str]
    target: str
    task: str
    label_encoder: LabelEncoder | None = None

    feature_info: FeatureInfo | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on raw data with the same columns as the training file."""
        missing = [c for c in self.feature_info.used if c not in df.columns]
        if missing:
            raise ValueError(f"Missing column(s): {', '.join(missing)}")
        X = clean_features(df, self.feature_info)
        pred = self.pipeline.predict(X)
        if self.label_encoder is not None:
            pred = self.label_encoder.inverse_transform(pred.astype(int))
        return pred


@dataclass
class Experiment:
    """Everything shown on the results page. JSON-serializable via ``to_dict``."""

    id: str
    name: str
    created_at: str
    dataset: str
    n_rows: int
    target: str
    task: str
    features: dict
    classes: list[str]
    settings: dict
    primary_metric: str
    results: list[dict] = field(default_factory=list)

    @property
    def successful(self) -> list[dict]:
        return [r for r in self.results if r["status"] == "ok"]

    @property
    def best(self) -> dict | None:
        ok = self.successful
        if not ok:
            return None
        higher = METRICS[self.task][self.primary_metric][1]
        return (max if higher else min)(ok, key=lambda r: r["metrics"][self.primary_metric])

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        return cls(**data)


ProgressCallback = Callable[[int, int, str], None]


def run_experiment(
    df: pd.DataFrame,
    target: str,
    algorithms: list[str] | None = None,
    task: str | None = None,
    settings: Settings | None = None,
    dataset_name: str = "dataset",
    experiment_name: str | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Experiment, dict[str, ModelBundle]]:
    """Train every requested algorithm and return the experiment plus fitted models.

    ``algorithms=None`` means "test all algorithms available for the task".
    """
    settings = settings or Settings()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    task = task or detect_task(df[target])
    info = select_features(df, target)
    X, y_raw = prepare_xy(df, target, info)

    available = get_algorithms(task)
    names = algorithms or list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"Unknown algorithm(s) for {task}: {', '.join(unknown)}")

    label_encoder = None
    classes: list[str] = []
    stratify = None
    if task == CLASSIFICATION:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        classes = [str(c) for c in label_encoder.classes_]
        if len(classes) < 2:
            raise ValueError("The target column needs at least two different classes.")
        if np.bincount(y).min() >= 2:
            stratify = y
    else:
        y = pd.to_numeric(y_raw, errors="raise").to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state, stratify=stratify
    )
    cv = _make_cv(task, y_train, settings)

    experiment = Experiment(
        id=datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        name=experiment_name or f"{dataset_name} → {target}",
        created_at=datetime.now().isoformat(timespec="seconds"),
        dataset=dataset_name,
        n_rows=len(X),
        target=target,
        task=task,
        features={"numeric": info.numeric, "categorical": info.categorical, "dropped": info.dropped},
        classes=classes,
        settings={**settings.__dict__, "cv_folds": cv.get_n_splits() if cv else 0,
                  "n_train": len(X_train), "n_test": len(X_test)},
        primary_metric=PRIMARY_METRIC[task],
    )

    bundles: dict[str, ModelBundle] = {}
    preprocessor = build_preprocessor(info)
    for i, name in enumerate(names):
        if progress:
            progress(i, len(names), name)
        pipeline = Pipeline(
            [("preprocess", clone(preprocessor)),
             ("model", available[name].factory(settings.random_state))]
        )
        try:
            result = _evaluate(pipeline, task, X_train, X_test, y_train, y_test, cv,
                               PRIMARY_METRIC[task], len(classes))
            result["name"] = name
            experiment.results.append(result)
            bundles[name] = ModelBundle(name, pipeline, info.used, target, task,
                                        label_encoder, info)
        except Exception as exc:  # one failing algorithm must not stop the others
            experiment.results.append({"name": name, "status": "error", "error": str(exc)})
    if progress:
        progress(len(names), len(names), "done")
    return experiment, bundles


def _make_cv(task: str, y_train: np.ndarray, settings: Settings):
    folds = settings.cv_folds
    if folds < 2:
        return None
    if task == CLASSIFICATION:
        folds = min(folds, int(np.bincount(y_train).min()))
        if folds < 2:
            return None
        return StratifiedKFold(folds, shuffle=True, random_state=settings.random_state)
    return KFold(min(folds, len(y_train)), shuffle=True, random_state=settings.random_state)


def _evaluate(pipeline, task, X_train, X_test, y_train, y_test, cv, primary, n_classes) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = []
        if cv is not None:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                        scoring=primary, error_score="raise").tolist()

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        predict_time = time.perf_counter() - start

        result = {
            "status": "ok",
            "cv_scores": cv_scores,
            "train_time": train_time,
            "predict_time": predict_time,
            "feature_importance": _feature_importance(pipeline),
        }
        if task == CLASSIFICATION:
            y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            result.update(_classification_details(y_test, y_pred, y_proba, n_classes))
        else:
            result.update(_regression_details(y_test, y_pred))
    return result


def _classification_details(y_true, y_pred, y_proba, n_classes) -> dict:
    labels = list(range(n_classes))
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    roc = []
    if y_proba is not None:
        present = np.unique(y_true)
        try:
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            elif len(present) == n_classes:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                                   average="weighted", labels=labels)
        except ValueError:
            pass
        # Binary: one curve for the positive class. Multiclass: one-vs-rest per class.
        for cls in ([1] if n_classes == 2 else labels):
            if cls not in present or len(present) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true == cls, y_proba[:, cls])
            fpr, tpr = _thin_curve(fpr, tpr)
            roc.append({"class": cls, "fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))})
    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "roc": roc,
    }


def _thin_curve(fpr: np.ndarray, tpr: np.ndarray, max_points: int = 300) -> tuple[list, list]:
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr.tolist(), tpr.tolist()


def _regression_details(y_true, y_pred) -> dict:
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
    }
    idx = np.arange(len(y_true))
    if len(idx) > MAX_STORED_POINTS:
        idx = np.random.default_rng(0).choice(idx, MAX_STORED_POINTS, replace=False)
    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "y_true": np.asarray(y_true)[idx].tolist(),
        "y_pred": np.asarray(y_pred)[idx].tolist(),
    }


def _feature_importance(pipeline: Pipeline) -> dict[str, float]:
    """Built-in importances (trees) or absolute coefficients (linear models)."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return {}
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        return {}
    if len(names) != len(values) or not np.isfinite(values).all() or values.sum() == 0:
        return {}
    values = values / values.sum()
    order = np.argsort(values)[::-1][:MAX_IMPORTANCES]
    return {str(names[i]): float(values[i]) for i in order}
