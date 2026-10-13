"""Registry of the algorithms MLperformance can train and compare."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .preprocessing import CLASSIFICATION, REGRESSION


@dataclass(frozen=True)
class Algorithm:
    name: str
    task: str
    factory: Callable[[int], object]  # random_state -> unfitted estimator
    description: str
    # Kernel/neighbour methods scale badly; the UI warns above this many rows.
    slow_above_rows: int | None = None


def _algorithms() -> list[Algorithm]:
    algos = [
        # --- classification ---
        Algorithm("Logistic Regression", CLASSIFICATION,
                  lambda rs: LogisticRegression(max_iter=2000, random_state=rs),
                  "Linear model that estimates class probabilities. Fast, interpretable baseline."),
        Algorithm("K-Nearest Neighbors", CLASSIFICATION,
                  lambda rs: KNeighborsClassifier(),
                  "Predicts the majority class of the closest training samples.", 50_000),
        Algorithm("Decision Tree", CLASSIFICATION,
                  lambda rs: DecisionTreeClassifier(random_state=rs),
                  "A single tree of if/else rules. Very interpretable, prone to overfitting."),
        Algorithm("Random Forest", CLASSIFICATION,
                  lambda rs: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=rs),
                  "Many decorrelated trees voting together. Strong general-purpose model."),
        Algorithm("Extra Trees", CLASSIFICATION,
                  lambda rs: ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=rs),
                  "Like Random Forest with more randomized splits; often faster."),
        Algorithm("Gradient Boosting", CLASSIFICATION,
                  lambda rs: GradientBoostingClassifier(random_state=rs),
                  "Trees built sequentially, each one correcting the errors of the previous."),
        Algorithm("Histogram Gradient Boosting", CLASSIFICATION,
                  lambda rs: HistGradientBoostingClassifier(random_state=rs),
                  "Fast gradient boosting for larger datasets (LightGBM-style)."),
        Algorithm("AdaBoost", CLASSIFICATION,
                  lambda rs: AdaBoostClassifier(random_state=rs),
                  "Boosting that re-weights misclassified samples."),
        Algorithm("Support Vector Machine", CLASSIFICATION,
                  lambda rs: SVC(probability=True, random_state=rs),
                  "Finds the maximum-margin boundary (RBF kernel). Slow on big data.", 20_000),
        Algorithm("Naive Bayes", CLASSIFICATION,
                  lambda rs: GaussianNB(),
                  "Probabilistic model assuming independent features. Extremely fast."),
        Algorithm("Linear Discriminant Analysis", CLASSIFICATION,
                  lambda rs: LinearDiscriminantAnalysis(),
                  "Linear boundaries assuming Gaussian classes with shared covariance."),
        Algorithm("Neural Network (MLP)", CLASSIFICATION,
                  lambda rs: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                                           early_stopping=True, random_state=rs),
                  "Multi-layer perceptron: a small fully connected neural network."),
        # --- regression ---
        Algorithm("Linear Regression", REGRESSION,
                  lambda rs: LinearRegression(),
                  "Ordinary least squares. The simplest regression baseline."),
        Algorithm("Ridge Regression", REGRESSION,
                  lambda rs: Ridge(random_state=rs),
                  "Linear regression with L2 regularization; robust to correlated features."),
        Algorithm("Lasso Regression", REGRESSION,
                  lambda rs: Lasso(random_state=rs, max_iter=10_000),
                  "Linear regression with L1 regularization; can zero out useless features."),
        Algorithm("Elastic Net", REGRESSION,
                  lambda rs: ElasticNet(random_state=rs, max_iter=10_000),
                  "Mix of Ridge and Lasso regularization."),
        Algorithm("K-Nearest Neighbors", REGRESSION,
                  lambda rs: KNeighborsRegressor(),
                  "Averages the target of the closest training samples.", 50_000),
        Algorithm("Decision Tree", REGRESSION,
                  lambda rs: DecisionTreeRegressor(random_state=rs),
                  "A single tree of if/else rules. Very interpretable, prone to overfitting."),
        Algorithm("Random Forest", REGRESSION,
                  lambda rs: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=rs),
                  "Many decorrelated trees averaged together. Strong general-purpose model."),
        Algorithm("Extra Trees", REGRESSION,
                  lambda rs: ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=rs),
                  "Like Random Forest with more randomized splits; often faster."),
        Algorithm("Gradient Boosting", REGRESSION,
                  lambda rs: GradientBoostingRegressor(random_state=rs),
                  "Trees built sequentially, each one correcting the errors of the previous."),
        Algorithm("Histogram Gradient Boosting", REGRESSION,
                  lambda rs: HistGradientBoostingRegressor(random_state=rs),
                  "Fast gradient boosting for larger datasets (LightGBM-style)."),
        Algorithm("AdaBoost", REGRESSION,
                  lambda rs: AdaBoostRegressor(random_state=rs),
                  "Boosting that focuses on samples with large errors."),
        Algorithm("Support Vector Machine", REGRESSION,
                  lambda rs: SVR(),
                  "Support vector regression with an RBF kernel. Slow on big data.", 20_000),
        Algorithm("Neural Network (MLP)", REGRESSION,
                  lambda rs: MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000,
                                          early_stopping=True, random_state=rs),
                  "Multi-layer perceptron: a small fully connected neural network."),
    ]
    algos.extend(_optional_algorithms())
    return algos


def _optional_algorithms() -> list[Algorithm]:
    """XGBoost and LightGBM are offered only when they are installed."""
    extra: list[Algorithm] = []
    try:
        from xgboost import XGBClassifier, XGBRegressor

        extra += [
            Algorithm("XGBoost", CLASSIFICATION,
                      lambda rs: XGBClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
                      "Highly optimized gradient boosting library."),
            Algorithm("XGBoost", REGRESSION,
                      lambda rs: XGBRegressor(n_estimators=300, random_state=rs, n_jobs=-1),
                      "Highly optimized gradient boosting library."),
        ]
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        extra += [
            Algorithm("LightGBM", CLASSIFICATION,
                      lambda rs: LGBMClassifier(random_state=rs, verbose=-1),
                      "Fast, memory-efficient gradient boosting by Microsoft."),
            Algorithm("LightGBM", REGRESSION,
                      lambda rs: LGBMRegressor(random_state=rs, verbose=-1),
                      "Fast, memory-efficient gradient boosting by Microsoft."),
        ]
    except ImportError:
        pass
    return extra


ALGORITHMS: list[Algorithm] = _algorithms()


def get_algorithms(task: str) -> dict[str, Algorithm]:
    """All algorithms available for ``task``, keyed by display name."""
    return {a.name: a for a in ALGORITHMS if a.task == task}
