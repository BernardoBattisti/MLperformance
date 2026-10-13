"""Persist experiments to disk so they can be reopened later.

Layout::

    runs/<experiment-id>/
        experiment.json          # metrics, curves, settings (what the UI shows)
        models/<algorithm>.joblib  # fitted ModelBundle, ready for .predict()
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import joblib

from .trainer import Experiment, ModelBundle


def runs_root(runs_dir: str | Path | None = None) -> Path:
    """Explicit folder, else $MLPERFORMANCE_RUNS_DIR, else ./runs."""
    return Path(runs_dir or os.environ.get("MLPERFORMANCE_RUNS_DIR", "runs"))


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def save_experiment(
    experiment: Experiment,
    bundles: dict[str, ModelBundle],
    runs_dir: str | Path | None = None,
) -> Path:
    run_dir = runs_root(runs_dir) / experiment.id
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in bundles.items():
        joblib.dump(bundle, models_dir / f"{_slug(name)}.joblib")
    (run_dir / "experiment.json").write_text(
        json.dumps(experiment.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def list_experiments(runs_dir: str | Path | None = None) -> list[Experiment]:
    """All saved experiments, newest first. Unreadable folders are skipped."""
    runs_dir = runs_root(runs_dir)
    if not runs_dir.exists():
        return []
    experiments = []
    for path in runs_dir.glob("*/experiment.json"):
        try:
            experiments.append(Experiment.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, TypeError):
            continue
    return sorted(experiments, key=lambda e: e.created_at, reverse=True)


def load_experiment(experiment_id: str, runs_dir: str | Path | None = None) -> Experiment:
    path = runs_root(runs_dir) / experiment_id / "experiment.json"
    return Experiment.from_dict(json.loads(path.read_text(encoding="utf-8")))


def model_path(experiment_id: str, algorithm: str, runs_dir: str | Path | None = None) -> Path:
    return runs_root(runs_dir) / experiment_id / "models" / f"{_slug(algorithm)}.joblib"


def load_model(path: str | Path) -> ModelBundle:
    """Load a saved model. Only open files you trust: joblib can execute code."""
    return joblib.load(path)


def delete_experiment(experiment_id: str, runs_dir: str | Path | None = None) -> None:
    run_dir = (runs_root(runs_dir) / experiment_id).resolve()
    if run_dir.parent != runs_root(runs_dir).resolve():
        raise ValueError("Invalid experiment id.")
    shutil.rmtree(run_dir)
