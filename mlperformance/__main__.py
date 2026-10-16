"""Command-line interface.

    python -m mlperformance train data.csv --target price
    python -m mlperformance train data.csv --target species --algorithms "Random Forest" "Naive Bayes"
    python -m mlperformance algorithms
    python -m mlperformance list
"""

from __future__ import annotations

import argparse
import sys

from .data import load_dataset
from .models import get_algorithms
from .preprocessing import CLASSIFICATION, REGRESSION
from .storage import list_experiments, save_experiment
from .trainer import METRICS, Settings, run_experiment


def _train(args: argparse.Namespace) -> int:
    df = load_dataset(args.file)

    def progress(i: int, total: int, name: str) -> None:
        if i < total:
            print(f"[{i + 1}/{total}] {name}…", flush=True)

    experiment, bundles = run_experiment(
        df, args.target, args.algorithms, task=args.task,
        settings=Settings(args.test_size, args.cv_folds, args.seed),
        dataset_name=args.file, progress=progress,
    )
    run_dir = save_experiment(experiment, bundles, args.runs_dir)

    metrics = METRICS[experiment.task]
    higher = metrics[experiment.primary_metric][1]
    ranked = sorted(experiment.successful, key=lambda r: r["metrics"][experiment.primary_metric],
                    reverse=higher)
    print(f"\nTask: {experiment.task} · ranked by {metrics[experiment.primary_metric][0]} (test set)\n")
    for rank, r in enumerate(ranked, 1):
        print(f"{rank:>3}. {r['name']:<32} {r['metrics'][experiment.primary_metric]:.4f}"
              f"   ({r['train_time']:.2f}s)")
    for r in experiment.results:
        if r["status"] == "error":
            print(f"  ✗ {r['name']}: {r['error']}")
    print(f"\nSaved to {run_dir}. Open the app (python MLperformance.py) to see the charts.")
    return 0


def _algorithms(_: argparse.Namespace) -> int:
    for task in (CLASSIFICATION, REGRESSION):
        print(f"{task.capitalize()}:")
        for name, algo in get_algorithms(task).items():
            print(f"  - {name}: {algo.description}")
    return 0


def _list(args: argparse.Namespace) -> int:
    for exp in list_experiments(args.runs_dir):
        best = exp.best
        summary = f"{best['name']} ({best['metrics'][exp.primary_metric]:.4f})" if best else "no model"
        print(f"{exp.id}  {exp.name:<40} best: {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Legacy Windows consoles (cp1252) cannot print every character in dataset names.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="mlperformance",
                                     description="Compare machine learning algorithms on your data.")
    parser.add_argument("--runs-dir", default=None, help="Where experiments are stored (default: user data folder)")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train and compare algorithms on a dataset")
    train.add_argument("file", help="CSV, TSV, Excel, JSON or Parquet file")
    train.add_argument("--target", required=True, help="Column to predict")
    train.add_argument("--algorithms", nargs="+", help="Algorithms to train (default: all)")
    train.add_argument("--task", choices=[CLASSIFICATION, REGRESSION], help="Override task detection")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--cv-folds", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=_train)

    sub.add_parser("algorithms", help="List available algorithms").set_defaults(func=_algorithms)
    sub.add_parser("list", help="List saved experiments").set_defaults(func=_list)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
