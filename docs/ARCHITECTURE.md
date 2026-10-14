# Architecture

MLperformance is a desktop application written in Python. The interface uses [PySide6](https://doc.qt.io/qtforpython-6/) (the official Qt bindings), charts are drawn with [matplotlib](https://matplotlib.org) inside the window, and the models come from [scikit-learn](https://scikit-learn.org). [PyInstaller](https://pyinstaller.org) packages everything into a Windows executable.

The machine learning core has no dependency on the interface, so it can also be used from the command line or from your own scripts.

## Modules

```
mlperformance/
├── data.py            load_dataset(): CSV/TSV/Excel/JSON/Parquet -> DataFrame
├── preprocessing.py   detect_task(), select_features(), build_preprocessor(), prepare_xy()
├── models.py          Algorithm registry: name, task, factory, description
├── trainer.py         run_experiment() -> (Experiment, {name: ModelBundle})
├── storage.py         save/list/load/delete experiments in the user data folder
├── __main__.py        Command-line interface
└── gui/
    ├── app.py           Entry point: QApplication setup, --self-test
    ├── main_window.py   Sidebar navigation, theme selection
    ├── train_view.py    "New experiment" screen
    ├── results_view.py  "Results" screen
    ├── worker.py        Runs training on a background QThread
    ├── charts.py        matplotlib figures built from an Experiment
    ├── widgets.py       Cards, stat tiles, chart canvas with tooltips, table model
    └── theme.py         Light/dark color tokens and the Qt style sheet
MLperformance.py       Launcher, also the PyInstaller entry point
MLperformance.spec     PyInstaller recipe (bundles assets/ and examples/)
```

Dependencies only point downwards: `gui` → `storage` → `trainer` → `models`/`preprocessing` → `data`.

## How an experiment flows

```
file ──load_dataset──▶ DataFrame
                          │
                select_features(target)       drops IDs, empty/constant, date columns
                          │
                 prepare_xy ─▶ X, y           drops rows with an empty target
                          │
              train_test_split (stratified for classification)
                          │
        for each algorithm:
            Pipeline[ preprocess ─▶ model ]
            ├─ cross_val_score on train       ─▶ cv_scores
            ├─ fit on train                   ─▶ train_time
            ├─ predict on test                ─▶ metrics, confusion matrix,
            │                                    ROC curves / predictions
            └─ feature importances
                          │
                 Experiment (JSON) + ModelBundles (joblib)
                          │
                    save_experiment ─▶ runs/<id>/
                          │
                  Results screen reads runs/ and draws charts
```

A failure in one algorithm is recorded in its result (`status: "error"`) and the others keep running.

## Preprocessing

Built by `build_preprocessor()` as a `ColumnTransformer`, and placed **inside** each model's `Pipeline`, so it is re-fitted on the training data of every fold and never sees test data:

| Column type | Steps |
|---|---|
| Numeric (and boolean) | Median imputation → standard scaling |
| Categorical (text) | Most-frequent imputation → one-hot encoding (at most 50 categories, rare ones grouped, unseen categories ignored) |

Classification targets are label-encoded; the encoder is stored with the model so predictions are returned as the original labels.

## Data model

`Experiment` (in `trainer.py`) is a plain dataclass serialized to `experiment.json`:

| Field | Description |
|---|---|
| `id`, `name`, `created_at` | Identification |
| `dataset`, `n_rows`, `target`, `task` | What was trained |
| `features` | `numeric`, `categorical` and `dropped` (column → reason) |
| `classes` | Class labels (classification) |
| `settings` | Test size, folds, seed, train/test sizes |
| `primary_metric` | `f1_weighted` or `r2` |
| `results` | One entry per algorithm (below) |

Each result contains `name`, `status`, `metrics`, `cv_scores`, `train_time`, `predict_time` and `feature_importance`, plus:

- classification: `confusion_matrix` and `roc` (a list of `{class, fpr, tpr, auc}`; curves are thinned to 300 points);
- regression: `y_true` / `y_pred` for the test set (sampled to at most 5,000 points).

Because the Results screen only needs `experiment.json`, opening an old experiment is instant and does not require re-training.

`ModelBundle` wraps the fitted pipeline with the feature list and the label encoder and exposes `predict(df)` for raw data.

## Desktop interface

- **Threading**: training runs in `TrainingWorker` on a `QThread`, so the window stays responsive. Progress is reported to the UI through Qt signals.
- **Charts**: each function in `charts.py` draws onto an empty matplotlib `Figure` and returns *hover targets* (bars, points or heatmap cells with their tooltip text). `ChartView` embeds the figure and shows the tooltips. It re-runs the drawing function when the theme changes.
- **Theme**: `theme.py` holds the color tokens for light and dark mode and generates the Qt style sheet from them. Chart colors come from a colorblind-safe palette with separate light and dark steps. Single-series charts use one hue with the best model highlighted, and ROC plots are limited to three curves so the lines stay distinguishable.
- **Resources**: `widgets.resource_path()` finds `assets/` and `examples/` both when running from source and inside the PyInstaller bundle.

## Extending

- **New algorithm**: add an `Algorithm(...)` entry in `models.py`. It must follow the scikit-learn estimator API. It then appears in the UI, the CLI and the tests automatically.
- **New metric**: compute it in `_classification_details` / `_regression_details` in `trainer.py` and add it to `METRICS`.
- **New chart**: add a function to `gui/charts.py` and place a `ChartView` for it in `gui/results_view.py`.
- **New file format**: extend `load_dataset()` in `data.py` and `SUPPORTED_EXTENSIONS`.

## Testing and packaging

`tests/` contains unit tests for loading, preprocessing, training and storage. It also contains interface tests that open the real windows with Qt's off-screen platform and run complete experiments. Tests use the `MLPERFORMANCE_RUNS_DIR` environment variable to write into a temporary folder. GitHub Actions runs the suite on Linux and Windows.

`build.bat` (locally) and `.github/workflows/release.yml` (on GitHub, for `v*` tags) build the executable with PyInstaller. Both then run `MLperformance.exe --self-test`, which trains every algorithm on the bundled examples, saves the models, reloads them and predicts. This confirms that nothing was left out of the bundle.
