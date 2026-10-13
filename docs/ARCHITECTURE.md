# Architecture

MLperformance is a local Python application. The web interface is built with [Streamlit](https://streamlit.io), the models come from [scikit-learn](https://scikit-learn.org), and charts are drawn with [Plotly](https://plotly.com/python/). The machine learning core has no dependency on the UI, so it can be used from the command line or from your own scripts.

## Modules

```
mlperformance/
├── data.py            load_dataset(): CSV/TSV/Excel/JSON/Parquet -> DataFrame
├── preprocessing.py   detect_task(), select_features(), build_preprocessor(), prepare_xy()
├── models.py          Algorithm registry: name, task, factory, description
├── trainer.py         run_experiment() -> (Experiment, {name: ModelBundle})
├── storage.py         save/list/load/delete experiments under runs/
├── plots.py           Plotly figures built from an Experiment
├── __main__.py        Command-line interface
└── ui/
    ├── train_page.py    "New experiment" page
    └── results_page.py  "Results" page
app.py                 Streamlit entry point (navigation)
app_pages/             Thin page scripts that call ui.*.render()
```

Dependencies only point downwards: `ui` → `plots`/`storage` → `trainer` → `models`/`preprocessing` → `data`.

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
                  Results page reads runs/ and draws charts
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

Because the results page only needs `experiment.json`, opening an old experiment is instant and does not require re-training.

`ModelBundle` wraps the fitted pipeline with the feature list and the label encoder and exposes `predict(df)` for raw data.

## Charts

`plots.py` builds every figure from an `Experiment` or a single result. Colors come from a colorblind-safe palette with separate light and dark variants (the page detects Streamlit's theme). Single-series charts use one hue with the best model highlighted. Multi-model ROC plots are limited to three curves so the lines stay distinguishable.

## Extending

- **New algorithm**: add an `Algorithm(...)` entry in `models.py`. It must follow the scikit-learn estimator API. It then appears in the UI, the CLI and the tests automatically.
- **New metric**: compute it in `_classification_details` / `_regression_details` in `trainer.py` and add it to `METRICS`.
- **New chart**: add a function to `plots.py` and call it from `ui/results_page.py`.
- **New file format**: extend `load_dataset()` in `data.py` and `SUPPORTED_EXTENSIONS`.

## Testing

`tests/` contains unit tests for loading, preprocessing, training and storage, plus headless UI tests using Streamlit's `AppTest`. Tests use the `MLPERFORMANCE_RUNS_DIR` environment variable to write into a temporary folder. GitHub Actions runs the suite on Linux and Windows.
