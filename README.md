# MLperformance

**Find the best machine learning algorithm for your data, without writing code.**

People who are just getting started with machine learning often don't know which algorithm to use for their data. MLperformance trains and compares many algorithms on your dataset and shows charts that explain which one performs best, and why.

It runs **locally on your computer**. Your data never leaves your machine.

> Open source under the [MIT License](LICENSE). Built for academic and educational purposes.

---

## Features

- **Import your own data**: CSV/TSV (the separator is detected automatically), Excel, JSON and Parquet.
- **Choose what to predict**: pick any column as the target. MLperformance detects whether the problem is **classification** (predicting a category) or **regression** (predicting a number), and you can override it.
- **Test one algorithm or all of them**: up to 12 classifiers and 13 regressors from scikit-learn, plus XGBoost and LightGBM if they are installed.
- **Automatic preprocessing**: fills missing values, scales numeric columns, one-hot encodes text columns and ignores columns that cannot help (IDs, empty or constant columns).
- **Sound evaluation**: k-fold cross-validation on the training split plus a held-out test set.
- **Saved experiments**: every run is stored in `runs/`, so you can reopen it later and compare charts.
- **Charts for each algorithm**:
  - Leaderboard and metric comparison
  - Cross-validation stability (score spread across folds)
  - Training time
  - Confusion matrix and ROC curves (classification)
  - Predicted vs. actual, error distribution and error comparison (regression)
  - Feature importance
- **Download trained models** (`.joblib`) and use them in your own Python code.
- **Command-line interface** for scripting and batch use.

## Quick start

You need **Python 3.10 or newer**.

### Windows

Double-click `run.bat`, or in a terminal:

```bat
run.bat
```

### Linux / macOS

```sh
chmod +x run.sh
./run.sh
```

The first run creates a virtual environment and installs the dependencies. The app then opens in your browser at <http://localhost:8501>.

### Manual installation

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional extra algorithms:

```sh
pip install xgboost lightgbm
```

## How to use

1. Open the **New experiment** page.
2. Upload a file, or pick one of the example datasets in [`examples/`](examples).
3. Choose the **column you want to predict**.
4. Choose **Test all algorithms** or **Choose specific algorithms**.
5. Click **Train models**.
6. Open the **Results** page to see the leaderboard and the charts for every algorithm.

A step-by-step walkthrough is in the [User Guide](docs/USER_GUIDE.md).

## Command line

```sh
# Compare all algorithms
python -m mlperformance train examples/iris.csv --target species

# Compare only some algorithms
python -m mlperformance train examples/diabetes.csv --target disease_progression \
    --algorithms "Random Forest" "Ridge Regression"

# List the available algorithms and saved experiments
python -m mlperformance algorithms
python -m mlperformance list
```

Experiments created from the command line also appear on the app's **Results** page.

## Using a trained model in Python

```python
import pandas as pd
from mlperformance.storage import load_model

model = load_model("runs/<experiment-id>/models/random-forest.joblib")
new_data = pd.read_csv("new_samples.csv")  # same columns as the training file
print(model.predict(new_data))
```

> Only load `.joblib` files that you trust: loading one can run arbitrary code.

## Documentation

| Document | Contents |
|---|---|
| [User Guide](docs/USER_GUIDE.md) | Step-by-step usage, data format, how to read each chart |
| [Algorithms & Metrics](docs/ALGORITHMS_AND_METRICS.md) | What each algorithm does and what each metric means |
| [Architecture](docs/ARCHITECTURE.md) | How the code is organized and how a run flows through it |
| [Contributing](CONTRIBUTING.md) | Development setup, tests and how to add an algorithm |

## Project structure

```
MLperformance/
├── app.py                  # Streamlit entry point
├── app_pages/              # Page scripts (New experiment, Results)
├── mlperformance/
│   ├── data.py             # File loading
│   ├── preprocessing.py    # Task detection, feature selection, preprocessing
│   ├── models.py           # Algorithm registry
│   ├── trainer.py          # Training, cross-validation and metrics
│   ├── storage.py          # Saving and loading experiments
│   ├── plots.py            # Plotly charts
│   ├── __main__.py         # Command-line interface
│   └── ui/                 # Page implementations
├── examples/               # Sample datasets
├── tests/                  # Automated tests (pytest)
└── docs/                   # Documentation
```

## Running the tests

```sh
pip install pytest
pytest
```

## Limitations

- Date/time columns are ignored for now.
- Hyperparameters use sensible defaults; there is no automatic tuning yet.
- Very large datasets (hundreds of thousands of rows) make some algorithms (SVM, KNN) slow. The app warns you before training.

## License

[MIT](LICENSE) © 2026 Bernardo Battisti. This project is intended for academic and educational use.
