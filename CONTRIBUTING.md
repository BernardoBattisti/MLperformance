# Contributing

Thanks for your interest in MLperformance. This is an academic, open source project and contributions of any size are welcome: bug reports, documentation fixes, new algorithms, new charts.

## Development setup

```sh
git clone https://github.com/<your-user>/MLperformance.git
cd MLperformance
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt pytest
streamlit run app.py
```

## Running the tests

```sh
pytest
```

All tests must pass before a pull request is merged. GitHub Actions runs them automatically on Linux and Windows.

## Adding an algorithm

1. Add an `Algorithm(...)` entry to `mlperformance/models.py` with a name, the task (`CLASSIFICATION` or `REGRESSION`), a factory that receives the random seed, and a one-sentence description.
2. If it is slow on large datasets, set `slow_above_rows` so the UI warns the user.
3. Document it in `docs/ALGORITHMS_AND_METRICS.md`.
4. Run `pytest`: `test_classification_all_algorithms` trains every classifier on the Iris dataset.

## Guidelines

- Keep the machine learning core (`data`, `preprocessing`, `models`, `trainer`, `storage`) independent of Streamlit.
- Write user-facing text and documentation in English, aimed at beginners.
- Add or update tests for behavior changes.
- Keep pull requests focused on one change.

## Reporting bugs

Open an issue with the steps to reproduce, the error message, and, if possible, a small anonymized sample of the data that triggers the problem.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
