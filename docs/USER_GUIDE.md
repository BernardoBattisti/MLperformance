# User Guide

This guide walks you through a complete experiment and explains how to read every chart.

## 1. Preparing your data

MLperformance expects **tabular data**: one row per sample (a customer, a house, a flower...) and one column per variable.

| Supported format | Extensions | Notes |
|---|---|---|
| Delimited text | `.csv`, `.tsv`, `.txt` | Comma, semicolon, tab or pipe separators are detected automatically. UTF-8 and Latin-1 encodings are supported. |
| Excel | `.xlsx`, `.xls` | The first sheet is used. The first row must contain the column names. |
| JSON | `.json` | A list of records, e.g. `[{"age": 30, "city": "A"}, ...]`. |
| Parquet | `.parquet` | |

Tips:

- The first row must contain the column names.
- Missing values are fine: numeric gaps are filled with the column median, text gaps with the most frequent value.
- At least 10 rows with a non-empty target are required. Results become meaningful from a few hundred rows.

### Columns that are ignored automatically

| Column | Why |
|---|---|
| Completely empty | No information |
| A single repeated value | Cannot distinguish samples |
| Text with too many distinct values (e.g. IDs, names, free text) | Would be memorized instead of learned |
| Date/time | Not supported yet |

The **New experiment** page lists every ignored column and the reason.

## 2. Running an experiment

1. **Data**: upload your file or select an example dataset. Check the row and column counts and preview the data.
2. **Target**: choose the column you want to predict.
   - **Classification** is used when the target is text, boolean, or a whole number with at most 20 distinct values.
   - **Regression** is used for other numeric targets.
   - You can override the detected type.
3. **Algorithms**:
   - **Test all algorithms**: recommended when you don't know where to start.
   - **Choose specific algorithms**: train only the ones you select.
   - Expand *What does each algorithm do?* for a short description of each.
4. **Advanced settings** (optional):
   - *Test set size*: fraction of rows held out for the final evaluation (default 20%).
   - *Cross-validation folds*: number of folds used on the training split (default 5, 0 disables it).
   - *Random seed*: fixes the randomness so results are reproducible.
   - *Experiment name*: how the run appears on the Results page.
5. Click **Train models**. A progress bar shows which algorithm is being trained.

When training finishes, the best algorithm is shown, and the experiment is saved under `runs/`.

## 3. Reading the results

Open the **Results** page and pick an experiment. The header shows the dataset, target, problem type and split sizes. The cards show the best algorithm and its score on the test set.

Models are ranked by:

- **F1 (weighted)** for classification, which balances precision and recall and handles unbalanced classes.
- **R²** for regression, the share of the target's variance the model explains.

### Leaderboard tab

- **Leaderboard table**: all metrics for every algorithm, best first. *CV mean* / *CV std* are the average and spread of the cross-validation scores. You can download the table as CSV.
- **Compare by metric**: horizontal bars, one per algorithm. The best one is highlighted. The axis title says whether higher or lower is better.
- **Cross-validation stability**: each dot is the score on one fold. A narrow box means the algorithm performs consistently. A wide box means its performance depends heavily on which rows it sees, so be careful with it.
- **Training time**: how long each algorithm took to train. Useful when two models score about the same.

### Algorithm details tab

Select an algorithm to inspect it.

**Classification**

- **Confusion matrix**: rows are the actual classes, columns the predicted classes. The diagonal shows correct predictions. With *Show percentages per actual class* enabled, each row adds up to 100%, so you can see which classes are confused with which.

**Regression**

- **Predicted vs. actual**: each dot is a test sample. Dots on the dashed line are perfect predictions. Dots far from the line are large errors.
- **Error distribution**: histogram of *predicted − actual*. A good model is centered on 0 with a narrow spread. A shifted histogram means the model consistently over- or under-estimates.

**Both**

- **Most influential features**: how much each input column contributed to the model, as relative importance. Tree models report their own importances; linear models use the absolute size of their coefficients. Some algorithms (KNN, SVM with RBF kernel, Naive Bayes, MLP) don't provide importances.
- **Download trained model**: saves the fitted model as a `.joblib` file (see the README for how to use it).

### ROC curves tab (classification)

Shows how well up to three algorithms separate one class from the others at every decision threshold. The closer a curve is to the top-left corner, the better. **AUC** (area under the curve) summarizes it: 1.0 is perfect and 0.5 is random guessing. For more than two classes, choose which class to look at (one-vs-rest).

### Error comparison tab (regression)

One box per algorithm with the absolute error on each test sample, most accurate first. The line inside each box is the median error. Shorter boxes further left mean smaller, more consistent errors.

### Experiment info tab

Lists the features that were used, the ignored columns, the classes, and the settings. You can also delete the experiment here.

## 4. Choosing the "best" algorithm

The top of the leaderboard is a good starting point, but also consider:

1. **Score differences**: if several algorithms are within about 0.01 of each other, they are effectively tied.
2. **Stability**: prefer a model with a narrow cross-validation box.
3. **Speed**: when scores are tied, a faster model is cheaper to retrain and run.
4. **Interpretability**: linear models and single decision trees are easier to explain than ensembles or neural networks.
5. **Overfitting**: if the cross-validation mean is much lower than the test score (or the other way round), the result may be due to luck in the split. Try another random seed.

## 5. Where things are stored

```
runs/<experiment-id>/
├── experiment.json          # all metrics and chart data
└── models/<algorithm>.joblib
```

Set the `MLPERFORMANCE_RUNS_DIR` environment variable to store experiments somewhere else. Deleting a folder in `runs/` removes that experiment.
