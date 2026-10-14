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

The **New experiment** screen lists every ignored column and the reason.

## 2. Running an experiment

The **New experiment** screen is organized in numbered steps.

1. **Data**: click **Open file…**, or choose an example dataset and click **Load example**. Check the row, column and missing-value counts and the preview table.
2. **Target**: choose the column to predict.
   - **Classification** is used when the target is text, boolean, or a whole number with at most 20 distinct values.
   - **Regression** is used for other numeric targets.
   - The detected type is marked *(detected)*. You can override it.
   - The step lists how many columns are used as input and which ones are ignored, and why.
3. **Algorithms**:
   - **Test all algorithms**: recommended when you don't know where to start.
   - **Choose specific algorithms**: tick only the ones you want.
   - Hover over an algorithm for a one-line description.
4. **Settings** (optional):
   - *Test set size*: share of rows held out for the final evaluation (default 20%).
   - *Cross-validation folds*: number of folds used on the training split (default 5, 0 disables it).
   - *Random seed*: fixes the randomness so results are reproducible.
   - *Experiment name*: how the run appears on the Results screen.
5. Click **Train models**. A progress bar shows which algorithm is being trained. The window stays responsive while you wait.

When training finishes, a green banner shows the best algorithm. Click **Open results →** to see the details.

## 3. Reading the results

On the **Results** screen, choose an experiment in the list at the top. **Refresh** reloads the list and **Delete** removes the selected experiment and its models.

The line under the list shows the dataset, target, problem type and split sizes. The cards show the best algorithm and its score on the test set.

Models are ranked by:

- **F1 (weighted)** for classification, which balances precision and recall and handles unbalanced classes.
- **R²** for regression, the share of the target's variance the model explains.

**Hover over any bar, dot or cell** in a chart to see its exact value.

### Leaderboard tab

- **Leaderboard**: all metrics for every algorithm, best first. *CV mean* / *CV std* are the average and spread of the cross-validation scores. **Export CSV…** saves the table.
- **Comparison by metric**: one bar per algorithm. The best one is highlighted. Pick any metric in the drop-down; the axis title says whether higher or lower is better.
- **Cross-validation stability**: each dot is the score on one fold. A narrow box means the algorithm performs consistently. A wide box means its performance depends heavily on which rows it sees, so be careful with it.
- **Training time**: how long each algorithm took to train. Useful when two models score about the same.

### Algorithm details tab

Choose an algorithm at the top. The cards show all of its metrics and its training time. **Export trained model…** saves it as a `.joblib` file (see the README for how to use it).

**Classification**

- **Confusion matrix**: rows are the actual classes, columns the predicted classes. The diagonal shows correct predictions. With *Show percentages per actual class* enabled, each row adds up to 100%, so you can see which classes are confused with which.

**Regression**

- **Predicted vs. actual**: each dot is a test sample. Dots on the dashed line are perfect predictions. Dots far from the line are large errors.
- **Error distribution**: histogram of *predicted − actual*. A good model is centered on 0 with a narrow spread. A shifted histogram means the model consistently over- or under-estimates.

**Both**

- **Most influential features**: how much each input column contributed to the model, as relative importance. Tree models report their own importances; linear models use the absolute size of their coefficients. Some algorithms (KNN, SVM with RBF kernel, Naive Bayes, MLP, Histogram Gradient Boosting) don't provide importances.

### ROC curves tab (classification)

Shows how well up to three algorithms separate one class from the others at every decision threshold. Tick the algorithms to compare in the list on the left. The closer a curve is to the top-left corner, the better. **AUC** (area under the curve) summarizes it: 1.0 is perfect and 0.5 is random guessing. For more than two classes, choose which class to look at (one-vs-rest).

### Error comparison tab (regression)

One box per algorithm with the absolute error on each test sample, most accurate first. The line inside each box is the median error. Shorter boxes further left mean smaller, more consistent errors.

### Experiment info tab

Lists the features that were used, the ignored columns, the classes, the settings and the folder where the experiment is stored.

## 4. Choosing the "best" algorithm

The top of the leaderboard is a good starting point, but also consider:

1. **Score differences**: if several algorithms are within about 0.01 of each other, they are effectively tied.
2. **Stability**: prefer a model with a narrow cross-validation box.
3. **Speed**: when scores are tied, a faster model is cheaper to retrain and run.
4. **Interpretability**: linear models and single decision trees are easier to explain than ensembles or neural networks.
5. **Overfitting**: if the cross-validation mean is much lower than the test score (or the other way round), the result may be due to luck in the split. Try another random seed.

## 5. Appearance

Use the **Appearance** selector at the bottom of the sidebar to choose **System** (follows your Windows/macOS setting), **Light** or **Dark**. The choice is remembered.

## 6. Where things are stored

```
<data folder>/MLperformance/runs/<experiment-id>/
├── experiment.json          # all metrics and chart data
└── models/<algorithm>.joblib
```

The data folder is `%LOCALAPPDATA%` on Windows and `~/.local/share` on Linux/macOS. Set the `MLPERFORMANCE_RUNS_DIR` environment variable to store experiments somewhere else.
