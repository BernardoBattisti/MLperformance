# Algorithms and Metrics

All algorithms use scikit-learn's default hyperparameters, except where noted, so that comparisons are fair and reproducible. Every algorithm receives the same preprocessed data (see [Architecture](ARCHITECTURE.md#preprocessing)).

## Classification algorithms

| Algorithm | Idea | Strengths | Weaknesses |
|---|---|---|---|
| Logistic Regression | Linear model of class probabilities | Fast, interpretable, good baseline | Only linear boundaries |
| K-Nearest Neighbors | Majority vote of the 5 closest samples | No training, captures local patterns | Slow to predict on large data, sensitive to irrelevant features |
| Decision Tree | A tree of if/else rules | Very interpretable | Overfits easily |
| Random Forest (200 trees) | Many decorrelated trees vote | Strong, robust default | Larger model, less interpretable |
| Extra Trees (200 trees) | Random Forest with random split thresholds | Often faster, lower variance | Similar to Random Forest |
| Gradient Boosting | Sequential trees that fix previous errors | Very accurate on tabular data | Slower to train |
| Histogram Gradient Boosting | Gradient boosting on binned features | Fast on large datasets | Less effective on very small data |
| AdaBoost | Re-weights misclassified samples | Simple boosting | Sensitive to noise and outliers |
| Support Vector Machine (RBF) | Maximum-margin non-linear boundary | Effective in high dimensions | Scales poorly beyond ~20k rows |
| Naive Bayes (Gaussian) | Bayes' theorem with independent features | Extremely fast | Independence assumption rarely holds |
| Linear Discriminant Analysis | Gaussian classes with a shared covariance | Fast, works with few samples | Linear boundaries only |
| Neural Network (MLP, 100 units) | Small fully connected network | Learns non-linear patterns | Needs more data, less interpretable |
| XGBoost *(optional)* | Optimized gradient boosting | State of the art on tabular data | Extra dependency |
| LightGBM *(optional)* | Leaf-wise gradient boosting | Very fast | Extra dependency, can overfit small data |

## Regression algorithms

| Algorithm | Idea |
|---|---|
| Linear Regression | Ordinary least squares |
| Ridge Regression | Linear regression with L2 penalty (shrinks coefficients) |
| Lasso Regression | Linear regression with L1 penalty (can set coefficients to zero) |
| Elastic Net | Combination of L1 and L2 penalties |
| K-Nearest Neighbors | Average target of the 5 closest samples |
| Decision Tree | A tree of if/else rules |
| Random Forest | Average of 200 decorrelated trees |
| Extra Trees | Average of 200 extra-randomized trees |
| Gradient Boosting | Sequential trees fitting residual errors |
| Histogram Gradient Boosting | Fast binned gradient boosting |
| AdaBoost | Boosting focused on samples with large errors |
| Support Vector Machine (SVR) | Fits a tube of tolerance around the data (RBF kernel) |
| Neural Network (MLP) | Small fully connected network |
| XGBoost / LightGBM *(optional)* | Optimized gradient boosting libraries |

## Evaluation protocol

1. Rows with an empty target are removed.
2. The data is split into a **training set** (default 80%) and a **test set** (20%). For classification, the split is stratified so each class keeps its proportion.
3. **k-fold cross-validation** (default k = 5, stratified for classification) is run on the training set with the primary metric. This estimates how stable the algorithm is.
4. The algorithm is trained on the full training set and evaluated **once** on the test set. All metrics in the leaderboard come from the test set.

Preprocessing is part of the model pipeline, so it is fitted only on training data in every fold. This prevents information from the test data leaking into training.

## Classification metrics

| Metric | Meaning | Range |
|---|---|---|
| **F1 (weighted)** *(primary)* | Harmonic mean of precision and recall per class, weighted by class size | 0 to 1, higher is better |
| Accuracy | Fraction of correct predictions | 0 to 1, higher is better |
| Balanced accuracy | Average recall per class; not inflated by a dominant class | 0 to 1, higher is better |
| Precision (weighted) | Of the samples predicted as a class, how many really are | 0 to 1, higher is better |
| Recall (weighted) | Of the samples of a class, how many were found | 0 to 1, higher is better |
| ROC AUC | Probability that the model ranks a random positive above a random negative (one-vs-rest, weighted for multiclass) | 0.5 (random) to 1, higher is better |

Why F1 instead of accuracy? With unbalanced data (e.g. 95% "no", 5% "yes") a model that always says "no" reaches 95% accuracy while being useless. F1 penalizes that.

## Regression metrics

| Metric | Meaning | Better |
|---|---|---|
| **R²** *(primary)* | Share of the target's variance explained by the model. 1 is perfect, 0 is as good as predicting the mean, negative is worse than the mean | Higher |
| RMSE | Root mean squared error, in the target's unit; penalizes large errors | Lower |
| MAE | Mean absolute error, in the target's unit | Lower |
| Median absolute error | Typical error, robust to outliers | Lower |

## Feature importance

- **Tree-based models** (Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM) report how much each feature reduces impurity.
- **Linear models** (Logistic, Linear, Ridge, Lasso, Elastic Net, LDA) use the absolute value of their coefficients. Since numeric features are standardized, the coefficients are comparable.
- Values are normalized to add up to 100%. One-hot encoded columns appear as `column_value`.
- Other algorithms (KNN, SVM with RBF kernel, Naive Bayes, MLP, Histogram Gradient Boosting) don't expose importances.

Importance shows what the model *uses*, not causality.
