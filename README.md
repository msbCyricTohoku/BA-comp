# tse_ba_comp

[![PyPI version](https://badge.fury.io/py/tse-ba-comp.svg)](https://badge.fury.io/py/tse-ba-comp)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

**tse_ba_comp** stands for Tse Biological Age Comparator (named after Prof. Gary Tse's Research Group) is a robust, easy-to-use Python library for estimating Biological Age (BA) from clinical biomarkers.

Developed by **Mehrdad S. Beni** & **Gary Tse**, this package evaluates and ensembles classical mathematical approaches against modern Machine Learning models to provide highly accurate, cross-validated age estimations.

### Key Features
* **Classical Models:** Fast, vectorized implementations of the Klemera-Doubal Method (KDM) and PCA-Dubina.
* **Machine Learning:** Automated pipelines for Elastic Net, Random Forest, and XGBoost.
* **Smart Ensembling:** Automatically combine predictions using Mean or Median strategies to smooth out variance.
* **Automated Preprocessing:** Handles train/test splitting, scaling, and missing data imputation safely to prevent data leakage. 
* **Built-in Visualization:** Generates standardized, publication-ready scatter plots of Biological Age vs. Chronological Age.

---

## Installation

Install directly from PyPI:

```bash
pip install tse-ba-comp
```

---

## Examples & Usage

### Example 1: Quick Start (Default & Shortest)
The easiest way to use the library is to pass a CSV dataset file path directly to the `run` function. The library uses highly optimized defaults out-of-the-box.

```python
import tse_ba_comp

my_biomarkers = ["albumin", "alp", "bun", "creat", "hba1c", "glucose", "sbp"]

results = tse_ba_comp.run(
    data="nhanes4_model_input.csv",
    biomarkers=my_biomarkers,
    out="my_results_folder"  # Automatically saves plots and CSVs here
)

print(results["metrics"])
```

---

### Example 2: ML Auto Tuning & Control
Pass a list of models to `ml` and set `tune=True` to let the library automatically grid search the best hyperparameters for those specific models.

```python
import tse_ba_comp

my_biomarkers = ["albumin", "alp", "bun", "creat", "hba1c", "glucose", "sbp"]

results = tse_ba_comp.run(
    data="nhanes4_model_input.csv",
    biomarkers=my_biomarkers,
    impute="knn",              # switch imputation to knn
    seed=101,                  # fix random seed for reproducibility
    ml=["rf", "xgb"],          # only run Random Forest & XGBoost
    tune=True,                 # trigger automated Grid Search
    out="advanced_results"
)

print(results["metrics"])
```

---

### Example 3: Comprehensive Pipeline (All parameters that our lib offer)
For full programmatic control, you can utilize every argument the library offers and pass an exact dictionary of fixed hyperparameters to bypass the automated grid search. 

```python
import tse_ba_comp

my_biomarkers = ["albumin", "alp", "bun", "creat", "hba1c", "glucose", "sbp"]

# Define exact parameters to pass to scikit-learn / XGBoost
custom_ml_settings = {
    "en": {"l1_ratio": 0.7},
    "rf": {"n_estimators": 200, "max_depth": 10},
    "xgb": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4}
}

results = tse_ba_comp.run(
    data="nhanes4_model_input.csv",
    biomarkers=my_biomarkers,
    age="age",                      # target column name
    impute="mean",                  # imputation strategy
    test_size=None,                 # set to None to evaluate on the ENTIRE dataset without splitting
    seed=42,                        # random seed
    kdm=True,                       # enable Klemera-Doubal Method
    kdm_s2_floor=0.05,              # tweak KDM variance floor
    pca=True,                       # enable PCA-Dubina
    ml=custom_ml_settings,          # apply custom ML settings directly
    tune=False,                     # skip grid search
    cv_folds=5,                     # cross-validation folds
    ensemble="mean",                # use arithmetic mean for ensemble
    out="all_keywords_results"
)

print(results["metrics"])
```

---

## Complete API Reference

Below is the complete list of arguments accepted by the `run` function.

### Core Data Settings
* `data` *(str or pandas.DataFrame)*: Path to your CSV file, or a loaded Pandas DataFrame.
* `biomarkers` *(list of str)*: List of column names representing the biomarkers to be used.
* `age` *(str)*: The column name containing chronological age. Default: `"age"`.

### Processing & Splitting
* `impute` *(str)*: How to handle missing data. Options: `"median"`, `"mean"`, `"zero"`, `"knn"`. Default: `"median"`.
* `test_size` *(float or None)*: The fraction of the dataset to hold out for testing and evaluation. **Set to `None` to bypass splitting and train/evaluate on the entire dataset** (traditional for classical BA methods). Default: `0.2`.
* `seed` *(int)*: Random seed to ensure reproducible train/test splits. Default: `42`.

### Classical Model Toggles
* `kdm` *(bool)*: Toggle the Klemera-Doubal Method. Default: `True`.
* `kdm_s2_floor` *(float)*: Minimum variance floor for KDM calculations to prevent division by near-zero. Default: `0.1`.
* `pca` *(bool)*: Toggle the PCA-Dubina method. Default: `True`.

### Machine Learning Settings
* `ml` *(bool, list, or dict)*: Controls the ML models. 
  * `True`: Runs Elastic Net (`"en"`), Random Forest (`"rf"`), and XGBoost (`"xgb"`) with defaults.
  * `False`: Skips ML entirely.
  * `list`: E.g., `["rf", "en"]` runs only specific models.
  * `dict`: Passes explicit `kwargs` to the underlying model constructors.
* `tune` *(bool)*: If `True`, runs a `GridSearchCV` over a predefined parameter grid for the active ML models. Default: `False`.
* `cv_folds` *(int)*: Number of cross-validation folds used during training/tuning. Default: `5`.

### Ensemble & Outputs
* `ensemble` *(str or None)*: How to combine the model predictions. Options: `"median"`, `"mean"`, or `None` (to skip). Default: `"median"`.
* `out` *(str or None)*: Directory path to save the generated scatter plots and prediction CSVs. If `None`, no files are saved to the disk.

---

## Outputs

The `run` function returns a dictionary with two keys:

1. `results["metrics"]`: A Pandas DataFrame containing the Pearson *r*, R², RMSE, and MAE for all executed models. Evaluated strictly on the test set (or the full dataset if `test_size=None`).
2. `results["predictions"]`: A Pandas DataFrame mapping the Chronological Age to the estimated Biological Ages for every patient in the test set (or full dataset).

## Developers

Developed by Dr. Mehrdad S. Beni and Prof. Gary Tse at Hong Kong Metropolitan University, 2026.
